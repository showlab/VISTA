import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def _split_heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
    batch, length, hidden = tensor.shape
    head_dim = hidden // heads
    return tensor.view(batch, length, heads, head_dim).permute(0, 2, 1, 3)


def _merge_heads(tensor: torch.Tensor) -> torch.Tensor:
    batch, heads, length, head_dim = tensor.shape
    return tensor.permute(0, 2, 1, 3).reshape(batch, length, heads * head_dim)


def _apply_rotary_emb(
    tensor: torch.Tensor,
    rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    if rotary_emb is None:
        return tensor

    freqs_cos, freqs_sin = rotary_emb
    freqs_cos = freqs_cos.to(device=tensor.device, dtype=tensor.dtype)
    freqs_sin = freqs_sin.to(device=tensor.device, dtype=tensor.dtype)

    seq_len = tensor.shape[2]
    if freqs_cos.shape[-2] < seq_len:
        pad = seq_len - freqs_cos.shape[-2]
        pad_cos = freqs_cos[..., -1:, :].expand(*freqs_cos.shape[:-2], pad, freqs_cos.size(-1))
        pad_sin = freqs_sin[..., -1:, :].expand(*freqs_sin.shape[:-2], pad, freqs_sin.size(-1))
        freqs_cos = torch.cat([freqs_cos, pad_cos], dim=-2)
        freqs_sin = torch.cat([freqs_sin, pad_sin], dim=-2)
    elif freqs_cos.shape[-2] > seq_len:
        freqs_cos = freqs_cos[..., :seq_len, :]
        freqs_sin = freqs_sin[..., :seq_len, :]

    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]

    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]

    channel_dim = even.shape[-1]
    if cos.shape[-1] != channel_dim:
        if cos.shape[-1] < channel_dim:
            pad = channel_dim - cos.shape[-1]
            cos = torch.cat([cos, cos[..., -1:].expand(*cos.shape[:-1], pad)], dim=-1)
        else:
            cos = cos[..., :channel_dim]
    if sin.shape[-1] != channel_dim:
        if sin.shape[-1] < channel_dim:
            pad = channel_dim - sin.shape[-1]
            sin = torch.cat([sin, sin[..., -1:].expand(*sin.shape[:-1], pad)], dim=-1)
        else:
            sin = sin[..., :channel_dim]

    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos

    out = torch.empty_like(tensor)
    out[..., 0::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out


class WanIPAAttentionProcessor(nn.Module):
    """Light-weight attention adapter for IPA tokens."""

    def __init__(self, hidden_size: int, num_tokens: int, heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.heads = heads
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)

    def _project_kv(self, tokens: torch.Tensor, attn_module: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        key = self.to_k(tokens)
        if getattr(attn_module, "norm_k", None) is not None:
            key = attn_module.norm_k(key)
        value = self.to_v(tokens)
        key = _split_heads(key, self.heads)
        value = _split_heads(value, self.heads)
        return key, value

    def self_attention_delta(
        self,
        attn_module: nn.Module,
        query_input: torch.Tensor,
        rotary_emb_main: Optional[Tuple[torch.Tensor, torch.Tensor]],
        main_length: int,
        ipa_tokens: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if ipa_tokens is None:
            return None
        if main_length <= 0:
            return None

        dtype = query_input.dtype
        device = query_input.device

        tokens = ipa_tokens.to(device=device, dtype=dtype)

        query = attn_module.to_q(query_input[:, :main_length])
        if getattr(attn_module, "norm_q", None) is not None:
            query = attn_module.norm_q(query)
        query = _split_heads(query, attn_module.heads)
        query = _apply_rotary_emb(query, rotary_emb_main)

        key, value = self._project_kv(tokens, attn_module)

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_output = _merge_heads(attn_output)
        attn_output = attn_module.to_out[0](attn_output)
        attn_output = attn_module.to_out[1](attn_output)

        delta = query_input.new_zeros(query_input.shape)
        delta[:, :main_length, :] = attn_output
        return delta

    def cross_attention_delta(
        self,
        attn_module: nn.Module,
        query_input: torch.Tensor,
        ipa_tokens: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if ipa_tokens is None:
            return None

        dtype = query_input.dtype
        device = query_input.device

        tokens = ipa_tokens.to(device=device, dtype=dtype)

        query = attn_module.to_q(query_input)
        if getattr(attn_module, "norm_q", None) is not None:
            query = attn_module.norm_q(query)
        query = _split_heads(query, attn_module.heads)

        key, value = self._project_kv(tokens, attn_module)

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_output = _merge_heads(attn_output)
        attn_output = attn_module.to_out[0](attn_output)
        attn_output = attn_module.to_out[1](attn_output)
        return attn_output


class WanIPAProjector(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int, num_tokens: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, hidden_size * num_tokens),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        target_device = self.proj[0].weight.device
        target_dtype = self.proj[0].weight.dtype
        x = self.proj(embeddings.to(device=target_device, dtype=target_dtype))
        x = x.view(-1, self.num_tokens, self.hidden_size)
        x = self.norm(x.to(dtype=self.norm.weight.dtype))
        return x
