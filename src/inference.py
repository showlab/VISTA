"""Minimal inference script for Wan22Video style transfer."""

import os
import argparse
import copy
import re
import warnings
from pickle import UnpicklingError

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from decord import VideoReader
from omegaconf import OmegaConf
from typing import Optional

import pytorch_lightning as L

from transformers import AutoTokenizer, UMT5EncoderModel, AutoImageProcessor, SiglipVisionModel
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from safetensors.torch import load_file as safetensors_load

from models.wan2.transformer_wan import WanTransformer3DModel
from models.wan2.custom_pipeline import CustomWanPipeline as WanPipeline
from models.wan2.attn_process import ConditionAttnProcessor2_0
from models.wan2.ipa_adapter import WanIPAProjector
from datasets.custom_dataset import CustomDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _is_placeholder_path(value: Optional[str]) -> bool:
    if value in (None, "", "null", "None"):
        return False
    normalized = str(value).strip().lower()
    return (
        normalized.startswith("/path/to/")
        or normalized.startswith("path/to/")
        or normalized.startswith("./path/to/")
    )


def _resolve_local_path(value: Optional[str], project_root: str) -> Optional[str]:
    if value in (None, "", "null", "None"):
        return None

    path = os.path.expanduser(str(value).strip())
    if os.path.isabs(path):
        return path

    abs_cwd = os.path.abspath(path)
    if os.path.exists(abs_cwd):
        return abs_cwd

    abs_project = os.path.join(project_root, path)
    if os.path.exists(abs_project):
        return abs_project

    return path


def _resolve_runtime_assets(opt, args_dict) -> None:
    """Allow README placeholder paths while preferring bundled project assets."""
    project_root = _project_root()
    bundled_ckpt = os.path.join(project_root, "checkpoints", "20000.ckpt")
    bundled_ckpt_exists = os.path.isfile(bundled_ckpt)

    ckpt_arg = args_dict.get("ckpt_path", "")
    if _is_placeholder_path(ckpt_arg) and bundled_ckpt_exists:
        opt.ckpt_path = bundled_ckpt
        print(f"[Infer] Using bundled checkpoint for --ckpt_path: {bundled_ckpt}")
    else:
        opt.ckpt_path = _resolve_local_path(ckpt_arg, project_root)

    ipa_arg = args_dict.get("ipa_checkpoint", "")
    if _is_placeholder_path(ipa_arg) and bundled_ckpt_exists:
        opt.ipa_checkpoint = bundled_ckpt
        print(f"[Infer] Using bundled checkpoint for --ipa-checkpoint: {bundled_ckpt}")
    else:
        opt.ipa_checkpoint = _resolve_local_path(ipa_arg, project_root)

    siglip_arg = args_dict.get("siglip_model", "")
    if _is_placeholder_path(siglip_arg):
        opt.siglip_model = "google/siglip-so400m-patch14-384"
        print("[Infer] Using default SigLIP model id: google/siglip-so400m-patch14-384")
    elif siglip_arg in (None, "", "null", "None"):
        opt.siglip_model = None
    else:
        opt.siglip_model = siglip_arg


# ---------------------------------------------------------------------------
# Dataset for inference-only mode (no ground truth needed)
# ---------------------------------------------------------------------------

class InferenceDataset(Dataset):
    """Loads input videos with optional reference images and text captions."""

    def __init__(
        self,
        video_root: str,
        first_root: str,
        height: int,
        width: int,
        sample_n_frames: int,
        caption_root: str = None,
        caption_ext: str = ".txt",
        training_len: int = -1,
    ) -> None:
        self.training_len = training_len
        self.video_root = video_root
        self.caption_root = caption_root or video_root
        self.caption_ext = caption_ext
        self.sample_n_frames = sample_n_frames

        def _natural_key(name: str):
            parts = re.split(r"(\d+)", name)
            return [int(p) if p.isdigit() else p.lower() for p in parts]

        img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        if first_root and os.path.isdir(first_root):
            first_list_all = sorted(os.listdir(first_root), key=_natural_key)
            first_list = [x for x in first_list_all if x.lower().endswith(img_exts)]
            if len(first_list) > 0:
                self.first_root = first_root
                self.first_paths = [os.path.join(first_root, x) for x in first_list]
                self.use_first = True
            else:
                self.first_root = None
                self.first_paths = []
                self.use_first = False
        else:
            self.first_root = None
            self.first_paths = []
            self.use_first = False

        video_exts = (".mp4", ".avi", ".mov", ".mkv")
        video_list = sorted(
            [x for x in os.listdir(self.video_root) if x.lower().endswith(video_exts)],
            key=_natural_key,
        )
        self.video_paths = [os.path.join(self.video_root, v) for v in video_list]
        if len(self.video_paths) == 0:
            raise RuntimeError(f"No videos found under video_root={self.video_root}")

        self.height = height
        self.width = width
        self.train_video_transforms = transforms.Compose(
            [
                transforms.CenterCrop((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.len_videos = len(self.video_paths)
        self.len_firsts = len(self.first_paths)

        print(
            f"InferenceDataset: video_root: {video_root}, "
            f"caption_root: {self.caption_root}, first_root: {self.first_root or ''}"
        )

    def __len__(self) -> int:
        if self.training_len != -1:
            return self.training_len
        if self.use_first:
            return min(self.len_videos, self.len_firsts)
        return self.len_videos

    def _load_caption(self, stem: str) -> str:
        cap_path = os.path.join(self.caption_root, stem + self.caption_ext)
        if os.path.exists(cap_path) and os.path.isfile(cap_path):
            try:
                with open(cap_path, "r", encoding="utf-8") as handle:
                    return handle.read().strip()
            except Exception:
                return ""
        return ""

    def __getitem__(self, index: int) -> dict:
        if self.use_first:
            min_len = min(self.len_videos, self.len_firsts)
        else:
            min_len = self.len_videos

        if min_len <= 0:
            raise RuntimeError(
                f"No valid samples: len_videos={self.len_videos}, len_firsts={self.len_firsts}."
            )

        index = index % min_len

        video_path = self.video_paths[index]
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)

        first_frame = None
        if self.use_first:
            first_frame_path = self.first_paths[index]
            first_frame = Image.open(first_frame_path).convert("RGB")
            first_frame = self.train_video_transforms(first_frame)

        assert self.sample_n_frames <= video_length, "sample_n_frames > video length"

        stride = 1
        available = video_length - (self.sample_n_frames - 1) * stride
        available = max(available, 1)
        if available <= 4:
            start_index = 0
        else:
            start_index = np.random.randint(0, available - 3)
        frame_indices = start_index + np.arange(self.sample_n_frames) * stride

        video = video_reader.get_batch(frame_indices).asnumpy()
        video = [Image.fromarray(frame) for frame in video]
        pixel_values = [self.train_video_transforms(frame) for frame in video]
        pixel_values = torch.stack(pixel_values)

        stem = os.path.splitext(os.path.basename(video_path))[0]
        prompt = self._load_caption(stem)

        sample = {
            "pixel_values": pixel_values.permute(1, 0, 2, 3),
            "prompts": prompt,
        }
        if self.use_first:
            sample["first_frames"] = first_frame

        return sample


# ---------------------------------------------------------------------------
# Inference system
# ---------------------------------------------------------------------------

class StyleTransferInference(torch.nn.Module):
    """Inference system for video style transfer with IPA and LoRA."""

    def __init__(self, opt):
        super().__init__()
        self.hparams = opt
        self.is_configured = False
        self.siglip_model = None
        self.siglip_processor = None
        self.ipa_projector = None
        self.ipa_enabled = False
        self.ipa_force_zero = False
        self.ipa_num_tokens = 0

    def configure_model(self):
        if self.is_configured:
            return
        self.is_configured = True

        model_id = self.hparams.model_id

        # Tokenizer & text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )

        # VAE
        self.vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.bfloat16
        )

        # Sampling scheduler
        base_sampler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        self.sample_scheduler = UniPCMultistepScheduler.from_config(
            base_sampler.config, flow_shift=5
        )

        # Transformer
        self.transformer = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )

        # Freeze all backbone parameters
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)

        # Latent normalization parameters
        self.register_buffer(
            'latents_mean',
            torch.tensor(self.vae.config.latents_mean).to(torch.bfloat16).view(1, self.vae.config.z_dim, 1, 1, 1),
            persistent=False
        )
        self.register_buffer(
            'latents_std',
            torch.tensor(self.vae.config.latents_std).to(torch.bfloat16).view(1, self.vae.config.z_dim, 1, 1, 1),
            persistent=False
        )

        self.vae_config = self.vae.config
        self.model_config = self.transformer.module.config if hasattr(self.transformer, "module") else self.transformer.config

        # LoRA adapter
        self.using_lora = bool(self.hparams.use_lora)
        if self.using_lora:
            from peft import LoraConfig
            transformer_lora_config = LoraConfig(
                r=96, lora_alpha=96, init_lora_weights=True,
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.transformer.add_adapter(transformer_lora_config)

        # Condition attention processor
        for blk in self.transformer.blocks:
            blk.attn1.set_processor(ConditionAttnProcessor2_0())

        # Extra patch embedding for condition input
        self.transformer.patch_embedding_extra = copy.deepcopy(
            self.transformer.patch_embedding
        ).requires_grad_(True)

    @torch.no_grad()
    def encode_prompt(self, prompt_list, device):
        max_sequence_length = 512
        text_inputs = self.tokenizer(
            prompt_list,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids.to(device), text_inputs.attention_mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        text_embeds = self.text_encoder(ids, mask).last_hidden_state
        text_embeds = [u[:v] for u, v in zip(text_embeds, seq_lens)]
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in text_embeds], dim=0
        )
        return text_embeds

    def _load_lora_from_ckpt(self, ckpt_path, device):
        """Load LoRA / patch_embedding_extra weights from a training checkpoint."""
        if ckpt_path in (None, "", "None", "null"):
            print("[Infer] No ckpt_path provided. Skip loading LoRA.")
            return

        print(f"[Infer] Loading checkpoint: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        except (TypeError, UnpicklingError):
            print("[Infer] weights_only load failed; falling back to full torch.load.")
            ckpt = torch.load(ckpt_path, map_location=device)
        if "state_dict" not in ckpt:
            print("[Infer] checkpoint has no 'state_dict' key; skip.")
            return

        sd_all = ckpt["state_dict"]

        # LoRA / attn processor
        if "transformer_processor" in sd_all:
            sd = sd_all["transformer_processor"]
            cur = self.transformer.state_dict()
            filtered = {k: v for k, v in sd.items() if (k in cur and cur[k].shape == v.shape)}
            skipped = [k for k in sd.keys() if k not in filtered]
            print(f"[Infer][LoRA] Load {len(filtered)}/{len(sd)} keys. Skipped {len(skipped)} mismatched keys.")
            self.transformer.load_state_dict(filtered, strict=False)
        else:
            print("[Infer] 'transformer_processor' not found in ckpt.state_dict; skip LoRA.")

        # patch_embedding_extra
        if "patch_embedding_extra" in sd_all:
            sd2 = sd_all["patch_embedding_extra"]
            cur2 = self.transformer.state_dict()
            filtered2 = {k: v for k, v in sd2.items() if (k in cur2 and cur2[k].shape == v.shape)}
            skipped2 = [k for k in sd2.keys() if k not in filtered2]
            print(f"[Infer][patch_embedding_extra] Load {len(filtered2)}/{len(sd2)} keys. Skipped {len(skipped2)} mismatched keys.")
            self.transformer.load_state_dict(filtered2, strict=False)
        else:
            print("[Infer] 'patch_embedding_extra' not found in ckpt.state_dict; skip.")

    def _tensor_to_pil(self, ref_tensor: torch.Tensor):
        tensor = ref_tensor.detach().cpu().clamp(-1, 1)
        tensor = (tensor + 1.0) * 0.5
        return to_pil_image(tensor)

    def _setup_ipa(self, device: torch.device) -> None:
        if self.ipa_enabled:
            return
        ipa_ckpt = getattr(self.hparams, "ipa_checkpoint", None)
        if ipa_ckpt in (None, "", "None", "null"):
            self.ipa_enabled = False
            return

        siglip_name = getattr(self.hparams, "siglip_model", None)
        if siglip_name in (None, "", "None", "null"):
            raise ValueError("siglip_model must be provided when using IPA checkpoint")

        self.ipa_num_tokens = int(getattr(self.hparams, "ipa_num_tokens", 128))
        ipa_alpha = float(getattr(self.hparams, "ipa_alpha", 0.5))
        self.ipa_force_zero = bool(getattr(self.hparams, "ipa_zero", False))

        self.transformer.enable_ipa(self.ipa_num_tokens, ipa_alpha)

        hidden_size = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim

        self.siglip_processor = AutoImageProcessor.from_pretrained(siglip_name)
        self.siglip_model = SiglipVisionModel.from_pretrained(siglip_name)
        self.siglip_model.to(device)
        self.siglip_model.eval()

        embed_dim = getattr(self.siglip_model.config, "hidden_size", None)
        if embed_dim is None:
            embed_dim = getattr(self.siglip_model.config, "vision_embed_dim", None)
        if embed_dim is None:
            embed_dim = getattr(self.siglip_model.config, "projection_dim", 512)

        self.ipa_projector = WanIPAProjector(embed_dim, hidden_size, self.ipa_num_tokens).to(device=device, dtype=self.transformer.dtype)
        self._load_ipa_checkpoint(ipa_ckpt, device)
        self.ipa_enabled = True

    def _load_ipa_checkpoint(self, ckpt_path: str, device: torch.device) -> None:
        load_device = device
        if isinstance(load_device, torch.device):
            if load_device.type == "cuda":
                index = load_device.index
                if index is None:
                    index = torch.cuda.current_device()
                load_device = f"cuda:{index}"
            else:
                load_device = load_device.type
        elif load_device is None:
            load_device = "cpu"

        if ckpt_path.endswith(".safetensors"):
            state_dict = safetensors_load(ckpt_path, device=load_device)
        else:
            raw = torch.load(ckpt_path, map_location=device)
            state_dict = raw.get("state_dict", raw)

        def _strip(name: str, prefixes):
            for prefix in prefixes:
                if name.startswith(prefix):
                    return name[len(prefix):]
            return None

        projector_prefixes = (
            "pipe.image_proj.", "image_proj.",
            "module.pipe.image_proj.", "module.image_proj.",
            "model.pipe.image_proj.", "model.image_proj.",
        )
        projector_state = {}
        if "ipa_projector" in state_dict:
            projector_state.update(state_dict["ipa_projector"])

        dit_prefixes = (
            "pipe.dit.", "dit.",
            "module.pipe.dit.", "module.dit.",
            "model.pipe.dit.", "model.dit.",
        )
        transformer_state = {}
        if "ipa_transformer" in state_dict:
            transformer_state.update(state_dict["ipa_transformer"])

        for key, value in state_dict.items():
            proj_key = _strip(key, projector_prefixes)
            if proj_key is not None:
                projector_state[proj_key] = value
                continue
            dit_key = _strip(key, dit_prefixes)
            if dit_key is not None and ("ipa_self_processor" in dit_key or "ipa_cross_processor" in dit_key):
                transformer_state[dit_key] = value

        if projector_state:
            cur_proj = self.ipa_projector.state_dict()
            proj_filtered = {k: v for k, v in projector_state.items() if (k in cur_proj and cur_proj[k].shape == v.shape)}
            skipped = [k for k in projector_state if k not in proj_filtered]
            print(f"[Infer][IPA] Projector load {len(proj_filtered)}/{len(projector_state)} keys. Skipped {len(skipped)}")
            self.ipa_projector.load_state_dict(proj_filtered, strict=False)

        if transformer_state:
            cur_tf = self.transformer.state_dict()
            tf_filtered = {k: v for k, v in transformer_state.items() if (k in cur_tf and cur_tf[k].shape == v.shape)}
            skipped_tf = [k for k in transformer_state if k not in tf_filtered]
            print(f"[Infer][IPA] Adapter load {len(tf_filtered)}/{len(transformer_state)} keys. Skipped {len(skipped_tf)}")
            self.transformer.load_state_dict(tf_filtered, strict=False)

    def _compute_ipa_tokens(self, ref_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.ipa_enabled:
            return None
        transformer_device = next(self.transformer.parameters()).device
        transformer_dtype = next(self.transformer.parameters()).dtype

        if self.ipa_force_zero:
            batch = 1 if ref_tensor is None else (ref_tensor.shape[0] if ref_tensor.dim() == 4 else 1)
            hidden_size = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
            return torch.zeros(batch, self.ipa_num_tokens, hidden_size, device=transformer_device, dtype=transformer_dtype)

        hidden_size = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim

        if ref_tensor is None:
            return torch.zeros(1, self.ipa_num_tokens, hidden_size, device=transformer_device, dtype=transformer_dtype)

        if ref_tensor.dim() == 3:
            ref_tensor = ref_tensor.unsqueeze(0)

        images = [self._tensor_to_pil(tensor) for tensor in ref_tensor]
        inputs = self.siglip_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.siglip_model.device)
        with torch.no_grad():
            outputs = self.siglip_model(pixel_values=pixel_values)
            if getattr(outputs, "pooler_output", None) is not None and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state.mean(dim=1)

        embeddings = embeddings.to(self.ipa_projector.proj[0].weight.device)
        tokens = self.ipa_projector(embeddings)
        tokens = tokens.to(device=transformer_device, dtype=transformer_dtype)
        return tokens

    def _resolve_sample_id(self, dataset, index: int) -> str:
        if hasattr(dataset, "video_paths2") and index < len(dataset.video_paths2):
            src_path = dataset.video_paths2[index]
        elif hasattr(dataset, "video_paths") and index < len(dataset.video_paths):
            src_path = dataset.video_paths[index]
        else:
            return f"sample_{index:06d}"
        return os.path.splitext(os.path.basename(src_path))[0]

    @torch.no_grad()
    def run_infer(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configure_model()
        self.to(device)
        self._setup_ipa(device)

        ds_cfg = self.hparams.dataset
        is_one2three = bool(getattr(ds_cfg, "is_one2three", False))

        if is_one2three:
            caption_root = getattr(ds_cfg, "caption_root", None)
            if caption_root in (None, "", "null", "None"):
                caption_root = None
            if caption_root is None:
                video_root2 = getattr(ds_cfg, "video_root2", None)
                if video_root2 and os.path.isdir(video_root2):
                    caption_root = video_root2
            dataset = InferenceDataset(
                video_root=ds_cfg.video_root,
                first_root=ds_cfg.first_root,
                height=ds_cfg.height,
                width=ds_cfg.width,
                sample_n_frames=ds_cfg.sample_n_frames,
                caption_root=caption_root or ds_cfg.video_root,
                caption_ext=getattr(ds_cfg, "caption_ext", ".txt"),
            )
        else:
            dataset = CustomDataset(
                video_root=ds_cfg.video_root,
                video_root2=ds_cfg.video_root2,
                first_root=ds_cfg.first_root,
                height=ds_cfg.height,
                width=ds_cfg.width,
                sample_n_frames=ds_cfg.sample_n_frames,
                is_one2three=ds_cfg.is_one2three,
            )

        data_loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=int(getattr(ds_cfg, "num_workers", 0)),
            drop_last=False,
            pin_memory=bool(getattr(ds_cfg, "pin_memory", False)),
            shuffle=False,
        )

        start_index = int(getattr(self.hparams, "start_index", 0) or 0)
        if start_index < 0:
            start_index = 0
        if start_index:
            print(f"[Infer] Start from index: {start_index}")
        max_samples = int(getattr(self.hparams, "max_samples", -1) or -1)
        processed = 0

        if self.using_lora:
            ckpt_path = getattr(self.hparams, "ckpt_path", None)
            self._load_lora_from_ckpt(ckpt_path, device)

        pipeline = WanPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            scheduler=self.sample_scheduler,
        )

        save_root = os.path.join(
            self.hparams.output_root,
            self.hparams.experiment_name,
            "infer_samples",
        )
        os.makedirs(save_root, exist_ok=True)
        print(f"[Infer] Save to: {save_root}")

        fps = int(getattr(ds_cfg, "fps", 24))
        guidance_scale = float(getattr(self.hparams, "guidance_scale", 5.0))
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx < start_index:
                continue
            if max_samples > 0 and processed >= max_samples:
                print(f"[Infer] Reached max_samples={max_samples}. Stop inference.")
                break
            model_input = batch["pixel_values"].to(device)   # [1, C, F, H, W]
            model_input2 = batch.get("pixel_values2", None)
            if model_input2 is not None:
                model_input2 = model_input2.to(device)
            elif not is_one2three:
                raise RuntimeError("video_root2 videos are required when is_one2three is False.")

            first_frames_full = batch.get("first_frames", None)
            ref_image = None
            if first_frames_full is not None:
                ref_image = first_frames_full.clone()
                first_frames = first_frames_full.to(device=device, dtype=self.vae.dtype).unsqueeze(2)
            else:
                first_frames = None

            if self.ipa_enabled:
                ipa_tokens = self._compute_ipa_tokens(ref_image if ref_image is not None else None)
                self.transformer.set_ipa_tokens(ipa_tokens)

            model_input = model_input.to(dtype=self.vae.dtype)
            if model_input2 is not None:
                model_input2 = model_input2.to(dtype=self.vae.dtype)
            prompts = batch["prompts"]

            if is_one2three:
                model_input_lat = self.vae.encode(model_input).latent_dist.sample()
                model_input_lat = (model_input_lat - self.latents_mean) / self.latents_std

                if first_frames is None:
                    first_frames_lat = model_input_lat[:, :, :1].detach() * 0
                else:
                    first_frames_lat = self.vae.encode(first_frames).latent_dist.sample()
                    first_frames_lat = (first_frames_lat - self.latents_mean) / self.latents_std

                attention_kwargs = {
                    "encoder_contion_states": model_input_lat,
                    "encoder_first_states": first_frames_lat,
                }
            else:
                model_input2_lat = self.vae.encode(model_input2).latent_dist.sample()
                model_input2_lat = (model_input2_lat - self.latents_mean) / self.latents_std
                attention_kwargs = {
                    "encoder_contion_states": model_input2_lat,
                }

            out = pipeline(
                prompt=prompts,
                height=ds_cfg.height,
                width=ds_cfg.width,
                num_frames=ds_cfg.sample_n_frames,
                guidance_scale=guidance_scale,
                attention_kwargs=attention_kwargs,
            )
            video_generate = out.frames[0]

            sample_id = self._resolve_sample_id(dataset, batch_idx)
            save_path = os.path.join(save_root, f"{sample_id}.mp4")
            export_to_video(video_generate, output_video_path=save_path, fps=fps)
            print(f"[Infer] Saved: {save_path}")
            processed += 1

            if self.ipa_enabled:
                self.transformer.clear_ipa_tokens()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Wan22Video Inference")
    parser.add_argument("--config", required=True, help="Path to the yaml config file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_path", type=str, default="", help="Path to a .ckpt with LoRA/extra weights")
    parser.add_argument("--start-index", type=int, default=0, help="Start from dataset index (skip earlier samples)")
    parser.add_argument("--siglip-model", type=str, default="", help="SigLIP vision encoder for IPA tokens")
    parser.add_argument("--ipa-checkpoint", type=str, default="", help="Path to IPA adapter checkpoint")
    parser.add_argument("--ipa-num-tokens", type=int, default=128, help="Number of IPA tokens")
    parser.add_argument("--ipa-alpha", type=float, default=0.5, help="Blending factor for IPA contributions")
    parser.add_argument("--ipa-zero", action="store_true", help="Force zero IPA tokens (debug)")
    parser.add_argument("--max-samples", type=int, default=-1, help="Maximum number of samples to run (-1 for all)")
    args, extras = parser.parse_known_args()
    args_dict = vars(args)

    opt = OmegaConf.merge(
        OmegaConf.load(args_dict["config"]),
        OmegaConf.from_cli(extras),
        OmegaConf.create(args_dict),
        OmegaConf.create({"num_nodes": int(os.environ.get("NUM_NODES", 1))}),
        OmegaConf.create({"num_gpus": int(torch.cuda.device_count())}),
    )

    _resolve_runtime_assets(opt, args_dict)

    L.seed_everything(opt.seed)

    system = StyleTransferInference(opt)
    system.run_infer()


if __name__ == "__main__":
    main()
