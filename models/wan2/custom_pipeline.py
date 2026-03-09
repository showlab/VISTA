from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import replace_example_docstring
import torch
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.utils import is_torch_xla_available

from dataclasses import dataclass
from diffusers.utils import BaseOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

EXAMPLE_DOC_STRING = ""


class CustomWanPipeline(WanPipeline):
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        The call function to the pipeline for generation.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`
        """
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2

        attention_kwargs = attention_kwargs or {}

        if "encoder_contion_states" not in attention_kwargs:
            for alt in ("encoder_condition_states", "encoder_cond_states", "encoder_first_states"):
                if alt in attention_kwargs and attention_kwargs[alt] is not None:
                    attention_kwargs["encoder_contion_states"] = attention_kwargs.pop(alt)
                    break

        if "encoder_first_states" in attention_kwargs:
            attention_kwargs.pop("encoder_first_states", None)

        cond_states = attention_kwargs.get("encoder_contion_states", None)
        if isinstance(cond_states, torch.Tensor):
            attention_kwargs["encoder_contion_states"] = cond_states.to(
                device=self._execution_device, dtype=self.transformer.dtype
            )

        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        expand_factor = getattr(self.config, "expand_timesteps_factor", None)
        if expand_factor is None:
            expand_factor = 2 if attention_kwargs.get("encoder_first_states", None) is not None else 1

        device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        def _scheduler_tensors_to_numpy(sched):
            import numpy as np
            _maybe_np = ["betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev", "sigmas", "timesteps"]
            for name in _maybe_np:
                if hasattr(sched, name):
                    val = getattr(sched, name)
                    if isinstance(val, torch.Tensor):
                        setattr(sched, name, val.detach().cpu().float().numpy())
            return sched

        _scheduler_tensors_to_numpy(self.scheduler)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                current_model = self.transformer
                current_guidance_scale = guidance_scale

                latent_model_input = latents.to(transformer_dtype)
                B, C, F, H, W = latent_model_input.shape
                device = latent_model_input.device

                with torch.no_grad():
                    q_tokens = self.transformer.patch_embedding(latent_model_input)
                    N = (q_tokens.flatten(2).transpose(1, 2)).shape[1]

                p_t, p_h, p_w = self.transformer.config.patch_size
                mask_grid = torch.ones((B, 1, F, H, W), device=device, dtype=transformer_dtype)[:, 0, ::p_t, ::p_h, ::p_w]
                kkk = (t.to(device=device, dtype=transformer_dtype).view(1, 1, 1, 1).expand(B, 1, 1, 1) * mask_grid).flatten(1)

                if kkk.shape[1] != N:
                    if kkk.shape[1] > N:
                        kkk = kkk[:, :N]
                    else:
                        kkk = torch.nn.functional.pad(kkk, (0, N - kkk.shape[1]))

                has_cond = attention_kwargs.get("encoder_contion_states", None) is not None
                t_embed = torch.cat([kkk, torch.zeros_like(kkk, dtype=kkk.dtype, device=kkk.device)], dim=-1) if has_cond else kkk
                assert t_embed.shape[1] == (2 * N if has_cond else N), \
                    f"t_embed tokens ({t_embed.shape[1]}) != expected ({2*N if has_cond else N})"

                with current_model.cache_context("cond"):
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=t_embed,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if self.do_classifier_free_guidance:
                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=t_embed,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)


@dataclass
class WanPipelineOutput(BaseOutput):
    r"""
    Output class for Wan pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs.
    """

    frames: torch.Tensor
