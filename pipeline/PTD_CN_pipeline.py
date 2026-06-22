"""Combined ControlNet + PTDiffusion inference pipeline.

Inherits the single-CN pipeline (`WanTextToVideoControlnetPipeline`) and adds a
per-step spatial-FFT phase substitution that pulls the phase spectrum of the
current main latent toward a face-derived reference latent.

The reference latents are **precomputed** and passed in as a single tensor of
shape `(num_inference_steps, 1, 16, T_lat, H_lat, W_lat)` in denoising-step
order — i.e. `ref_latents[0]` is the noisy end of the inversion trajectory
and `ref_latents[N-1]` is the clean face latent. Two compatible sources:
  * `inference/precompute_inverts.py` — Euler ODE inversion via the Wan
    transformer (one .pt per face containing the stacked trajectory).
  * `inference/run_WanIversionPipeline.py` — `deterministic_invert` (linear
    FlowMatch formula, no model forward). On disk it's 100 step_*.pt files
    per face; `run_inference.py:--invert_type=deterministic` loads them
    sorted and stacks into the same tensor shape before passing in.

The phase-substitute strength `alpha` follows the (direct, decayed) schedule
of the original PTDiffusion runner, modulated by a cosine-distance PI
controller (heuristic-3, always on) against the 100-entry
`GOOD_AVG_COSINE_DIST_LIST`.

Designed for `num_inference_steps == 100`. Removes teacache, prompt_embeds
input, and controlnet_latents input vs. the parent pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from PIL import Image

import wandb

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.utils import is_torch_xla_available

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.wan_t2v_controlnet_pipeline import (
    WanTextToVideoControlnetPipeline,
    prepare_controlnet_frames,
)
from pipeline.constants import GOOD_AVG_COSINE_DIST_LIST

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class WanPTDCNPipeline(WanTextToVideoControlnetPipeline):
    """ControlNet + PTDiffusion phase-substitute pipeline.

    Same constructor as `WanTextToVideoControlnetPipeline`. Overrides `__call__`
    and adds a `_phase_substitute` helper. Reference latents are passed in
    precomputed (see `inference/precompute_inverts.py`).
    """

    @staticmethod
    def _phase_substitute(x_dec: torch.Tensor, ref_latent: torch.Tensor,
                          alpha: float, step: int):
        """Spatial-FFT phase substitution on the last two dims.

        Mixes the phase of `x_dec` toward the phase of `ref_latent` with weight
        `alpha`, keeping the magnitude of `x_dec`. Logs five metrics to wandb
        at step=`step`. Returns (x_dec_new, ref_cosine_dist, energy_ratio).
        """
        # torch.fft.fft2 does not support BFloat16/Float16; cast to fp32 for
        # the FFT math and cast the result back to the caller's dtype.
        orig_dtype = x_dec.dtype
        x_dec_f32 = x_dec.to(torch.float32)
        ref_f32 = ref_latent.to(torch.float32)

        ref_latent_fft = torch.fft.fft2(ref_f32)
        ref_latent_angle = torch.angle(ref_latent_fft)

        x_dec_fft = torch.fft.fft2(x_dec_f32)
        x_dec_mag = torch.abs(x_dec_fft)
        x_dec_angle = torch.angle(x_dec_fft)
        mixed_angle = ref_latent_angle * alpha + (1 - alpha) * x_dec_angle

        energy_before = (x_dec_f32 ** 2).sum()

        x_dec_fft_new = x_dec_mag * torch.cos(mixed_angle) + \
            x_dec_mag * torch.sin(mixed_angle) * torch.complex(
                torch.zeros_like(x_dec_mag), torch.ones_like(x_dec_mag)
            )
        x_dec_new = torch.fft.ifft2(x_dec_fft_new).real

        energy_after = (x_dec_new ** 2).sum()
        energy_ratio = (energy_after / (energy_before + 1e-10)).item()

        ref_cosine_dist = torch.cos(ref_latent_angle - x_dec_angle).mean()

        # Only log when the caller has an active wandb run (training does;
        # the public inference.py runner doesn't). `wandb.run is not None`
        # is the canonical SDK check for "is a run currently active".
        if wandb.run is not None:
            wandb.log({
                "ref_cosine_dist_mean": ref_cosine_dist.item(),
                "alpha": alpha,
                "energy_ratio": energy_ratio,
                "energy_before": energy_before.item(),
                "energy_after": energy_after.item(),
            }, step=step)

        return x_dec_new.to(orig_dtype), ref_cosine_dist.item(), energy_ratio

    @torch.no_grad()
    def __call__(
        self,
        controlnet_frames: List[Image.Image] = None,
        ref_latents: torch.Tensor = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 512,
        width: int = 512,
        num_frames: int = 9,
        num_inference_steps: int = 100,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,

        controlnet_weight: float = 1.0,
        controlnet_guidance_start: float = 0.0,
        controlnet_guidance_end: Optional[float] = None,
        controlnet_stride: int = 3,

        # PTDiffusion phase-substitute schedule
        direct_transfer_steps: int = 45,
        decayed_transfer_steps: int = 22,
        initial_blending_coeff: float = 0.4,
        exponent: float = 0.5,
        Kp: float = 0.5,
        Ki: float = 0.2,
        max_blending_coeff_delta: float = 0.05,
    ):
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Input checks
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            None,
            None,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )
        assert num_inference_steps in (100, 101), (
            "WanPTDCNPipeline requires num_inference_steps in (100, 101) "
            "because heuristic-3 indexes GOOD_AVG_COSINE_DIST_LIST (101 "
            "entries) by step."
        )
        assert ref_latents is not None, (
            "ref_latents is required: a tensor of shape "
            "(num_inference_steps, 1, 16, T_lat, H_lat, W_lat) produced by "
            "inference/precompute_inverts.py."
        )
        assert ref_latents.shape[0] == num_inference_steps, (
            f"ref_latents has {ref_latents.shape[0]} steps but "
            f"num_inference_steps={num_inference_steps}."
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError("`prompt` must be provided as str or list[str].")

        # 3. Encode prompt (always raw-text path)
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Resolve controlnet_guidance_end: when None, auto-compute the step
        # fraction at which sigma first drops below boundary_ratio (the
        # dynamic-CN-end logic used to train beta-007_v3+).
        if controlnet_guidance_end is None:
            if self.config.boundary_ratio is None:
                controlnet_guidance_end = 1.0
            else:
                sigmas = self.scheduler.sigmas[:-1].detach().cpu()
                below = (sigmas < self.config.boundary_ratio).nonzero(as_tuple=False)
                if below.numel() == 0:
                    controlnet_guidance_end = 1.0
                else:
                    controlnet_guidance_end = int(below[0].item()) / num_inference_steps

        # 6. Prepare latent variables
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
            None,
        )
        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # Stage the precomputed ref_latents trajectory onto the same device as
        # `latents`. Kept in fp32; `_phase_substitute` does its own dtype
        # handling. ref_latents[i] is consumed at denoising step i.
        if ref_latents.shape[1:] != latents.shape:
            raise ValueError(
                f"ref_latents shape {tuple(ref_latents.shape)} is incompatible "
                f"with latents shape {tuple(latents.shape)}: dims after the "
                f"step axis must match exactly."
            )
        ref_latents = ref_latents.to(device=device, dtype=torch.float32)

        # 7. Encode controlnet frames
        controlnet_latents = None
        if controlnet_frames is not None:
            duplicate_frames_count = num_frames - len(controlnet_frames)
            if duplicate_frames_count > 0:
                reversed_controlnet_frames = list(reversed(controlnet_frames))
                controlnet_sum_frames = controlnet_frames + reversed_controlnet_frames
                reversed_chunks_count = num_frames // len(controlnet_sum_frames)
                controlnet_frames = [*controlnet_sum_frames]
                for _ in range(reversed_chunks_count):
                    controlnet_frames += controlnet_sum_frames
            controlnet_frames = controlnet_frames[:num_frames]
            controlnet_latents = prepare_controlnet_frames(
                controlnet_frames,
                height,
                width,
                dtype=self.controlnet.dtype,
                device=self.controlnet.device,
            )

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        # PI controller state
        ref_cosine_dist = 0.0
        prev_alpha = initial_blending_coeff
        integral_error = 0.0

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                # Stage selection + CN gate
                if boundary_timestep is None or t >= boundary_timestep:
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                    in_high_noise_stage = True
                else:
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2
                    in_high_noise_stage = False

                latent_model_input = latents.to(transformer_dtype)
                if self.config.expand_timesteps:
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                # ControlNet only fires on the high-noise expert AND within
                # the [start, end) gate.
                current_sampling_percent = i / len(timesteps)
                cn_active = (
                    in_high_noise_stage
                    and controlnet_latents is not None
                    and controlnet_guidance_start <= current_sampling_percent < controlnet_guidance_end
                )

                controlnet_states = None
                if cn_active:
                    controlnet_states = self.controlnet(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        controlnet_states=controlnet_latents,
                        return_dict=False,
                    )[0]
                    if isinstance(controlnet_states, (tuple, list)):
                        controlnet_states = [x.to(dtype=self.transformer.dtype) for x in controlnet_states]
                    else:
                        controlnet_states = controlnet_states.to(dtype=self.transformer.dtype)

                # Main forward
                with current_model.cache_context("cond"):
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        controlnet_states=controlnet_states,
                        controlnet_weight=controlnet_weight,
                        controlnet_stride=controlnet_stride,
                        teacache=None,
                        return_dict=False,
                    )[0]

                if self.do_classifier_free_guidance:
                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            controlnet_states=controlnet_states,
                            controlnet_weight=controlnet_weight,
                            controlnet_stride=controlnet_stride,
                            teacache=None,
                            return_dict=False,
                        )[0]
                    noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # ref_latents[i] is the precomputed Euler-ODE inversion latent
                # corresponding to denoising step i (saved in denoising order
                # by `inference/precompute_inverts.py`).
                ref_latent = ref_latents[i]

                # (direct, decayed) base alpha schedule
                if i < direct_transfer_steps:
                    base_alpha = initial_blending_coeff
                elif i < (direct_transfer_steps + decayed_transfer_steps - 1):
                    progress = (i - direct_transfer_steps) / max(decayed_transfer_steps - 1, 1)
                    base_alpha = initial_blending_coeff * (1 - progress ** exponent)
                else:
                    base_alpha = 0.0

                # Heuristic-3 PI controller (cosine distance), always on.
                if i > 0:
                    target = GOOD_AVG_COSINE_DIST_LIST[i]
                    current = ref_cosine_dist
                    norm = max(abs(target), 1e-4)
                    normalized_error = (current - target) / norm
                    integral_error += normalized_error
                    integral_clamp = 0.5 / max(Ki, 1e-8)
                    integral_error = max(-integral_clamp, min(integral_clamp, integral_error))
                    correction = Kp * normalized_error + Ki * integral_error
                    scale = max(0.0, min(1.5, 1.0 - correction))
                    alpha = base_alpha * scale
                    alpha = max(prev_alpha - max_blending_coeff_delta,
                                min(prev_alpha + max_blending_coeff_delta, alpha))
                else:
                    alpha = base_alpha
                alpha = max(0.0, alpha)
                prev_alpha = alpha

                # Phase substitute
                latents, ref_cosine_dist, energy_ratio = self._phase_substitute(
                    x_dec=latents,
                    ref_latent=ref_latent,
                    alpha=alpha,
                    step=i,
                )

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

        # Decode
        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
