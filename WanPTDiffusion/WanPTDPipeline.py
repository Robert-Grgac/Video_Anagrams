from typing import Any, Callable, Dict, List, Optional, Union

import torch
import glob
import os
import math

from diffusers import WanPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
import wandb
from utils.constants import GOOD_AVG_COSINE_DIST_LIST

class WanPTDiffusionPipeline(WanPipeline):
    
    def _load_reference_latents(
        self,
        latents_dir: str,
        device: torch.device = None,
    ):
        ref_latents = []
        for latent_file in sorted(glob.glob(os.path.join(latents_dir, "*.pt"))):
            ref_latent = torch.load(latent_file, weights_only=True, map_location=device)
            ref_latents.append(ref_latent)
        return ref_latents
    
    @staticmethod
    def _phase_substitute(x_dec: torch.Tensor, ref_latent: torch.Tensor, alpha: float, step: int, conditional_latent: torch.Tensor) -> torch.Tensor:
        ref_latent_fft = torch.fft.fft2(ref_latent)
        ref_latent_angle = torch.angle(ref_latent_fft)
        
        x_dec_fft = torch.fft.fft2(x_dec)
        x_dec_mag = torch.abs(x_dec_fft)
        x_dec_angle = torch.angle(x_dec_fft)
        mixed_angle = ref_latent_angle * alpha + (1 - alpha) * x_dec_angle
        
        conditional_latent_fft = torch.fft.fft2(conditional_latent)
        conditional_latent_mag = torch.abs(conditional_latent_fft)
        
        #Log the cosine distance between the reference and the x_dec angle
        ref_cosine_dist = torch.cos(ref_latent_angle - x_dec_angle).mean() 
        mse_mag = torch.nn.functional.mse_loss(conditional_latent_mag, x_dec_mag)
        wandb.log({"ref_cosine_dist_mean": ref_cosine_dist.item(), "cond_mse_mag": mse_mag.item(), "alpha": alpha}, step=step)
        
        x_dec_fft = x_dec_mag * torch.cos(mixed_angle) + \
                    x_dec_mag * torch.sin(mixed_angle) * torch.complex(torch.zeros_like(x_dec_mag),
                                                                       torch.ones_like(x_dec_mag))
        x_dec = torch.fft.ifft2(x_dec_fft).real
        
        return x_dec, ref_cosine_dist.item()
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 100,
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
        
        #Additional args for PTM
        direct_transfer_steps: int = 60,
        decayed_transfer_steps: int = 20,
        exponent: float = 0.5,
        initial_alpha: float = 1,
        inital_ref_latent: Optional[torch.Tensor] = None,
        use_preloaded_latents: bool = True,
        use_blending_heuristic_version_1: bool = False,
        use_blending_heuristic_version_2: bool = False,
        use_blending_heuristic_version_3: bool = False,
        ref_latents_dir: Optional[str] = "./precomputed_deterministic_inversion_latents",
        steepness: float = None,
        gain: float = 2.0,
        Kp: float = None,
        Ki: float = None,
        max_alpha_delta: float = None
        
    ):

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # --- original input checks ---
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

        # Adjust num_frames to fit VAE temporal scale
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

        # Batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Encode prompts
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

        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
        
        # Unconditional prompt embeds
        uncoditional_prompt_embeds = self._get_t5_prompt_embeds(
                prompt="",
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )
        
        uncoditional_prompt_embeds = uncoditional_prompt_embeds.to(transformer_dtype)


        # Timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Latents (sampling)
        num_channels_latents = (
            self.transformer.config.in_channels
            if self.transformer is not None
            else self.transformer_2.config.in_channels
        )

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

        # Needed for expand_timesteps mode
        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # Boundary timestep for Wan2.2 2-stage denoising
        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        #Load reference latents for PTM
        if use_preloaded_latents:
            ref_latents_list = self._load_reference_latents(latents_dir=ref_latents_dir, device=device)
        
        
        if inital_ref_latent is not None:
            ref_latent = inital_ref_latent.to(device)
            
        conditional_latent = latents.clone()
        
        #Initialize heurisitc vlaues
        ref_cosine_dist = 0.0
        ema_cosine_dist = 0.0
        ema_decay = 0.9
        prev_alpha = initial_alpha
        integral_error = 0.0
        prev_alpha = initial_alpha
        

        # -------------------------
        # Denoising loop + PTM
        # -------------------------
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                # choose stage model
                if boundary_timestep is None or t >= boundary_timestep:
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                    in_low_noise_stage = False
                else:
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2
                    in_low_noise_stage = True

                latent_model_input = latents.to(transformer_dtype)

                if self.config.expand_timesteps:
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                conditional_latent_input = conditional_latent.to(transformer_dtype)
                batched_hidden = torch.cat([latent_model_input, conditional_latent_input], dim=0)
                batched_timestep = timestep.repeat(2)
                batched_prompt_embeds = prompt_embeds.repeat(2, *([1] * (prompt_embeds.dim() - 1)))
    
                # Cond forward
                with current_model.cache_context("cond"):
                    batched_noise_pred = current_model(
                        hidden_states=batched_hidden,
                        timestep=batched_timestep,
                        encoder_hidden_states=batched_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                noise_pred, cond_noise_pred = batched_noise_pred.chunk(2, dim=0)
                
                # CFG
                if self.do_classifier_free_guidance:
                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                if not use_preloaded_latents:
                    ref_latent_model_input = ref_latent.to(transformer_dtype)
                    with current_model.cache_context("uncond"):
                        ref_noise_uncond = current_model(
                            hidden_states=ref_latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=uncoditional_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                
                    batched_noise = torch.cat([noise_pred, ref_noise_uncond], dim=0)
                    batched_latents = torch.cat([latents, ref_latent], dim=0)
                    
                    # scheduler step
                    batched_result = self.scheduler.step(batched_noise, t, batched_latents, return_dict=False)[0]

                    latents, ref_latent = batched_result.chunk(2, dim=0)
                else:
                    # scheduler step for main latents
                    batched_noise_for_scheduler = torch.cat([noise_pred, cond_noise_pred], dim=0)
                    batched_latents = torch.cat([latents, conditional_latent], dim=0)
                    batched_result = self.scheduler.step(batched_noise_for_scheduler, t, batched_latents, return_dict=False)[0]
                    latents, conditional_latent = batched_result.chunk(2, dim=0)
                    ref_latent = ref_latents_list[i]
                
                if i < direct_transfer_steps:
                    base_alpha = initial_alpha
                elif i < (direct_transfer_steps + decayed_transfer_steps - 1):
                    decay_progress = (i - direct_transfer_steps) / max(decayed_transfer_steps - 1, 1)
                    base_alpha = initial_alpha * (1 - decay_progress ** exponent)
                else:
                    base_alpha = 0.0
                
                if use_blending_heuristic_version_1 and i > 0:  # skip step 0 (no valid metric yet)
                    good_ref = GOOD_AVG_COSINE_DIST_LIST[i]
                    margin = 2 * math.sqrt(max(good_ref, 0.0))
                    excess = ref_cosine_dist - (good_ref + margin)
                    damping = 1.0 / (1.0 + math.exp(steepness * excess)) #apply sigmoid damping based on how much the ref_cosine_dist exceeds the good_ref + margin
                    alpha = base_alpha * damping
                elif use_blending_heuristic_version_2 and i > 0:
                        good_ref = GOOD_AVG_COSINE_DIST_LIST[i]
    
                        # (1) Smooth
                        ema_cosine_dist = ema_decay * ema_cosine_dist + (1 - ema_decay) * ref_cosine_dist
                        
                        # (2) Compute margin (dead zone) — recomputed every step, uses abs()
                        margin_band = 2 * math.sqrt(max(abs(good_ref), 1e-4))
                        
                        # (3) Proportional response with dead zone
                        error = ema_cosine_dist - good_ref
                        if abs(error) <= margin_band:
                            scale = 1.0  # inside dead zone, no correction
                        else:
                            sign = 1.0 if error > 0 else -1.0
                            overshoot = abs(error) - margin_band
                            gain = gain 
                            scale = 1.0 - sign * gain * overshoot / max(abs(good_ref) + margin_band, 1e-6)
                            scale = max(0.5, min(1.5, scale))  # clamp: alpha stays within 50%-150% of base
                        
                        alpha = base_alpha * scale
                        
                        # (4) Rate-limit alpha changes
                        alpha = max(prev_alpha - max_alpha_delta, min(prev_alpha + max_alpha_delta, alpha))
                        prev_alpha = alpha
                elif use_blending_heuristic_version_3 and i > 0:
                    target = GOOD_AVG_COSINE_DIST_LIST[i]
                    current = ref_cosine_dist 

                    error = current - target  
                    norm = max(abs(target), 1e-4)
                    normalized_error = error / norm

                    integral_error += normalized_error
                    max_integral_contribution = 0.5
                    integral_clamp = max_integral_contribution / max(Ki, 1e-8)
                    integral_error = max(-integral_clamp, min(integral_clamp, integral_error))  # anti-windup clamp

                    correction = Kp * normalized_error + Ki * integral_error
                    scale = max(0.0, min(1.5, 1.0 - correction))

                    alpha = base_alpha * scale
                    alpha = max(prev_alpha - max_alpha_delta, min(prev_alpha + max_alpha_delta, alpha))
                    prev_alpha = alpha
                else:
                    alpha = base_alpha
                    prev_alpha = alpha

                alpha = max(0.0, alpha)
                    
                latents, ref_cosine_dist = self._phase_substitute(x_dec=latents, ref_latent=ref_latent, alpha=alpha, step=i, conditional_latent=conditional_latent)
                
                # callback
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # progress bar update
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        # Decode
        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std_inv = (
                1.0 / torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = latents / latents_std_inv + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
