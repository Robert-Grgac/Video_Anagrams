from typing import Any, Callable, Dict, List, Optional, Union

import torch
import glob
import os
import math
import torch.nn.functional as F
import numpy as np
from PIL import Image
from skimage.feature import hog

from diffusers import WanPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers import WanTransformer3DModel
import wandb
from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision import transforms

from inference.constants import GOOD_AVG_COSINE_DIST_LIST
from wan_transformer import CustomWanTransformer3DModel
from wan_controlnet import WanControlnet
from wan_teacache import TeaCache



def resize_for_crop(image, crop_h, crop_w):
    img_h, img_w = image.shape[-2:]
    if img_h >= crop_h and img_w >= crop_w:
        coef = max(crop_h / img_h, crop_w / img_w)
    elif img_h <= crop_h and img_w <= crop_w:
        coef = max(crop_h / img_h, crop_w / img_w)
    else:
        coef = crop_h / img_h if crop_h > img_h else crop_w / img_w 
    out_h, out_w = int(img_h * coef), int(img_w * coef)
    resized_image = transforms.functional.resize(image, (out_h, out_w), antialias=True)
    return resized_image


def prepare_frames(input_images, video_size, do_resize=True, do_crop=True):
    input_images = np.stack([np.array(x) for x in input_images])
    images_tensor = torch.from_numpy(input_images).permute(0, 3, 1, 2) / 127.5 - 1
    if do_resize:
        images_tensor = [resize_for_crop(x, crop_h=video_size[0], crop_w=video_size[1]) for x in images_tensor]
    if do_crop:
        images_tensor = [transforms.functional.center_crop(x, video_size) for x in images_tensor]
    if isinstance(images_tensor, list):
        images_tensor = torch.stack(images_tensor)
    return images_tensor.unsqueeze(0) 


def prepare_controlnet_frames(controlnet_frames, height, width, dtype, device):
    prepared_frames = prepare_frames(controlnet_frames, (height, width))
    controlnet_encoded_frames = prepared_frames.to(dtype=dtype, device=device)
    return controlnet_encoded_frames.permute(0, 2, 1, 3, 4).contiguous()


class WanCNPTDiffusionPipeline(WanPipeline):
    
    model_cpu_offload_seq = "text_encoder->transformer->transformer_2->vae->controlnet"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer_2"]

    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: CustomWanTransformer3DModel,
        vae: AutoencoderKLWan,
        controlnet: WanControlnet,
        scheduler: FlowMatchEulerDiscreteScheduler,
        transformer_2: WanTransformer3DModel = None,
        boundary_ratio: Optional[float] = None,
        expand_timesteps: bool = False,
    ):
        super().__init__(
            tokenizer=tokenizer, text_encoder=text_encoder, transformer=transformer,
            vae=vae, scheduler=scheduler, transformer_2=transformer_2,
            boundary_ratio=boundary_ratio, expand_timesteps=expand_timesteps,
        )
        self.register_modules(controlnet=controlnet)
        if transformer_2 is not None:
            assert transformer.dtype == transformer_2.dtype, (
                f"dtype mismatch: transformer={transformer.dtype}, transformer_2={transformer_2.dtype}"
            )
 
        
    @staticmethod
    def _compute_hog_features(frames: torch.Tensor, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        video = frames[0].cpu().float()
        C, T, H, W = video.shape
        
        all_hog_features = []
        for t_idx in range(T):
            frame = video[:, t_idx, :, :]  # (C, H, W)
            if C == 3:
                gray = 0.2989 * frame[0] + 0.5870 * frame[1] + 0.1140 * frame[2]
            else:
                gray = frame[0]
            
            gray_np = gray.numpy()
            
            hog_feat = hog(
                gray_np,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                feature_vector=True,
            )
            all_hog_features.append(torch.from_numpy(hog_feat))
        
        # Concatenate all frames' HOG features into one vector
        return torch.cat(all_hog_features, dim=0)
    
    def _decode_latents_to_pixel(self, latents: torch.Tensor) -> torch.Tensor:
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
        # Clamp to [0, 1]
        video = video.clamp(0, 1)
        return video
        
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
        
        energy_before = (x_dec ** 2).sum()
        
        #Reconstuction
        x_dec_fft = x_dec_mag * torch.cos(mixed_angle) + \
                    x_dec_mag * torch.sin(mixed_angle) * torch.complex(torch.zeros_like(x_dec_mag),
                                                                       torch.ones_like(x_dec_mag))
        x_dec = torch.fft.ifft2(x_dec_fft).real
        
        energy_after = (x_dec ** 2).sum()
        energy_ratio = (energy_after / (energy_before + 1e-10)).item()
        
        #Log the cosine distance between the reference and the x_dec angle
        ref_cosine_dist = torch.cos(ref_latent_angle - x_dec_angle).mean() 
        mse_mag = torch.nn.functional.mse_loss(conditional_latent_mag, x_dec_mag)
        wandb.log({
        "ref_cosine_dist_mean": ref_cosine_dist.item(), 
        "cond_mse_mag": mse_mag.item(), 
        "alpha": alpha,
        "energy_ratio": energy_ratio,
        "energy_before": energy_before.item(),
        "energy_after": energy_after.item(),
        }, step=step)
        
        return x_dec, ref_cosine_dist.item(), energy_ratio
    
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
        use_blending_heuristic_version_4: bool = False,
        ref_latents_dir: Optional[str] = "./precomputed_deterministic_inversion_latents",
        steepness: float = None,
        gain: float = 2.0,
        Kp: float = None,
        Ki: float = None,
        max_alpha_delta: float = None,
        energy_target: float = 0.95,
        Kp_energy: float = 2.0,
        Ki_energy: float = 0.1,
        do_additional_logging: bool = False,
        
        #ControlNet args
        controlnet_frames: List[Image.Image] = None,
        controlnet_latents: Optional[torch.FloatTensor] = None,
        controlnet_weight: float = 1.0,
        controlnet_guidance_start: float = 0.0,
        controlnet_guidance_end: float = 1.0,
        controlnet_stride: int = 3,

        teacache_state: Optional[TeaCache]= None,
        teacache_treshold: float = 0.0,
    ):
        self.teacache = teacache_state or None
        if (self.teacache is None) and (teacache_treshold > 0.0):
            self.teacache = TeaCache(
                num_inference_steps=num_inference_steps, 
                model_name="DEFAULT",
                treshold=teacache_treshold
            )
        
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
        
        # Encode controlnet frames
        if (controlnet_latents is None) and (controlnet_frames is not None):
            duplicate_frames_count = num_frames - len(controlnet_frames)
            print(f'Using controlnet frames: {len(controlnet_frames)}. Extended frames count: {duplicate_frames_count}')
            if duplicate_frames_count > 0:
                # Simple duplicate first frame
                # controlnet_frames = [controlnet_frames[0]] * duplicate_frames_count + controlnet_frames
                # Or reversed duplicate frames ?
                reversed_controlnet_frames = list(reversed(controlnet_frames))
                controlnet_sum_frames = controlnet_frames + reversed_controlnet_frames
                reversed_chunks_count = num_frames // len(controlnet_sum_frames)
                controlnet_frames = [*controlnet_sum_frames]
                for _ in range(reversed_chunks_count):
                    controlnet_frames += controlnet_sum_frames

            # If controlnet frames count greater than num_frames parameter
            controlnet_frames = controlnet_frames[:num_frames]
            
            controlnet_latents = prepare_controlnet_frames(
                controlnet_frames,
                height, 
                width,
                dtype=self.controlnet.dtype, 
                device=self.controlnet.device
            )
        
        #Initialize heurisitc vlaues
        ref_cosine_dist = 0.0
        ema_cosine_dist = 0.0
        ema_decay = 0.9
        prev_alpha = initial_alpha
        integral_error = 0.0
        prev_alpha = initial_alpha
        energy_ratio = 1.0          
        integral_error_energy = 0.0
        

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

                controlnet_states = None
                current_sampling_percent = i / len(timesteps)
                if (controlnet_latents is not None) and (controlnet_guidance_start <= current_sampling_percent < controlnet_guidance_end) and (not in_low_noise_stage):
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
                        
                # Cond forward
                with current_model.cache_context("cond"):
                    if not in_low_noise_stage:
                        batched_noise_pred = current_model(
                            hidden_states=batched_hidden,
                            timestep=batched_timestep,
                            encoder_hidden_states=batched_prompt_embeds,
                            controlnet_states=controlnet_states,
                            controlnet_weight=controlnet_weight,
                            controlnet_stride=controlnet_stride,
                            teacache=self.teacache,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    else:
                        batched_noise_pred = current_model(
                            hidden_states=batched_hidden,
                            timestep=batched_timestep,
                            encoder_hidden_states=batched_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        
                noise_pred, cond_noise_pred = batched_noise_pred.chunk(2, dim=0)
                
                cosine_similarity_of_noise = F.cosine_similarity(noise_pred.flatten(), cond_noise_pred.flatten(), dim=0).item()
                
                # CFG
                if self.do_classifier_free_guidance:
                    if not in_low_noise_stage:
                        with current_model.cache_context("uncond"):
                            noise_uncond = current_model(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                controlnet_states=controlnet_states,
                                controlnet_weight=controlnet_weight,
                                controlnet_stride=controlnet_stride,
                                teacache=self.teacache,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]
                    else:
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
                        if not in_low_noise_stage:
                            ref_noise_uncond = current_model(
                                hidden_states=ref_latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=uncoditional_prompt_embeds,
                                controlnet_states=controlnet_states,
                                controlnet_weight=controlnet_weight,
                                controlnet_stride=controlnet_stride,
                                teacache=self.teacache,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]
                        else:
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
                
                elif use_blending_heuristic_version_4 and i > 0:
                    # PI controller based on spectral energy ratio
                    target = energy_target
                    current = energy_ratio
                    
                    # Error: negative means energy was lost (alpha too high)
                    error = target - current 
                    
                    normalized_error = error
                    
                    integral_error_energy += normalized_error
                    max_integral_contribution = 0.3
                    integral_clamp = max_integral_contribution / max(Ki_energy, 1e-8)
                    integral_error_energy = max(-integral_clamp, min(integral_clamp, integral_error_energy))
                    
                    correction = Kp_energy * normalized_error + Ki_energy * integral_error_energy
                    
                    scale = max(0.0, min(1.5, 1.0 + correction))
                    
                    alpha = base_alpha * scale
                    alpha = max(prev_alpha - max_alpha_delta, min(prev_alpha + max_alpha_delta, alpha))
                    prev_alpha = alpha
                else:
                    alpha = base_alpha
                    prev_alpha = alpha

                alpha = max(0.0, alpha)
                    
                latents, ref_cosine_dist, energy_ratio  = self._phase_substitute(x_dec=latents, ref_latent=ref_latent, alpha=alpha, step=i, conditional_latent=conditional_latent)
                
                if do_additional_logging:
                    #testing different measuments to use as a referen for PI contorller
                    mse_on_latents = F.mse_loss(latents, conditional_latent).item()
                    latent_cos_sim = F.cosine_similarity(latents.flatten(), conditional_latent.flatten(), dim=0).item()
                    normalized_mse_on_latents = F.mse_loss(latents, conditional_latent) / (conditional_latent ** 2).mean()

                    # Select 3 frames: first, middle, last from the LATENT temporal dimension
                    T_latent = latents.shape[2]  # temporal dim of latent tensor
                    frame_indices = [0, T_latent // 2, T_latent - 1]

                    # Slice latents at those temporal positions → (B, C, 3, H_lat, W_lat)
                    blended_latent_frames = latents[:, :, frame_indices, :, :]
                    ref_latent_frames = ref_latent[:, :, frame_indices, :, :]

                    # Decode only the 3 selected frames (much cheaper than full video BUT COULD MESS WITH THE DECODED VALUES SINCE VAE HAS TEMPORAL CONTEXT)
                    blended_pixels = self._decode_latents_to_pixel(blended_latent_frames)
                    ref_pixels = self._decode_latents_to_pixel(ref_latent_frames)
                    
                    # Compute HOG features
                    hog_blended = self._compute_hog_features(blended_pixels)
                    hog_ref = self._compute_hog_features(ref_pixels)
                    
                    # Cosine similarity between HOG descriptors
                    hog_cos_sim = F.cosine_similarity(
                        hog_blended.unsqueeze(0), 
                        hog_ref.unsqueeze(0), 
                        dim=1
                    ).item()
                    
                    
                    wandb.log({
                        "mse_on_latents": mse_on_latents,
                        "latent_cos_sim": latent_cos_sim,
                        "normalized_mse_on_latents": normalized_mse_on_latents.item(),
                        "cosine_similarity_of_noise": cosine_similarity_of_noise,
                        "hog_cosine_similarity": hog_cos_sim,
                    }, step=i)
                
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
        self.teacache = None

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
