from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable
import logging as logger
import torch
from diffusers import WanPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from utils.views import ViewTransform


class WanAnagramPipeline2(WanPipeline):
    """
    Input docstring...
    """
    
    @torch.no_grad()
    def __call__(
        self,
        #Import 2 prompts instead of 1
        prompt_a: Union[str, List[str]] = None,
        prompt_b: Union[str, List[str]] = None,
        
        #joint diffusion parameters
        conditional_guidance_scale: float = None,
        negative_guidance_scale: float = None,
        joint_diffusion_steps: int = None,
        
        #anagram parameters
        view: Optional[ViewTransform] = None,
        apply_mean_reduction: bool = True,
        
        #pixel-space tansformations
        
        #Original WanPipeline inputs
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
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt_a,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if prompt_b is None or not isinstance(prompt_b, str):
            raise ValueError("WanAnagramPipeline2 requires a second prompt (string) via the `prompt_b` argument.")
        
        if joint_diffusion_steps is None:
            raise ValueError("WanAnagramPipeline2 requires the number of joint diffusion steps via the `joint_diffusion_steps` argument.")
        
        if joint_diffusion_steps < num_inference_steps and view is None:
            raise ValueError("WanAnagramPipeline2 requires a view transform via the `view` argument when `joint_diffusion_steps` is less than `num_inference_steps`.")
        
        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
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

        # 2. Define call parameters
        if isinstance(prompt_a, list) or isinstance(prompt_b, list):
            raise ValueError("WanAnagramPipeline2 currently supports only single-string prompts (not lists).")
        batch_size = 1

        # 3. Encode input prompt
        prompt_embeds_a, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt_a,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds_a = prompt_embeds_a.to(transformer_dtype)
        
        uncond_prompt_embeds = self._get_t5_prompt_embeds(
                prompt="",
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )
        uncond_prompt_embeds = uncond_prompt_embeds.to(transformer_dtype)
        
        if prompt_b is not None:
            prompt_embeds_b = self._get_t5_prompt_embeds(
                prompt=prompt_b,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            prompt_embeds_b = prompt_embeds_b.to(transformer_dtype)

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
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
        
        if joint_diffusion_steps < num_inference_steps and not apply_mean_reduction:
            latents_b = latents.clone()

        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)
        
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device, latents.dtype)
        )
        latents_std = (
            1.0 / torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device, latents.dtype)
        )

        # 6. Denoising loop
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

                if boundary_timestep is None or t >= boundary_timestep:
                    # wan2.1 or high-noise stage in wan2.2
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    # low-noise stage in wan2.2
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2
                    
                if self.config.expand_timesteps:
                    # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])
                
                #Batch everything for faster computation
                if i > joint_diffusion_steps:
                    if not apply_mean_reduction:
                        latents_flipped = view.forward(latents)
                        latents_batched = torch.cat([latents, latents_flipped], dim=0)
                        latents_batched = latents_batched.to(transformer_dtype)
                    else:
                        latents_flipped = view.forward(latents)
                        latents_batched = torch.cat([latents, latents_flipped], dim=0)
                        latents_batched = latents_batched.to(transformer_dtype)
                else:
                    latents_batched = torch.cat([latents, latents], dim=0)
                    latents_batched = latents_batched.to(transformer_dtype)
                    
                prompt_embeds_batched = torch.cat([prompt_embeds_a, prompt_embeds_b], dim=0)
                
                with current_model.cache_context("cond"):
                    noise_cond = current_model(
                        hidden_states=latents_batched,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds_batched,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                
                if conditional_guidance_scale is not None and negative_guidance_scale is not None:
                    #Do cfg with negative and unconditional flow estimations
                    neg_batched = negative_prompt_embeds.repeat(2, 1, 1)
                    uncond_batched = uncond_prompt_embeds.repeat(2, 1, 1)
                    with current_model.cache_context("neg"):
                        noise_neg = current_model(
                            hidden_states=latents_batched,
                            timestep=timestep,
                            encoder_hidden_states=neg_batched, 
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    
                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latents_batched,
                            timestep=timestep,
                            encoder_hidden_states=uncond_batched, 
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    
                    noise_cond_a, noise_cond_b = noise_cond.chunk(2, dim=0)
                    noise_neg_a, noise_neg_b = noise_neg.chunk(2, dim=0)
                    noise_uncond_a, noise_uncond_b = noise_uncond.chunk(2, dim=0)
                    
                    if i > joint_diffusion_steps:
                        #Invert back the predicted noise for prompt_b
                        noise_cond_b = view.inverse(noise_cond_b)
                        noise_neg_b = view.inverse(noise_neg_b)
                        noise_uncond_b = view.inverse(noise_uncond_b)
                    
                    noise_pred_a = noise_uncond_a + conditional_guidance_scale * (noise_cond_a - noise_uncond_a) - negative_guidance_scale * (noise_neg_a - noise_uncond_a)
                    noise_pred_b = noise_uncond_b + conditional_guidance_scale * (noise_cond_b - noise_uncond_b) - negative_guidance_scale * (noise_neg_b - noise_uncond_b)
                    noise_pred = (noise_pred_a + noise_pred_b) / 2.0

                        
                elif conditional_guidance_scale is not None and negative_guidance_scale is None:
                    #Do cfg with unconditional flow estimation only
                    uncond_batched = uncond_prompt_embeds.repeat(2, 1, 1)
                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latents_batched,
                            timestep=timestep,
                            encoder_hidden_states=uncond_batched, 
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        
                    noise_cond_a, noise_cond_b = noise_cond.chunk(2, dim=0)
                    noise_uncond_a, noise_uncond_b = noise_uncond.chunk(2, dim=0)
                    
                    if i > joint_diffusion_steps:
                        #Invert back the predicted noise for prompt_b
                        noise_cond_b = view.inverse(noise_cond_b)
                        noise_uncond_b = view.inverse(noise_uncond_b)
                        
                        
                    noise_pred_a = noise_uncond_a + conditional_guidance_scale * (noise_cond_a - noise_uncond_a)
                    noise_pred_b = noise_uncond_b + conditional_guidance_scale * (noise_cond_b - noise_uncond_b)
                    noise_pred = (noise_pred_a + noise_pred_b) / 2.0
                elif conditional_guidance_scale is None and negative_guidance_scale is not None:
                    #Do cfg with negative flow estimation only
                    neg_batched = negative_prompt_embeds.repeat(2, 1, 1)
                    with current_model.cache_context("neg"):
                        noise_neg = current_model(
                            hidden_states=latents_batched,
                            timestep=timestep,
                            encoder_hidden_states=neg_batched, 
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    
                    noise_cond_a, noise_cond_b = noise_cond.chunk(2, dim=0)
                    noise_neg_a, noise_neg_b = noise_neg.chunk(2, dim=0)
                    
                    if i > joint_diffusion_steps:
                        #Invert back the predicted noise for prompt_b
                        noise_cond_b = view.inverse(noise_cond_b)
                        noise_neg_b = view.inverse(noise_neg_b)
                    
                    noise_pred_a = noise_cond_a - negative_guidance_scale * (noise_neg_a - noise_cond_a)
                    noise_pred_b = noise_cond_b - negative_guidance_scale * (noise_neg_b - noise_cond_b)
                    noise_pred = (noise_pred_a + noise_pred_b) / 2.0
                else:
                    
                    noise_cond_a, noise_cond_b = noise_cond.chunk(2, dim=0)
                    
                    if i > joint_diffusion_steps:
                        #Invert back the predicted noise for prompt_b
                        noise_cond_b = view.inverse(noise_cond_b)
                    
                    noise_pred_a = noise_cond_a
                    noise_pred_b = noise_cond_b
                    
                    noise_pred = (noise_pred_a + noise_pred_b) / 2.0
                
                # (SCHEDULER STEP) compute the previous noisy sample x_t -> x_t-1
                if i > joint_diffusion_steps and not apply_mean_reduction:
                    # Batch both latents and noise predictions for single scheduler call
                    latents_batched_for_scheduler = torch.cat([latents, latents_b], dim=0)
                    noise_pred_batched = torch.cat([noise_pred_a, noise_pred_b], dim=0)
                    
                    # Single scheduler step with batched inputs
                    denoised_batched = self.scheduler.step(noise_pred_batched, t, latents_batched_for_scheduler, return_dict=False)[0]
                    
                    # Unbatch the results
                    latents, latents_b = denoised_batched.chunk(2, dim=0)
                    latents = latents.to(transformer_dtype)
                    latents_b = latents_b.to(transformer_dtype)
                else:  
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0].to(transformer_dtype)
                

                #-----------------NO CALLBACKS SO IGNORE THIS PART-----------------#
                #Call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                #-------------------------------------------------------------------#
                
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

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

            if joint_diffusion_steps < num_inference_steps and not apply_mean_reduction:
                latents_b = latents_b.to(self.vae.dtype)
                latents_b = latents_b / latents_std + latents_mean
                video_b = self.vae.decode(latents_b, return_dict=False)[0]
                video_b = self.video_processor.postprocess_video(video_b, output_type=output_type)
                
            
        else:
            video = latents
            if joint_diffusion_steps < num_inference_steps and not apply_mean_reduction:
                video_b = latents_b
            

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            if joint_diffusion_steps < num_inference_steps and not apply_mean_reduction:
                return (video, video_b)
            return (video)
        
        pipeline_output = WanPipelineOutput(frames=video)
        if joint_diffusion_steps < num_inference_steps and not apply_mean_reduction:
            pipeline_output_b = WanPipelineOutput(frames=video_b)
            return (pipeline_output, pipeline_output_b)

        return pipeline_output