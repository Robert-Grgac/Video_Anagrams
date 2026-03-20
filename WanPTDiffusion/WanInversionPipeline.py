from fileinput import filename
import os
from typing import List, Optional, Union
import glob

import torch
import PIL.Image
from tqdm import tqdm

from diffusers import WanPipeline
from diffusers.video_processor import VideoProcessor


class WanInversionPipeline(WanPipeline):
    """
    Wan pipeline extended with an `invert()` method that performs Euler ODE
    inversion on a reference image and stores the latent at each step.
    """

    def _encode_reference_image_to_latents(
        self,
        reference_image: Union[PIL.Image.Image, torch.Tensor],
        *,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Encode a single reference image into Wan VAE latent space,
        treating it as a static video (same frame repeated).

        Returns normalized latents with shape [1, z_dim, T_lat, H_lat, W_lat].
        """
        if isinstance(reference_image, PIL.Image.Image):
            img = self.video_processor.preprocess(reference_image, height=height, width=width)
        elif isinstance(reference_image, torch.Tensor):
            img = reference_image
            if img.ndim == 3:
                img = img.unsqueeze(0)
            img = self.video_processor.preprocess(img, height=height, width=width)
        else:
            raise TypeError(
                "`reference_image` must be a PIL.Image.Image or torch.Tensor."
            )

        img = img.to(device=device, dtype=torch.float32)

        # Build static video: [B, 3, T, H, W]
        video = img.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)

        # Encode with VAE
        video = video.to(device=device, dtype=self.vae.dtype)
        posterior = self.vae.encode(video).latent_dist
        latents = posterior.mode()  # deterministic (mean)

        # Normalize to match diffusion latent normalization
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean, device=device, dtype=latents.dtype)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std, device=device, dtype=latents.dtype)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
        )
        latents = (latents - latents_mean) / latents_std

        return latents.to(dtype=dtype)

    @torch.no_grad()
    def invert(
        self,
        reference_image: Union[PIL.Image.Image, torch.Tensor],
        height: int = 528,
        width: int = 528,
        num_frames: int = 61,
        num_inference_steps: int = 50,
        prompt: str = "",
        negative_prompt: str = "",
        guidance_scale: float = 1.0,
        save_dir: str = "./non_deterministic_inversion_trajectory",
        save_dtype: torch.dtype = torch.bfloat16,
        max_sequence_length: int = 512,
        attention_kwargs=None,
        latent_filename: Optional[str] = None,
    ):

        os.makedirs(save_dir, exist_ok=True)

        device = self._execution_device

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = (
                num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
            )
        num_frames = max(num_frames, 1)

        #z_hat_0
        z = self._encode_reference_image_to_latents(
            reference_image,
            height=height,
            width=width,
            num_frames=num_frames,
            device=device,
            dtype=torch.float32,
        )

        self._guidance_scale = guidance_scale
        do_cfg = guidance_scale > 1.0

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt if do_cfg else None,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        inv_sigmas = torch.flip(self.scheduler.sigmas, dims=[0]).to(device=device)

        num_train_timesteps = self.scheduler.config.num_train_timesteps
        
        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * num_train_timesteps
        else:
            boundary_timestep = None

        print(f"Running Euler ODE inversion with {len(inv_sigmas) - 1} steps")
        print(f"Sigma range: {inv_sigmas[0].item():.4f} → {inv_sigmas[-1].item():.4f}")

        previous_model = None
        for i in tqdm(range(len(inv_sigmas) - 1), desc="Euler Inversion"):
            sigma = inv_sigmas[i]
            sigma_next = inv_sigmas[i + 1]

            t = sigma * num_train_timesteps
            timestep = t.expand(z.shape[0]).to(device=device)

            latent_model_input = z.to(transformer_dtype)
            
            if boundary_timestep is None or t >= boundary_timestep:
                current_model = self.transformer
                other_model = self.transformer_2
                #print(f"High noise model used at step {i}")      
            else:
                current_model = self.transformer_2
                other_model = self.transformer
                #print(f"Low noise model used at step {i}")  

            if previous_model is not None and current_model is not previous_model:
                other_model.to("cpu")
                torch.cuda.empty_cache()
                #print(f"  → Offloaded previous model to CPU at step {i}")

            previous_model = current_model

            noise_pred = current_model( # switch to self.transformer
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

            if do_cfg:
                noise_uncond = current_model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            dt = sigma_next - sigma
            z = z.to(torch.float32) + dt * noise_pred.to(torch.float32)#  Z = z.to(torch.float32) - i/inference_steps * noise_pred.to(torch.float32)

        if latent_filename is None:
            latent_filename = f"z_final.pt"
        final_path = os.path.join(save_dir, latent_filename)
        torch.save(z.to(save_dtype).cpu(), final_path)

    def decode_inversion_trajectory_to_images(
    self,                                       
    latent_dir: str = "./inversion_trajectory",
    output_dir: str = "./decoded_latents",
    frame_index: int = 0,
    ):
        os.makedirs(output_dir, exist_ok=True)

        pt_files = sorted(glob.glob(os.path.join(latent_dir, "*.pt")))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {latent_dir}")

        device = self._execution_device

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
        )
        latents_std_inv = (
            1.0 / torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
        )

        print(f"Decoding {len(pt_files)} latent files from {latent_dir} → {output_dir}")

        for pt_path in pt_files:
            step_name = os.path.splitext(os.path.basename(pt_path))[0]  # e.g. "step_0000"

            latents = torch.load(pt_path, map_location="cpu")

            latents = latents.to(device=device, dtype=self.vae.dtype)

            mean = latents_mean.to(device=device, dtype=latents.dtype)
            std_inv = latents_std_inv.to(device=device, dtype=latents.dtype)
            latents = latents / std_inv + mean

            with torch.no_grad():
                video = self.vae.decode(latents, return_dict=False)[0]

            frames = self.video_processor.postprocess_video(video, output_type="pil")

            if isinstance(frames[0], list):
                frame_img = frames[0][frame_index]
            else:
                frame_img = frames[frame_index]
            image_filename = f"{step_name}.png"
            save_path = os.path.join(output_dir, image_filename)
            frame_img.save(save_path)

    @torch.no_grad()
    def deterministic_invert(
        self,
        reference_image: Union[PIL.Image.Image, torch.Tensor],
        height: int = 528,
        width: int = 528,
        num_frames: int = 61,
        num_inference_steps: int = 50,
        save_latent_dir: str = "./deterministic_inversion_latents",
        save_image_dir: str = "./deterministic_inversion_images",
        save_dtype: torch.dtype = torch.float32,
        frame_index: int = 0,
        save_images: bool = True,
    ):

        os.makedirs(save_latent_dir, exist_ok=True)
        os.makedirs(save_image_dir, exist_ok=True)

        device = self._execution_device

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = (
                num_frames // self.vae_scale_factor_temporal
                * self.vae_scale_factor_temporal + 1
            )
        num_frames = max(num_frames, 1)

        z_hat_0 = self._encode_reference_image_to_latents(
            reference_image,
            height=height,
            width=width,
            num_frames=num_frames,
            device=device,
            dtype=torch.float32,
        )

        eps = torch.randn(
            z_hat_0.shape,
            device=device,
            dtype=torch.float32,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        inv_sigmas = torch.flip(self.scheduler.sigmas, dims=[0]).to(
            device=device, dtype=torch.float32
        )

        print(
            f"Running deterministic inversion with {len(inv_sigmas)} sigma values "
            f"(0 → 1)"
        )
        print(
            f"Sigma range: {inv_sigmas[0].item():.6f} → "
            f"{inv_sigmas[-1].item():.6f}"
        )

        if save_images:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean, device=device)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
            )
            latents_std_inv = (
                1.0
                / torch.tensor(self.vae.config.latents_std, device=device)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
            )

        for i, t in enumerate(tqdm(inv_sigmas, desc="Deterministic Inversion")):
            z_t = t * z_hat_0 + (1.0 - t) * eps

            # Save latent
            step_name = f"step_{i:04d}"
            latent_path = os.path.join(save_latent_dir, f"{step_name}.pt")
            torch.save(z_t.to(save_dtype).cpu(), latent_path)
            
            if save_images:
                # Decode latent
                z_decode = z_t.to(device=device, dtype=self.vae.dtype)
                mean = latents_mean.to(dtype=z_decode.dtype)
                std_inv = latents_std_inv.to(dtype=z_decode.dtype)
                z_decode = z_decode / std_inv + mean

                video = self.vae.decode(z_decode, return_dict=False)[0]
                frames = self.video_processor.postprocess_video(video, output_type="pil")

                if isinstance(frames[0], list):
                    frame_img = frames[0][frame_index]
                else:
                    frame_img = frames[frame_index]

                image_path = os.path.join(save_image_dir, f"{step_name}.png")
                frame_img.save(image_path)

