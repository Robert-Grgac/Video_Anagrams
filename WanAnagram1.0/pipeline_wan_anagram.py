from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

# Import the original pipeline
from diffusers import WanPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

# Custom class imports
from utils.views import ViewTransform, IdentityView
from utils.schedules import AnagramSchedule
from utils.mixers import cfg, mix_joint, mix_anagram

class WanAnagramPipeline(WanPipeline):
    """
    WanAnagramPipeline: single-latent, two-stage Joint -> Anagram.

    - Stage 1 (0..K-1): average the guided model outputs from prompt A and prompt B on the same latent
    - Stage 2 (K..T-1): compute guided model output for (view1,promptA) and (view2,promptB),
                       inverse-align them, then average.

    Everything else (scheduler, VAE decode, etc.) is inherited from WanPipeline.
    """

    @torch.no_grad()
    def __call__(
        self,
        # New: two prompts (A and B)
        prompt_a: Union[str, List[str]] = None,
        prompt_b: Union[str, List[str]] = None,

        # Original args from WAN
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,

        # Advanced: allow passing precomputed embeds (optional)
        prompt_a_embeds: Optional[torch.Tensor] = None,
        prompt_b_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,

        # View transform for prompt B (view A is always identity)
        view_b: Optional[ViewTransform] = None,

        # Anagram schedule config
        joint_steps: Optional[int] = 20,
        
        #Guidance scale for joint stage and anagram stage respectively
        guidance_scale_joint: float = 6.0,
        guidance_scale_anagram: Optional[float] = 9.0,
        
        #Flag for alternating vs mean reduction
        alternating_reduction: bool = False,

        # Output formatting (same as WAN)
        output_type: Optional[str] = "np",
        return_dict: bool = True,

        # Pass-through to transformer attention processor
        attention_kwargs: Optional[Dict[str, Any]] = None,

        # Callback support (same as WAN)
        callback_on_step_end: Optional[Any] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        """
        Generate a *single* anagram video that matches:
        - prompt_a when viewed normally
        - prompt_b when viewed under view_b (e.g., horizontal flip)

        Notes:
        - This implementation assumes batch_size == 1 for prompts (typical for research).
          You can generalize to lists later, but start simple and correct.
        """

        # ---------- 0) Defaults ----------
        schedule = AnagramSchedule(joint_steps=joint_steps)
        
        if view_b is None:
            raise ValueError("You must pass a `view_b` transform (e.g., HorizontalFlipView).")
        view_a = IdentityView()

        # ---------- 1) Basic validation ----------
        # Reuse WAN's input checks for height/width/etc.
        self.check_inputs(
            prompt_a,
            negative_prompt,
            height,
            width,
            prompt_embeds=prompt_a_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale_2=guidance_scale_2,
        )

        if prompt_b is None and prompt_b_embeds is None:
            raise ValueError("Provide either `prompt_b` or `prompt_b_embeds`.")

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

        if isinstance(prompt_a, list) or isinstance(prompt_b, list):
            raise ValueError("WanAnagramPipeline currently supports only single-string prompts (not lists).")
        batch_size = 1

        # ---------- 2) Encode prompts ----------
        #   eA : embedding for prompt_a
        #   eB : embedding for prompt_b
        #   eU : embedding for "uncond" / negative prompt (used for CFG)
        #
        # WAN's encode_prompt returns (prompt_embeds, negative_prompt_embeds).
        # We call it separately for A and B so we can reuse its tokenizer/text_encoder logic.

        do_cfg = guidance_scale is not None and guidance_scale > 1.0

        # Prompt A
        eA, eU_A = self.encode_prompt(
            prompt=prompt_a,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_a_embeds,
            negative_prompt_embeds=negative_prompt_embeds, 
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Prompt B
        eB, eU_B = self.encode_prompt(
            prompt=prompt_b,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_b_embeds,
            negative_prompt_embeds=negative_prompt_embeds,  # same uncond by default
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Use the unconditional embeds from A (they should match B if negative_prompt is same)
        if do_cfg and eU_A is not None and eU_B is not None:
            if not torch.equal(eU_A, eU_B):
                raise ValueError("Negative/unconditional embeds differ between prompt A and B. Use a shared negative_prompt.")
        
        eU = eU_A

        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        eA = eA.to(transformer_dtype)
        eB = eB.to(transformer_dtype)
        if eU is not None:
            eU = eU.to(transformer_dtype)

        # ---------- 3) Prepare timesteps ----------
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        # ---------- 4) Prepare latents ----------
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

        # This mask is only relevant for expand_timesteps mode (WAN2.2 ti2v path)
        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # ---------- 5) Helper: build timestep tensor ----------
        def build_timestep_tensor(t_scalar: torch.Tensor, base_latents: torch.Tensor, repeat_k: int) -> torch.Tensor:
            """
            Returns a timestep tensor matching WAN's expectation.

            - If expand_timesteps: timestep is (B, seq_len)
            - Else: timestep is (B,)
            Then we repeat it repeat_k times along batch dimension to match latent batching.
            """
            if self.config.expand_timesteps:
                # Copy WAN behavior:
                # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                temp_ts = (mask[0][0][:, ::2, ::2] * t_scalar).flatten()
                base_ts = temp_ts.unsqueeze(0).expand(base_latents.shape[0], -1)
            else:
                base_ts = t_scalar.expand(base_latents.shape[0])

            if repeat_k == 1:
                return base_ts

            # Repeat batch dimension: (B, ...) -> (B*repeat_k, ...)
            return base_ts.repeat_interleave(repeat_k, dim=0)

        # ---------- 6) Helper: single batched transformer forward ----------
        def model_forward(
            current_model,
            latent_batch: torch.Tensor,
            timestep_batch: torch.Tensor,
            embed_batch: torch.Tensor,
        ) -> torch.Tensor:
            out = current_model(
                hidden_states=latent_batch,
                timestep=timestep_batch,
                encoder_hidden_states=embed_batch,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            return out

        # ---------- 7) Denoising loop ----------
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        cosine_similarity_list = []

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                # Pick model (WAN two-stage) exactly like original
                if boundary_timestep is None or t >= boundary_timestep:
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2

                # Ensure dtype matches transformer
                base_latents = latents.to(transformer_dtype)

                if schedule.is_joint(i):
                    # ========== STAGE 1: Joint diffusion ==========
                    # We want: mean( CFG(z, eA), CFG(z, eB) )
                    #
                    # Batched forward pass:
                    #   [z, z, z] with [eU, eA, eB] (if CFG)
                    # or:
                    #   [z, z] with [eA, eB] (if no CFG)

                    if do_cfg:
                        k = 3
                        latent_batch = torch.cat([base_latents, base_latents, base_latents], dim=0)
                        embed_batch = torch.cat([eU, eA, eB], dim=0)
                        ts_batch = build_timestep_tensor(t, base_latents, repeat_k=k)

                        out = model_forward(current_model, latent_batch, ts_batch, embed_batch)
                        out_u, out_a, out_b = out.chunk(3, dim=0)

                        out_a_cfg = cfg(out_u, out_a, guidance_scale_joint)
                        out_b_cfg = cfg(out_u, out_b, guidance_scale_joint)
                        
                        #For debugging purposes
                        cos = F.cosine_similarity(
                                out_a_cfg.flatten(1),
                                out_b_cfg.flatten(1),
                                dim=1
                            )
                        cosine_similarity_list.append(cos.mean().item())
                        
                        model_out = mix_joint(out_a_cfg, out_b_cfg)
                    else:
                        k = 2
                        latent_batch = torch.cat([base_latents, base_latents], dim=0)
                        embed_batch = torch.cat([eA, eB], dim=0)
                        ts_batch = build_timestep_tensor(t, base_latents, repeat_k=k)

                        out = model_forward(current_model, latent_batch, ts_batch, embed_batch)
                        out_a, out_b = out.chunk(2, dim=0)
                        model_out = mix_joint(out_a, out_b)

                else:
                    # ========== STAGE 2: Visual Anagram diffusion ==========
                    # We want: mean( inv(v1)(CFG(v1(z), eA)), inv(v2)(CFG(v2(z), eB)) )
                    #
                    # v1 = identity
                    # v2 = view_b
                    # alt_view = the current view
                    # zav = latent under alt_view
                    alt = (i - schedule.joint_steps) % 2 == 0
                    if alt == 0:
                        view = view_a # identity
                        alt_cond = eA
                    else:
                        view = view_b
                        alt_cond = eB

                    zv = view.forward(base_latents)
                    z1 = view_a.forward(base_latents)  # identity
                    z2 = view_b.forward(base_latents)

                    if do_cfg:
                        if alternating_reduction:
                            # Batch: [zv, zv] with [eU, alt_cond]
                            latent_batch = torch.cat([zv, zv], dim=0)
                            embed_batch = torch.cat([eU, alt_cond], dim=0)
                            ts_batch = build_timestep_tensor(t, base_latents, repeat_k=2)

                            out = model_forward(current_model, latent_batch, ts_batch, embed_batch)
                            u, c = out.chunk(2, dim=0)

                            out_cfg = cfg(u, c, guidance_scale_anagram)

                            out_cfg_aligned = view.inverse(out_cfg)
                            model_out = out_cfg_aligned
                        else:
                            # Batch: [z1, z1, z2, z2] with [eU, eA, eU, eB]
                            k = 4
                            latent_batch = torch.cat([z1, z1, z2, z2], dim=0)
                            embed_batch = torch.cat([eU, eA, eU, eB], dim=0)
                            ts_batch = build_timestep_tensor(t, base_latents, repeat_k=k)

                            out = model_forward(current_model, latent_batch, ts_batch, embed_batch)
                            u1, c1, u2, c2 = out.chunk(4, dim=0)

                            out1_cfg = cfg(u1, c1, guidance_scale_anagram)
                            out2_cfg = cfg(u2, c2, guidance_scale_anagram)
                            
                            #For debugging purposes
                            cos = F.cosine_similarity(
                                    out1_cfg.flatten(1),
                                    out2_cfg.flatten(1),
                                    dim=1
                                )
                            cosine_similarity_list.append(cos.mean().item())

                            out1_aligned = view_a.inverse(out1_cfg)  # identity
                            out2_aligned = view_b.inverse(out2_cfg)

                            model_out = mix_anagram([out1_aligned, out2_aligned])
                    else:
                        if alternating_reduction:
                            latent_batch = zv
                            embed_batch = alt_cond
                            ts_batch = build_timestep_tensor(t, base_latents, repeat_k=1)

                            out = model_forward(current_model, latent_batch, ts_batch, embed_batch)
                            model_out = view.inverse(out)
                        else:
                            # Batch: [z1, z2] with [eA, eB]
                            k = 2
                            latent_batch = torch.cat([z1, z2], dim=0)
                            embed_batch = torch.cat([eA, eB], dim=0)
                            ts_batch = build_timestep_tensor(t, base_latents, repeat_k=k)

                            out = model_forward(current_model, latent_batch, ts_batch, embed_batch)
                            out1, out2 = out.chunk(2, dim=0)

                            model_out = mix_anagram([view_a.inverse(out1), view_b.inverse(out2)])

                # ----- Scheduler step: identical to WAN -----
                latents = self.scheduler.step(model_out, t, latents, return_dict=False)[0]

                # ----- Optional callback hook (same as WAN) -----
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for kname in callback_on_step_end_tensor_inputs:
                        callback_kwargs[kname] = locals()[kname]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                # ----- Progress bar update (same as WAN) -----
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        # ---------- 8) Decode (copied from WAN) ----------
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

        print("Average cosine similarity between view outputs during anagram stage:")
        print(cosine_similarity_list)
        return WanPipelineOutput(frames=video)

        