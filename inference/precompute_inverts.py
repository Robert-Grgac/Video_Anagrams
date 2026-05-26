"""Precompute Euler ODE inversion trajectories for all 100 face images.

Subclasses `WanInversionPipeline` (`PTD_Pipeline/WanInversionPipeline.py`)
and adds `invert_with_trajectory()` — same Euler ODE math as the original
`invert()` but collects z at every step instead of saving just the final z.

Each transformer expert is ~28 GB resident in bf16 (54 GB sharded on disk).
A single expert exceeds the headroom on a 45 GB A40 if the VAE / text encoder
are also fully on GPU, so we rely on `enable_model_cpu_offload()` (the same
mechanism the inference pipeline uses successfully) to hop components on and
off GPU between forward calls. Manual `.to(device)` on the transformers is
explicitly avoided.

For each face in `raw_face/face_{i}.pt`:
  1. Encode as a static-video Wan VAE latent via the pipeline helper.
  2. Walk sigma from ~0 (clean) to ~1 (noise) via the boundary-aware expert
     switch, collecting z each step.
  3. Flip the trajectory to denoising-step order so the inference pipeline
     can index `ref_latents[i]` directly at denoising step i (i=0 -> noisy,
     i=N-1 -> clean face).
  4. Save to `<output_dir>/face_{i}.pt` as a single tensor of shape
     `(num_inference_steps, 1, 16, T_lat, H_lat, W_lat)`.

Usage:
    python -m inference.precompute_inverts \
        --base_model_path "$WAN_MODEL" \
        --cache_dir       "$WAN_BETA_CACHE" \
        --output_dir      "$WAN_BETA_CACHE/invert_face"
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

import torch
import PIL.Image
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from PTD_Pipeline.WanInversionPipeline import WanInversionPipeline


_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_path", type=str, required=True,
                   help="Wan-AI Wan2.2-T2V-A14B-Diffusers snapshot dir.")
    p.add_argument("--cache_dir", type=str, required=True,
                   help="Wan-beta cache root (holds raw_face/face_{i}.pt).")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Where to write face_{i}.pt invert files.")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num_frames", type=int, default=9)
    p.add_argument("--num_inference_steps", type=int, default=100)
    p.add_argument("--num_faces", type=int, default=100,
                   help="Process face indices 0..N-1 (default 100).")
    p.add_argument("--start_idx", type=int, default=0,
                   help="Starting face index (resume support).")
    p.add_argument("--save_dtype", type=str, default="bf16",
                   choices=list(_DTYPE_MAP.keys()))
    p.add_argument("--prompt", type=str, default="",
                   help="Inversion prompt (default empty = unconditional).")
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run faces whose output file already exists.")
    return p.parse_args()


class WanInversionTrajectoryPipeline(WanInversionPipeline):
    """Adds `invert_with_trajectory()` to `WanInversionPipeline`.

    Math is identical to the parent's `invert()` (Euler ODE, boundary-aware
    expert switch, `z = z + dt * noise_pred`). The only differences are:
      - collect `z` BEFORE each step into a trajectory list,
      - return the stacked trajectory instead of saving just the final z.
    """

    @torch.no_grad()
    def invert_with_trajectory(
        self,
        reference_image: Union[PIL.Image.Image, torch.Tensor],
        height: int = 512,
        width: int = 512,
        num_frames: int = 9,
        num_inference_steps: int = 100,
        prompt: str = "",
        negative_prompt: str = "",
        guidance_scale: float = 1.0,
        max_sequence_length: int = 512,
        attention_kwargs=None,
    ) -> torch.Tensor:
        """Euler ODE inversion clean->noise, returning the per-step trajectory.

        Returns a tensor of shape `(num_inference_steps, 1, 16, T_lat, H_lat,
        W_lat)` in fp32 on the pipeline's execution device, in INVERSION
        order (inv-step 0 -> clean, inv-step N-1 -> just before the last
        Euler step). The caller is responsible for flipping to denoising
        order if needed."""
        device = self._execution_device

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = (
                num_frames // self.vae_scale_factor_temporal
                * self.vae_scale_factor_temporal + 1
            )
        num_frames = max(num_frames, 1)

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
        boundary_timestep = (
            self.config.boundary_ratio * num_train_timesteps
            if self.config.boundary_ratio is not None
            else None
        )

        trajectory: List[torch.Tensor] = []
        previous_model = None

        for i in tqdm(range(len(inv_sigmas) - 1),
                      desc="Euler inversion", leave=False):
            sigma = inv_sigmas[i]
            sigma_next = inv_sigmas[i + 1]

            t = sigma * num_train_timesteps
            timestep = t.expand(z.shape[0]).to(device=device)

            # Snapshot z BEFORE the step. inv-step 0 is the clean face latent.
            # Keep on GPU; the 100 snapshots together are ~80 MB at fp32, much
            # smaller than the active transformer.
            trajectory.append(z.detach().clone())

            latent_model_input = z.to(transformer_dtype)

            if boundary_timestep is None or t >= boundary_timestep:
                current_model = self.transformer
                other_model = self.transformer_2
            else:
                current_model = self.transformer_2
                other_model = self.transformer

            # Mirror the parent invert()'s fast-path: when crossing the
            # boundary, manually evict the now-idle expert. enable_model_cpu_
            # offload would do this on its own via hooks, but doing it
            # explicitly avoids holding both experts on GPU during the swap.
            if previous_model is not None and current_model is not previous_model:
                other_model.to("cpu")
                torch.cuda.empty_cache()
            previous_model = current_model

            noise_pred = current_model(
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
            z = z.to(torch.float32) + dt * noise_pred.to(torch.float32)

        return torch.stack(trajectory, dim=0)


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    save_dtype = _DTYPE_MAP[args.save_dtype]

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_face_dir = cache_dir / "raw_face"
    if not raw_face_dir.exists():
        raise FileNotFoundError(f"raw_face dir not found: {raw_face_dir}")

    print(f"[load] building inversion pipeline from {args.base_model_path}")
    pipe = WanInversionTrajectoryPipeline.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
    )
    # `enable_model_cpu_offload` is what makes this fit on 45GB A40 — each
    # 28GB expert is materialized on GPU only during its own forward calls
    # and evicted between calls. Same mechanism the inference pipeline uses.
    pipe.enable_model_cpu_offload()
    print(f"[detect] boundary_ratio={pipe.config.boundary_ratio}")

    end_idx = args.start_idx + args.num_faces
    print(f"[run] faces {args.start_idx} .. {end_idx - 1}  "
          f"steps={args.num_inference_steps}  "
          f"hw={args.height}x{args.width}  frames={args.num_frames}")

    t_start_all = time.time()
    for face_idx in range(args.start_idx, end_idx):
        out_path = output_dir / f"face_{face_idx}.pt"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] face_{face_idx}: {out_path} already exists "
                  f"(use --overwrite to redo)")
            continue

        raw_path = raw_face_dir / f"face_{face_idx}.pt"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        raw_u8 = torch.load(raw_path, map_location="cpu", weights_only=True)
        face_img = Image.fromarray(raw_u8.permute(1, 2, 0).numpy())

        t0 = time.time()
        traj_inv = pipe.invert_with_trajectory(
            reference_image=face_img,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            prompt=args.prompt,
            max_sequence_length=args.max_sequence_length,
        )

        # Inversion goes clean -> noise (inv-step 0 is clean, inv-step N-1 is
        # noisy). The inference pipeline indexes by denoising step (i=0 is
        # noisy, i=N-1 is clean). Flip so ref_latents[i] is the right ref for
        # denoising step i.
        traj_denoise = torch.flip(traj_inv, dims=[0]).to(save_dtype).cpu()
        del traj_inv
        gc.collect()
        torch.cuda.empty_cache()

        torch.save(traj_denoise, out_path)
        dt = time.time() - t0
        size_mb = out_path.stat().st_size / (1024 ** 2)
        print(f"[done] face_{face_idx}: shape={tuple(traj_denoise.shape)} "
              f"dtype={traj_denoise.dtype} size={size_mb:.1f} MB "
              f"wall={dt:.1f}s -> {out_path}")

        del traj_denoise, face_img, raw_u8
        gc.collect()

    dt_all = time.time() - t_start_all
    print(f"[summary] {args.num_faces} faces in {dt_all/60:.1f} min "
          f"({dt_all / max(args.num_faces, 1):.1f}s per face avg)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
