"""Inference over 10 unseen faces x 10 new prompts (100 videos).

Unlike `inference/run_inference.py` (which pairs precomputed-invert faces with
slugs from `training/input_prompts.py`), this runner:

  * takes a directory of unseen face images (`--faces_dir`, default
    /home/s2710099/data/unseen_faces holding face_0.png .. face_9.png),
  * uses a fresh set of 10 prompts defined here (`UNSEEN_PROMPTS`), distinct
    from `training/input_prompts.py` (5 short + 5 verbose),
  * runs the FULL cross product: every face against every prompt => 100 videos,
  * computes the required `ref_latents` inversion trajectory INLINE via the
    deterministic (VAE-only, linear FlowMatch) formula — the unseen faces have
    no precomputed invert on disk, and the deterministic path needs no
    transformer forwards, so no second model load.

The heavy pipeline (Wan 2.2 A14B + both experts + ControlNet) is built ONCE
(reusing `run_inference.build_pipeline`). The outer loop is over faces: each
face's deterministic ref_latents is computed once and reused across all 10
prompts (10 inversions total, not 100). One mp4 + one wandb run per (face,
prompt) pair => 100 videos total.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import wandb

# Reuse the proven pipeline builder, video writer, and the resolution / frame /
# negative-prompt constants so this runner stays bit-identical to the main one.
from inference.run_inference import (
    build_pipeline,
    save_video,
    NEGATIVE_PROMPT,
    HEIGHT,
    WIDTH,
    NUM_FRAMES,
)

WANDB_PROJECT = "CN_PTD_unseen_inference"

# ---- 10 NEW prompts (none of these slugs/scenes appear in input_prompts.py) --
# First 5 are short "<scene>, <art style>" prompts (the input_prompts.py idiom);
# last 5 are verbose "<scene>, static, no movement, in style of <X>, high
# quality" prompts (the old PTD-runner style the user provided as examples).
UNSEEN_PROMPTS = {
    # short
    "salt_marsh":           "salt marsh at dawn, oil painting",
    "lava_field":           "obsidian lava field, oil painting",
    "tea_hills":            "terraced tea hills, watercolor painting",
    "driftwood_beach":      "driftwood beach, oil painting",
    "pine_ridge":           "misty pine ridge, oil painting",
    # long / verbose
    "tundra_impressionist": "frozen tundra, static, no movement, in style of impressionist painting, high quality",
    "reef_photoreal":       "coral reef, static, no movement, photorealistic, high quality",
    "canyon_watercolor":    "desert canyon, static, no movement, in style of watercolor painting, high quality",
    "meadow_streetart":     "alpine meadow, static, no movement, in style of street art, high quality",
    "dunes_cubist":         "rolling dunes, static, no movement, in style of cubist painting, high quality",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True,
                   help="Trained ControlNet .safetensors.")
    p.add_argument("--base_model_path", type=str, required=True,
                   help="Wan-AI Wan2.2-T2V-A14B-Diffusers snapshot dir.")
    p.add_argument("--controlnet_config_repo", type=str, required=True,
                   help="HED config snapshot dir (architecture only).")
    p.add_argument("--faces_dir", type=str, required=True,
                   help="Directory of unseen face PNGs (face_*.png). Every face "
                        "is run against every prompt (full cross product).")
    p.add_argument("--project_name", type=str, required=True,
                   help="Output subdir name AND wandb run-name prefix.")
    p.add_argument("--controlnet_stride", type=int, default=3)
    p.add_argument("--controlnet_weight", type=float, default=1.0)
    p.add_argument("--initial_blending_coeff", type=float, default=0.4)
    p.add_argument("--direct_transfer_steps", type=int, default=45)
    p.add_argument("--decayed_transfer_steps", type=int, default=22)
    p.add_argument("--Kp", type=float, default=0.5)
    p.add_argument("--Ki", type=float, default=0.2)
    p.add_argument("--max_blending_coeff_delta", type=float, default=0.05)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--num_inference_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=8)
    return p.parse_args()


@torch.no_grad()
def deterministic_ref_latents(
    pipe,
    face_img: Image.Image,
    *,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    seed: int,
) -> torch.Tensor:
    """Build the (num_inference_steps, 1, 16, T_lat, H_lat, W_lat) ref_latents
    trajectory for `face_img` using the deterministic linear-FlowMatch formula
    `z_t = sigma * z_face + (1 - sigma) * eps`.

    Mirrors `PTD_Pipeline.WanInversionPipeline.deterministic_invert` exactly
    (same VAE encode + normalization, same flipped scheduler sigmas, same fixed
    eps) but keeps the trajectory in memory and returns it in denoising-step
    order (index 0 = noisy, index N-1 = near-clean face), matching what
    `run_inference.py --invert_type=deterministic` produces by loading the
    first N step_*.pt files. VAE-only: no transformer forward.
    """
    device = pipe._execution_device

    # VAE-encode the face as a static video latent (replica of
    # WanInversionPipeline._encode_reference_image_to_latents).
    img = pipe.video_processor.preprocess(face_img, height=height, width=width)
    img = img.to(device=device, dtype=torch.float32)
    video = img.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)
    video = video.to(device=device, dtype=pipe.vae.dtype)
    posterior = pipe.vae.encode(video).latent_dist
    z_face = posterior.mode()  # deterministic mean

    z_dim = pipe.vae.config.z_dim
    latents_mean = torch.tensor(
        pipe.vae.config.latents_mean, device=device, dtype=z_face.dtype
    ).view(1, z_dim, 1, 1, 1)
    latents_std = torch.tensor(
        pipe.vae.config.latents_std, device=device, dtype=z_face.dtype
    ).view(1, z_dim, 1, 1, 1)
    z_face = ((z_face - latents_mean) / latents_std).to(torch.float32)

    # Fixed noise (seeded for reproducibility across the 10 prompts).
    gen = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn(z_face.shape, generator=gen, device=device, dtype=torch.float32)

    # Scheduler sigmas flipped to ascend 0 -> 1; inv_sigmas[i] is the sigma at
    # denoising step i. deterministic_invert writes N+1 files; the inference
    # loop consumes only the first N, so we stack i = 0 .. N-1.
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    inv_sigmas = torch.flip(pipe.scheduler.sigmas, dims=[0]).to(
        device=device, dtype=torch.float32
    )

    refs = []
    for i in range(num_inference_steps):
        sigma = inv_sigmas[i]
        z_t = sigma * z_face + (1.0 - sigma) * eps
        refs.append(z_t)
    ref_latents = torch.stack(refs, dim=0)  # (N, 1, 16, T_lat, H_lat, W_lat)

    print(f"[invert] deterministic ref_latents: shape={tuple(ref_latents.shape)} "
          f"dtype={ref_latents.dtype}  sigma range "
          f"{inv_sigmas[0].item():.4f} -> {inv_sigmas[num_inference_steps - 1].item():.4f}")
    return ref_latents


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    faces_dir = Path(args.faces_dir)
    if not faces_dir.is_dir():
        raise NotADirectoryError(f"faces_dir not found: {faces_dir}")
    # Sort numerically by the index in face_<N>.png so face_10 would sort after
    # face_9 (plain lexical sort would not).
    face_paths = sorted(
        faces_dir.glob("face_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not face_paths:
        raise FileNotFoundError(f"no face_*.png found in {faces_dir}")

    out_root = Path.home() / "outputs" / "inference" / args.project_name
    out_root.mkdir(parents=True, exist_ok=True)

    slugs = list(UNSEEN_PROMPTS.keys())
    total = len(face_paths) * len(slugs)
    print(f"[run] faces={len(face_paths)}  prompts={len(slugs)}  "
          f"total_videos={total}  cn_weight={args.controlnet_weight}  "
          f"cfg={args.guidance_scale}")

    pipe = build_pipeline(args)

    from accelerate.hooks import remove_hook_from_module

    job_i = 0
    for face_path in face_paths:
        face_idx = int(face_path.stem.split("_")[1])
        face_img = Image.open(face_path).convert("RGB")

        # Compute the deterministic inversion ONCE PER FACE — reused across all
        # 10 prompts (the trajectory is prompt-independent), so 10 inversions
        # total rather than 100.
        ref_latents = deterministic_ref_latents(
            pipe,
            face_img,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
        )

        for slug in slugs:
            job_i += 1
            prompt_text = UNSEEN_PROMPTS[slug]
            run_name = f"{args.project_name}_face{face_idx}_{slug}"
            mp4_path = out_root / f"face_{face_idx}_{slug}.mp4"

            run_config = dict(
                project_name=args.project_name,
                checkpoint=args.checkpoint_path,
                face_image=str(face_path),
                face_idx=face_idx,
                controlnet_stride=args.controlnet_stride,
                controlnet_weight=args.controlnet_weight,
                initial_blending_coeff=args.initial_blending_coeff,
                direct_transfer_steps=args.direct_transfer_steps,
                decayed_transfer_steps=args.decayed_transfer_steps,
                Kp=args.Kp,
                Ki=args.Ki,
                max_blending_coeff_delta=args.max_blending_coeff_delta,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                invert_type="deterministic",
                slug=slug,
                prompt=prompt_text,
            )
            with wandb.init(project=WANDB_PROJECT, name=run_name, config=run_config,
                            reinit=True):
                print(f"[{job_i}/{total}] run={run_name}  "
                      f"cn_weight={args.controlnet_weight}  prompt={prompt_text!r}")

                # Re-pin CN to GPU (accelerate re-attaches its hook per __call__).
                remove_hook_from_module(pipe.controlnet, recurse=True)
                pipe.controlnet.to("cuda")
                generator = torch.Generator().manual_seed(args.seed)

                cn_frames = ([face_img] * NUM_FRAMES) if args.controlnet_weight > 0 else None
                out = pipe(
                    controlnet_frames=cn_frames,
                    ref_latents=ref_latents,
                    prompt=prompt_text,
                    negative_prompt=NEGATIVE_PROMPT,
                    height=HEIGHT,
                    width=WIDTH,
                    num_frames=NUM_FRAMES,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    controlnet_weight=args.controlnet_weight,
                    controlnet_stride=args.controlnet_stride,
                    direct_transfer_steps=args.direct_transfer_steps,
                    decayed_transfer_steps=args.decayed_transfer_steps,
                    initial_blending_coeff=args.initial_blending_coeff,
                    Kp=args.Kp,
                    Ki=args.Ki,
                    max_blending_coeff_delta=args.max_blending_coeff_delta,
                    generator=generator,
                    output_type="np",
                )
                frames = out.frames[0]  # (T, H, W, 3) float in [0, 1]
                save_video(frames, mp4_path, fps=args.fps)
                print(f"[done] wrote {mp4_path}  (face={face_idx}, slug={slug})")

                # Drop per-run transients before the next prompt (ref_latents is
                # reused across this face's prompts, so it is NOT freed here).
                del out, frames
                gc.collect()
                torch.cuda.empty_cache()

        # Done with this face — free its ref_latents and image before the next.
        del ref_latents, face_img
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[summary] all {total} videos done -> {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
