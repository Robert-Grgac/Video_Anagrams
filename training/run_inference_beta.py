"""Standalone inference for a trained beta ControlNet.

Loads a saved ControlNet checkpoint, builds the full Wan 2.2 pipeline with
both experts (high-noise + low-noise), runs once on a (canny, prompt) pair
from the precomputed cache, and writes one mp4.

Uses pipe.enable_model_cpu_offload() so the two-expert pipeline fits on a
single 44GB A40 — both experts at bf16 ≈ 56GB if both stayed on GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_beta import cast_respecting_fp32_modules, detect_boundary_ratio
from training.input_prompts import PROMPTS_BATCH_1, PROMPTS_BATCH_2

if TYPE_CHECKING:
    from wan_t2v_controlnet_pipeline import WanTextToVideoControlnetPipeline

ALL_PROMPTS = {**PROMPTS_BATCH_1, **PROMPTS_BATCH_2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True,
                   help="Trained ControlNet .safetensors (e.g. .../beta-001_final.safetensors).")
    p.add_argument("--base_model_path", type=str, required=True,
                   help="Wan-AI Wan2.2-T2V-A14B-Diffusers snapshot dir.")
    p.add_argument("--controlnet_config_repo", type=str, required=True,
                   help="HED config snapshot dir (architecture only).")
    p.add_argument("--cache_dir", type=str, required=True,
                   help="Wan-beta precompute cache (for canny + slugs).")
    p.add_argument("--output_path", type=str, required=True,
                   help="Output mp4 path.")
    p.add_argument("--face_idx", type=int, default=0,
                   help="Which face's Canny to condition on.")
    p.add_argument("--slug", type=str, default=None,
                   help="Slug name from PROMPTS_BATCH_*. If unset, picks the first slug "
                        "available for face_idx in the cache manifest.")
    p.add_argument("--negative_prompt", type=str,
                   default="bad quality, worst quality")
    p.add_argument("--num_frames", type=int, default=9)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num_inference_steps", type=int, default=100)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--controlnet_weight", type=float, default=1.0)
    p.add_argument("--controlnet_stride", type=int, default=3)
    # ControlNet was trained only against the high-noise expert (sigma >= 0.875).
    # Limit injection to that regime: with FlowMatch's roughly-linear sigma
    # schedule, current_sampling_percent < (1 - boundary_ratio) = 0.125 covers
    # exactly the steps where the high-noise expert is active.
    p.add_argument("--controlnet_guidance_start", type=float, default=0.0)
    p.add_argument("--controlnet_guidance_end", type=float, default=0.125)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=8)
    return p.parse_args()


def resolve_slug(cache_dir: Path, face_idx: int, slug_arg: str | None) -> str:
    if slug_arg is not None:
        if slug_arg not in ALL_PROMPTS:
            raise ValueError(f"Unknown slug '{slug_arg}'.")
        return slug_arg
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    for rec in manifest:
        if rec["face_idx"] == face_idx:
            return rec["slug"]
    raise RuntimeError(f"No manifest entry found for face_idx={face_idx}.")


def load_canny_image(cache_dir: Path, face_idx: int) -> Image.Image:
    canny_path = cache_dir / "canny" / f"face_{face_idx}.pt"
    if not canny_path.exists():
        raise FileNotFoundError(canny_path)
    canny_u8 = torch.load(canny_path, map_location="cpu", weights_only=True)  # (3, H, W) uint8
    return Image.fromarray(canny_u8.permute(1, 2, 0).numpy())


def save_video(frames_np: np.ndarray, path: Path, fps: int = 8) -> None:
    import imageio
    arr = (np.clip(frames_np, 0.0, 1.0) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), list(arr), fps=fps, codec="libx264")


def build_pipeline(args: argparse.Namespace) -> "WanTextToVideoControlnetPipeline":
    from diffusers import AutoencoderKLWan
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel
    from safetensors.torch import load_file

    from wan_transformer import CustomWanTransformer3DModel
    from wan_controlnet import WanControlnet
    from wan_t2v_controlnet_pipeline import WanTextToVideoControlnetPipeline

    base = args.base_model_path

    print(f"[load] tokenizer + text_encoder ...")
    tokenizer = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        base, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    ).eval()

    print(f"[load] vae ...")
    vae = AutoencoderKLWan.from_pretrained(
        base, subfolder="vae", torch_dtype=torch.bfloat16,
    ).eval()

    print(f"[load] high-noise transformer ...")
    transformer = CustomWanTransformer3DModel.from_pretrained(
        base, subfolder="transformer", torch_dtype=torch.bfloat16,
    ).eval()

    # Load transformer_2 (low-noise expert) as the SAME custom subclass so it
    # accepts the controlnet_states kwarg the pipeline always passes. The class
    # only overrides forward(); state_dict keys match WanTransformer3DModel
    # exactly, so the checkpoint loads cleanly. With controlnet_states=None
    # (or unused stride misalignment), the residuals contribute nothing here.
    print(f"[load] low-noise transformer_2 ...")
    transformer_2 = CustomWanTransformer3DModel.from_pretrained(
        base, subfolder="transformer_2", torch_dtype=torch.bfloat16,
    ).eval()

    print(f"[load] scheduler ...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base, subfolder="scheduler",
    )

    print(f"[load] controlnet config from {args.controlnet_config_repo} ...")
    config = WanControlnet.load_config(args.controlnet_config_repo)
    controlnet = WanControlnet.from_config(config)
    cast_respecting_fp32_modules(controlnet, torch.bfloat16)
    print(f"[load] controlnet weights from {args.checkpoint_path} ...")
    sd = load_file(args.checkpoint_path)
    missing, unexpected = controlnet.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys when loading controlnet: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys when loading controlnet: {len(unexpected)}")
    controlnet.eval()

    boundary_ratio, src = detect_boundary_ratio(base, dict(transformer.config))
    print(f"[detect] boundary_ratio={boundary_ratio} ({src})")

    pipe = WanTextToVideoControlnetPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        controlnet=controlnet,
        scheduler=scheduler,
        boundary_ratio=boundary_ratio,
    )
    pipe.enable_model_cpu_offload()

    # Pin the ControlNet to GPU. The pipeline's prepare_controlnet_frames reads
    # self.controlnet.device at prep time; under model_cpu_offload that returns
    # "cpu", and accelerate's pre-forward hook does not migrate the resulting
    # input tensor → CPU/GPU device mismatch inside the first Conv3D. The
    # ControlNet is small (~1 GB at bf16) and runs every step, so pinning has
    # no memory cost and avoids the offload round-trip on each call.
    from accelerate.hooks import remove_hook_from_module
    remove_hook_from_module(pipe.controlnet, recurse=True)
    pipe.controlnet.to("cuda")
    return pipe


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir)

    slug = resolve_slug(cache_dir, args.face_idx, args.slug)
    prompt_text = ALL_PROMPTS[slug]
    canny_img = load_canny_image(cache_dir, args.face_idx)

    print(f"[input] face_idx={args.face_idx} slug='{slug}'")
    print(f"[input] prompt: {prompt_text!r}")

    pipe = build_pipeline(args)

    generator = torch.Generator().manual_seed(args.seed)
    out = pipe(
        controlnet_frames=[canny_img] * args.num_frames,
        prompt=prompt_text,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_weight=args.controlnet_weight,
        controlnet_stride=args.controlnet_stride,
        controlnet_guidance_start=args.controlnet_guidance_start,
        controlnet_guidance_end=args.controlnet_guidance_end,
        generator=generator,
        output_type="np",
    )
    frames = out.frames[0]  # (T, H, W, 3) float in [0, 1]
    save_video(frames, Path(args.output_path), fps=args.fps)
    print(f"[done] wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
