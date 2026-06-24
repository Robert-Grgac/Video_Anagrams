"""ControlNet ablation — single-CN vs dual-CN over the canonical 100 pairs.

Two consecutive inference passes over the SAME 100 face↔slug pairs used by
inference/run_inference.py (the CN+PTD pipeline) and the PTD baselines:

  Pass A — single shared CN (beta-002, trained on BOTH experts) driving the
           two-expert WanTextToVideoControlnetPipeline.
           Output: /home/s2710099/outputs/inference/single_cn_dual_experts
  Pass B — dual CN: beta-007 high-noise + beta-004 low-noise, via
           WanTextToVideoDualControlnetPipeline.
           Output: /home/s2710099/outputs/inference/dual_cn_dual_experts

Video-output-only: NO wandb / no analytics. Each pass is built ONCE and loops
100 faces internally (the pipeline is never rebuilt per video).

The two pipelines are NEVER co-resident: pass A is fully torn down (del + gc +
empty_cache) before pass B is built, so only one set of transformers is ever in
memory. Within a pass, enable_model_cpu_offload() keeps just the active expert
on the GPU (the high/low experts swap on/off as the denoise crosses the
boundary), which is what fits the two-expert model on a single 45 GB card.

Conditions (apples-to-apples with the CN+PTD reference run):
  528×528, 61 frames, 100 steps, guidance_scale=5.0, canny control input,
  controlnet_weight=1.0, controlnet_stride=3, controlnet_guidance_end=1.0.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Reuse the tested loaders/helpers from the standalone inference scripts so the
# CN build + accelerate-hook pinning logic lives in exactly one place.
from training.run_inference_beta import (
    ALL_PROMPTS,
    _ALL_SLUGS,
    build_pipeline as build_single_pipeline,
    load_canny_image,
    resolve_slug,
    save_video,
)
from training.run_inference_beta_dual import build_pipeline as build_dual_pipeline

PAIRS = [(i, _ALL_SLUGS[i]) for i in range(100)]

# Same negative prompt as inference/run_inference.py (the CN+PTD reference set),
# so the conditioning matches the combined pipeline exactly.
NEGATIVE_PROMPT = (
    "blurry, low quality, worst quality, jpeg artifacts, text, subtitles, "
    "watermark, static image, still frame, distorted anatomy, inconsistent motion"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_path", type=str, required=True,
                   help="Wan-AI Wan2.2-T2V-A14B-Diffusers snapshot dir.")
    p.add_argument("--controlnet_config_repo", type=str, required=True,
                   help="HED config snapshot dir (architecture only).")
    p.add_argument("--cache_dir", type=str, required=True,
                   help="Wan-beta precompute cache (for canny + manifest).")
    # Checkpoints (defaults point at the on-cluster /home paths; sbatch stages
    # them to scratch and overrides these).
    p.add_argument("--single_checkpoint", type=str,
                   default="/home/s2710099/checkpoints/wan-beta/beta-002_final.safetensors")
    p.add_argument("--high_checkpoint", type=str,
                   default="/home/s2710099/checkpoints/wan-beta/beta-007_v2_final.safetensors")
    p.add_argument("--low_checkpoint", type=str,
                   default="/home/s2710099/checkpoints/wan-beta/beta-004_final.safetensors")
    p.add_argument("--output_root", type=str,
                   default="/home/s2710099/outputs/inference",
                   help="Base dir; passes write to <output_root>/single_cn_dual_experts "
                        "and <output_root>/dual_cn_dual_experts. On Snellius pass "
                        "--output_root /home/astergiou/outputs/inference.")
    p.add_argument("--setup", type=str, default="both",
                   choices=["both", "single", "dual"],
                   help="Run only the single-CN pass, only the dual-CN pass, or "
                        "both consecutively (default). Use single/dual to split "
                        "across two Snellius jobs.")
    p.add_argument("--single_output_dir", type=str, default=None,
                   help="Override for the single-CN output dir (default "
                        "<output_root>/single_cn_dual_experts).")
    p.add_argument("--dual_output_dir", type=str, default=None,
                   help="Override for the dual-CN output dir (default "
                        "<output_root>/dual_cn_dual_experts).")
    p.add_argument("--control_subdir", type=str, default="canny",
                   help="Control modality the CNs were trained on (canny).")
    p.add_argument("--height", type=int, default=528)
    p.add_argument("--width", type=int, default=528)
    p.add_argument("--num_frames", type=int, default=61)
    p.add_argument("--num_inference_steps", type=int, default=100)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--controlnet_weight", type=float, default=1.0)
    p.add_argument("--controlnet_stride", type=int, default=3)
    p.add_argument("--controlnet_guidance_start", type=float, default=0.0)
    p.add_argument("--controlnet_guidance_end", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Cap on pairs per pass (debug). None = all 100.")
    return p.parse_args()


def _pairs(args: argparse.Namespace):
    return PAIRS if args.max_pairs is None else PAIRS[: args.max_pairs]


def run_single_cn(args: argparse.Namespace) -> None:
    """Pass A — single shared CN (beta-002) on the two-expert pipeline."""
    from accelerate.hooks import remove_hook_from_module

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.single_output_dir
                   or Path(args.output_root) / "single_cn_dual_experts")
    out_dir.mkdir(parents=True, exist_ok=True)

    build_args = SimpleNamespace(
        base_model_path=args.base_model_path,
        controlnet_config_repo=args.controlnet_config_repo,
        checkpoint_path=args.single_checkpoint,
    )
    print(f"\n========== PASS A: single CN (beta-002) -> {out_dir} ==========")
    pipe = build_single_pipeline(build_args)

    try:
        for face_idx, slug in _pairs(args):
            slug = resolve_slug(cache_dir, face_idx, slug)
            target = out_dir / f"face_{face_idx}_{slug}.mp4"
            # Resumability: skip already-written videos so a timed-out job can be
            # re-submitted to continue where it left off.
            if target.exists() and target.stat().st_size > 0:
                print(f"[single][skip] {target.name} — exists")
                continue
            prompt_text = ALL_PROMPTS[slug]
            canny_img = load_canny_image(cache_dir, face_idx, args.control_subdir)

            # accelerate re-attaches a hook to the CN on every __call__; strip it
            # and pin the CN to GPU so the first Conv3D sees a GPU input.
            remove_hook_from_module(pipe.controlnet, recurse=True)
            pipe.controlnet.to("cuda")
            generator = torch.Generator().manual_seed(args.seed)

            out = pipe(
                controlnet_frames=[canny_img] * args.num_frames,
                prompt=prompt_text,
                negative_prompt=NEGATIVE_PROMPT,
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
            frames = out.frames[0]
            save_video(frames, target, fps=args.fps)
            print(f"[single][done] {target.name}")
            del out, frames, canny_img
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        # Fully release the single-CN pipeline before building the dual one so
        # the two are never co-resident.
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


def run_dual_cn(args: argparse.Namespace) -> None:
    """Pass B — dual CN: beta-007 high + beta-004 low."""
    from accelerate.hooks import remove_hook_from_module

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.dual_output_dir
                   or Path(args.output_root) / "dual_cn_dual_experts")
    out_dir.mkdir(parents=True, exist_ok=True)

    build_args = SimpleNamespace(
        base_model_path=args.base_model_path,
        controlnet_config_repo=args.controlnet_config_repo,
        high_checkpoint=args.high_checkpoint,
        low_checkpoint=args.low_checkpoint,
    )
    print(f"\n========== PASS B: dual CN (beta-007 high + beta-004 low) -> {out_dir} ==========")
    pipe = build_dual_pipeline(build_args)

    try:
        for face_idx, slug in _pairs(args):
            slug = resolve_slug(cache_dir, face_idx, slug)
            target = out_dir / f"face_{face_idx}_{slug}.mp4"
            if target.exists() and target.stat().st_size > 0:
                print(f"[dual][skip] {target.name} — exists")
                continue
            prompt_text = ALL_PROMPTS[slug]
            canny_img = load_canny_image(cache_dir, face_idx, args.control_subdir)

            for cn in (pipe.controlnet_high, pipe.controlnet_low):
                remove_hook_from_module(cn, recurse=True)
                cn.to("cuda")
            generator = torch.Generator().manual_seed(args.seed)

            out = pipe(
                controlnet_frames=[canny_img] * args.num_frames,
                prompt=prompt_text,
                negative_prompt=NEGATIVE_PROMPT,
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
            frames = out.frames[0]
            save_video(frames, target, fps=args.fps)
            print(f"[dual][done] {target.name}")
            del out, frames, canny_img
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    print(f"[config] {args.height}x{args.width} x {args.num_frames}f, "
          f"{args.num_inference_steps} steps, guidance={args.guidance_scale}, "
          f"cn_weight={args.controlnet_weight}, cn_stride={args.controlnet_stride}, "
          f"cn_end={args.controlnet_guidance_end}, control={args.control_subdir}")
    print(f"[ckpt] single={args.single_checkpoint}")
    print(f"[ckpt] high={args.high_checkpoint}")
    print(f"[ckpt] low={args.low_checkpoint}")

    print(f"[plan] setup={args.setup}")
    if args.setup in ("both", "single"):
        run_single_cn(args)
    if args.setup in ("both", "dual"):
        run_dual_cn(args)

    print(f"\n[all done] CN ablation complete (setup={args.setup}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
