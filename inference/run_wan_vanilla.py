"""Vanilla Wan 2.2 T2V A14B baseline runner — 100 clean videos.

No ControlNet, no PTDiffusion. Plain text-to-video over the canonical 100
slugs (PROMPTS_BATCH_1 + PROMPTS_BATCH_2, declaration order), one mp4 per
prompt named {slug}.mp4. This is the reference distribution for the
quantitative analysis (FVD target + LPIPS-vanilla reference).

Loading mirrors PTD_Pipeline/run_WanPTDPipeline_100_fair.py: VAE in fp32
(avoids decode artifacts), the rest in bf16, via stock
WanPipeline.from_pretrained (diffusers 0.36 loads both MoE experts +
boundary_ratio + the snapshot's UniPC scheduler automatically).
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

from training.input_prompts import PROMPTS_BATCH_1, PROMPTS_BATCH_2

ALL_PROMPTS = {**PROMPTS_BATCH_1, **PROMPTS_BATCH_2}
_ALL_SLUGS = list(PROMPTS_BATCH_1.keys()) + list(PROMPTS_BATCH_2.keys())
assert len(_ALL_SLUGS) == 100, f"expected 100 slugs, got {len(_ALL_SLUGS)}"

# Same negative prompt as every method runner, for consistency.
NEGATIVE_PROMPT = (
    "blurry, low quality, worst quality, jpeg artifacts, text, subtitles, "
    "watermark, static image, still frame, distorted anatomy, inconsistent motion"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True,
                   help="Local Wan2.2-T2V-A14B-Diffusers snapshot dir (or HF id).")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Where {slug}.mp4 files are written.")
    p.add_argument("--height", type=int, default=528)
    p.add_argument("--width", type=int, default=528)
    p.add_argument("--num_frames", type=int, default=61)
    p.add_argument("--num_inference_steps", type=int, default=40)
    p.add_argument("--guidance_scale", type=float, default=4.0,
                   help="CFG for the high-noise expert (Wan A14B recommended 4.0).")
    p.add_argument("--guidance_scale_2", type=float, default=3.0,
                   help="CFG for the low-noise expert (Wan A14B recommended 3.0).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=16,
                   help="mp4 playback fps only; does not affect frame-indexed eval.")
    p.add_argument("--max_prompts", type=int, default=None,
                   help="Cap on number of slugs processed (debug). None = all 100.")
    p.add_argument("--start_idx", type=int, default=0,
                   help="Skip the first N slugs (resume support).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] vae (fp32) ...")
    vae = AutoencoderKLWan.from_pretrained(
        args.model_path, subfolder="vae", torch_dtype=torch.float32,
    )
    print("[load] WanPipeline (bf16, both experts) ...")
    pipe = WanPipeline.from_pretrained(
        args.model_path, vae=vae, torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    print("[load] pipeline ready.")

    slugs = _ALL_SLUGS[args.start_idx:]
    if args.max_prompts is not None:
        slugs = slugs[: args.max_prompts]
    print(f"[run] {len(slugs)} prompt(s); start_idx={args.start_idx} "
          f"cfg={args.guidance_scale}/{args.guidance_scale_2} "
          f"steps={args.num_inference_steps} seed={args.seed}")

    for i, slug in enumerate(slugs):
        prompt = ALL_PROMPTS[slug]
        mp4_path = out_dir / f"{slug}.mp4"
        print(f"[{i + 1}/{len(slugs)}] slug={slug!r} prompt={prompt!r}")
        generator = torch.Generator().manual_seed(args.seed)
        out = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            guidance_scale_2=args.guidance_scale_2,
            generator=generator,
            output_type="pil",
        )
        frames = out.frames[0]  # list[PIL.Image], length num_frames
        export_to_video(frames, str(mp4_path), fps=args.fps)
        print(f"[done] wrote {mp4_path}")

        del out, frames
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[summary] wrote {len(slugs)} video(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
