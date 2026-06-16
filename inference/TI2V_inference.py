"""Wan 2.2 TI2V-5B image+text-to-video runner — 100 face/scene videos.

Companion to ``inference/run_wan_vanilla.py``, with two deliberate differences:

1. It loads the **TI2V-5B** model (single-transformer Text-Image-to-Video, the
   high-compression 16x16x4 VAE variant) via diffusers'
   ``WanImageToVideoPipeline`` instead of the MoE ``WanPipeline``. Note the
   model_path must point at the *diffusers-format* snapshot
   ``Wan-AI/Wan2.2-TI2V-5B-Diffusers`` (model_index.json + transformer/ vae/
   text_encoder/ tokenizer/ scheduler/). The raw Wan-format dir
   ``/home/s2710099/models/Wan2.2-TI2V-5B`` is NOT loadable this way.

2. The same 100 (face, slug) pairs that ``inference/run_inference.py`` uses:
   face image ``face_{i}.pt`` is paired with the i-th slug in
   PROMPTS_BATCH_1 + PROMPTS_BATCH_2 (declaration order). Each slug's short
   prompt is rewritten into a longer, TI2V-oriented instruction that names the
   face as the thing to blend into the scene (see ``verbosify``). The face .pt
   tensor is fed as the conditioning ``image=`` for image-to-video.

One mp4 per pair is written, named ``face_{i}_{slug}.mp4``.
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

from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
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


def verbosify(short_prompt: str) -> str:
    """Rewrite a slug's short prompt into the verbose TI2V instruction.

    Template:
        "aurora borealis over mountains, painting"
          -> "Blend this face structure seamlessly into the background scene
              of aurora borealis over mountains, in the art style of a painting"

    Rule: the short prompts are either ``"<scene>, <style> painting"`` or just
    ``"<scene>"`` (no comma). When a trailing style clause is present (every
    prompt with a comma in input_prompts.py ends in a "... painting" style),
    split it off and reattach it as "in the artsytle of a <style>". Otherwise
    emit the scene-only form.
    """
    base = "Blend this face structure seamlessly into the background scene of "
    if ", " in short_prompt:
        scene, style = short_prompt.rsplit(", ", 1)
        return f"{base}{scene}, in the art style of a {style}"
    return f"{base}{short_prompt}"


# slug -> verbose prompt, built once in declaration order. Inspect/print to
# review the 100 modified prompts before a run.
VERBOSE_PROMPTS = {slug: verbosify(ALL_PROMPTS[slug]) for slug in _ALL_SLUGS}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True,
                   help="Local Wan2.2-TI2V-5B-Diffusers snapshot dir (or HF id). "
                        "Must be diffusers format, NOT the raw Wan checkpoint.")
    p.add_argument("--faces_dir", type=str, required=True,
                   help="Dir of conditioning faces face_{i}.pt (uint8 (3,H,W)). "
                        "Same set run_inference.py uses: "
                        "<cache>/wan-beta/raw_face.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Where face_{i}_{slug}.mp4 files are written.")
    # TI2V-5B VAE is 16x spatial + transformer patch 2 => H/W must be a
    # multiple of 32 (unlike the 8x Wan2.1 VAE in run_wan_vanilla, where 528
    # was valid). 528 is NOT divisible by 32 and triggers a 16-vs-17 token
    # grid mismatch in the transformer; 512 is the nearest valid size.
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num_frames", type=int, default=61,
                   help="TI2V VAE temporal compression is 4, so use 4k+1.")
    p.add_argument("--num_inference_steps", type=int, default=40)
    p.add_argument("--guidance_scale", type=float, default=5.0,
                   help="TI2V-5B is single-transformer; one CFG (recommended 5.0).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=16,
                   help="mp4 playback fps only; does not affect frame-indexed eval.")
    p.add_argument("--max_prompts", type=int, default=None,
                   help="Cap on number of pairs processed (debug). None = all 100.")
    p.add_argument("--start_idx", type=int, default=0,
                   help="Skip the first N pairs (resume support).")
    return p.parse_args()


def load_face_image(faces_dir: Path, face_idx: int) -> Image.Image:
    """Load face_{i}.pt (uint8 (3,H,W)) and return it as a PIL RGB image.

    The pipeline's video processor resizes the image to (height, width); we
    hand it the native crop and let it handle the fit.
    """
    raw_path = faces_dir / f"face_{face_idx}.pt"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    raw_u8 = torch.load(raw_path, map_location="cpu", weights_only=True)  # (3, H, W) uint8
    return Image.fromarray(raw_u8.permute(1, 2, 0).numpy())


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    faces_dir = Path(args.faces_dir)
    if not faces_dir.is_dir():
        raise FileNotFoundError(f"faces_dir not found: {faces_dir}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] vae (fp32) ...")
    vae = AutoencoderKLWan.from_pretrained(
        args.model_path, subfolder="vae", torch_dtype=torch.float32,
    )
    print("[load] WanImageToVideoPipeline (bf16, TI2V-5B single transformer) ...")
    pipe = WanImageToVideoPipeline.from_pretrained(
        args.model_path, vae=vae, torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    print("[load] pipeline ready.")

    slugs = _ALL_SLUGS[args.start_idx:]
    if args.max_prompts is not None:
        slugs = slugs[: args.max_prompts]
    print(f"[run] {len(slugs)} pair(s); start_idx={args.start_idx} "
          f"cfg={args.guidance_scale} steps={args.num_inference_steps} "
          f"seed={args.seed} {args.height}x{args.width}x{args.num_frames}")

    for i, slug in enumerate(slugs):
        face_idx = args.start_idx + i  # face_{i} pairs with the i-th slug
        prompt = VERBOSE_PROMPTS[slug]
        face_img = load_face_image(faces_dir, face_idx)
        mp4_path = out_dir / f"face_{face_idx}_{slug}.mp4"
        print(f"[{i + 1}/{len(slugs)}] face={face_idx} slug={slug!r} "
              f"prompt={prompt!r}")
        generator = torch.Generator().manual_seed(args.seed)
        out = pipe(
            image=face_img,
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_type="pil",
        )
        frames = out.frames[0]  # list[PIL.Image], length num_frames
        export_to_video(frames, str(mp4_path), fps=args.fps)
        print(f"[done] wrote {mp4_path}")

        del out, frames, face_img
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[summary] wrote {len(slugs)} video(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
