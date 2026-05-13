"""Standalone dual-CN inference for beta-003 / beta-004.

Loads two trained ControlNet checkpoints (one per expert), builds the dual
Wan 2.2 pipeline with both transformers and both controlnets, and runs the
same (weights × ends) sweep as ``run_inference_beta.py`` — one mp4 per
(weight, end) cell.

Mirrors ``run_inference_beta.py`` in CLI shape, with ``--checkpoint_path``
replaced by ``--high_checkpoint`` and ``--low_checkpoint``.
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

from training.utils import cast_respecting_fp32_modules, detect_boundary_ratio
from training.input_prompts import PROMPTS_BATCH_1, PROMPTS_BATCH_2

if TYPE_CHECKING:
    from wan_t2v_controlnet_pipeline_dual import WanTextToVideoDualControlnetPipeline

ALL_PROMPTS = {**PROMPTS_BATCH_1, **PROMPTS_BATCH_2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--high_checkpoint", type=str, required=True,
                   help="Trained ControlNet .safetensors used as the high-noise CN "
                        "(typically beta-001_final.safetensors).")
    p.add_argument("--low_checkpoint", type=str, required=True,
                   help="Trained ControlNet .safetensors used as the low-noise CN "
                        "(beta-003 EMA or beta-004 EMA).")
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
    p.add_argument("--weights", type=str, default=None,
                   help="Optional comma-separated list of controlnet_weight values. "
                        "If set, builds the pipeline once and writes one mp4 per weight; "
                        "output filenames get a '_w{weight}' suffix. Overrides --controlnet_weight.")
    p.add_argument("--controlnet_stride", type=int, default=3)
    p.add_argument("--controlnet_guidance_start", type=float, default=0.0)
    p.add_argument("--controlnet_guidance_end", type=float, default=1.0)
    p.add_argument("--ends", type=str, default=None,
                   help="Optional comma-separated list of controlnet_guidance_end values. "
                        "Combined with --weights as a Cartesian product. "
                        "When set, output filenames get a '_w{weight}_e{end}' suffix. "
                        "Overrides --controlnet_guidance_end.")
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
    canny_u8 = torch.load(canny_path, map_location="cpu", weights_only=True)
    return Image.fromarray(canny_u8.permute(1, 2, 0).numpy())


def save_video(frames_np: np.ndarray, path: Path, fps: int = 8) -> None:
    import imageio
    arr = (np.clip(frames_np, 0.0, 1.0) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), list(arr), fps=fps, codec="libx264")


def _load_controlnet(controlnet_config_repo: str, ckpt_path: str, label: str):
    from safetensors.torch import load_file
    from wan_controlnet import WanControlnet

    print(f"[load] controlnet config from {controlnet_config_repo} ({label}) ...")
    config = WanControlnet.load_config(controlnet_config_repo)
    cn = WanControlnet.from_config(config)
    cast_respecting_fp32_modules(cn, torch.bfloat16)
    print(f"[load] controlnet weights from {ckpt_path} ({label}) ...")
    sd = load_file(ckpt_path)
    missing, unexpected = cn.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] {label}: missing keys when loading controlnet: {len(missing)}")
    if unexpected:
        print(f"[warn] {label}: unexpected keys when loading controlnet: {len(unexpected)}")
    cn.eval()
    return cn


def build_pipeline(args: argparse.Namespace) -> "WanTextToVideoDualControlnetPipeline":
    from diffusers import AutoencoderKLWan
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel

    from wan_transformer import CustomWanTransformer3DModel
    from wan_t2v_controlnet_pipeline_dual import WanTextToVideoDualControlnetPipeline

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

    print(f"[load] low-noise transformer_2 ...")
    transformer_2 = CustomWanTransformer3DModel.from_pretrained(
        base, subfolder="transformer_2", torch_dtype=torch.bfloat16,
    ).eval()

    print(f"[load] scheduler ...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base, subfolder="scheduler",
    )

    controlnet_high = _load_controlnet(args.controlnet_config_repo,
                                       args.high_checkpoint, label="high")
    controlnet_low = _load_controlnet(args.controlnet_config_repo,
                                      args.low_checkpoint, label="low")

    boundary_ratio, src = detect_boundary_ratio(base, dict(transformer.config))
    print(f"[detect] boundary_ratio={boundary_ratio} ({src})")

    pipe = WanTextToVideoDualControlnetPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        controlnet_high=controlnet_high,
        controlnet_low=controlnet_low,
        scheduler=scheduler,
        boundary_ratio=boundary_ratio,
    )
    pipe.enable_model_cpu_offload()

    # Same CPU/GPU mismatch on first Conv3D as run_inference_beta.py; pin BOTH
    # controlnets and strip their accelerate hooks. Total CN-on-GPU is ~1.4 GB
    # at bf16 — well within the A40 budget.
    from accelerate.hooks import remove_hook_from_module
    for cn in (pipe.controlnet_high, pipe.controlnet_low):
        remove_hook_from_module(cn, recurse=True)
        cn.to("cuda")
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

    if args.weights:
        weights = [float(w) for w in args.weights.split(",") if w.strip()]
    else:
        weights = [args.controlnet_weight]
    if args.ends:
        ends = [float(e) for e in args.ends.split(",") if e.strip()]
    else:
        ends = [args.controlnet_guidance_end]
    sweep_active = bool(args.weights) or bool(args.ends)
    print(f"[sweep] {len(weights)} weight(s) x {len(ends)} end(s) = "
          f"{len(weights) * len(ends)} videos")

    out_path = Path(args.output_path)
    from accelerate.hooks import remove_hook_from_module
    for w in weights:
        for e in ends:
            # Re-pin both CNs each iteration: model_cpu_offload re-attaches an
            # accelerate hook on every __call__, which would route inputs to
            # CPU and trigger a device mismatch on the first Conv3D.
            for cn in (pipe.controlnet_high, pipe.controlnet_low):
                remove_hook_from_module(cn, recurse=True)
                cn.to("cuda")
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
                controlnet_weight=w,
                controlnet_stride=args.controlnet_stride,
                controlnet_guidance_start=args.controlnet_guidance_start,
                controlnet_guidance_end=e,
                generator=generator,
                output_type="np",
            )
            frames = out.frames[0]
            if not sweep_active:
                target = out_path
            else:
                wstr = f"{w:.2f}".replace(".", "p")
                estr = f"{e:.3f}".rstrip("0").rstrip(".").replace(".", "p")
                target = out_path.with_name(
                    f"{out_path.stem}_w{wstr}_e{estr}{out_path.suffix}"
                )
            save_video(frames, target, fps=args.fps)
            print(f"[done] wrote {target}  (w={w}, end={e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
