"""CLI runner for the combined ControlNet + PTDiffusion inference pipeline.

Builds the WanPTDCNPipeline ONCE (heavy: full Wan 2.2 A14B + both experts +
ControlNet) and loops over a hardcoded `PAIRS = [(face_idx, slug), ...]`
list, writing one mp4 + one wandb run per pair.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import wandb

from training.utils import cast_respecting_fp32_modules, detect_boundary_ratio
from training.input_prompts import PROMPTS_BATCH_1, PROMPTS_BATCH_2

ALL_PROMPTS = {**PROMPTS_BATCH_1, **PROMPTS_BATCH_2}

# ---- Hardcoded run config (edit between runs by hand) -----------------------
# face_idx i is paired with the i-th slug in PROMPTS_BATCH_1 + PROMPTS_BATCH_2
# declaration order: face_0 -> 'snowy_mountain', ..., face_99 -> 'ivy_wall'.
_ALL_SLUGS = list(PROMPTS_BATCH_1.keys()) + list(PROMPTS_BATCH_2.keys())
assert len(_ALL_SLUGS) == 100, f"expected 100 slugs, got {len(_ALL_SLUGS)}"
PAIRS = [(i, _ALL_SLUGS[i]) for i in range(100)]
HEIGHT       = 512
WIDTH        = 512
NUM_FRAMES   = 9
NEGATIVE_PROMPT = (
    "blurry, low quality, worst quality, jpeg artifacts, text, subtitles, "
    "watermark, static image, still frame, distorted anatomy, inconsistent motion"
)
WANDB_PROJECT = "CN_PTD_inference_2"
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True,
                   help="Trained ControlNet .safetensors.")
    p.add_argument("--base_model_path", type=str, required=True,
                   help="Wan-AI Wan2.2-T2V-A14B-Diffusers snapshot dir.")
    p.add_argument("--controlnet_config_repo", type=str, required=True,
                   help="HED config snapshot dir (architecture only).")
    p.add_argument("--cache_dir", type=str, required=True,
                   help="Wan-beta cache root (holds raw_face/ and face_latent/).")
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


def save_video(frames_np: np.ndarray, path: Path, fps: int = 8) -> None:
    import imageio
    arr = (np.clip(frames_np, 0.0, 1.0) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), list(arr), fps=fps, codec="libx264")


def build_pipeline(args: argparse.Namespace):
    from diffusers import AutoencoderKLWan
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel
    from safetensors.torch import load_file

    from wan_transformer import CustomWanTransformer3DModel
    from wan_controlnet import WanControlnet
    from inference.PTD_CN_pipeline import WanPTDCNPipeline

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

    pipe = WanPTDCNPipeline(
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

    # Pin CN to GPU and strip its accelerate hook — see comment in
    # training/run_inference_beta.build_pipeline. Accelerate re-attaches per
    # __call__; we re-strip in the run loop.
    from accelerate.hooks import remove_hook_from_module
    remove_hook_from_module(pipe.controlnet, recurse=True)
    pipe.controlnet.to("cuda")
    return pipe


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cache_dir = Path(args.cache_dir)
    out_root = Path.home() / "outputs" / "inference" / args.project_name
    out_root.mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline(args)
    device = pipe._execution_device

    from accelerate.hooks import remove_hook_from_module

    for face_idx, slug in PAIRS:
        if slug not in ALL_PROMPTS:
            raise ValueError(f"Unknown slug '{slug}' (face_idx={face_idx}).")
        prompt_text = ALL_PROMPTS[slug]

        raw_path = cache_dir / "raw_face" / f"face_{face_idx}.pt"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        raw_u8 = torch.load(raw_path, map_location="cpu", weights_only=True)  # (3, H, W) uint8
        face_img = Image.fromarray(raw_u8.permute(1, 2, 0).numpy())

        run_name = f"face_{face_idx}_{slug}"
        run_config = dict(
            project_name=args.project_name,
            checkpoint=args.checkpoint_path,
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
            face_idx=face_idx,
            slug=slug,
            prompt=prompt_text,
        )
        with wandb.init(project=WANDB_PROJECT, name=run_name, config=run_config,
                        reinit=True) as run:
            print(f"[run] {run_name}  prompt={prompt_text!r}")
            # Re-pin CN to GPU (accelerate re-attaches per call)
            remove_hook_from_module(pipe.controlnet, recurse=True)
            pipe.controlnet.to("cuda")
            generator = torch.Generator().manual_seed(args.seed)
            # Independent seed for the noise tensor mixed into the ref latent
            # — must NOT share entropy with `generator` (used by the denoising
            # init), otherwise the ref's noise term equals the latent init
            # noise and the phase-substitute becomes a near-identity at step 0.
            ref_noise_generator = torch.Generator().manual_seed(args.seed + 1)

            out = pipe(
                controlnet_frames=[face_img] * NUM_FRAMES,
                face_image=face_img,
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
                ref_noise_generator=ref_noise_generator,
                output_type="np",
            )
            frames = out.frames[0]  # (T, H, W, 3) float in [0, 1]
            mp4_path = out_root / f"{run_name}.mp4"
            save_video(frames, mp4_path, fps=args.fps)
            print(f"[done] wrote {mp4_path}  (face={face_idx}, slug={slug})")

    print("[summary] all pairs done:")
    for face_idx, slug in PAIRS:
        print(f"  face={face_idx} slug={slug} -> "
              f"{out_root / f'face_{face_idx}_{slug}.mp4'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
