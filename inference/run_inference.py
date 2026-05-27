"""CLI runner for the combined ControlNet + PTDiffusion inference pipeline.

Builds the WanPTDCNPipeline ONCE (heavy: full Wan 2.2 A14B + both experts +
ControlNet) and loops over a hardcoded `PAIRS = [(face_idx, slug), ...]`
list, writing one mp4 + one wandb run per pair.
"""
from __future__ import annotations

import argparse
import gc
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
HEIGHT       = 528
WIDTH        = 528
NUM_FRAMES   = 61
NEGATIVE_PROMPT = (
    "blurry, low quality, worst quality, jpeg artifacts, text, subtitles, "
    "watermark, static image, still frame, distorted anatomy, inconsistent motion"
)
WANDB_PROJECT = "CN_PTD_inference_6"

# ---- Sweep config (used only when --sweep is passed) -----------------------
# Each entry is one sub-experiment that runs `face_idxs` against the same
# (guidance_scale, controlnet_weight). Output goes to
# ~/outputs/inference/<project_name>/<variant_name>/face_{i}_<slug>.mp4 and
# the wandb run name is "<variant_name>_face_{i}_<slug>". Reuses the loaded
# pipeline across all entries (one model load for the whole sweep).
#
# `prompt_overrides`: None → use ALL_PROMPTS[slug] for each face (i.e. the
# slug-derived short prompts in input_prompts.py). Or a list of strings the
# same length as `face_idxs` to override per-face (used for the verbose-prompt
# variant matching the old PTD_Pipeline runner's prompt style).
SWEEP_CONFIGS = [
    # CFG sweep at CN=1.0, slug-derived prompts
    {"variant_name": "cfg1p5_cnw1p0", "guidance_scale": 1.5, "controlnet_weight": 1.0,
     "face_idxs": [0, 1, 2, 3, 4], "prompt_overrides": None},
    {"variant_name": "cfg2p0_cnw1p0", "guidance_scale": 2.0, "controlnet_weight": 1.0,
     "face_idxs": [0, 1, 2, 3, 4], "prompt_overrides": None},
    {"variant_name": "cfg3p0_cnw1p0", "guidance_scale": 3.0, "controlnet_weight": 1.0,
     "face_idxs": [0, 1, 2, 3, 4], "prompt_overrides": None},
    {"variant_name": "cfg4p0_cnw1p0", "guidance_scale": 4.0, "controlnet_weight": 1.0,
     "face_idxs": [0, 1, 2, 3, 4], "prompt_overrides": None},
    # CN weight sweep at CFG=5, slug-derived prompts
    {"variant_name": "cfg5p0_cnw0p5", "guidance_scale": 5.0, "controlnet_weight": 0.5,
     "face_idxs": [0, 1, 2, 3, 4], "prompt_overrides": None},
    {"variant_name": "cfg5p0_cnw2p5", "guidance_scale": 5.0, "controlnet_weight": 2.5,
     "face_idxs": [0, 1, 2, 3, 4], "prompt_overrides": None},
    # CFG=5, CN=1.0 with the verbose OLD-PTD-runner-style prompts.
    {"variant_name": "cfg5p0_cnw1p0_verbose_prompts",
     "guidance_scale": 5.0, "controlnet_weight": 1.0,
     "face_idxs": [0, 1, 2, 3, 4],
     "prompt_overrides": [
         "snowy mountain,static, no movement, in style of cubist painting, high quality",
         "grand canyon, static, no movement, photorealistic, high quality",
         "park, static, no movement, in style of oil painting, high quality",
         "seaflor, static, no movement, in style of watercolor painting, high quality",
         "flowers, static, no movement, in style of street art, high quality",
     ]},
]
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
                   help="Wan-beta cache root (holds raw_face/ and the chosen "
                        "invert dir).")
    p.add_argument("--invert_type", type=str, default="euler",
                   choices=["euler", "deterministic"],
                   help="euler: load single stacked .pt from "
                        "<cache>/invert_face/face_{i}.pt (Euler ODE inversion "
                        "via the transformer). deterministic: load 100 "
                        "step_*.pt files from "
                        "<cache>/deterministic_invert_faces/face_{i}/ and "
                        "stack them (linear FlowMatch formula, no model forward).")
    p.add_argument("--invert_dir", type=str, default=None,
                   help="Override for precomputed invert dir. Defaults to "
                        "<cache_dir>/invert_face (euler) or "
                        "<cache_dir>/deterministic_invert_faces (deterministic).")
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Limit number of (face, slug) pairs processed. None = "
                        "all 100. Use 10 for the deterministic-invert "
                        "sanity test (only 10 face dirs exist there).")
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
    p.add_argument("--sweep", action="store_true",
                   help="Iterate over SWEEP_CONFIGS instead of the single-config "
                        "(--guidance_scale, --controlnet_weight, --max_pairs) "
                        "path. Each variant runs its face_idxs against its own "
                        "cfg/cn_weight; outputs go to "
                        "<project_name>/<variant_name>/face_*.mp4.")
    return p.parse_args()


def build_jobs(args: argparse.Namespace) -> list[dict]:
    """Flatten the sweep config (or the single-config args) into a list of
    per-run job dicts. Each job carries everything needed to run one inference:
    variant_name (or None), face_idx, slug, prompt, guidance_scale, controlnet_weight.
    """
    jobs: list[dict] = []
    if args.sweep:
        for variant in SWEEP_CONFIGS:
            face_idxs = variant["face_idxs"]
            overrides = variant.get("prompt_overrides")
            if overrides is not None and len(overrides) != len(face_idxs):
                raise ValueError(
                    f"variant {variant['variant_name']!r}: "
                    f"prompt_overrides has {len(overrides)} entries but "
                    f"face_idxs has {len(face_idxs)}."
                )
            for slot_i, face_idx in enumerate(face_idxs):
                slug = _ALL_SLUGS[face_idx]
                prompt_text = overrides[slot_i] if overrides is not None else ALL_PROMPTS[slug]
                jobs.append(dict(
                    variant_name=variant["variant_name"],
                    face_idx=face_idx,
                    slug=slug,
                    prompt=prompt_text,
                    guidance_scale=variant["guidance_scale"],
                    controlnet_weight=variant["controlnet_weight"],
                ))
    else:
        pairs = PAIRS[: args.max_pairs] if args.max_pairs is not None else PAIRS
        for face_idx, slug in pairs:
            if slug not in ALL_PROMPTS:
                raise ValueError(f"Unknown slug '{slug}' (face_idx={face_idx}).")
            jobs.append(dict(
                variant_name=None,
                face_idx=face_idx,
                slug=slug,
                prompt=ALL_PROMPTS[slug],
                guidance_scale=args.guidance_scale,
                controlnet_weight=args.controlnet_weight,
            ))
    return jobs


def save_video(frames_np: np.ndarray, path: Path, fps: int = 8) -> None:
    import imageio
    arr = (np.clip(frames_np, 0.0, 1.0) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), list(arr), fps=fps, codec="libx264")


def build_pipeline(args: argparse.Namespace):
    from diffusers import AutoencoderKLWan
    # The Wan 2.2 A14B model_index/scheduler_config specifies
    # UniPCMultistepScheduler (flow_prediction + use_flow_sigmas). The previous
    # FlowMatchEulerDiscreteScheduler choice here was wrong — it produces a
    # different denoising trajectory AND downcasts step() output to bf16, which
    # is what made `_phase_substitute`'s cosine curve saturate vs. the OLD
    # WanPTDiffusionPipeline (which loads UniPC via from_pretrained).
    from diffusers.schedulers import UniPCMultistepScheduler
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

    print(f"[load] scheduler (UniPCMultistepScheduler) ...")
    scheduler = UniPCMultistepScheduler.from_pretrained(
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
    if args.invert_dir:
        invert_dir = Path(args.invert_dir)
    elif args.invert_type == "euler":
        invert_dir = cache_dir / "invert_face"
    else:  # deterministic
        invert_dir = cache_dir / "deterministic_invert_faces"
    if not invert_dir.exists():
        precompute_cmd = (
            "sbatch slurm/precompute_inverts.sbatch"
            if args.invert_type == "euler"
            else "python -m inference.run_WanIversionPipeline"
        )
        raise FileNotFoundError(
            f"invert dir not found: {invert_dir}. Run `{precompute_cmd}` first."
        )
    out_root = Path.home() / "outputs" / "inference" / args.project_name
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args)
    mode = "sweep" if args.sweep else "single-config"
    print(f"[run] mode={mode}  invert_type={args.invert_type}  "
          f"invert_dir={invert_dir}  num_jobs={len(jobs)}")
    if args.sweep:
        for variant in SWEEP_CONFIGS:
            print(f"  variant={variant['variant_name']!r}  "
                  f"cfg={variant['guidance_scale']}  "
                  f"cn_weight={variant['controlnet_weight']}  "
                  f"faces={variant['face_idxs']}  "
                  f"prompt_overrides={'yes' if variant.get('prompt_overrides') else 'no'}")

    pipe = build_pipeline(args)
    device = pipe._execution_device

    from accelerate.hooks import remove_hook_from_module

    for job_i, job in enumerate(jobs):
        face_idx = job["face_idx"]
        slug = job["slug"]
        prompt_text = job["prompt"]
        guidance_scale = job["guidance_scale"]
        controlnet_weight = job["controlnet_weight"]
        variant_name = job["variant_name"]

        raw_path = cache_dir / "raw_face" / f"face_{face_idx}.pt"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        raw_u8 = torch.load(raw_path, map_location="cpu", weights_only=True)  # (3, H, W) uint8
        face_img = Image.fromarray(raw_u8.permute(1, 2, 0).numpy())

        if args.invert_type == "euler":
            invert_path = invert_dir / f"face_{face_idx}.pt"
            if not invert_path.exists():
                raise FileNotFoundError(invert_path)
            ref_latents = torch.load(invert_path, map_location="cpu", weights_only=True)
        else:  # deterministic — 100 separate step_*.pt files per face
            face_dir = invert_dir / f"face_{face_idx}"
            if not face_dir.is_dir():
                raise FileNotFoundError(face_dir)
            step_files = sorted(face_dir.glob("step_*.pt"))
            # deterministic_invert writes num_inference_steps + 1 files (one
            # per inv_sigma, including the final pure-face one). The denoising
            # loop only consumes num_inference_steps refs, so take the first N
            # — step_0000 (pure noise) through step_{N-1} — and stack.
            if len(step_files) < args.num_inference_steps:
                raise ValueError(
                    f"{face_dir} has only {len(step_files)} step_*.pt files; "
                    f"need at least {args.num_inference_steps}."
                )
            latents_list = [
                torch.load(p, map_location="cpu", weights_only=True)
                for p in step_files[: args.num_inference_steps]
            ]
            ref_latents = torch.stack(latents_list, dim=0)

        if variant_name is not None:
            run_name = f"{variant_name}_face_{face_idx}_{slug}"
            mp4_path = out_root / variant_name / f"face_{face_idx}_{slug}.mp4"
        else:
            run_name = f"face_{face_idx}_{slug}"
            mp4_path = out_root / f"face_{face_idx}_{slug}.mp4"

        run_config = dict(
            project_name=args.project_name,
            variant_name=variant_name,
            checkpoint=args.checkpoint_path,
            controlnet_stride=args.controlnet_stride,
            controlnet_weight=controlnet_weight,
            initial_blending_coeff=args.initial_blending_coeff,
            direct_transfer_steps=args.direct_transfer_steps,
            decayed_transfer_steps=args.decayed_transfer_steps,
            Kp=args.Kp,
            Ki=args.Ki,
            max_blending_coeff_delta=args.max_blending_coeff_delta,
            guidance_scale=guidance_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            face_idx=face_idx,
            slug=slug,
            prompt=prompt_text,
        )
        with wandb.init(project=WANDB_PROJECT, name=run_name, config=run_config,
                        reinit=True) as run:
            print(f"[{job_i + 1}/{len(jobs)}] run={run_name}  "
                  f"cfg={guidance_scale}  cn_weight={controlnet_weight}  "
                  f"prompt={prompt_text!r}")
            # Re-pin CN to GPU (accelerate re-attaches per call)
            remove_hook_from_module(pipe.controlnet, recurse=True)
            pipe.controlnet.to("cuda")
            generator = torch.Generator().manual_seed(args.seed)

            # When controlnet_weight=0, skip CN entirely so cn_active==False
            # and the CN forward never runs (cleaner null-out than weight=0,
            # which still executes the CN forward and multiplies by zero).
            cn_frames = ([face_img] * NUM_FRAMES) if controlnet_weight > 0 else None
            out = pipe(
                controlnet_frames=cn_frames,
                ref_latents=ref_latents,
                prompt=prompt_text,
                negative_prompt=NEGATIVE_PROMPT,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_weight=controlnet_weight,
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

            # Drop per-run transients before the next face. Without this, the
            # last run's frames / out / ref_latents and any accelerate hook
            # state stick around and the 32G --mem footprint creeps up across
            # 100 wandb sessions (OOM-kill observed at face 17 in job 500023).
            del out, frames, ref_latents, raw_u8, face_img
            gc.collect()
            torch.cuda.empty_cache()

    print("[summary] all jobs done:")
    for job in jobs:
        if job["variant_name"] is not None:
            path = out_root / job["variant_name"] / f"face_{job['face_idx']}_{job['slug']}.mp4"
        else:
            path = out_root / f"face_{job['face_idx']}_{job['slug']}.mp4"
        print(f"  variant={job['variant_name']!r:>40s}  "
              f"face={job['face_idx']}  slug={job['slug']}  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
