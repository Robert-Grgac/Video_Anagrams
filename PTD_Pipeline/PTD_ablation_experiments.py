"""PTD heuristic ablation — 4 blending schedules over the canonical 100 pairs.

Loads the WanPTDiffusionPipeline ONCE and calls it four times consecutively,
once per blending schedule, reusing the same 100 face↔slug pairs (the same set
used by inference/run_inference.py for the CN+PTD pipeline and by
run_WanPTDPipeline_100_fair.py for heuristic-3). Each pass writes 100 mp4s to
its own output dir:

    no heuristic (static schedule)  -> /home/s2710099/outputs/inference/PTD_noHeuirsitc
    heuristic 1 (sigmoid damping)   -> /home/s2710099/outputs/inference/PTD_heuristic_1
    heuristic 2 (dead-zone P ctrl)  -> /home/s2710099/outputs/inference/PTD_heuristic_2
    heuristic 4 (energy-ratio PI)   -> /home/s2710099/outputs/inference/heuristic_4

This runner is video-output-only: NO wandb, NO HOG / additional logging
(do_additional_logging=False), NO conditional-baseline trajectory
(track_conditional_baseline=False — halves per-step transformer cost), and NO
CFG (guidance_scale=0.0). The only thing produced is the 4×100 mp4s.

Settings mirror run_WanPTDPipeline_100_fair.py: 528×528, 61 frames, 101 steps,
direct_transfer_steps=45, decayed_transfer_steps=22, initial_alpha=0.4,
exponent=0.5, deterministic-invert reference latents.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import wandb
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PTD_Pipeline.WanPTDPipeline import WanPTDiffusionPipeline
from training.input_prompts import PROMPTS_BATCH_1, PROMPTS_BATCH_2

ALL_PROMPTS = {**PROMPTS_BATCH_1, **PROMPTS_BATCH_2}
_ALL_SLUGS = list(PROMPTS_BATCH_1.keys()) + list(PROMPTS_BATCH_2.keys())
assert len(_ALL_SLUGS) == 100, f"expected 100 slugs, got {len(_ALL_SLUGS)}"
PAIRS = [(i, _ALL_SLUGS[i]) for i in range(100)]

NEGATIVE_PROMPT = (
    "blurry, low quality, worst quality, jpeg artifacts, text, subtitles, "
    "watermark, static image, still frame, distorted anatomy, inconsistent motion"
)

# Shared PTM schedule (identical to the 100-fair / grid_search_2 runs).
DIRECT_TRANSFER_STEPS = 45
DECAYED_TRANSFER_STEPS = 22
INITIAL_ALPHA = 0.4
EXPONENT = 0.5

# The four schedules to run consecutively. Each entry carries the heuristic flag
# set plus the controller params that flag consumes (everything else stays at
# the pipeline default and is ignored when its flag is False).
#   - h1 (sigmoid damping) needs `steepness`.
#   - h2 (dead-zone P controller) needs `gain` + `max_alpha_delta`.
#   - h4 (energy-ratio PI controller) needs energy_target/Kp_energy/Ki_energy
#         + max_alpha_delta (from grid_search_2).
# `subdir` is appended to --output_root (default /home/s2710099/outputs/inference)
# so the same script runs on the local cluster and on Snellius
# (--output_root /home/astergiou/outputs/inference) writing identical subdir
# names (which the eval scripts key off).
HEURISTIC_RUNS = [
    {
        "name": "no_heuristic",
        "subdir": "PTD_noHeuirsitc",
        "kwargs": dict(
            use_blending_heuristic_version_1=False,
            use_blending_heuristic_version_2=False,
            use_blending_heuristic_version_3=False,
            use_blending_heuristic_version_4=False,
        ),
    },
    {
        "name": "heuristic_1",
        "subdir": "PTD_heuristic_1",
        "kwargs": dict(
            use_blending_heuristic_version_1=True,
            use_blending_heuristic_version_2=False,
            use_blending_heuristic_version_3=False,
            use_blending_heuristic_version_4=False,
            steepness=10.0,
        ),
    },
    {
        "name": "heuristic_2",
        "subdir": "PTD_heuristic_2",
        "kwargs": dict(
            use_blending_heuristic_version_1=False,
            use_blending_heuristic_version_2=True,
            use_blending_heuristic_version_3=False,
            use_blending_heuristic_version_4=False,
            gain=2.0,
            max_alpha_delta=0.05,
        ),
    },
    {
        "name": "heuristic_4",
        "subdir": "heuristic_4",
        "kwargs": dict(
            use_blending_heuristic_version_1=False,
            use_blending_heuristic_version_2=False,
            use_blending_heuristic_version_3=False,
            use_blending_heuristic_version_4=True,
            energy_target=0.95,
            Kp_energy=2.0,
            Ki_energy=0.1,
            max_alpha_delta=0.05,
        ),
    },
]
HEURISTIC_NAMES = [r["name"] for r in HEURISTIC_RUNS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str,
                   default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                   help="HF id or local snapshot dir for Wan 2.2 A14B.")
    p.add_argument("--ref_latents_root", type=str,
                   default="/home/s2710099/cache/wan-beta/deterministic_invert_faces_528x528x61",
                   help="Directory with face_{0..99}/ subdirs of step_*.pt latents.")
    p.add_argument("--output_root", type=str,
                   default="/home/s2710099/outputs/inference",
                   help="Base dir; each heuristic writes to <output_root>/<subdir>.")
    p.add_argument("--heuristic", type=str, default="all",
                   choices=["all"] + HEURISTIC_NAMES,
                   help="Run only this heuristic (one per Snellius job), or 'all' "
                        "to run the four consecutively (default, local cluster).")
    p.add_argument("--height", type=int, default=528)
    p.add_argument("--width", type=int, default=528)
    p.add_argument("--num_frames", type=int, default=61)
    p.add_argument("--num_inference_steps", type=int, default=101)
    p.add_argument("--seed", type=int, default=0,
                   help="Global RNG seed, re-applied at the start of each "
                        "heuristic pass so every face draws identical initial "
                        "noise across all four schedules.")
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Cap on pairs per heuristic (debug). None = all 100.")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


def main() -> int:
    args = parse_args()

    # Determinism knobs identical to the 100-fair runner.
    seed_everything(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # WanPTDPipeline._phase_substitute calls wandb.log() unconditionally every
    # step. We want video output only — open a single disabled run so every
    # log is a no-op (no network, no junk runs, no per-video init/finish).
    os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(mode="disabled")

    # --- Load the pipeline ONCE; reused for all four heuristic passes. ---
    dtype = torch.bfloat16
    vae = AutoencoderKLWan.from_pretrained(args.model_path, subfolder="vae",
                                           torch_dtype=torch.float32)
    print('--- VAE loaded ---')
    pipe = WanPTDiffusionPipeline.from_pretrained(args.model_path, vae=vae,
                                                  torch_dtype=dtype)
    print('--- PT Diffusion pipeline loaded ---')
    pipe.enable_model_cpu_offload()
    print("Pipeline setup done...")

    pairs = PAIRS if args.max_pairs is None else PAIRS[: args.max_pairs]

    runs = (HEURISTIC_RUNS if args.heuristic == "all"
            else [r for r in HEURISTIC_RUNS if r["name"] == args.heuristic])
    print(f"[plan] heuristics to run: {[r['name'] for r in runs]}")

    for run in runs:
        out_dir = os.path.join(args.output_root, run["subdir"])
        os.makedirs(out_dir, exist_ok=True)
        # Re-seed so all four schedules see identical initial noise per face.
        seed_everything(args.seed)
        print(f"\n========== heuristic pass: {run['name']} -> {out_dir} ==========")

        for face_idx, slug in pairs:
            prompt = ALL_PROMPTS[slug]
            ref_dir = os.path.join(args.ref_latents_root, f"face_{face_idx}")
            if not os.path.isdir(ref_dir):
                raise FileNotFoundError(f"Reference latents dir missing: {ref_dir}")

            mp4_path = os.path.join(out_dir, f"face_{face_idx}_{slug}.mp4")
            # Resumability: skip videos already written so a timed-out job can be
            # re-submitted to continue where it left off.
            if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
                print(f"[{run['name']}] face={face_idx} slug={slug!r} — exists, skip")
                continue
            print(f"[{run['name']}] face={face_idx} slug={slug!r}")

            video = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                latents=None,
                guidance_scale=0.0,  # NO CFG
                # PTM schedule
                direct_transfer_steps=DIRECT_TRANSFER_STEPS,
                decayed_transfer_steps=DECAYED_TRANSFER_STEPS,
                exponent=EXPONENT,
                initial_alpha=INITIAL_ALPHA,
                ref_latents_dir=ref_dir,
                # Video-only: skip every analytics path.
                do_additional_logging=False,          # no HOG / latent metrics
                track_conditional_baseline=False,      # no parallel baseline pass
                # Per-heuristic flags + controller params.
                **run["kwargs"],
            )

            frames = video.get('frames', video.get('images', video))
            frames = frames[0]  # drop batch dim -> [T, H, W, C]
            frame_list = [frames[i] for i in range(frames.shape[0])]
            export_to_video(frame_list, mp4_path)
            print(f"[done] wrote {mp4_path}")

        print(f"[summary] {run['name']}: wrote {len(pairs)} mp4(s) to {out_dir}")

    print("\n[all done] 4 heuristic passes complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
