"""WanPTDPipeline baseline runner — 100 fair-comparison videos.

Derived from PTD_Pipeline/run_WanPTDPipeline_grid_search_2.py. Same pipeline,
same hyperparameters, same negative prompt. The only structural change is the
input set: instead of cross-producing a handful of hand-written verbose prompts
against the first few faces, this runner walks the canonical 100 face↔slug
pairs used by inference/run_inference.py and training/run_inference_beta.py
(face_i ↔ PROMPTS_BATCH_1+PROMPTS_BATCH_2[i] in declaration order, e.g.
face_0↔'snowy_mountain', face_99↔'ivy_wall'). Prompts are the short
slug-derived strings from training/input_prompts.py, matching what the
controlnet/combined pipeline conditions on — so all three (CN-only, CN+PTD,
PTD-only) condition on identical (face, prompt) tuples and the comparison is
apples-to-apples.

Output: one mp4 per pair in --output_dir, named face_{i}_{slug}.mp4.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str,
                   default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                   help="HF id or local snapshot dir for Wan 2.2 A14B.")
    p.add_argument("--ref_latents_root", type=str,
                   default="/home/s2710099/cache/wan-beta/deterministic_invert_faces_528x528x61",
                   help="Directory containing face_{0..99}/ subdirs, each with "
                        "the 101 step_*.pt deterministic-invert latents.")
    p.add_argument("--output_dir", type=str,
                   default="/home/s2710099/outputs/inference/ptd_og_pipeline_100_fair",
                   help="Where face_{i}_{slug}.mp4 files are written.")
    p.add_argument("--wandb_project", type=str, default="PTD_inference_100_fair")
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Cap on number of pairs processed (debug). None = all 100.")
    p.add_argument("--start_idx", type=int, default=0,
                   help="Skip the first N pairs (resume support).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # --- Reproducibility (mirrors grid_search_2's `use_same_seed=True` branch) ---
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)

    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Pipeline setup (identical to grid_search_2) ---
    dtype = torch.bfloat16
    vae = AutoencoderKLWan.from_pretrained(args.model_path, subfolder="vae",
                                           torch_dtype=torch.float32)
    print('--- VAE loaded ---')
    pipe = WanPTDiffusionPipeline.from_pretrained(args.model_path, vae=vae,
                                                  torch_dtype=dtype)
    print('--- PT Diffusion pipeline loaded ---')
    pipe.enable_model_cpu_offload()
    print("Pipeline setup done...")

    # --- Video generation parameters (identical to grid_search_2) ---
    height = 528
    width = 528
    num_frames = 61
    num_inference_steps = 101
    direct_transfer_steps = 45
    decayed_transfer_steps = 22
    initial_alpha = 0.4

    pairs = PAIRS[args.start_idx:]
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    print(f"[run] processing {len(pairs)} pair(s); start_idx={args.start_idx} "
          f"max_pairs={args.max_pairs}")

    for face_idx, slug in pairs:
        prompt = ALL_PROMPTS[slug]
        ref_dir = os.path.join(args.ref_latents_root, f"face_{face_idx}")
        if not os.path.isdir(ref_dir):
            raise FileNotFoundError(f"Reference latents dir missing: {ref_dir}")

        run_name = f"face_{face_idx}_{slug}"
        mp4_path = os.path.join(args.output_dir, f"{run_name}.mp4")
        with wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "face_idx": face_idx,
                "slug": slug,
                "prompt": prompt,
                "ref_latents_dir": ref_dir,
                "direct_transfer_steps": direct_transfer_steps,
                "decayed_transfer_steps": decayed_transfer_steps,
                "initial_alpha": initial_alpha,
                "Kp": 0.5,
                "Ki": 0.2,
                "max_alpha_delta": 0.05,
                "num_inference_steps": num_inference_steps,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "guidance_scale_call": 0.0,
                "seed": seed,
            },
            reinit=True,
        ) as run:
            print(f"[run] face={face_idx} slug={slug!r} prompt={prompt!r}")
            video = pipe(
                # Standard params
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                latents=None,
                guidance_scale=0.0,  # NO CFG (matches grid_search_2 heuristic mode)
                # PTM params
                direct_transfer_steps=direct_transfer_steps,
                decayed_transfer_steps=decayed_transfer_steps,
                exponent=0.5,
                initial_alpha=initial_alpha,
                ref_latents_dir=ref_dir,
                use_blending_heuristic_version_1=False,
                use_blending_heuristic_version_2=False,
                use_blending_heuristic_version_3=True,
                use_blending_heuristic_version_4=False,
                Kp=0.5,
                Ki=0.2,
                max_alpha_delta=0.05,
                # Skip the parallel "what-if no PTD" baseline trajectory — it
                # exists only to compute two wandb logging scalars
                # (cond_mse_mag, cosine_similarity_of_noise) and doubles the
                # per-step transformer batch. Halves wall time per video; we
                # don't need the metrics for the 100-fair analysis set.
                track_conditional_baseline=False,
            )

            frames = video.get('frames', video.get('images', video))
            frames = frames[0]  # drop batch dim -> [T, H, W, C]
            frame_list = [frames[i] for i in range(frames.shape[0])]
            export_to_video(frame_list, mp4_path)
            print(f"[done] wrote {mp4_path}")

    print(f"[summary] processed {len(pairs)} pair(s); outputs in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
