"""Quick-and-dirty: deterministic-invert 10 faces via WanInversionPipeline.

For each face in raw_face/face_{0..9}.pt, runs the existing
`WanInversionPipeline.deterministic_invert` (no model forwards — just the
linear `z_t = inv_sigmas[i]*face + (1 - inv_sigmas[i])*eps` formula) and
writes 100 step_*.pt files into a per-face subdir of
`<cache>/deterministic_invert_faces/`. That on-disk layout is what
`WanPTDPipeline._load_reference_latents` consumed for the historical GOOD
runs (wan-heuristic-experiments-4.2), so this is bit-for-bit the same path.

height/width/num_frames/num_inference_steps are pinned to match
`inference/run_inference.py`, so the saved latents are shape-compatible
without any reshape.

Env:
    WAN_MODEL       Wan snapshot dir (set by slurm/inference.sbatch).
                    Falls back to the HF repo id if unset (online only).
    WAN_BETA_CACHE  Wan-beta cache root (raw_face/ + this script's output).
                    Falls back to $HOME/cache/wan-beta.
"""
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from diffusers import AutoencoderKLWan

sys.path.insert(0, str(Path(__file__).parent.parent))

from PTD_Pipeline.WanInversionPipeline import WanInversionPipeline


WAN_MODEL = os.environ.get("WAN_MODEL", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
WAN_BETA_CACHE = Path(os.environ.get(
    "WAN_BETA_CACHE", str(Path.home() / "cache" / "wan-beta")
))
RAW_FACE_DIR = WAN_BETA_CACHE / "raw_face"
OUT_DIR = WAN_BETA_CACHE / "deterministic_invert_faces_528x528x61"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_FACES = 10
NUM_FRAMES = 61
HEIGHT = 528
WIDTH = 528
N_STEPS = 100

dtype = torch.bfloat16
print(f"[load] vae from {WAN_MODEL}")
vae = AutoencoderKLWan.from_pretrained(WAN_MODEL, subfolder="vae", torch_dtype=torch.float32)
print(f"[load] WanInversionPipeline from {WAN_MODEL}")
pipe = WanInversionPipeline.from_pretrained(WAN_MODEL, vae=vae, torch_dtype=dtype)
pipe.enable_model_cpu_offload()
print("[ready] pipeline setup done")

for face_idx in range(NUM_FACES):
    raw_path = RAW_FACE_DIR / f"face_{face_idx}.pt"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    raw_u8 = torch.load(raw_path, map_location="cpu", weights_only=True)  # (3,H,W) uint8
    face_img = Image.fromarray(raw_u8.permute(1, 2, 0).numpy())

    save_dir = OUT_DIR / f"face_{face_idx}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] face_{face_idx} -> {save_dir}")
    pipe.deterministic_invert(
        reference_image=face_img,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=N_STEPS,
        save_latent_dir=str(save_dir),
        save_image_dir=str(save_dir / "_decoded"),
        save_images=False,
    )
    print(f"[done] face_{face_idx}")

print(f"[summary] wrote {NUM_FACES} face dirs to {OUT_DIR}")
