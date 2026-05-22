"""One-shot VAE encoder for the face_latent cache.

Reads every $WAN_BETA_CACHE/raw_face/face_*.pt (uint8 (3, H, W)) and writes a
normalized fp32 latent to $WAN_BETA_CACHE/face_latent/face_*.pt of shape
(1, 16, 1, H_lat, W_lat) ready to be consumed by WanPTDCNPipeline.

Normalization direction (silent-corruption trap, see CLAUDE.md):
The pipeline's decode does `latents / (1/std) + mean = latents*std + mean`, so
the encode-side inverse is `(z - mean) / std` using the raw vae.config values.

Loads the Wan 2.2 VAE only — tiny GPU footprint.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_path", type=str, required=True,
                   help="Wan-AI Wan2.2-T2V-A14B-Diffusers snapshot dir.")
    p.add_argument("--raw_face_dir", type=str, required=True,
                   help="Dir of face_{idx}.pt uint8 (3, H, W) tensors.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Dir to write face_{idx}.pt fp32 latents into.")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N faces.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-encode even if the output .pt already exists.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_face_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pts = sorted(raw_dir.glob("face_*.pt"),
                 key=lambda p: int(p.stem.split("_")[1]))
    if args.limit is not None:
        pts = pts[:args.limit]
    if not pts:
        print(f"[error] no face_*.pt files under {raw_dir}")
        return 1
    print(f"[input]  {len(pts)} raw face tensors from {raw_dir}")
    print(f"[output] {out_dir}")

    from diffusers import AutoencoderKLWan

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] vae from {args.base_model_path} (device={device}) ...")
    vae = AutoencoderKLWan.from_pretrained(
        args.base_model_path, subfolder="vae", torch_dtype=torch.bfloat16,
    ).to(device).eval()

    z_dim = vae.config.z_dim
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    t0 = time.time()
    written = 0
    skipped = 0
    with torch.no_grad():
        for pt in tqdm(pts, desc="encode"):
            out_path = out_dir / pt.name
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            face_u8 = torch.load(pt, map_location="cpu", weights_only=True)  # (3, H, W) uint8
            if face_u8.shape[-2:] != (args.height, args.width):
                from torchvision.transforms.functional import resize
                face_u8 = resize(face_u8, [args.height, args.width], antialias=True)

            # [-1, 1] normalization, single-frame "static video" shape (1, 3, 1, H, W)
            x = face_u8.to(torch.float32) / 127.5 - 1.0
            x = x.unsqueeze(0).unsqueeze(2)  # (1, 3, 1, H, W)
            x = x.to(device=device, dtype=vae.dtype)

            z = vae.encode(x).latent_dist.sample(generator)  # (1, 16, 1, H_lat, W_lat)
            z = z.to(torch.float32)

            mean = latents_mean.to(z)
            std = latents_std.to(z)
            z_norm = (z - mean) / std

            torch.save(z_norm.cpu().to(torch.float32), out_path)
            written += 1

    dt = time.time() - t0
    print(f"[done] wrote {written}, skipped {skipped} pre-existing  ({dt:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
