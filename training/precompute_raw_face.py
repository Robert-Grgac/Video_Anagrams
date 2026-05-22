"""Precompute raw-face inputs for the ControlNet (information upper bound).

Writes one ``<output_dir>/raw_face/face_{idx}.pt`` per face as a (3, H, W) uint8
tensor — same on-disk schema as the canny and silhouette caches. Each tensor is
just the input PNG resized to (H, W) and converted to RGB. No detection, no
edges, no segmentation.

This is the "what if we feed the model the literal face" benchmark — the strict
upper bound on what structural information the ControlNet's encoder can pull
out of a single input image. Train against it with
``--control_subdir raw_face`` in beta007/train.py.

Pure PIL + torch; runs in any env (wan22-bw, base, silhouette).

    python training/precompute_raw_face.py \\
        --input_faces_dir $HOME/data/wan-beta/input_faces \\
        --output_dir      $HOME/cache/wan-beta
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_faces_dir", type=str, required=True,
                   help="Dir containing face_{idx}.png files.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Top-level cache dir. Raw faces go to "
                        "<output_dir>/raw_face/face_{idx}.pt.")
    p.add_argument("--subdir_name", type=str, default="raw_face",
                   help="Subdir under output_dir to write to.")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N faces.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-render even if the output .pt already exists.")
    return p.parse_args()


def make_raw_face_uint8(face_png: Path, height: int, width: int) -> torch.Tensor:
    img = Image.open(face_png).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)              # (H, W, 3)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)


def main() -> int:
    args = parse_args()
    in_dir = Path(args.input_faces_dir)
    out_dir = Path(args.output_dir) / args.subdir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(in_dir.glob("face_*.png"),
                  key=lambda p: int(p.stem.split("_")[1]))
    if args.limit is not None:
        pngs = pngs[:args.limit]
    if not pngs:
        print(f"[error] no face_*.png files found under {in_dir}")
        return 1
    print(f"[input]  {len(pngs)} face PNGs from {in_dir}")
    print(f"[output] {out_dir}  size={args.height}x{args.width}")

    t0 = time.time()
    written = 0
    skipped = 0
    for png in tqdm(pngs, desc="raw_face"):
        out_path = out_dir / f"{png.stem}.pt"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        t = make_raw_face_uint8(png, args.height, args.width)
        torch.save(t, out_path)
        written += 1

    dt = time.time() - t0
    print(f"[done] wrote {written}, skipped {skipped} pre-existing  ({dt:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
