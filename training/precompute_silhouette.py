"""Precompute option-H face silhouettes for every face in input_faces/.

Writes one ``<output_dir>/silhouette/face_{idx}.pt`` per face as a (3, H, W)
uint8 tensor — same on-disk schema as the canny cache so the dataset can
swap modalities by switching the control subdir (see train.py's
--control_subdir flag).

Latents and prompt embeddings are modality-independent and are NOT touched
by this script. Keep the existing canny cache around — both `canny/` and
`silhouette/` can sit under the same top-level cache dir.

Run from the silhouette env (mediapipe required, CPU only):

    python training/precompute_silhouette.py \\
        --input_faces_dir $HOME/data/wan-beta/input_faces \\
        --output_dir      $HOME/cache/wan-beta
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.face_silhouette import make_option_h_uint8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_faces_dir", type=str, required=True,
                   help="Dir containing face_{idx}.png files.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Top-level cache dir. Silhouettes go to "
                        "<output_dir>/silhouette/face_{idx}.pt.")
    p.add_argument("--subdir_name", type=str, default="silhouette",
                   help="Subdir under output_dir to write to. Override if you "
                        "want to test a tweaked option-H variant alongside the "
                        "first run, e.g. 'silhouette_thick'.")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--fill_value", type=int, default=128,
                   help="Gray value for the filled FACE_OVAL polygon.")
    p.add_argument("--line_value", type=int, default=255,
                   help="Pixel value for drawn interior contour lines.")
    p.add_argument("--line_thickness", type=int, default=2,
                   help="Thickness in pixels of interior contour lines.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N faces (for smoke tests).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-render even if the output .pt already exists.")
    return p.parse_args()


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
    print(f"[output] {out_dir}")
    print(f"[params] fill={args.fill_value} line={args.line_value} "
          f"thickness={args.line_thickness}px size={args.height}x{args.width}")

    t0 = time.time()
    failures: list[str] = []
    skipped = 0
    written = 0
    for png in tqdm(pngs, desc="silhouette"):
        out_path = out_dir / f"{png.stem}.pt"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        t = make_option_h_uint8(
            png, args.height, args.width,
            fill_value=args.fill_value,
            line_value=args.line_value,
            line_thickness=args.line_thickness,
        )
        if t is None:
            failures.append(png.name)
            continue
        torch.save(t, out_path)
        written += 1

    dt = time.time() - t0
    print(f"[done] wrote {written}, skipped {skipped} pre-existing, "
          f"failed {len(failures)}  ({dt:.1f}s)")

    if failures:
        print(f"[fail] mediapipe could not detect a face in {len(failures)} image(s):")
        for f in failures:
            print(f"  {f}")
        print("Aborting — training would mismatch the manifest if these are "
              "left missing. Fix the inputs or rerun with --limit to exclude.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
