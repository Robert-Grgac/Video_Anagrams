"""Side-by-side visualisation of ControlNet input modalities.

For one face PNG, renders four panels:
    source RGB | canny | binary silhouette (opt A) | option H (silhouette + features)
into a single composite PNG. Lets us eyeball what each modality looks like
before committing to a full precompute / training run.

Run from the silhouette env (mediapipe required):

    python training/visualize_control_inputs.py \\
        --face_png $HOME/data/wan-beta/input_faces/face_0.png \\
        --output /tmp/control_inputs_face_0.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.face_silhouette import (
    make_binary_silhouette_uint8,
    make_option_h_uint8,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--face_png", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--canny_low", type=int, default=100)
    p.add_argument("--canny_high", type=int, default=200)
    return p.parse_args()


def _canny_arr(face_png: Path, h: int, w: int, low: int, high: int) -> np.ndarray:
    img = Image.open(face_png).convert("RGB")
    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    return cv2.Canny(gray, low, high)


def _label_panel(arr: np.ndarray, text: str) -> np.ndarray:
    """Pad a 28-px caption strip under an HxW(xC) panel."""
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    h, w = arr.shape[:2]
    strip = np.full((28, w, 3), 30, dtype=np.uint8)
    pil = Image.fromarray(strip)
    ImageDraw.Draw(pil).text((6, 6), text, fill=(220, 220, 220))
    return np.vstack([arr, np.asarray(pil)])


def main() -> int:
    args = parse_args()
    face_png = Path(args.face_png)
    out_path = Path(args.output)

    src_img = Image.open(face_png).convert("RGB")
    if src_img.size != (args.width, args.height):
        src_img = src_img.resize((args.width, args.height), Image.LANCZOS)
    src_arr = np.asarray(src_img)

    canny_arr = _canny_arr(face_png, args.height, args.width,
                           args.canny_low, args.canny_high)

    sil = make_binary_silhouette_uint8(face_png, args.height, args.width)
    if sil is None:
        print(f"[error] mediapipe found no face in {face_png}")
        return 1
    sil_arr = sil.permute(1, 2, 0).numpy()

    h_t = make_option_h_uint8(face_png, args.height, args.width)
    h_arr = h_t.permute(1, 2, 0).numpy()

    canny_density = (canny_arr > 0).mean() * 100
    sil_density = (sil_arr[..., 0] > 0).mean() * 100
    h_density = (h_arr[..., 0] > 0).mean() * 100

    panels = [
        _label_panel(src_arr, f"source ({face_png.name})"),
        _label_panel(
            canny_arr,
            f"canny  low={args.canny_low} hi={args.canny_high}  density={canny_density:.2f}%",
        ),
        _label_panel(
            sil_arr,
            f"opt A: binary silhouette (FACE_OVAL fill)  density={sil_density:.2f}%",
        ),
        _label_panel(
            h_arr,
            f"opt H: silhouette (gray) + feature contours (white)  density={h_density:.2f}%",
        ),
    ]
    composite = np.hstack(panels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite).save(out_path)
    print(f"[done] wrote {out_path}  shape={composite.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
