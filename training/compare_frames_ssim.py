#!/usr/bin/env python3
"""Compare the first frame of two mp4 videos with SSIM (and pixel MSE).

Uses pytorch_msssim on CPU, identical to the training-time eval metric, so the
printed SSIM is directly comparable to values in beta-007_eval.json.

No GPU required.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_msssim import ssim as pmsssim_ssim


DEFAULT_A = "/home/s2710099/outputs/wan-beta/beta-007/periodic/step00010_sample000_face0_misty_morning.mp4"
DEFAULT_B = "/home/s2710099/outputs/wan-beta/beta-007/periodic/step00300_sample000_face0_misty_morning.mp4"


def first_frame_rgb01(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"could not read first frame from {path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb.astype(np.float32) / 255.0


def to_nchw(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).contiguous()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", default=DEFAULT_A, help="path to first mp4")
    ap.add_argument("--b", default=DEFAULT_B, help="path to second mp4")
    args = ap.parse_args()

    a = first_frame_rgb01(args.a)
    b = first_frame_rgb01(args.b)
    if a.shape != b.shape:
        print(f"shape mismatch: a={a.shape} b={b.shape}", file=sys.stderr)
        return 1

    ssim_val = pmsssim_ssim(to_nchw(a), to_nchw(b), data_range=1.0, size_average=True).item()
    mse_val = float(((a - b) ** 2).mean())

    print(f"a:     {Path(args.a).name}")
    print(f"b:     {Path(args.b).name}")
    print(f"shape: {a.shape}  (HxWxC, RGB, [0,1])")
    print(f"ssim:  {ssim_val:.6f}")
    print(f"mse:   {mse_val:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
