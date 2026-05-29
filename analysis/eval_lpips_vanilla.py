"""Prompt-axis metric: LPIPS(first_frame(video), first_frame(vanilla[slug])).

score = 1 - lpips  (higher = closer to clean Wan). The `vanilla` condition is the
reference, so its own value is 0 by construction -> skipped by default.
See docs/IMPLEMENTATION_PLAN_EVAL.md §6.2. CPU is fine.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.eval_common import (  # noqa: E402
    CONDITIONS,
    add_common_args,
    first_frame,
    iter_videos,
    resolve_conditions,
    write_rows,
    write_summary,
)

METRIC = "lpips_vanilla"


def to_lpips_tensor(img_u8: np.ndarray, device: str) -> torch.Tensor:
    """uint8 HWC RGB -> float [1,3,H,W] in [-1,1]."""
    t = torch.from_numpy(img_u8.astype(np.float32) / 127.5 - 1.0)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    import lpips

    conds = resolve_conditions(args)
    # Build slug -> vanilla mp4 path lookup from the vanilla condition dir.
    vanilla_dir = CONDITIONS["vanilla"]
    vanilla_by_slug = {slug: path for path, _idx, slug in iter_videos(vanilla_dir)}
    if not vanilla_by_slug:
        raise SystemExit(f"no vanilla reference videos found in {vanilla_dir}")

    loss_fn = lpips.LPIPS(net=args.net).to(args.device).eval()

    rows = []
    for cond, cdir in conds.items():
        if cond == "vanilla":
            print("[skip] 'vanilla' is the reference (lpips=0 by construction)")
            continue
        n = 0
        for path, face_idx, slug in iter_videos(cdir):
            ref_path = vanilla_by_slug.get(slug)
            if ref_path is None:
                print(f"[warn] no vanilla reference for slug {slug!r}; skipping")
                continue
            a = first_frame(path)
            b = first_frame(ref_path)
            if a.shape != b.shape:
                import cv2

                b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
            with torch.no_grad():
                d = loss_fn(
                    to_lpips_tensor(a, args.device), to_lpips_tensor(b, args.device)
                ).item()
            rows.append(
                {"condition": cond, "face_idx": face_idx, "slug": slug,
                 "lpips": d, "score": 1.0 - d}
            )
            n += 1
        print(f"[{cond}] {n} videos")

    if not rows:
        raise SystemExit("no rows produced")

    write_rows(METRIC, rows, args.results_dir)
    import pandas as pd

    write_summary(
        METRIC, pd.read_csv(Path(args.results_dir) / f"{METRIC}.csv"),
        args.results_dir, ["score", "lpips"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
