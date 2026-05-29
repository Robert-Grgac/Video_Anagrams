"""Face-axis metric: LPIPS(first_frame(video), source_face(face_idx)).

score = 1 - lpips (higher = more face structure). Computed for ALL 4 conditions
incl. vanilla (the pure-scene anchor). Domain gap (face photo vs painted scene)
compresses the dynamic range -> treat RetinaFace as the primary face signal.
See docs/IMPLEMENTATION_PLAN_EVAL.md §6.3. CPU is fine.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.eval_common import (  # noqa: E402
    add_common_args,
    first_frame,
    iter_videos,
    load_source_face,
    resolve_conditions,
    write_rows,
    write_summary,
)

METRIC = "lpips_face"


def to_lpips_tensor(img_u8: np.ndarray, device: str) -> torch.Tensor:
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
    loss_fn = lpips.LPIPS(net=args.net).to(args.device).eval()

    face_cache: dict[int, np.ndarray] = {}
    rows = []
    for cond, cdir in conds.items():
        n = 0
        for path, face_idx, slug in iter_videos(cdir):
            frame = first_frame(path)
            h, w = frame.shape[:2]
            key = (face_idx, h, w)
            if key not in face_cache:
                face_cache[key] = load_source_face(face_idx, size=(h, w))
            face = face_cache[key]
            with torch.no_grad():
                d = loss_fn(
                    to_lpips_tensor(frame, args.device),
                    to_lpips_tensor(face, args.device),
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
