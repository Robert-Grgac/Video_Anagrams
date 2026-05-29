"""Face-axis metric: RetinaFace detection over a video's frames (CPU fine).

Model: insightface `buffalo_l` (RetinaFace detector). Per video, run the detector
on every (or every Nth) frame and record:
  det_rate  - fraction of frames with >=1 face at threshold tau
  mean_conf - mean per-frame max det_score over frames that had a detection
  max_conf  - max det_score over all frames
Higher = more face present. All 4 conditions.
See docs/IMPLEMENTATION_PLAN_EVAL.md §6.4.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.eval_common import (  # noqa: E402
    add_common_args,
    iter_videos,
    read_frames,
    resolve_conditions,
    write_rows,
    write_summary,
)

METRIC = "retinaface"


def _build_app(tau: float, device: str, det_size: int):
    from insightface.app import FaceAnalysis

    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1
    root = os.environ.get("INSIGHTFACE_HOME")
    kw = {"name": "buffalo_l", "providers": providers, "allowed_modules": ["detection"]}
    if root:
        kw["root"] = root
    app = FaceAnalysis(**kw)
    app.prepare(ctx_id=ctx_id, det_thresh=tau, det_size=(det_size, det_size))
    return app


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--tau", type=float, default=0.5, help="detection score threshold")
    p.add_argument("--every", type=int, default=1, help="run detector every Nth frame")
    p.add_argument("--det_size", type=int, default=640)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    conds = resolve_conditions(args)
    app = _build_app(args.tau, args.device, args.det_size)

    rows = []
    for cond, cdir in conds.items():
        n = 0
        for path, face_idx, slug in iter_videos(cdir):
            frames = read_frames(path, every=args.every)
            per_frame_max = []  # max det_score per frame (0.0 if no detection)
            for fr in frames:
                bgr = np.ascontiguousarray(fr[:, :, ::-1])  # insightface wants BGR
                faces = app.get(bgr)
                scores = [float(f.det_score) for f in faces]
                per_frame_max.append(max(scores) if scores else 0.0)
            per_frame_max = np.asarray(per_frame_max, dtype=float)
            detected = per_frame_max > 0.0
            det_rate = float(detected.mean()) if per_frame_max.size else 0.0
            mean_conf = float(per_frame_max[detected].mean()) if detected.any() else 0.0
            max_conf = float(per_frame_max.max()) if per_frame_max.size else 0.0
            rows.append(
                {"condition": cond, "face_idx": face_idx, "slug": slug,
                 "det_rate": det_rate, "mean_conf": mean_conf, "max_conf": max_conf}
            )
            n += 1
        print(f"[{cond}] {n} videos")

    if not rows:
        raise SystemExit("no rows produced")

    write_rows(METRIC, rows, args.results_dir)
    import pandas as pd

    write_summary(
        METRIC, pd.read_csv(Path(args.results_dir) / f"{METRIC}.csv"),
        args.results_dir, ["det_rate", "mean_conf", "max_conf"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
