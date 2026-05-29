"""Supplementary metric: LAION v1 aesthetic score per video (GPU rec.).

Model = openai CLIP ViT-L/14 image features (L2-normalized) -> linear head.
The head is the LAION-AI/aesthetic-predictor V1 "simple" Linear(768->1)
(`sa_0_4_vit_l_14_linear.pth`). NB: the plan §6.6 names `sac+logos+ava1-l14-
linearMSE.pth`, but that file belongs to the *V2* MLP predictor
(christophschuhmann/improved-aesthetic-predictor) — we use the V1 head the repo
actually ships. Per video: aesthetic per sampled frame, mean -> `aesthetic`.
All 4 conditions. See docs/IMPLEMENTATION_PLAN_EVAL.md §6.6.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.eval_common import (  # noqa: E402
    add_common_args,
    iter_videos,
    read_frames,
    resolve_conditions,
    write_rows,
    write_summary,
)

METRIC = "aesthetic"
_HEAD_NAME = "sa_0_4_vit_l_14_linear.pth"


def _resolve_aesthetic_ckpt() -> str:
    cands = []
    if os.environ.get("AESTHETIC_CKPT"):
        cands.append(Path(os.environ["AESTHETIC_CKPT"]))
    cands += [
        Path.home() / "src" / "aesthetic-predictor" / _HEAD_NAME,
        Path.home() / "checkpoints" / "aesthetic" / _HEAD_NAME,
        Path(__file__).parent / "third_party" / "aesthetic" / _HEAD_NAME,
    ]
    for c in cands:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        f"aesthetic head {_HEAD_NAME} not found. Set AESTHETIC_CKPT "
        f"(tried: {[str(c) for c in cands]})"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--num_frames", type=int, default=8, help="frames sampled per video")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    import clip
    from PIL import Image

    conds = resolve_conditions(args)
    download_root = os.environ.get("CLIP_CACHE")  # else clip's default ~/.cache/clip
    model, preprocess = clip.load("ViT-L/14", device=args.device, download_root=download_root)
    model.eval()

    head = nn.Linear(768, 1)
    head.load_state_dict(torch.load(_resolve_aesthetic_ckpt(), map_location="cpu"))
    head = head.to(args.device).eval()

    rows = []
    for cond, cdir in conds.items():
        n = 0
        for path, face_idx, slug in iter_videos(cdir):
            frames = read_frames(path)
            idxs = np.linspace(0, len(frames) - 1, args.num_frames).round().astype(int)
            batch = torch.stack(
                [preprocess(Image.fromarray(frames[i])) for i in idxs]
            ).to(args.device)
            with torch.no_grad():
                feats = model.encode_image(batch).float()
                feats = feats / feats.norm(dim=-1, keepdim=True)
                scores = head(feats).squeeze(-1)  # [num_frames]
            rows.append(
                {"condition": cond, "face_idx": face_idx, "slug": slug,
                 "aesthetic": float(scores.mean().item())}
            )
            n += 1
        print(f"[{cond}] {n} videos")

    if not rows:
        raise SystemExit("no rows produced")

    write_rows(METRIC, rows, args.results_dir)
    import pandas as pd

    write_summary(
        METRIC, pd.read_csv(Path(args.results_dir) / f"{METRIC}.csv"),
        args.results_dir, ["aesthetic"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
