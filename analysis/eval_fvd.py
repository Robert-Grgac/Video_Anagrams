"""Set-level metric: FVD of each method's 100 clips vs the 100 vanilla clips (GPU).

Backbone: cd-fvd's I3D (Kinetics-400). Real distribution = vanilla; for each
method condition we accumulate I3D stats over its 100 clips and Frechet-distance
them against the vanilla stats (computed once, reused). Includes vanilla-vs-vanilla
as a near-zero sanity floor.

cd-fvd's preprocess_i3d resamples each clip to 224x224 and keeps all frames, so we
feed the raw 61 frames per clip. N=100 biases FVD high -> comparative only (§10).
See docs/IMPLEMENTATION_PLAN_EVAL.md §6.5.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.eval_common import (  # noqa: E402
    CONDITIONS,
    DEFAULT_RESULTS_DIR,
    add_common_args,
    iter_videos,
    read_frames,
    resolve_conditions,
)

METRIC = "fvd"


def _resolve_i3d_ckpt() -> str | None:
    cands = []
    if os.environ.get("CDFVD_I3D_CKPT"):
        cands.append(Path(os.environ["CDFVD_I3D_CKPT"]))
    import cdfvd

    bundled = Path(cdfvd.__file__).parent / "third_party" / "i3d" / "i3d_pretrained_400.pt"
    cands.append(bundled)
    cands.append(Path.home() / "checkpoints" / "cdfvd" / "i3d_pretrained_400.pt")
    for c in cands:
        if c.exists():
            return str(c)
    print(f"[fvd] no cached I3D ckpt (tried {[str(c) for c in cands]}); "
          "cd-fvd will attempt to download (needs outbound internet)")
    return None


def _accumulate(evaluator, cdir: Path, kind: str) -> int:
    """Stream a condition's clips through the I3D feature extractor (kind in
    {'real','fake'}). Returns the number of clips added."""
    n = 0
    for path, _idx, _slug in iter_videos(cdir):
        vid = read_frames(path)[None]  # (1, T, H, W, C) uint8
        if kind == "real":
            evaluator.add_real_stats(vid)
        else:
            evaluator.add_fake_stats(vid)
        n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from cdfvd import fvd

    conds = resolve_conditions(args)
    methods = [c for c in conds if c != "vanilla"]
    vanilla_dir = CONDITIONS["vanilla"]
    if not vanilla_dir.is_dir():
        raise SystemExit(f"vanilla reference dir missing: {vanilla_dir}")

    i3d_ckpt = _resolve_i3d_ckpt()
    evaluator = fvd.cdfvd(
        model="i3d", n_real="full", n_fake="full",
        ckpt_path=i3d_ckpt, device=args.device,
    )

    # Reference (real) stats over the 100 vanilla clips, computed once.
    print(f"[fvd] accumulating real stats from {vanilla_dir.name}")
    n_real = _accumulate(evaluator, vanilla_dir, "real")
    clip_len = read_frames(next(iter_videos(vanilla_dir))[0]).shape[0]

    scores: dict[str, float] = {}
    for cond in methods:
        evaluator.empty_fake_stats()
        n_fake = _accumulate(evaluator, conds[cond], "fake")
        score = float(evaluator.compute_fvd_from_stats())
        scores[cond] = score
        print(f"[fvd] {cond}: {score:.3f}  (n_fake={n_fake})")

    # Sanity floor: FVD(vanilla, vanilla) using the shared real stats -> ~0.
    scores["vanilla"] = float(
        evaluator.compute_fvd_from_stats(
            fake_stats=evaluator.real_stats, real_stats=evaluator.real_stats
        )
    )
    print(f"[fvd] vanilla (sanity floor): {scores['vanilla']:.6f}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{METRIC}.json"

    # Idempotent per-condition merge (mirrors write_rows for CSV metrics): if a
    # prior fvd.json exists, keep its scores for conditions we didn't recompute
    # this run, and overwrite the ones we did. Without this, passing
    # EVAL_CONDITIONS="cn_only" would clobber ptd_only / cn_ptd entries.
    merged_scores: dict[str, float] = {}
    if path.exists():
        try:
            merged_scores = dict(json.loads(path.read_text()).get("scores", {}))
        except Exception as e:
            print(f"[warn] could not parse existing {path} ({e}); overwriting")
            merged_scores = {}
    merged_scores.update(scores)

    out = {
        "backbone": "i3d",
        "ref": vanilla_dir.name,
        "scores": merged_scores,
        "n_per_set": n_real,
        "clip_len": int(clip_len),
    }
    path.write_text(json.dumps(out, indent=2))
    print(f"[write] {path}  (scored this run: {sorted(scores)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
