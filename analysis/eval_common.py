"""Shared import surface for every eval script in analysis/.

One place for: the canonical slug/index lookups, the four condition dirs,
video decoding, source-face loading, and the tidy-CSV + summary-JSON writers.
See docs/IMPLEMENTATION_PLAN_EVAL.md §5.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

# sys.path shim: analysis/<file>.py reaches repo root for `training.*` imports.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.input_prompts import PROMPTS_BATCH_1, PROMPTS_BATCH_2  # noqa: E402

# ---------------------------------------------------------------------------
# Canonical pairing — slug order is BATCH_1 then BATCH_2, index i <-> face_i.
# ---------------------------------------------------------------------------
ALL_PROMPTS: dict[str, str] = {**PROMPTS_BATCH_1, **PROMPTS_BATCH_2}
ALL_SLUGS: list[str] = list(PROMPTS_BATCH_1.keys()) + list(PROMPTS_BATCH_2.keys())
assert len(ALL_SLUGS) == 100, f"expected 100 slugs, got {len(ALL_SLUGS)}"
SLUG_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(ALL_SLUGS)}
IDX_TO_SLUG: dict[int, str] = {i: s for i, s in enumerate(ALL_SLUGS)}

# ---------------------------------------------------------------------------
# The four conditions (§2). Dirs overridable via env:
#   EVAL_INFERENCE_ROOT  -> base dir holding all condition subdirs
#   EVAL_DIR_<LABEL>     -> absolute override for one condition (e.g. EVAL_DIR_VANILLA)
# ---------------------------------------------------------------------------
INFER_ROOT = Path(
    os.environ.get("EVAL_INFERENCE_ROOT", str(Path.home() / "outputs" / "inference"))
)

_DEFAULT_DIRS = {
    # The "vanilla" slot is the reference baseline for FVD / LPIPS and the
    # source of the baseline aesthetic/RetinaFace/viclip scores. It now points
    # at the TI2V-5B baseline (wan_ti2v_100) instead of the plain T2V vanilla
    # set. The KEY stays "vanilla" because eval_fvd.py / eval_lpips_vanilla.py
    # resolve the reference via the hardcoded CONDITIONS["vanilla"].
    "vanilla": "wan_ti2v_100",
    "ptd_only": "ptd_og_pipeline_100_fair",
    "cn_only": "beta008_100_fair",
    "cn_ptd": "ptd_cn_final_cfg5_cnw2p5",
}


def _cond_dir(label: str, default_name: str) -> Path:
    env = os.environ.get(f"EVAL_DIR_{label.upper()}")
    return Path(env) if env else INFER_ROOT / default_name


CONDITIONS: dict[str, Path] = {
    lab: _cond_dir(lab, name) for lab, name in _DEFAULT_DIRS.items()
}

# Source faces (raw uint8 (3,H,W) tensors), used by the face-axis metrics.
RAW_FACE_DIR = Path(
    os.environ.get("RAW_FACE_DIR", str(Path.home() / "cache" / "wan-beta" / "raw_face"))
)

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"

_FACE_RE = re.compile(r"^face_(\d+)_(.+)\.mp4$")


# ---------------------------------------------------------------------------
# Video iteration / decoding
# ---------------------------------------------------------------------------
def iter_videos(condition_dir: Path | str) -> Iterator[tuple[Path, int, str]]:
    """Yield (path, face_idx, slug) for every mp4 in a condition dir.

    Method files are `face_{i}_{slug}.mp4`; vanilla files are `{slug}.mp4`
    whose face_idx is recovered via SLUG_TO_IDX.
    """
    condition_dir = Path(condition_dir)
    for path in sorted(condition_dir.glob("*.mp4")):
        m = _FACE_RE.match(path.name)
        if m:
            yield path, int(m.group(1)), m.group(2)
        else:
            slug = path.stem
            if slug not in SLUG_TO_IDX:
                print(f"[warn] unrecognized file, skipping: {path.name}")
                continue
            yield path, SLUG_TO_IDX[slug], slug


def read_frames(
    path: Path | str, every: int = 1, max_frames: int | None = None
) -> np.ndarray:
    """Decode an mp4 to uint8 [T,H,W,3] RGB. decord (fast) with imageio fallback."""
    path = str(path)
    try:
        import decord

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path)
        idxs = list(range(0, len(vr), every))
        if max_frames is not None:
            idxs = idxs[:max_frames]
        frames = vr.get_batch(idxs).asnumpy()
        return np.ascontiguousarray(frames).astype(np.uint8)
    except Exception:
        import imageio.v2 as imageio

        reader = imageio.get_reader(path)
        out = []
        for i, fr in enumerate(reader):
            if i % every != 0:
                continue
            out.append(np.asarray(fr)[..., :3])
            if max_frames is not None and len(out) >= max_frames:
                break
        reader.close()
        return np.stack(out).astype(np.uint8)


def first_frame(path: Path | str) -> np.ndarray:
    """First frame as uint8 [H,W,3] RGB."""
    return read_frames(path, every=1, max_frames=1)[0]


def load_source_face(face_idx: int, size: tuple[int, int] | None = None) -> np.ndarray:
    """Source face as uint8 [H,W,3] RGB. `size=(H,W)` resizes (for LPIPS match)."""
    import torch

    pt = RAW_FACE_DIR / f"face_{face_idx}.pt"
    if not pt.exists():
        raise FileNotFoundError(f"source face not found: {pt}")
    t = torch.load(pt, map_location="cpu", weights_only=True)  # (3,H,W) uint8
    arr = t.permute(1, 2, 0).contiguous().numpy().astype(np.uint8)  # HWC RGB
    if size is not None:
        import cv2

        h, w = size
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    return arr


# ---------------------------------------------------------------------------
# Result storage — tidy long CSV (idempotent per-condition) + summary JSON
# ---------------------------------------------------------------------------
def write_rows(metric: str, rows: list[dict], results_dir: Path | str) -> Path:
    """Append/overwrite `metric.csv`. Idempotent per-condition: any condition
    present in `rows` has its existing rows replaced; other conditions kept."""
    import pandas as pd

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv = results_dir / f"{metric}.csv"
    new = pd.DataFrame(rows)
    conds = set(new["condition"].unique())
    if csv.exists():
        old = pd.read_csv(csv)
        old = old[~old["condition"].isin(conds)]
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined = combined.sort_values(["condition", "face_idx"]).reset_index(drop=True)
    combined.to_csv(csv, index=False)
    print(f"[write] {csv}  ({len(new)} rows for {sorted(conds)})")
    return csv


def write_summary(
    metric: str, df, results_dir: Path | str, value_cols: Sequence[str]
) -> Path:
    """Per-condition mean/std/median/IQR/n for each value column -> summary JSON."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict] = {}
    for cond, g in df.groupby("condition"):
        out[cond] = {}
        for col in value_cols:
            vals = g[col].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                out[cond][col] = {"n": 0}
                continue
            q1, q3 = np.percentile(vals, [25, 75])
            out[cond][col] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "median": float(np.median(vals)),
                "iqr": float(q3 - q1),
                "n": int(vals.size),
            }
    path = results_dir / f"{metric}_summary.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[write] {path}")
    return path


# ---------------------------------------------------------------------------
# CLI helpers (every per-metric script shares these two args)
# ---------------------------------------------------------------------------
def add_common_args(parser):
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        help=f"subset of {list(CONDITIONS)}; default all",
    )
    parser.add_argument(
        "--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR),
        help="dir for the per-metric CSV + summary JSON",
    )
    return parser


def resolve_conditions(args) -> dict[str, Path]:
    labels = args.conditions or list(CONDITIONS.keys())
    out: dict[str, Path] = {}
    for lab in labels:
        if lab not in CONDITIONS:
            raise SystemExit(f"unknown condition {lab!r}; known: {list(CONDITIONS)}")
        d = CONDITIONS[lab]
        if not d.is_dir():
            print(f"[warn] condition dir missing: {lab} -> {d}")
        out[lab] = d
    return out
