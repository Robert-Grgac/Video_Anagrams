"""Render the 4 (prompt-axis x face-axis) scatter plots (CPU, login node).

Every video is a point in (prompt x face) space, colored by condition:
  prompt axis in { ViCLIP, 1 - LPIPS(frame1, vanilla) }
  face   axis in { 1 - LPIPS(frame1, source_face), RetinaFace det_rate }
  -> 2 x 2 = 4 plots. The "good illusion" lives in the interior, not the corners.
The PTD-only mean is drawn as a reference operating point.
See docs/IMPLEMENTATION_PLAN_EVAL.md §6.7.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.eval_common import CONDITIONS, DEFAULT_RESULTS_DIR  # noqa: E402

# (csv filename, value column, axis label)
PROMPT_AXES = {
    "viclip": ("viclip.csv", "viclip", "ViCLIP score"),
    "lpipsvanilla": ("lpips_vanilla.csv", "score", "1 - LPIPS(frame1, vanilla)"),
}
FACE_AXES = {
    "lpipsface": ("lpips_face.csv", "score", "1 - LPIPS(frame1, source face)"),
    "retinaface": ("retinaface.csv", None, "RetinaFace det_rate"),  # col set via CLI
}

COND_COLORS = {
    "vanilla": "#7f7f7f",
    "ptd_only": "#1f77b4",
    "cn_only": "#2ca02c",
    "cn_ptd": "#d62728",
}
COND_ORDER = ["vanilla", "ptd_only", "cn_only", "cn_ptd"]


def _load(results_dir: Path, fname: str, col: str) -> pd.DataFrame:
    path = results_dir / fname
    if not path.exists():
        raise SystemExit(f"missing input CSV: {path} (run its eval script first)")
    df = pd.read_csv(path)
    return df[["condition", "face_idx", "slug", col]].rename(columns={col: "value"})


def _scatter(ax, merged: pd.DataFrame, xlabel: str, ylabel: str):
    for cond in COND_ORDER:
        g = merged[merged["condition"] == cond]
        if g.empty:
            continue
        ax.scatter(
            g["x"], g["y"], s=28, alpha=0.55, label=cond,
            color=COND_COLORS.get(cond, None), edgecolors="none",
        )
    # Per-condition cluster means (centroid of each color group).
    for cond in COND_ORDER:
        g = merged[merged["condition"] == cond]
        if g.empty:
            continue
        ax.scatter(
            [g["x"].mean()], [g["y"].mean()],
            marker="X", s=240, color=COND_COLORS.get(cond, None),
            edgecolors="black", linewidths=1.4, zorder=5,
        )
    # Single legend entry covering all cluster-mean markers (color matches each
    # condition's scatter; the X-with-black-edge form is what's shared).
    mean_handle = Line2D(
        [], [], marker="X", linestyle="None", markersize=11,
        markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.4,
        label="condition mean",
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mean_handle)
    labels.append(mean_handle.get_label())
    ax.legend(handles, labels, fontsize=7, loc="best")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--plots_dir", type=str,
                   default=str(Path(__file__).parent / "plots"))
    p.add_argument("--retina_col", default="det_rate",
                   choices=["det_rate", "mean_conf", "max_conf"],
                   help="which RetinaFace column to use on the face axis")
    p.add_argument("--vanilla_mask", type=str, default=None,
                   help="Path to a Python file exposing 'manual_face_labeling' "
                        "(dict[filename->flag]). When set, restrict to slugs "
                        "where flag==1. Default scope = vanilla only (cleans "
                        "its detector noise floor without altering the methods); "
                        "use --mask_all_conditions to apply the slug subset to "
                        "every condition.")
    p.add_argument("--mask_all_conditions", action="store_true",
                   help="With --vanilla_mask: filter ALL conditions to the "
                        "flag==1 slug subset (apples-to-apples on the same "
                        "prompt subset).")
    p.add_argument("--retina_only", action="store_true",
                   help="Only render the two RetinaFace-axis plots.")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    face_axes = dict(FACE_AXES)
    face_axes["retinaface"] = ("retinaface.csv", args.retina_col,
                               f"RetinaFace {args.retina_col}")

    # Optional vanilla flag mask. Default scope = vanilla only (cleans its
    # detector noise floor caused by natural scene people, without touching
    # the methods).
    flag1_slugs = None
    conds_to_filter = None
    if args.vanilla_mask:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_mask", args.vanilla_mask)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        labels = mod.manual_face_labeling
        flag1_slugs = {k.rsplit(".", 1)[0] for k, v in labels.items() if v == 1}
        conds_to_filter = COND_ORDER if args.mask_all_conditions else ["vanilla"]
        print(f"[mask] {len(flag1_slugs)}/{len(labels)} flag==1 slugs; "
              f"filter applied to: {conds_to_filter}")

    def _apply_mask(df):
        if flag1_slugs is None:
            return df
        keep = (~df["condition"].isin(conds_to_filter)) | df["slug"].isin(flag1_slugs)
        return df[keep]

    suffix = "_facefree" if args.vanilla_mask else ""

    written = []
    for pkey, (pf, pcol, plabel) in PROMPT_AXES.items():
        px = _apply_mask(_load(results_dir, pf, pcol)).rename(columns={"value": "x"})
        for fkey, (ff, fcol, flabel) in face_axes.items():
            if args.retina_only and fkey != "retinaface":
                continue
            fy = _apply_mask(_load(results_dir, ff, fcol)).rename(columns={"value": "y"})
            merged = px.merge(fy[["condition", "face_idx", "y"]],
                              on=["condition", "face_idx"], how="inner")
            fig, ax = plt.subplots(figsize=(6.5, 6))
            _scatter(ax, merged, plabel, flabel)
            ax.set_title(f"{plabel}  vs  {flabel}", fontsize=10)
            out = plots_dir / f"scatter_{pkey}__{fkey}{suffix}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            written.append(out)
            print(f"[write] {out}  ({len(merged)} points)")

    print(f"[done] wrote {len(written)} plots to {plots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
