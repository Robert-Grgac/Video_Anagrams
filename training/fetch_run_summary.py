"""Fetch wandb run history, summarize it, write one compact JSON.

Usage:
    python training/fetch_run_summary.py \
        --entity   robert-grgac2-university-of-twente \
        --project  wan-controlnet-beta \
        --run_id   l64hi0jk \
        --out      training_cards/beta-001_wandb_summary.json

Credentials: read from ~/.netrc (machine api.wandb.ai). No env var needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import wandb

NUMERIC_METRICS = [
    "loss",
    "grad_norm",
    "lr",
    "controlnet_residual_norm",
    "timestep",
    "sigma",
    "gpu_mem_gb",
    "active_expert",
    "phase_step",
    "cycle_idx",
]


def window_stats(arr: np.ndarray) -> dict:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def downsample(steps: np.ndarray, values: np.ndarray, n: int = 80) -> list[list[float]]:
    """Pick ~n evenly-spaced points; return list of [step, value]."""
    mask = np.isfinite(values)
    steps, values = steps[mask], values[mask]
    if values.size == 0:
        return []
    if values.size <= n:
        return [[int(s), float(v)] for s, v in zip(steps, values)]
    idx = np.linspace(0, values.size - 1, n).astype(int)
    return [[int(steps[i]), float(values[i])] for i in idx]


def loss_by_sigma_bin(loss: np.ndarray, sigma: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Bin loss by sigma to see if certain noise levels are systematically harder."""
    mask = np.isfinite(loss) & np.isfinite(sigma)
    loss, sigma = loss[mask], sigma[mask]
    if loss.size == 0:
        return []
    edges = np.linspace(sigma.min(), sigma.max(), n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (sigma >= lo) & (sigma <= hi if i == n_bins - 1 else sigma < hi)
        if in_bin.sum() == 0:
            continue
        out.append(
            {
                "sigma_lo": float(lo),
                "sigma_hi": float(hi),
                "n": int(in_bin.sum()),
                "loss_mean": float(loss[in_bin].mean()),
                "loss_median": float(np.median(loss[in_bin])),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--run_id", required=True, help="wandb run id (the slug in the URL)")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--curve_points", type=int, default=80)
    args = ap.parse_args()

    api = wandb.Api()
    run = api.run(f"{args.entity}/{args.project}/{args.run_id}")

    # scan_history streams all rows; this is the lowest-overhead path for a
    # ~10k-step run. We ask for only the keys we want.
    keys = NUMERIC_METRICS + ["step"]
    rows = list(run.scan_history(keys=keys, page_size=10000))
    n = len(rows)
    if n == 0:
        raise SystemExit("no rows returned — check run id / keys")

    # build numpy columns
    cols = {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys}
    steps = cols["step"]
    if not np.all(np.isfinite(steps)):
        steps = np.arange(n, dtype=float)

    # global + windowed stats per metric
    metrics_summary = {}
    for k in NUMERIC_METRICS:
        v = cols[k]
        early = v[: max(1, n // 10)]                       # first 10%
        mid = v[int(n * 0.4) : int(n * 0.6) or n]          # middle 20%
        late = v[int(n * 0.9) :]                           # last 10%
        metrics_summary[k] = {
            "global": window_stats(v),
            "early_10pct": window_stats(early),
            "mid_20pct": window_stats(mid),
            "late_10pct": window_stats(late),
            "first_value": float(v[0]) if np.isfinite(v[0]) else None,
            "last_value": float(v[-1]) if np.isfinite(v[-1]) else None,
            "curve": downsample(steps, v, args.curve_points),
        }

    # success-criterion check from the card:
    # "final FM loss ≤ 90% of the median of the first 100 steps"
    loss = cols["loss"]
    first100_median = float(np.nanmedian(loss[:100])) if n >= 100 else float(np.nanmedian(loss))
    last100_median = float(np.nanmedian(loss[-100:])) if n >= 100 else float(np.nanmedian(loss))
    success = {
        "first_100_steps_median_loss": first100_median,
        "last_100_steps_median_loss": last100_median,
        "ratio_last_over_first": last100_median / first100_median if first100_median > 0 else None,
        "passes_descent_bar_le_0p9": last100_median <= 0.9 * first100_median,
    }

    # cross-cuts: where does loss fall by sigma regime?
    sigma_bins = loss_by_sigma_bin(loss, cols["sigma"], n_bins=10)

    # phase split (beta-002+): split loss by active_expert (0=high, 1=low).
    # This is the central diagnostic for phase-alternating runs because the
    # global loss curve mixes both regimes and obscures whether either phase
    # actually converged.
    phase_summary = None
    swap_window_summary = None
    active = cols.get("active_expert")
    if active is not None and np.isfinite(active).any():
        high_mask = (active == 0) & np.isfinite(loss)
        low_mask = (active == 1) & np.isfinite(loss)
        if high_mask.sum() and low_mask.sum():
            high_loss = loss[high_mask]
            low_loss = loss[low_mask]
            # Within-phase windows: use indices into each phase's slice so
            # late_10pct really is the last 10% of THAT phase, not of the run.
            def _win(arr):
                if arr.size == 0:
                    return {}
                e = arr[: max(1, arr.size // 10)]
                m = arr[int(arr.size * 0.4) : int(arr.size * 0.6) or arr.size]
                l = arr[int(arr.size * 0.9) :]
                return {
                    "global": window_stats(arr),
                    "early_10pct": window_stats(e),
                    "mid_20pct": window_stats(m),
                    "late_10pct": window_stats(l),
                }
            phase_summary = {
                "high_phase": _win(high_loss),
                "low_phase": _win(low_loss),
                "high_minus_low_late": (
                    float(np.median(high_loss[int(high_loss.size * 0.9):]))
                    - float(np.median(low_loss[int(low_loss.size * 0.9):]))
                ),
            }

            # Swap-boundary analysis: per swap, compare median loss in the
            # 50 steps just before vs the 50 steps just after. A sawtooth
            # (loss spikes at every swap and only partly recovers) is a tell
            # for capacity contention or too-short K.
            swap_idx = np.where(np.diff(active) != 0)[0] + 1  # boundaries
            window = 50
            jumps = []
            for s in swap_idx:
                pre = loss[max(0, s - window): s]
                post = loss[s: s + window]
                pre = pre[np.isfinite(pre)]
                post = post[np.isfinite(post)]
                if pre.size and post.size:
                    jumps.append({
                        "step": int(steps[s]) if s < steps.size else int(s),
                        "from": int(active[s - 1]),
                        "to": int(active[s]),
                        "pre_median": float(np.median(pre)),
                        "post_median": float(np.median(post)),
                        "delta": float(np.median(post) - np.median(pre)),
                    })
            if jumps:
                deltas = np.array([j["delta"] for j in jumps])
                swap_window_summary = {
                    "n_swaps": len(jumps),
                    "mean_delta_post_minus_pre": float(deltas.mean()),
                    "max_positive_delta": float(deltas.max()),
                    "max_negative_delta": float(deltas.min()),
                    "swaps": jumps,
                }

    # residuals waking up: did controlnet_residual_norm grow over training?
    res = cols["controlnet_residual_norm"]
    residual_growth = {
        "early_10pct_mean": float(np.nanmean(res[: max(1, n // 10)])),
        "late_10pct_mean": float(np.nanmean(res[int(n * 0.9) :])),
        "ratio_late_over_early": (
            float(np.nanmean(res[int(n * 0.9) :]) / np.nanmean(res[: max(1, n // 10)]))
            if np.nanmean(res[: max(1, n // 10)]) > 0
            else None
        ),
    }

    out = {
        "run": {
            "entity": args.entity,
            "project": args.project,
            "run_id": args.run_id,
            "name": run.name,
            "state": run.state,
            "url": run.url,
            "created_at": str(run.created_at),
            "runtime_sec": run.summary.get("_runtime"),
            "n_logged_steps": n,
        },
        "config": {k: v for k, v in run.config.items() if not k.startswith("_")},
        "wandb_summary": {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in run.summary.items()
            if not k.startswith("_")
        },
        "success_criterion": success,
        "metrics": metrics_summary,
        "loss_by_sigma_bin": sigma_bins,
        "residual_growth": residual_growth,
        "phase_summary": phase_summary,
        "swap_window_summary": swap_window_summary,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path}  ({size_kb:.1f} KB, {n} steps summarized)")


if __name__ == "__main__":
    main()
