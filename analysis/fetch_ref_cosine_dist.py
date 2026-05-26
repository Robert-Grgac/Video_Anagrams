"""Pull `ref_cosine_dist_mean` from one good run and one bad run, dump to JSON.

Targets:
    BAD : project=CN_PTD_inference_2,             run name=face_17_iceberg
    GOOD: project=wan-heuristic-experiments-4.2,  run name=face1_prompt6

Credentials read from ~/.netrc (machine api.wandb.ai). Only this single metric
is fetched — history (every logged step) plus the summary scalar.

Usage:
    python analysis/fetch_ref_cosine_dist.py
    # override entity / runs / output:
    python analysis/fetch_ref_cosine_dist.py \
        --entity robert-grgac2-university-of-twente \
        --out    analysis/ref_cosine_dist.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import wandb

METRIC = "ref_cosine_dist_mean"

TARGETS = [
    {"label": "bad_CN_PTD_CN_trulyoff",  "project": "CN_PTD_inference_4_debugging", "run_name": "face_0_snowy_mountain"},
    {"label": "good_og_ptd_pipeline", "project": "PTD_inference_original_pipeline", "run_name": "face_1_prompt2"},
]


def find_run_by_name(api: wandb.Api, entity: str, project: str, run_name: str):
    """Resolve a run by display name. wandb names are not unique, so if more
    than one matches we pick the most recent and warn."""
    matches = list(
        api.runs(f"{entity}/{project}", filters={"display_name": run_name})
    )
    if not matches:
        raise SystemExit(
            f"no run named {run_name!r} in {entity}/{project} — "
            f"check entity / project / name spelling"
        )
    if len(matches) > 1:
        print(
            f"warning: {len(matches)} runs match name={run_name!r} in {project}; "
            f"using most recent (id={matches[0].id})"
        )
    return matches[0]


def fetch_metric_history(run, metric: str) -> list[dict]:
    """Stream every logged row for `metric`. Returns [{'step': int, 'value': float}, ...]."""
    rows = list(run.scan_history(keys=[metric], page_size=10000))
    out = []
    for i, r in enumerate(rows):
        v = r.get(metric)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        # `_step` is wandb's row index; fall back to enumerate if absent
        step = r.get("_step", i)
        out.append({"step": int(step), "value": v})
    return out


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="robert-grgac2-university-of-twente")
    ap.add_argument("--out", default="analysis/ref_cosine_dist_og_ptd_vs_cnptd_truly_off.json")
    args = ap.parse_args()

    api = wandb.Api()

    payload = {"metric": METRIC, "entity": args.entity, "runs": {}}

    for tgt in TARGETS:
        label, project, run_name = tgt["label"], tgt["project"], tgt["run_name"]
        print(f"[{label}] resolving {args.entity}/{project} :: {run_name!r} ...")
        run = find_run_by_name(api, args.entity, project, run_name)
        print(f"[{label}] found run id={run.id} state={run.state} url={run.url}")

        history = fetch_metric_history(run, METRIC)
        values = [h["value"] for h in history]
        summary_val = run.summary.get(METRIC)
        try:
            summary_val = float(summary_val) if summary_val is not None else None
        except (TypeError, ValueError):
            summary_val = None

        payload["runs"][label] = {
            "project": project,
            "run_name": run_name,
            "run_id": run.id,
            "state": run.state,
            "url": run.url,
            "created_at": str(run.created_at),
            "summary_value": summary_val,
            "history_stats": stats(values),
            "history": history,
        }
        print(
            f"[{label}] {METRIC}: n={len(values)} "
            f"summary={summary_val} "
            f"mean={payload['runs'][label]['history_stats'].get('mean')}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    size_kb = out_path.stat().st_size / 1024
    print(f"\nwrote {out_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
