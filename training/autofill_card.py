"""Substitute ``<AUTO:key>`` markers in a training card from sibling JSONs.

Reads ``training_cards/{run_id}_results.json`` and (optionally)
``training_cards/{run_id}_smoke_results.json``, then replaces every
``<AUTO:key>`` in the card with ``str(value)``. Markers without a JSON entry
are rewritten as ``<AUTO:key — MISSING>`` so they're visible in the rendered
card instead of silently dropped.

Standalone usage::

    python -m training.autofill_card training_cards/beta-001.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MARKER_RE = re.compile(r"<AUTO:([a-zA-Z0-9_]+)>")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[autofill] WARN: failed to parse {path}: {e}", file=sys.stderr)
        return {}


def autofill(card_path: Path) -> int:
    """Fill markers in ``card_path``. Returns number of substitutions made."""
    card_path = Path(card_path)
    if not card_path.exists():
        raise FileNotFoundError(card_path)
    run_id = card_path.stem  # e.g. "beta-001"
    results = _load_json(card_path.parent / f"{run_id}_results.json")
    smoke = _load_json(card_path.parent / f"{run_id}_smoke_results.json")
    # also pick up a generic precompute meta (no run-id scope) if present
    precompute = _load_json(card_path.parent / "_precompute_meta.json")
    merged = {**precompute, **smoke, **results}  # later wins

    text = card_path.read_text(encoding="utf-8")
    n_filled = 0
    n_missing = 0

    def repl(m: re.Match) -> str:
        nonlocal n_filled, n_missing
        key = m.group(1)
        if key in merged:
            n_filled += 1
            return str(merged[key])
        n_missing += 1
        return f"<AUTO:{key} — MISSING>"

    new_text = MARKER_RE.sub(repl, text)
    card_path.write_text(new_text, encoding="utf-8")
    print(f"[autofill] {card_path.name}: filled={n_filled} missing={n_missing}")
    return n_filled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("card_path", type=str)
    args = ap.parse_args()
    autofill(Path(args.card_path))


if __name__ == "__main__":
    main()
