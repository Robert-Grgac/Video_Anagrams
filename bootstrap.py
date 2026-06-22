"""Bootstrap: check assets, precompute the training cache, train the ControlNet.

Orchestrates the full path from a fresh clone with downloaded inputs to a
trained ``models/controlnet/controlnet.safetensors``:

  1. Verify the Wan 2.2 A14B model is present under ``models/wan2.2/``.
  2. Verify the HED architecture config is vendored at ``models/hed_config/``.
  3. Verify 100 ``face_*.png`` files in ``data/raw_faces/`` and 10 000
     ``face_*_*.jpg`` files in ``data/targets/``.
  4. If the precompute cache (``cache/training/manifest.json``) is missing,
     run ``training.precompute_training``.
  5. Run ``training.train`` to produce the ControlNet checkpoint.

Optional wandb logging: pass both ``--wandb_project`` and ``--wandb_run_name``
to enable; pass neither to keep it off.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_WAN_MODEL = REPO_ROOT / "models" / "wan2.2"
DEFAULT_HED_CONFIG = REPO_ROOT / "models" / "hed_config"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_CACHE_DIR = REPO_ROOT / "cache" / "training"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models" / "controlnet"

EXPECTED_FACES = 100
EXPECTED_TARGETS = 10_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bootstrap precompute + ControlNet training.",
    )
    p.add_argument(
        "--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
        help=f"Input data root. Expects raw_faces/ and targets/ underneath. "
             f"Default: {DEFAULT_DATA_DIR}.",
    )
    p.add_argument(
        "--cache_dir", type=str, default=str(DEFAULT_CACHE_DIR),
        help=f"Precompute cache dir (silhouette/latents/prompts/manifest.json). "
             f"Default: {DEFAULT_CACHE_DIR}.",
    )
    p.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help=f"Where to write controlnet.safetensors. "
             f"Default: {DEFAULT_OUTPUT_DIR}.",
    )
    p.add_argument("--wandb_project", type=str, default=None,
                   help="Wandb project. Must be passed together with --wandb_run_name.")
    p.add_argument("--wandb_run_name", type=str, default=None,
                   help="Wandb run name. Must be passed together with --wandb_project.")
    return p.parse_args()


# ---- checks ---------------------------------------------------------------

def check_wan_model(wan_dir: Path) -> None:
    print(f"[check] Wan 2.2 A14B model at {wan_dir}")
    if not (wan_dir / "model_index.json").exists():
        sys.exit(
            f"ERROR: Wan model not found at {wan_dir}.\n"
            f"Expected model_index.json plus the transformer/, transformer_2/, "
            f"vae/, text_encoder/, tokenizer/, scheduler/ subfolders.\n"
            f"Download with:\n"
            f"  huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers "
            f"--local-dir {wan_dir}"
        )
    print("        OK")


def check_hed_config(hed_dir: Path) -> None:
    print(f"[check] HED architecture config at {hed_dir}")
    if not (hed_dir / "config.json").exists():
        sys.exit(
            f"ERROR: HED config.json not found at {hed_dir}/config.json. "
            f"This file is vendored in the repo — re-fetch it if missing."
        )
    print("        OK")


def check_data(data_dir: Path) -> tuple[Path, Path]:
    faces_dir = data_dir / "raw_faces"
    targets_dir = data_dir / "targets"

    print(f"[check] raw faces at {faces_dir}")
    if not faces_dir.is_dir():
        sys.exit(f"ERROR: {faces_dir} does not exist.")
    face_pngs = list(faces_dir.glob("face_*.png"))
    if len(face_pngs) != EXPECTED_FACES:
        sys.exit(
            f"ERROR: expected {EXPECTED_FACES} face_*.png files in "
            f"{faces_dir}; found {len(face_pngs)}."
        )
    print(f"        OK ({EXPECTED_FACES} files)")

    print(f"[check] targets at {targets_dir}")
    if not targets_dir.is_dir():
        sys.exit(f"ERROR: {targets_dir} does not exist.")
    target_jpgs = list(targets_dir.glob("face_*_*.jpg"))
    if len(target_jpgs) != EXPECTED_TARGETS:
        sys.exit(
            f"ERROR: expected {EXPECTED_TARGETS} face_*_*.jpg files in "
            f"{targets_dir}; found {len(target_jpgs)}."
        )
    print(f"        OK ({EXPECTED_TARGETS} files)")

    return faces_dir, targets_dir


def precompute_done(cache_dir: Path) -> bool:
    manifest = cache_dir / "manifest.json"
    if not manifest.exists():
        return False
    try:
        records = json.loads(manifest.read_text())
    except Exception:
        return False
    return len(records) == EXPECTED_TARGETS


# ---- runner ---------------------------------------------------------------

def run_stage(cmd: list[str], label: str) -> None:
    print(f"\n[{label}] " + " ".join(cmd) + "\n")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        sys.exit(f"ERROR: {label} exited with code {result.returncode}")


def main() -> None:
    args = parse_args()

    if (args.wandb_project is None) != (args.wandb_run_name is None):
        sys.exit(
            "ERROR: --wandb_project and --wandb_run_name must be passed "
            "together (or neither — wandb is off by default)."
        )

    print("=" * 70)
    print("Bootstrap: video-anagrams ControlNet training")
    print("=" * 70)

    wan_dir = Path(os.environ.get("WAN_MODEL", str(DEFAULT_WAN_MODEL))).resolve()
    hed_dir = DEFAULT_HED_CONFIG
    data_dir = Path(args.data_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    check_wan_model(wan_dir)
    check_hed_config(hed_dir)
    faces_dir, targets_dir = check_data(data_dir)

    if precompute_done(cache_dir):
        print(f"\n[skip] precompute already complete at {cache_dir} "
              f"(manifest has {EXPECTED_TARGETS} entries).")
    else:
        print(f"\n[precompute] cache missing or incomplete; running ...")
        run_stage([
            sys.executable, "-m", "training.precompute_training",
            "--input_faces_dir", str(faces_dir),
            "--targets_dir", str(targets_dir),
            "--output_dir", str(cache_dir),
            "--base_model_path", str(wan_dir),
        ], "precompute")

    print("\n[train] launching training ...")
    train_cmd = [
        sys.executable, "-m", "training.train",
        "--cache_dir", str(cache_dir),
        "--base_model_path", str(wan_dir),
        "--controlnet_config_repo", str(hed_dir),
        "--output_dir", str(output_dir),
    ]
    if args.wandb_project is not None:
        train_cmd += [
            "--wandb_project", args.wandb_project,
            "--wandb_run_name", args.wandb_run_name,
        ]
    run_stage(train_cmd, "train")

    final = output_dir / "controlnet.safetensors"
    print(f"\n[done] ControlNet saved to {final}")


if __name__ == "__main__":
    main()
