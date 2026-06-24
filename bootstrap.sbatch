#!/bin/bash
# =============================================================================
# bootstrap.sbatch  (wan22-test smoke run)
#
# Runs `python bootstrap.py` far enough to prove the pipeline trains: it skips
# precompute (the cache below already has all 10 000 entries) and launches
# training.train, which forwards/backwards through the high-noise expert and
# asserts the ControlNet gets gradients. This is a SMOKE TEST — let it reach
# the first effective optimiser step (tqdm shows `step` / global_step=1, or the
# grad-flow assertion passes a few micro-steps in) then `scancel` the job.
#
# Submit from the repo root:
#     sbatch bootstrap.sbatch
# =============================================================================
#SBATCH --job-name=wan22test-bootstrap
#SBATCH --partition=dmb
# Suitable nodes: hpc-node07/09 (L40 48GB), ctit092 (A40 45GB). Exclude
# ctit084/085 (RTX6000 24GB) and hpc-node31 (Blackwell -> needs wan22-bw env).
#SBATCH --exclude=ctit084,ctit085,hpc-node31
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/home/s2710099/logs/wan22test-bootstrap-%j.out
#SBATCH --error=/home/s2710099/logs/wan22test-bootstrap-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams
# =============================================================================

set -euo pipefail

# ---- Conda env (the NEW test env, not the existing wan22) --------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wan22-test

# ---- Runtime env vars -------------------------------------------------------
# flash-attn kernels load on Ampere here. Use "native" on Blackwell.
export DIFFUSERS_ATTN_BACKEND="${DIFFUSERS_ATTN_BACKEND:-flash}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ---- Paths (override repo defaults to point at on-disk assets) ---------------
# Diffusers-format Wan 2.2 A14B (bootstrap reads the model via the WAN_MODEL env).
export WAN_MODEL="$PWD/models/wan2.2"

# Test data dir: raw_faces/ + targets/ symlinks (100 faces, 10 000 targets).
DATA_DIR="/home/s2710099/data/wan22-test"
# Precomputed cache with a control_path manifest (silhouette/latents/prompts).
# Already complete -> bootstrap skips precompute and goes straight to training.
CACHE_DIR="/home/s2710099/cache/wan22-test"
# Scratch output (we kill before any checkpoint is written anyway).
OUTPUT_DIR="/home/s2710099/checkpoints/wan22-test-out"

# ---- Run (no wandb; we only want to see it start training) ------------------
python -u bootstrap.py \
    --data_dir "$DATA_DIR" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$OUTPUT_DIR"
