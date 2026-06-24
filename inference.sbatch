#!/bin/bash
# =============================================================================
# inference.sbatch  (wan22-test full run)
#
# Runs `python inference.py` end-to-end: builds the deterministic FlowMatch
# inversion trajectory for one unseen face, then runs the PTD + ControlNet
# pipeline for the full 100 inference steps and writes one mp4. Let this one
# finish — it validates the whole front-to-back path on the new env.
#
# inference.py reads its paths from the repo (models/wan2.2, models/hed_config,
# models/controlnet/controlnet.safetensors) — those are symlinked to the
# on-disk diffusers model and the trained checkpoint, so no flags are needed.
#
# Submit from the repo root:
#     sbatch inference.sbatch
# Override the face / prompt on the command line:
#     IMAGE_PATH=/home/s2710099/data/unseen_faces/face_3.png \
#     PROMPT="a coral reef teeming with fish" sbatch inference.sbatch
# =============================================================================
#SBATCH --job-name=wan22test-infer
#SBATCH --partition=dmb
# Suitable nodes: hpc-node07/09 (L40 48GB), ctit092 (A40 45GB). Exclude
# ctit084/085 (RTX6000 24GB) and hpc-node31 (Blackwell -> needs wan22-bw env).
#SBATCH --exclude=ctit084,ctit085,hpc-node31
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/home/s2710099/logs/wan22test-infer-%j.out
#SBATCH --error=/home/s2710099/logs/wan22test-infer-%j.err
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

# ---- Inputs -----------------------------------------------------------------
IMAGE_PATH="${IMAGE_PATH:-/home/s2710099/data/unseen_faces/face_0.png}"
PROMPT="${PROMPT:-a dense autumn forest canopy with golden leaves, dappled sunlight, oil painting}"

OUTDIR="/home/s2710099/outputs/wan22-test"
mkdir -p "$OUTDIR"
OUTPUT_PATH="${OUTPUT_PATH:-$OUTDIR/output-${SLURM_JOB_ID}.mp4}"

# ---- Run --------------------------------------------------------------------
python -u inference.py "$IMAGE_PATH" "$PROMPT" --output "$OUTPUT_PATH"
