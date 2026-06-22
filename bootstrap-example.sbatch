#!/bin/bash
# =============================================================================
# bootstrap-example.sbatch
#
# Example Slurm submit script for `python bootstrap.py` — checks assets,
# precomputes the training cache (if missing), and trains the ControlNet.
# Final output: models/controlnet/controlnet.safetensors.
#
# This is a STARTING POINT, not a universal default. You will almost
# certainly need to adapt the SBATCH directives below for your cluster
# (partition / queue name, account, GPU type, time and memory limits).
#
# Submit from the repo root:
#     sbatch bootstrap-example.sbatch
#
# Optional wandb logging — set both env vars before submitting:
#     WANDB_PROJECT=my-project WANDB_RUN_NAME=run-001 sbatch bootstrap-example.sbatch
# =============================================================================
#SBATCH --job-name=video-anagrams-train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
# ---- Cluster-specific (uncomment and edit as needed) ------------------------
##SBATCH --partition=YOUR_PARTITION
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --constraint=YOUR_GPU_CONSTRAINT
# =============================================================================

set -euo pipefail

# ---- Conda env --------------------------------------------------------------
# Assumes `conda env create -f environment.yml` has been run once. Change the
# env name here if you used `-n <other-name>` during creation.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wan22

# ---- Runtime env vars -------------------------------------------------------
# flash-attn kernels load on Ampere / Hopper. Switch to "native" if you are
# on Blackwell (SM 12.x) or any GPU where flash-attn import works but the
# kernels do not.
export DIFFUSERS_ATTN_BACKEND="${DIFFUSERS_ATTN_BACKEND:-flash}"

# Skip Hub round trips if your local snapshots are complete.
export HF_HUB_OFFLINE=1

# Mitigates CUDA fragmentation on long-running training jobs.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ---- Optional wandb pair ----------------------------------------------------
# bootstrap.py errors if exactly one of (WANDB_PROJECT, WANDB_RUN_NAME) is set,
# so we forward both or neither.
WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT:-}" && -n "${WANDB_RUN_NAME:-}" ]]; then
    WANDB_ARGS+=(--wandb_project "$WANDB_PROJECT"
                 --wandb_run_name "$WANDB_RUN_NAME")
fi

# ---- Run --------------------------------------------------------------------
# `python -u` keeps stdout/stderr unbuffered so the Slurm log reflects
# progress live (matters for the long training run).
python -u bootstrap.py "${WANDB_ARGS[@]}"
