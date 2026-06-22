#!/bin/bash
# =============================================================================
# inference-example.sbatch
#
# Example Slurm submit script for `python inference.py` — generates one
# video from a face image + prompt. Requires that bootstrap.py has already
# trained the ControlNet (models/controlnet/controlnet.safetensors).
#
# STARTING POINT only. Adapt the SBATCH directives for your cluster.
#
# Submit from the repo root with your image and prompt:
#     IMAGE_PATH=data/raw_faces/face_0.png \
#     PROMPT="snowy mountain, oil painting" \
#     sbatch inference-example.sbatch
#
# Or edit the IMAGE_PATH / PROMPT defaults below and just `sbatch` it.
# =============================================================================
#SBATCH --job-name=video-anagrams-infer
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
# ---- Cluster-specific (uncomment and edit as needed) ------------------------
##SBATCH --partition=YOUR_PARTITION
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --constraint=YOUR_GPU_CONSTRAINT
# =============================================================================

set -euo pipefail

# ---- Conda env --------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wan22

# ---- Runtime env vars -------------------------------------------------------
export DIFFUSERS_ATTN_BACKEND="${DIFFUSERS_ATTN_BACKEND:-flash}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ---- Inputs -----------------------------------------------------------------
# Override on the command line or edit these defaults.
IMAGE_PATH="${IMAGE_PATH:-data/raw_faces/face_0.png}"
PROMPT="${PROMPT:-snowy mountain, oil painting}"

# Default output is ./output.mp4. With many concurrent jobs that would clash,
# so we suffix with $SLURM_JOB_ID. Edit OUTPUT_PATH to taste, or delete the
# --output flag below to revert to the default.
OUTPUT_PATH="${OUTPUT_PATH:-output-${SLURM_JOB_ID}.mp4}"

# ---- Run --------------------------------------------------------------------
python -u inference.py "$IMAGE_PATH" "$PROMPT" --output "$OUTPUT_PATH"
