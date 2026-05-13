#!/bin/bash
#SBATCH --job-name=wan-beta-precompute
#SBATCH --partition=dmb
#SBATCH --nodelist=hpc-node31
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/s2710099/logs/wan-beta-precompute-%j.out
#SBATCH --error=/home/s2710099/logs/wan-beta-precompute-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams


set -euo pipefail

# Activate conda environment.
# hpc-node31 is RTX PRO 6000 Blackwell (SM 12.0). flash-attn 2.8.3 imports cleanly
# but flash_attn_func is None at runtime on SM 12.0, so we use the Blackwell-only
# env (wan22-bw, no flash-attn) and force native SDPA below.
source $(conda info --base)/etc/profile.d/conda.sh
conda activate wan22-bw

# Wan-beta paths — flat top-level convention, one subdir per category
export WAN_BETA_DATA=${WAN_BETA_DATA:-$HOME/data/wan-beta}
export WAN_BETA_CACHE=${WAN_BETA_CACHE:-$HOME/cache/wan-beta}

# HF cache (default location, populated by head node) and offline mode for compute nodes
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1

# Diffusers attention backend — native SDPA on Blackwell (flash-attn not usable on SM 12.0).
export DIFFUSERS_ATTN_BACKEND="native"

# Deterministic-friendly cuBLAS workspace (matches inference scripts)
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Resolve cached snapshot dir. Diffusers' from_pretrained() calls model_info()
# (network) for sharded models even with HF_HUB_OFFLINE=1 when given a repo id;
# passing a local snapshot path bypasses that.
WAN_CACHE=$HOME/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers
export WAN_MODEL=${WAN_MODEL:-$(ls -d $WAN_CACHE/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')}
[ -z "$WAN_MODEL" ] && { echo "ERROR: Wan model not cached at $WAN_CACHE" >&2; exit 1; }
echo "[paths] WAN_MODEL=$WAN_MODEL"

mkdir -p $WAN_BETA_CACHE

python -m training.precompute_beta \
    --input_faces_dir $WAN_BETA_DATA/input_faces \
    --targets_dir    $WAN_BETA_DATA/targets \
    --output_dir     $WAN_BETA_CACHE \
    --base_model_path $WAN_MODEL \
    --height 512 --width 512 --num_frames 9 \
    --max_seq_len 226
