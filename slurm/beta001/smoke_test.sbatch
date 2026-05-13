#!/bin/bash
#SBATCH --job-name=wan-beta-smoke
#SBATCH --partition=dmb
#SBATCH --nodelist=ctit092
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=/home/s2710099/logs/wan-beta-smoke-%j.out
#SBATCH --error=/home/s2710099/logs/wan-beta-smoke-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams

set -euo pipefail

# Activate conda environment (interpreter: ~/.conda/envs/wan22/bin/python)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate wan22

# Wan-beta paths — flat top-level convention, one subdir per category
export WAN_BETA_DATA=${WAN_BETA_DATA:-$HOME/data/wan-beta}
export WAN_BETA_CACHE=${WAN_BETA_CACHE:-$HOME/cache/wan-beta}
export WAN_BETA_CKPT=${WAN_BETA_CKPT:-$HOME/checkpoints/wan-beta}
export WAN_BETA_OUT=${WAN_BETA_OUT:-$HOME/outputs/wan-beta}

# HF cache (default location, populated by head node) and offline mode for compute nodes
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1

# Diffusers attention backend
export DIFFUSERS_ATTN_BACKEND="flash"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Reduce CUDA allocator fragmentation; helps near-VRAM-cap on A40
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Resolve cached snapshot dirs. Diffusers' from_pretrained() calls model_info()
# (network) for sharded models even with HF_HUB_OFFLINE=1 when given a repo id;
# passing a local snapshot path bypasses that.
WAN_CACHE=$HOME/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers
HED_CACHE=$HOME/.cache/huggingface/hub/models--TheDenk--wan2.2-t2v-a14b-controlnet-hed-v1
export WAN_MODEL=${WAN_MODEL:-$(ls -d $WAN_CACHE/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')}
export HED_CONFIG=${HED_CONFIG:-$(ls -d $HED_CACHE/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')}
[ -z "$WAN_MODEL" ] && { echo "ERROR: Wan model not cached at $WAN_CACHE" >&2; exit 1; }
[ -z "$HED_CONFIG" ] && { echo "ERROR: HED config not cached at $HED_CACHE" >&2; exit 1; }
echo "[paths] WAN_MODEL=$WAN_MODEL"
echo "[paths] HED_CONFIG=$HED_CONFIG"

# Local fast storage for wandb (smoke runs offline by default)
export TMPDIR=/tmp/$USER
mkdir -p $TMPDIR
export WANDB_DIR=$TMPDIR/wandb
mkdir -p $WANDB_DIR
export WANDB__REQUIRE_SERVICE=false
export WANDB__SERVICE_WAIT=600
export WANDB_MODE=${WANDB_MODE:-offline}
rm -rf /tmp/wandb-service-* 2>/dev/null || true

mkdir -p $WAN_BETA_CACHE

python -m training.beta001.smoke_test \
    --base_model_path $WAN_MODEL \
    --controlnet_config_repo $HED_CONFIG \
    --input_faces_dir $WAN_BETA_DATA/input_faces \
    --targets_dir $WAN_BETA_DATA/targets \
    --cache_dir $WAN_BETA_CACHE \
    --card_run_id beta-001
