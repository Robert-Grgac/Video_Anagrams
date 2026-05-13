#!/bin/bash
#SBATCH --job-name=wan-beta3-smoke
#SBATCH --partition=dmb
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=/home/s2710099/logs/wan-beta3-smoke-%j.out
#SBATCH --error=/home/s2710099/logs/wan-beta3-smoke-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams

set -euo pipefail

# Activate conda environment (interpreter: ~/.conda/envs/wan22/bin/python)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${WAN22_ENV:-wan22}

# Wan-beta paths — flat top-level convention, one subdir per category
export WAN_BETA_CACHE=${WAN_BETA_CACHE:-$HOME/cache/wan-beta}
export WAN_BETA_CKPT=${WAN_BETA_CKPT:-$HOME/checkpoints/wan-beta}

# HF cache (default location, populated by head node) and offline mode for compute nodes
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1

# Diffusers attention backend. If unset, auto-detect: use flash when flash-attn
# imports cleanly (A40 / wan22 env), else fall back to native SDPA (Blackwell /
# wan22-bw env where flash-attn is uninstalled). Without this, an explicit
# DIFFUSERS_ATTN_BACKEND=flash causes a None-call TypeError when flash-attn is
# missing, since diffusers' dispatcher doesn't auto-fall-back at runtime.
if [ -z "${DIFFUSERS_ATTN_BACKEND:-}" ]; then
    if python -c "import flash_attn" 2>/dev/null; then
        export DIFFUSERS_ATTN_BACKEND="flash"
    else
        export DIFFUSERS_ATTN_BACKEND="native"
    fi
fi
echo "[attn] DIFFUSERS_ATTN_BACKEND=$DIFFUSERS_ATTN_BACKEND"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Reduce CUDA allocator fragmentation; helps near-VRAM-cap on A40
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Resolve cached snapshot dirs.
WAN_CACHE=$HOME/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers
HED_CACHE=$HOME/.cache/huggingface/hub/models--TheDenk--wan2.2-t2v-a14b-controlnet-hed-v1
export WAN_MODEL=${WAN_MODEL:-$(ls -d $WAN_CACHE/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')}
export HED_CONFIG=${HED_CONFIG:-$(ls -d $HED_CACHE/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')}
[ -z "$WAN_MODEL" ] && { echo "ERROR: Wan model not cached at $WAN_CACHE" >&2; exit 1; }
[ -z "$HED_CONFIG" ] && { echo "ERROR: HED config not cached at $HED_CACHE" >&2; exit 1; }
echo "[paths] WAN_MODEL=$WAN_MODEL"
echo "[paths] HED_CONFIG=$HED_CONFIG"

# Smoke runs without wandb; keep WANDB_MODE explicit anyway for safety.
export TMPDIR=/tmp/$USER
mkdir -p $TMPDIR
export WANDB_MODE=${WANDB_MODE:-offline}

# Smoke covers BOTH init paths in one job; the warm-start path needs beta-001.
export WARM_START=${WARM_START:-$WAN_BETA_CKPT/beta-001_final.safetensors}
[ -f "$WARM_START" ] || { echo "ERROR: warm-start checkpoint not found: $WARM_START" >&2; exit 1; }

python -m training.beta003.smoke_test \
    --cache_dir       $WAN_BETA_CACHE \
    --base_model_path $WAN_MODEL \
    --controlnet_config_repo $HED_CONFIG \
    --warm_start_checkpoint $WARM_START \
    --num_frames 9 --height 512 --width 512
