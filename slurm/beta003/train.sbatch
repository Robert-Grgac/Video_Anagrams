#!/bin/bash
#SBATCH --job-name=wan-beta3-train
#SBATCH --partition=dmb
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/s2710099/logs/wan-beta3-train-%j.out
#SBATCH --error=/home/s2710099/logs/wan-beta3-train-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams


set -euo pipefail

# Activate conda environment (interpreter: ~/.conda/envs/wan22/bin/python)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${WAN22_ENV:-wan22}

# Wan-beta paths — flat top-level convention, one subdir per category
export WAN_BETA_CACHE=${WAN_BETA_CACHE:-$HOME/cache/wan-beta}
export WAN_BETA_CKPT=${WAN_BETA_CKPT:-$HOME/checkpoints/wan-beta}
export WAN_BETA_OUT=${WAN_BETA_OUT:-$HOME/outputs/wan-beta}

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

# Local fast storage for wandb + service stability fixes
export TMPDIR=/tmp/$USER
mkdir -p $TMPDIR
export WANDB_DIR=$TMPDIR/wandb
mkdir -p $WANDB_DIR
export WANDB__REQUIRE_SERVICE=false
export WANDB__SERVICE_WAIT=600
export WANDB_MODE=${WANDB_MODE:-online}
rm -rf /tmp/wandb-service-* 2>/dev/null || true

# Make sure output dirs exist (training script writes here)
mkdir -p $WAN_BETA_CKPT $WAN_BETA_OUT

# beta-003: warm-start from beta-001_final.safetensors. Same file is reused as
# the high-CN for the end-of-run dual-CN smoke (the regression check).
export WARM_START=${WARM_START:-$WAN_BETA_CKPT/beta-001_final.safetensors}
[ -f "$WARM_START" ] || { echo "ERROR: warm-start checkpoint not found: $WARM_START" >&2; exit 1; }

python -m training.beta003.train \
    --cache_dir       $WAN_BETA_CACHE \
    --base_model_path $WAN_MODEL \
    --controlnet_config_repo $HED_CONFIG \
    --warm_start_checkpoint $WARM_START \
    --dual_smoke_high_checkpoint $WARM_START \
    --output_dir      $WAN_BETA_CKPT \
    --inference_output_dir $WAN_BETA_OUT \
    --card_path       training_cards/beta003/beta-003.md \
    --wandb_project   wan-controlnet-beta \
    --run_name        beta-003 \
    --num_frames 9 --height 512 --width 512 \
    --lr 1e-4 --weight_decay 1e-4 --grad_clip 1.0 \
    --num_epochs 1 \
    --max_steps 4000 \
    --gradient_accumulation_steps 2 \
    --ema_decay 0.999 \
    --ema_update_after_step 100 \
    --num_workers 1 \
    --checkpoint_every 1000
