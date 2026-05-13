#!/bin/bash
#SBATCH --job-name=wan-beta6-train
#SBATCH --partition=dmb
#SBATCH --nodelist=ctit092
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/s2710099/logs/wan-beta6-train-%j.out
#SBATCH --error=/home/s2710099/logs/wan-beta6-train-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams


set -euo pipefail

# A40 node uses the standard wan22 env (flash-attn works there).
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${WAN22_ENV:-wan22}

export WAN_BETA_CACHE=${WAN_BETA_CACHE:-$HOME/cache/wan-beta}
export WAN_BETA_CKPT=${WAN_BETA_CKPT:-$HOME/checkpoints/wan-beta}
export WAN_BETA_OUT=${WAN_BETA_OUT:-$HOME/outputs/wan-beta}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1

# A40 (compute_cap 8.6) supports flash-attn; pin explicitly so the dispatcher
# doesn't surprise us. The Blackwell guard inside the training script is a
# no-op here because compute_cap < 12.
export DIFFUSERS_ATTN_BACKEND="flash"
echo "[attn] DIFFUSERS_ATTN_BACKEND=$DIFFUSERS_ATTN_BACKEND"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

WAN_CACHE=$HOME/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers
HED_CACHE=$HOME/.cache/huggingface/hub/models--TheDenk--wan2.2-t2v-a14b-controlnet-hed-v1
export WAN_MODEL=${WAN_MODEL:-$(ls -d $WAN_CACHE/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')}
export HED_CONFIG=${HED_CONFIG:-$(ls -d $HED_CACHE/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')}
[ -z "$WAN_MODEL" ] && { echo "ERROR: Wan model not cached at $WAN_CACHE" >&2; exit 1; }
[ -z "$HED_CONFIG" ] && { echo "ERROR: HED config not cached at $HED_CACHE" >&2; exit 1; }
echo "[paths] WAN_MODEL=$WAN_MODEL"
echo "[paths] HED_CONFIG=$HED_CONFIG"

export TMPDIR=/tmp/$USER
mkdir -p $TMPDIR
export WANDB_DIR=$TMPDIR/wandb
mkdir -p $WANDB_DIR
export WANDB__REQUIRE_SERVICE=false
export WANDB__SERVICE_WAIT=600
export WANDB_MODE=${WANDB_MODE:-online}
rm -rf /tmp/wandb-service-* 2>/dev/null || true

mkdir -p $WAN_BETA_CKPT $WAN_BETA_OUT

# Optional: pair the end-of-run dual-CN smoke with beta-005's EMA if it's
# already on disk. Falls back to beta-001 (legacy) if not. Both are valid
# high-CNs for a sanity-check video; the eventual canonical pairing is
# beta-005 ⊕ beta-006 (modern recipe end-to-end).
DEFAULT_DUAL_HIGH=$WAN_BETA_CKPT/beta-005_final.safetensors
[ -f "$DEFAULT_DUAL_HIGH" ] || DEFAULT_DUAL_HIGH=$WAN_BETA_CKPT/beta-001_final.safetensors
export DUAL_SMOKE_HIGH=${DUAL_SMOKE_HIGH:-$DEFAULT_DUAL_HIGH}
[ -f "$DUAL_SMOKE_HIGH" ] || { echo "ERROR: dual-smoke high checkpoint not found: $DUAL_SMOKE_HIGH" >&2; exit 1; }
echo "[dual-smoke] high-CN = $DUAL_SMOKE_HIGH"

# beta-006: cold-start low-noise expert, pure accumulation on A40.
# eff_batch = micro_batch * accum = 1 * 32 = 32; max_steps = 125 ⇒ 4000 micro-steps.
python -m training.beta006.train \
    --cache_dir       $WAN_BETA_CACHE \
    --base_model_path $WAN_MODEL \
    --controlnet_config_repo $HED_CONFIG \
    --dual_smoke_high_checkpoint $DUAL_SMOKE_HIGH \
    --output_dir      $WAN_BETA_CKPT \
    --inference_output_dir $WAN_BETA_OUT \
    --card_path       training_cards/beta006/beta-006.md \
    --wandb_project   wan-controlnet-beta \
    --run_name        beta-006 \
    --num_frames 9 --height 512 --width 512 \
    --lr 1e-4 --weight_decay 1e-4 --grad_clip 1.0 \
    --num_epochs 1 \
    --max_steps 125 \
    --micro_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --ema_decay 0.999 \
    --ema_update_after_step 100 \
    --loss_ema_window 20 \
    --memory_tripwire_gb 43.0 \
    --num_workers 2 \
    --checkpoint_every 50
