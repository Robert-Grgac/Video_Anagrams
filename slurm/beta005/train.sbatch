#!/bin/bash
#SBATCH --job-name=wan-beta5-train
#SBATCH --partition=dmb
#SBATCH --nodelist=hpc-node31
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/s2710099/logs/wan-beta5-train-%j.out
#SBATCH --error=/home/s2710099/logs/wan-beta5-train-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams


set -euo pipefail

# Activate conda environment. Blackwell node uses wan22-bw (no flash-attn).
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${WAN22_ENV:-wan22-bw}

# Wan-beta paths
export WAN_BETA_CACHE=${WAN_BETA_CACHE:-$HOME/cache/wan-beta}
export WAN_BETA_CKPT=${WAN_BETA_CKPT:-$HOME/checkpoints/wan-beta}
export WAN_BETA_OUT=${WAN_BETA_OUT:-$HOME/outputs/wan-beta}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1

# Force NATIVE attention on Blackwell. The wan22-bw env can have flash_attn
# importable but its CUDA kernels don't load on SM 12.0 → flash_attn_func is
# None at runtime (this is what crashed beta-004 attempt 1). The env-var-only
# fallback in the legacy sbatch (`python -c 'import flash_attn'`) is fooled by
# this; we hard-pin native here and the training script also calls
# set_attention_backend("native") on the loaded models as belt-and-braces.
export DIFFUSERS_ATTN_BACKEND="native"
export WAN_FORCE_NATIVE_ATTN=1
echo "[attn] DIFFUSERS_ATTN_BACKEND=$DIFFUSERS_ATTN_BACKEND (forced native for Blackwell)"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
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

export TMPDIR=/tmp/$USER
mkdir -p $TMPDIR
export WANDB_DIR=$TMPDIR/wandb
mkdir -p $WANDB_DIR
export WANDB__REQUIRE_SERVICE=false
export WANDB__SERVICE_WAIT=600
export WANDB_MODE=${WANDB_MODE:-online}
rm -rf /tmp/wandb-service-* 2>/dev/null || true

mkdir -p $WAN_BETA_CKPT $WAN_BETA_OUT

# beta-005: cold-start high-noise expert with batching on Blackwell.
# eff_batch = micro_batch * accum = 8 * 4 = 32; max_steps = 125 effective
# optimizer steps ⇒ 4000 micro-steps total (mentor's "4000/32" budget).
# Memory tripwire 90 GB on the 96 GB Blackwell.
python -m training.beta005.train \
    --cache_dir       $WAN_BETA_CACHE \
    --base_model_path $WAN_MODEL \
    --controlnet_config_repo $HED_CONFIG \
    --output_dir      $WAN_BETA_CKPT \
    --inference_output_dir $WAN_BETA_OUT \
    --card_path       training_cards/beta005/beta-005.md \
    --wandb_project   wan-controlnet-beta \
    --run_name        beta-005 \
    --num_frames 9 --height 512 --width 512 \
    --lr 1e-4 --weight_decay 1e-4 --grad_clip 1.0 \
    --num_epochs 1 \
    --max_steps 125 \
    --micro_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --ema_decay 0.999 \
    --ema_update_after_step 100 \
    --loss_ema_window 20 \
    --memory_tripwire_gb 90.0 \
    --num_workers 2 \
    --checkpoint_every 50
