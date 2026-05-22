#!/bin/bash
#SBATCH --job-name=wan-beta9-sil
#SBATCH --partition=dmb
#SBATCH --nodelist=hpc-node31
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --output=/home/s2710099/logs/wan-beta9-sil-%j.out
#SBATCH --error=/home/s2710099/logs/wan-beta9-sil-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams

# beta-009 with --face_target_subdir=silhouette: additive face-region loss
#   loss = MSE(v_pred, v_target) + λ_face·face_region_MSE(v_pred, noise − z_face)
# where z_face is the VAE-encoded SILHOUETTE drawing (shape only, no identity).
# Silhouette also stays as the CN input and as the mask source — only the
# additive-loss target differs from the raw_face variant.
#
# Override RUN_NAME / LAMBDA_FACE on the sbatch command line, e.g.:
#   RUN_NAME=beta-009_silhouette sbatch slurm/beta009/train_silhouette.sbatch
#   RUN_NAME=beta-009_silhouette_l10 LAMBDA_FACE=10 sbatch slurm/beta009/train_silhouette.sbatch

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${WAN22_ENV:-wan22-bw}

export WAN_BETA_DATA=${WAN_BETA_DATA:-$HOME/data/wan-beta}
export WAN_BETA_CACHE=${WAN_BETA_CACHE:-$HOME/cache/wan-beta}
export WAN_BETA_CKPT=${WAN_BETA_CKPT:-$HOME/checkpoints/wan-beta}
export WAN_BETA_OUT=${WAN_BETA_OUT:-$HOME/outputs/wan-beta}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1

# Flush Python stdout/stderr immediately so SLURM .out/.err update in real time.
export PYTHONUNBUFFERED=1

export DIFFUSERS_ATTN_BACKEND="native"
export WAN_FORCE_NATIVE_ATTN=1
echo "[attn] DIFFUSERS_ATTN_BACKEND=$DIFFUSERS_ATTN_BACKEND (forced native for Blackwell)"
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

: "${RUN_NAME:?must be set, e.g. RUN_NAME=beta-009_silhouette}"
export FACE_TARGET_SUBDIR=${FACE_TARGET_SUBDIR:-silhouette}
export CONTROL_SUBDIR=${CONTROL_SUBDIR:-silhouette}
export FACE_MASK_SUBDIR=${FACE_MASK_SUBDIR:-silhouette}
export LAMBDA_FACE=${LAMBDA_FACE:-5.0}
export NUM_CN_LAYERS=${NUM_CN_LAYERS:-}

EXTRA_ARGS=()
if [ -n "$NUM_CN_LAYERS" ]; then
    EXTRA_ARGS+=(--num_cn_layers "$NUM_CN_LAYERS")
fi

echo "[run] RUN_NAME=$RUN_NAME"
echo "[run] FACE_TARGET_SUBDIR=$FACE_TARGET_SUBDIR (silhouette drawing VAE-encoded as z_face)"
echo "[run] CONTROL_SUBDIR=$CONTROL_SUBDIR"
echo "[run] FACE_MASK_SUBDIR=$FACE_MASK_SUBDIR"
echo "[run] LAMBDA_FACE=$LAMBDA_FACE"

python -m training.beta009.train \
    --cache_dir       $WAN_BETA_CACHE \
    --control_subdir  $CONTROL_SUBDIR \
    --face_mask_subdir $FACE_MASK_SUBDIR \
    --face_target_subdir $FACE_TARGET_SUBDIR \
    --lambda_face     $LAMBDA_FACE \
    --targets_dir     $WAN_BETA_DATA/targets \
    --base_model_path $WAN_MODEL \
    --controlnet_config_repo $HED_CONFIG \
    --output_dir      $WAN_BETA_CKPT \
    --inference_output_dir $WAN_BETA_OUT \
    --wandb_project   wan-controlnet-beta \
    --run_name        "$RUN_NAME" \
    --num_frames 9 --height 512 --width 512 \
    --lr 5e-5 \
    --weight_decay 1e-4 --grad_clip 1.0 \
    --num_epochs 1 \
    --micro_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --controlnet_stride 3 \
    --use_self_distillation \
    --lambda_consistency 0.5 \
    --ema_decay 0.99 \
    --ema_update_after_step 10 \
    --loss_ema_window 20 \
    --eval_size 100 \
    --periodic_eval_size 10 \
    --periodic_eval_every 10 \
    --inference_steps 50 \
    --inference_guidance_scale 5.0 \
    --inference_controlnet_weight 1.0 \
    --memory_tripwire_gb 90.0 \
    --num_workers 2 \
    --checkpoint_every 50 \
    "${EXTRA_ARGS[@]}"
