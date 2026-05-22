#!/bin/bash
#SBATCH --job-name=wan-beta9-smoke-l40s
#SBATCH --partition=dmb
#SBATCH --nodelist=hpc-node07, hpc-node07, ctit092
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/s2710099/logs/wan-beta9-smoke-l40s-%j.out
#SBATCH --error=/home/s2710099/logs/wan-beta9-smoke-l40s-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams

# L40S variant of the beta-009 smoke. Differs from smoke_test.sbatch in two
# things:
#   1. transformer_2 stays on CPU during training (via --cpu_offload_inference).
#   2. Before the inference call, pipe.enable_model_cpu_offload() swaps the
#      two experts in/out so only one is GPU-resident at a time, and the
#      ControlNet is re-pinned to GPU (otherwise CPU/GPU dtype mismatch in
#      its first Conv3D under model_cpu_offload).
# Result: peak GPU memory tracks the single active expert (~28 GB) plus
# activations, fitting L40S (48 GB) / A40 (45 GB) comfortably.

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${WAN22_ENV:-wan22}

export WAN_BETA_DATA=${WAN_BETA_DATA:-$HOME/data/wan-beta}
export WAN_BETA_CACHE=${WAN_BETA_CACHE:-$HOME/cache/wan-beta}
export WAN_BETA_CKPT=${WAN_BETA_CKPT:-$HOME/checkpoints/wan-beta}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1

# Flush Python stdout/stderr immediately so SLURM .out/.err update in real time.
export PYTHONUNBUFFERED=1

# L40S (compute_cap 8.9) supports flash-attn; use the standard wan22 env.
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
export WANDB_MODE=${WANDB_MODE:-offline}

export FACE_TARGET_SUBDIR=${FACE_TARGET_SUBDIR:-raw_face}
export CONTROL_SUBDIR=${CONTROL_SUBDIR:-silhouette}
export FACE_MASK_SUBDIR=${FACE_MASK_SUBDIR:-silhouette}
export LAMBDA_FACE=${LAMBDA_FACE:-5.0}

echo "[smoke] FACE_TARGET_SUBDIR=$FACE_TARGET_SUBDIR"
echo "[smoke] CONTROL_SUBDIR=$CONTROL_SUBDIR"
echo "[smoke] FACE_MASK_SUBDIR=$FACE_MASK_SUBDIR"
echo "[smoke] LAMBDA_FACE=$LAMBDA_FACE"

python -m training.beta009.smoke_test \
    --cache_dir       $WAN_BETA_CACHE \
    --targets_dir     $WAN_BETA_DATA/targets \
    --base_model_path $WAN_MODEL \
    --controlnet_config_repo $HED_CONFIG \
    --smoke_output_dir /tmp/$USER/wan_beta9_smoke \
    --control_subdir  $CONTROL_SUBDIR \
    --face_mask_subdir $FACE_MASK_SUBDIR \
    --face_target_subdir $FACE_TARGET_SUBDIR \
    --lambda_face     $LAMBDA_FACE \
    --num_frames 9 --height 512 --width 512 \
    --micro_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --effective_steps 2 \
    --ema_decay 0.99 \
    --ema_update_after_step 1 \
    --use_self_distillation \
    --lambda_consistency 0.5 \
    --controlnet_stride 3 \
    --eval_size 100 \
    --inference_steps 5 \
    --cpu_offload_inference \
    --mem_tripwire_gb 43.0
