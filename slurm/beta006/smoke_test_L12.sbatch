#!/bin/bash
#SBATCH --job-name=wan-beta6-L12-smoke
#SBATCH --partition=dmb
#SBATCH --nodelist=ctit092
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=/home/s2710099/logs/wan-beta6-L12-smoke-%j.out
#SBATCH --error=/home/s2710099/logs/wan-beta6-L12-smoke-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams

set -euo pipefail

# Worst-case memory check for the ablation set: L=12 + stride=2 (every CN
# residual is used) + self-distillation enabled (extra no_grad transformer
# forward per micro-step). If this fits on A40 (45 GB), the rest of the
# ablations fit too.

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${WAN22_ENV:-wan22}

export WAN_BETA_CACHE=${WAN_BETA_CACHE:-$HOME/cache/wan-beta}
export WAN_BETA_CKPT=${WAN_BETA_CKPT:-$HOME/checkpoints/wan-beta}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1

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

# L=12 + stride=2 (uses all 12 residuals at blocks 0,2,..,22) + SD on +
# the SD-suitable EMA recipe (decay=0.99, update_after=10).
python -m training.beta006.smoke_test \
    --cache_dir       $WAN_BETA_CACHE \
    --base_model_path $WAN_MODEL \
    --controlnet_config_repo $HED_CONFIG \
    --num_frames 9 --height 512 --width 512 \
    --micro_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --effective_steps 2 \
    --num_cn_layers 12 \
    --controlnet_stride 2 \
    --use_self_distillation \
    --lambda_consistency 0.5 \
    --ema_decay 0.99 \
    --ema_update_after_step 10 \
    --mem_tripwire_gb 43.0
