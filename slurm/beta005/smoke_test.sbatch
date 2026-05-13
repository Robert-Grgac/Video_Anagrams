#!/bin/bash
#SBATCH --job-name=wan-beta5-smoke
#SBATCH --partition=dmb
#SBATCH --nodelist=hpc-node31
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=/home/s2710099/logs/wan-beta5-smoke-%j.out
#SBATCH --error=/home/s2710099/logs/wan-beta5-smoke-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${WAN22_ENV:-wan22-bw}

export WAN_BETA_CACHE=${WAN_BETA_CACHE:-$HOME/cache/wan-beta}
export WAN_BETA_CKPT=${WAN_BETA_CKPT:-$HOME/checkpoints/wan-beta}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1

# Same Blackwell attention pinning as the real run.
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
export WANDB_MODE=${WANDB_MODE:-offline}

# Smoke covers exactly the batching path used by train_beta5. Defaults inside
# the smoke script (micro_batch=8, accum=2, eff_steps=3) keep this under ~3
# minutes; the real run uses accum=4, but loop logic is identical.
python -m training.beta005.smoke_test \
    --cache_dir       $WAN_BETA_CACHE \
    --base_model_path $WAN_MODEL \
    --controlnet_config_repo $HED_CONFIG \
    --num_frames 9 --height 512 --width 512 \
    --micro_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --effective_steps 3 \
    --mem_tripwire_gb 90.0
