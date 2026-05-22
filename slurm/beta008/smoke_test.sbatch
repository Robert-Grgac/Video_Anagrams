#!/bin/bash
#SBATCH --job-name=wan-beta7-smoke
#SBATCH --partition=dmb
#SBATCH --nodelist=hpc-node31
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=96G
#SBATCH --output=/home/s2710099/logs/wan-beta7-smoke-%j.out
#SBATCH --error=/home/s2710099/logs/wan-beta7-smoke-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${WAN22_ENV:-wan22-bw}

export WAN_BETA_DATA=${WAN_BETA_DATA:-$HOME/data/wan-beta}
export WAN_BETA_CACHE=${WAN_BETA_CACHE:-$HOME/cache/wan-beta}
export WAN_BETA_CKPT=${WAN_BETA_CKPT:-$HOME/checkpoints/wan-beta}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1

# Blackwell: force native attention (matches train_beta5/7 sbatch).
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

# Smoke covers what train_beta7 adds on top of train_beta5/6, plus the rerun
# bug-fixes / additions:
#   - Accelerate manual-gate accumulate path
#   - Eval pipeline construction with text_encoder=None
#   - Face/prompt-stratified split helper (Latin pairing — periodic ⊂ train, eval disjoint)
#   - Dynamic controlnet_guidance_end (CN confined to high-noise expert via σ schedule)
#   - One inference forward pass + MSE + SSIM (pytorch-msssim) against the target JPG
# Defaults: 2 eff-steps × accum=8 (16 micro-steps training) + 1 inference at 5 steps.
# Should finish in ~3-5 minutes on Blackwell.
python -m training.beta007.smoke_test \
    --cache_dir       $WAN_BETA_CACHE \
    --targets_dir     $WAN_BETA_DATA/targets \
    --base_model_path $WAN_MODEL \
    --controlnet_config_repo $HED_CONFIG \
    --smoke_output_dir /tmp/$USER/wan_beta7_smoke \
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
    --mem_tripwire_gb 90.0
