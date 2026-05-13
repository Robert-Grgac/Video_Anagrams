#!/bin/bash
#SBATCH --job-name=wan-beta7-train
#SBATCH --partition=dmb
#SBATCH --nodelist=hpc-node31
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --output=/home/s2710099/logs/wan-beta7-train-%j.out
#SBATCH --error=/home/s2710099/logs/wan-beta7-train-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams


set -euo pipefail

# Activate conda environment. Blackwell node uses wan22-bw (no flash-attn).
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${WAN22_ENV:-wan22-bw}

# Wan-beta paths
export WAN_BETA_DATA=${WAN_BETA_DATA:-$HOME/data/wan-beta}
export WAN_BETA_CACHE=${WAN_BETA_CACHE:-$HOME/cache/wan-beta}
export WAN_BETA_CKPT=${WAN_BETA_CKPT:-$HOME/checkpoints/wan-beta}
export WAN_BETA_OUT=${WAN_BETA_OUT:-$HOME/outputs/wan-beta}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1

# Force NATIVE attention on Blackwell. The wan22-bw env can have flash_attn
# importable but its CUDA kernels don't load on SM 12.0 → flash_attn_func is
# None at runtime. Hard-pin native here; the training script also calls
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

# beta-007_v3: same recipe as beta-007_v2 but trained for 5 epochs with the
# learning rate multiplied by 0.9 after each completed epoch (10% decay/epoch).
# No training card; wandb run + saved videos are the only artifacts.
python -m training.beta007.train \
    --cache_dir       $WAN_BETA_CACHE \
    --targets_dir     $WAN_BETA_DATA/targets \
    --base_model_path $WAN_MODEL \
    --controlnet_config_repo $HED_CONFIG \
    --output_dir      $WAN_BETA_CKPT \
    --inference_output_dir $WAN_BETA_OUT \
    --wandb_project   wan-controlnet-beta \
    --run_name        beta-007_v3 \
    --num_frames 9 --height 512 --width 512 \
    --lr 5e-5 --lr_decay_per_epoch 0.9 \
    --weight_decay 1e-4 --grad_clip 1.0 \
    --num_epochs 5 \
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
    --checkpoint_every 50
