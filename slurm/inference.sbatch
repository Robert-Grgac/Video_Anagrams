#!/bin/bash
#SBATCH --job-name=wan-ptd-cn-infer
#SBATCH --partition=dmb
#SBATCH --nodelist=hpc-node09
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=/home/s2710099/logs/wan-ptd-cn-infer-%j.out
#SBATCH --error=/home/s2710099/logs/wan-ptd-cn-infer-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams

set -euo pipefail

# ============================================================================
# Local scratch (per HPC admin: do NOT do heavy I/O on /home)
# ----------------------------------------------------------------------------
# This cluster has no default $SLURM_TMPDIR; use /local/<jobid> per the
# HPC wiki recommendation.
# ============================================================================
ScratchDir="/local/${SLURM_JOBID}"

cleanup_node() {
    set +e
    # Sync wandb run files back to /home before wiping scratch (small files,
    # only written once at job end, so /home cost is bounded).
    if [ -d "$ScratchDir/wandb" ]; then
        mkdir -p "$HOME/logs/wandb"
        cp -a "$ScratchDir/wandb"/. "$HOME/logs/wandb"/ 2>/dev/null || true
    fi
    if [ -d "$ScratchDir" ]; then
        echo "[cleanup] removing $ScratchDir"
        rm -rf "$ScratchDir"
    fi
}
trap cleanup_node EXIT SIGINT SIGTERM

if [ -d "$ScratchDir" ]; then
    echo "[scratch] '$ScratchDir' already exists (stale?). Reusing."
else
    echo "[scratch] creating '$ScratchDir'"
    mkdir -p "$ScratchDir"
fi

# ============================================================================
# Conda env
# ============================================================================
source $(conda info --base)/etc/profile.d/conda.sh
conda activate wan22

# ============================================================================
# Resolve source paths on /home (read once during stage-in)
# ============================================================================
WAN_CACHE_HOME=$HOME/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers
HED_CACHE_HOME=$HOME/.cache/huggingface/hub/models--TheDenk--wan2.2-t2v-a14b-controlnet-hed-v1
WAN_SRC=$(ls -d "$WAN_CACHE_HOME"/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
HED_SRC=$(ls -d "$HED_CACHE_HOME"/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
[ -z "$WAN_SRC" ] && { echo "ERROR: Wan model not cached at $WAN_CACHE_HOME" >&2; exit 1; }
[ -z "$HED_SRC" ] && { echo "ERROR: HED config not cached at $HED_CACHE_HOME" >&2; exit 1; }

BETA_CACHE_HOME=${WAN_BETA_CACHE_HOME:-$HOME/cache/wan-beta}
BETA_CKPT_HOME=${WAN_BETA_CKPT_HOME:-$HOME/checkpoints/wan-beta}

# Which checkpoint file to stage (overridable from the sbatch command line).
export CHECKPOINT_NAME=${CHECKPOINT_NAME:-beta-008_final.safetensors}
[ -f "$BETA_CKPT_HOME/$CHECKPOINT_NAME" ] \
    || { echo "ERROR: checkpoint $BETA_CKPT_HOME/$CHECKPOINT_NAME not found" >&2; exit 1; }

# ============================================================================
# Stage inputs to local scratch
# ----------------------------------------------------------------------------
# `cp -aL` follows symlinks (HF hub stores snapshots/<rev>/ as symlinks into
# blobs/<hash>) so the staged copies are real files, not pointers back to /home.
# ============================================================================
mkdir -p "$ScratchDir/wan_model" \
         "$ScratchDir/hed_config" \
         "$ScratchDir/wan-beta-cache" \
         "$ScratchDir/wan-beta-ckpt" \
         "$ScratchDir/hf_home" \
         "$ScratchDir/torch" \
         "$ScratchDir/wandb"

echo "[stage] copying Wan snapshot ($WAN_SRC) -> $ScratchDir/wan_model"
cp -aL "$WAN_SRC"/. "$ScratchDir/wan_model"/

echo "[stage] copying HED controlnet config ($HED_SRC) -> $ScratchDir/hed_config"
cp -aL "$HED_SRC"/. "$ScratchDir/hed_config"/

echo "[stage] copying beta cache ($BETA_CACHE_HOME) -> $ScratchDir/wan-beta-cache"
cp -aL "$BETA_CACHE_HOME"/. "$ScratchDir/wan-beta-cache"/

echo "[stage] copying checkpoint $CHECKPOINT_NAME -> $ScratchDir/wan-beta-ckpt"
cp -L "$BETA_CKPT_HOME/$CHECKPOINT_NAME" "$ScratchDir/wan-beta-ckpt/$CHECKPOINT_NAME"

echo "[stage] staged size:"
du -sh "$ScratchDir"/* 2>/dev/null || true

# ============================================================================
# Re-bind all heavy-I/O paths to scratch
# ============================================================================
export WAN_MODEL="$ScratchDir/wan_model"
export HED_CONFIG="$ScratchDir/hed_config"
export WAN_BETA_CACHE="$ScratchDir/wan-beta-cache"
export WAN_BETA_CKPT="$ScratchDir/wan-beta-ckpt"
export CHECKPOINT="$WAN_BETA_CKPT/$CHECKPOINT_NAME"

# HF / Torch / wandb caches on local disk — these are the worst NFS offenders
# (thousands of small files, frequent metadata ops).
export HF_HOME="$ScratchDir/hf_home"
export HF_HUB_CACHE="$ScratchDir/hf_home/hub"
export TRANSFORMERS_CACHE="$ScratchDir/hf_home"
export HF_HUB_OFFLINE=1
export TORCH_HOME="$ScratchDir/torch"
export WANDB_DIR="$ScratchDir/wandb"

# Keep current wandb mode; per project memory the cluster has no outbound
# internet so `offline` is normally safer — override with `WANDB_MODE=offline`
# on the sbatch command line if jobs hang in wandb.init().
export WANDB_MODE=${WANDB_MODE:-online}

# Diffusers attention backend
export DIFFUSERS_ATTN_BACKEND="flash"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[paths] WAN_MODEL=$WAN_MODEL"
echo "[paths] HED_CONFIG=$HED_CONFIG"
echo "[paths] CHECKPOINT=$CHECKPOINT"
echo "[paths] WAN_BETA_CACHE=$WAN_BETA_CACHE"
echo "[paths] HF_HOME=$HF_HOME  TORCH_HOME=$TORCH_HOME  WANDB_DIR=$WANDB_DIR"

# ============================================================================
# Tunables — override on the sbatch command line, e.g.:
#   PROJECT_NAME=ptd_cn_beta008_kp0p5_b0p4 sbatch slurm/inference.sbatch
# ============================================================================
export PROJECT_NAME=${PROJECT_NAME:-ptd_cn_default}
export CONTROLNET_STRIDE=${CONTROLNET_STRIDE:-3}
export CONTROLNET_WEIGHT=${CONTROLNET_WEIGHT:-1.0}
export INITIAL_BLENDING_COEFF=${INITIAL_BLENDING_COEFF:-0.4}
export DIRECT_TRANSFER_STEPS=${DIRECT_TRANSFER_STEPS:-45}
export DECAYED_TRANSFER_STEPS=${DECAYED_TRANSFER_STEPS:-22}
export KP=${KP:-0.5}
export KI=${KI:-0.2}
export MAX_BLENDING_COEFF_DELTA=${MAX_BLENDING_COEFF_DELTA:-0.05}
export GUIDANCE_SCALE=${GUIDANCE_SCALE:-5.0}
export NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-100}
export SEED=${SEED:-42}

echo "[run] PROJECT_NAME=$PROJECT_NAME"
echo "[run] CN_WEIGHT=$CONTROLNET_WEIGHT  INITIAL_BLENDING_COEFF=$INITIAL_BLENDING_COEFF"
echo "[run] OUTPUT_DIR=$HOME/outputs/inference/$PROJECT_NAME (final mp4s only)"

# ============================================================================
# Run
# ----------------------------------------------------------------------------
# run_inference.py still writes the final mp4s to $HOME/outputs/inference/...,
# which is a small number of large sequential writes — that's the kind of
# /home traffic the admin's email explicitly allows ("Only store final results
# in /home"). The heavy I/O (model loads, HF / Torch caches, wandb run files,
# beta raw-face cache reads) is now all on $ScratchDir.
# ============================================================================
python -m inference.run_inference \
    --checkpoint_path "$CHECKPOINT" \
    --base_model_path "$WAN_MODEL" \
    --controlnet_config_repo "$HED_CONFIG" \
    --cache_dir "$WAN_BETA_CACHE" \
    --project_name "$PROJECT_NAME" \
    --controlnet_stride "$CONTROLNET_STRIDE" \
    --controlnet_weight "$CONTROLNET_WEIGHT" \
    --initial_blending_coeff "$INITIAL_BLENDING_COEFF" \
    --direct_transfer_steps "$DIRECT_TRANSFER_STEPS" \
    --decayed_transfer_steps "$DECAYED_TRANSFER_STEPS" \
    --Kp "$KP" \
    --Ki "$KI" \
    --max_blending_coeff_delta "$MAX_BLENDING_COEFF_DELTA" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --seed "$SEED"

# Cleanup of $ScratchDir is handled by the EXIT trap above.
