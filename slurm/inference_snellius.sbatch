#!/bin/bash
#SBATCH --job-name=wan-ptd-cn-infer
#SBATCH --account=ttsei15225
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --chdir=/home/%u/robert_thesis/Video_Anagrams

set -euo pipefail

# ============================================================================
# ONE-TIME SETUP (run once on a Snellius login node before submitting):
#   mkdir -p /scratch-shared/$USER/{hf-cache,wan-beta-cache,wan-beta-ckpt,outputs,logs}
#   # symlink ~/outputs to scratch so run_inference.py's hardcoded Path.home()/outputs
#   # writes to scratch instead of home (which has a quota):
#   [ -e $HOME/outputs ] || ln -s /scratch-shared/$USER/outputs $HOME/outputs
# ============================================================================

# ============================================================================
# Local scratch — Snellius provides $TMPDIR per job and auto-cleans it.
# No manual mkdir/rm needed.
# ============================================================================
ScratchDir="$TMPDIR"
echo "[scratch] using TMPDIR=$ScratchDir"

# Sync wandb run files back to scratch-shared at exit (cheap, only at job end).
cleanup_node() {
    set +e
    if [ -d "$ScratchDir/wandb" ]; then
        mkdir -p "/scratch-shared/$USER/logs/wandb"
        cp -a "$ScratchDir/wandb"/. "/scratch-shared/$USER/logs/wandb"/ 2>/dev/null || true
    fi
}
trap cleanup_node EXIT SIGINT SIGTERM

# ============================================================================
# Modules + conda env
# ============================================================================
module purge
module load 2023
module load Miniconda3/23.5.2-0           # adjust if `module avail Miniconda3` shows a different version
source $(conda info --base)/etc/profile.d/conda.sh
conda activate wan22

# ============================================================================
# Resolve source paths on persistent storage (read once during stage-in)
# ============================================================================
WAN_CACHE_HOME=/scratch-shared/$USER/hf-cache/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers
HED_CACHE_HOME=/scratch-shared/$USER/hf-cache/hub/models--TheDenk--wan2.2-t2v-a14b-controlnet-hed-v1
WAN_SRC=$(ls -d "$WAN_CACHE_HOME"/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
HED_SRC=$(ls -d "$HED_CACHE_HOME"/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
[ -z "$WAN_SRC" ] && { echo "ERROR: Wan model not cached at $WAN_CACHE_HOME" >&2; exit 1; }
[ -z "$HED_SRC" ] && { echo "ERROR: HED config not cached at $HED_CACHE_HOME" >&2; exit 1; }

BETA_CACHE_HOME=${WAN_BETA_CACHE_HOME:-/scratch-shared/$USER/wan-beta-cache}
BETA_CKPT_HOME=${WAN_BETA_CKPT_HOME:-/scratch-shared/$USER/wan-beta-ckpt}

export CHECKPOINT_NAME=${CHECKPOINT_NAME:-beta-008_final.safetensors}
[ -f "$BETA_CKPT_HOME/$CHECKPOINT_NAME" ] \
    || { echo "ERROR: checkpoint $BETA_CKPT_HOME/$CHECKPOINT_NAME not found" >&2; exit 1; }

# ============================================================================
# Stage inputs to local scratch ($TMPDIR is on fast node-local disk)
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

export HF_HOME="$ScratchDir/hf_home"
export HF_HUB_CACHE="$ScratchDir/hf_home/hub"
export TRANSFORMERS_CACHE="$ScratchDir/hf_home"
export HF_HUB_OFFLINE=1
export TORCH_HOME="$ScratchDir/torch"
export WANDB_DIR="$ScratchDir/wandb"

# Snellius compute nodes have no outbound internet by default; keep wandb offline.
# Override with WANDB_MODE=online if you've confirmed proxy access.
export WANDB_MODE=${WANDB_MODE:-offline}

# Attention backend — H100 supports flash-attn 2.x fully.
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
#   PROJECT_NAME=ptd_cn_snellius_h100 sbatch slurm/inference_snellius.sbatch
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

export INVERT_TYPE=${INVERT_TYPE:-euler}
export MAX_PAIRS=${MAX_PAIRS:-}

if [ -n "${INVERT_SUBDIR:-}" ]; then
    INVERT_DIR="$WAN_BETA_CACHE/$INVERT_SUBDIR"
fi

echo "[run] PROJECT_NAME=$PROJECT_NAME"
echo "[run] CN_WEIGHT=$CONTROLNET_WEIGHT  INITIAL_BLENDING_COEFF=$INITIAL_BLENDING_COEFF"
echo "[run] INVERT_TYPE=$INVERT_TYPE  MAX_PAIRS=${MAX_PAIRS:-<all>}"
echo "[run] OUTPUT_DIR=$HOME/outputs/inference/$PROJECT_NAME (symlinked to /scratch-shared)"

# ============================================================================
# Run
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
    --seed "$SEED" \
    --invert_type "$INVERT_TYPE" \
    ${INVERT_DIR:+--invert_dir "$INVERT_DIR"} \
    ${MAX_PAIRS:+--max_pairs "$MAX_PAIRS"} \
    ${SWEEP:+--sweep}
