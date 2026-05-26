#!/bin/bash
#SBATCH --job-name=wan-precompute-inverts
#SBATCH --partition=dmb
#SBATCH --nodelist=hpc-node09
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --output=/home/s2710099/logs/wan-precompute-inverts-%j.out
#SBATCH --error=/home/s2710099/logs/wan-precompute-inverts-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams

set -euo pipefail

# ============================================================================
# Local scratch (per HPC admin: do NOT do heavy I/O on /home)
# ============================================================================
ScratchDir="/local/${SLURM_JOBID}"

# Final destination for the invert_face/ outputs. We write to scratch and
# rsync once at the end so /home only sees one batched write.
FINAL_INVERT_DIR=${WAN_BETA_INVERT_DIR:-$HOME/cache/wan-beta/invert_face}

cleanup_node() {
    set +e
    # Sync outputs back to /home before wiping scratch. This is the only
    # place /home gets touched during the job.
    if [ -d "$ScratchDir/invert_face" ]; then
        mkdir -p "$FINAL_INVERT_DIR"
        echo "[cleanup] syncing $ScratchDir/invert_face -> $FINAL_INVERT_DIR"
        rsync -a "$ScratchDir/invert_face/" "$FINAL_INVERT_DIR/" || true
    fi
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
WAN_SRC=$(ls -d "$WAN_CACHE_HOME"/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
[ -z "$WAN_SRC" ] && { echo "ERROR: Wan model not cached at $WAN_CACHE_HOME" >&2; exit 1; }

BETA_CACHE_HOME=${WAN_BETA_CACHE_HOME:-$HOME/cache/wan-beta}
[ -d "$BETA_CACHE_HOME/raw_face" ] \
    || { echo "ERROR: raw_face dir not found at $BETA_CACHE_HOME/raw_face" >&2; exit 1; }

# ============================================================================
# Stage inputs to local scratch
# ----------------------------------------------------------------------------
# `cp -aL` follows symlinks (HF hub stores snapshots/<rev>/ as symlinks into
# blobs/<hash>) so the staged copies are real files, not pointers back to /home.
# We only stage what this job actually needs: the Wan snapshot (model weights
# + scheduler + vae + tokenizer + text_encoder) and raw_face/ (face PNG-as-pt
# files). No ControlNet, no checkpoint, no rest of the beta cache.
# ============================================================================
mkdir -p "$ScratchDir/wan_model" \
         "$ScratchDir/raw_face" \
         "$ScratchDir/invert_face" \
         "$ScratchDir/hf_home" \
         "$ScratchDir/torch" \
         "$ScratchDir/wandb"

echo "[stage] copying Wan snapshot ($WAN_SRC) -> $ScratchDir/wan_model"
cp -aL "$WAN_SRC"/. "$ScratchDir/wan_model"/

echo "[stage] copying raw_face/ ($BETA_CACHE_HOME/raw_face) -> $ScratchDir/raw_face"
cp -aL "$BETA_CACHE_HOME/raw_face"/. "$ScratchDir/raw_face"/

echo "[stage] staged size:"
du -sh "$ScratchDir"/* 2>/dev/null || true

# ============================================================================
# Re-bind all heavy-I/O paths to scratch
# ============================================================================
export WAN_MODEL="$ScratchDir/wan_model"

# Build a `cache_dir`-shaped directory on scratch so precompute_inverts can
# read raw_face from it via `--cache_dir`. We only need raw_face/ here; the
# other subdirs (face_latent/, conds/, etc.) are not touched by this script.
mkdir -p "$ScratchDir/wan-beta-cache"
ln -sfn "$ScratchDir/raw_face" "$ScratchDir/wan-beta-cache/raw_face"
export WAN_BETA_CACHE="$ScratchDir/wan-beta-cache"

# HF / Torch / wandb caches on local disk
export HF_HOME="$ScratchDir/hf_home"
export HF_HUB_CACHE="$ScratchDir/hf_home/hub"
export TRANSFORMERS_CACHE="$ScratchDir/hf_home"
export HF_HUB_OFFLINE=1
export TORCH_HOME="$ScratchDir/torch"
export WANDB_DIR="$ScratchDir/wandb"
export WANDB_MODE=${WANDB_MODE:-offline}

# Diffusers attention backend
export DIFFUSERS_ATTN_BACKEND="flash"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[paths] WAN_MODEL=$WAN_MODEL"
echo "[paths] WAN_BETA_CACHE=$WAN_BETA_CACHE"
echo "[paths] OUTPUT_DIR(scratch)=$ScratchDir/invert_face"
echo "[paths] FINAL_INVERT_DIR(home)=$FINAL_INVERT_DIR"

# ============================================================================
# Tunables — override on the sbatch command line:
#   START_IDX=50 NUM_FACES=50 sbatch slurm/precompute_inverts.sbatch
# ============================================================================
export HEIGHT=${HEIGHT:-512}
export WIDTH=${WIDTH:-512}
export NUM_FRAMES=${NUM_FRAMES:-9}
export NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-100}
export NUM_FACES=${NUM_FACES:-100}
export START_IDX=${START_IDX:-0}
export SAVE_DTYPE=${SAVE_DTYPE:-bf16}
export PROMPT=${PROMPT:-""}
export OVERWRITE_FLAG=${OVERWRITE_FLAG:-""}  # set to "--overwrite" to redo

echo "[run] START_IDX=$START_IDX NUM_FACES=$NUM_FACES STEPS=$NUM_INFERENCE_STEPS"
echo "[run] HW=${HEIGHT}x${WIDTH} FRAMES=$NUM_FRAMES DTYPE=$SAVE_DTYPE"

# ============================================================================
# Run
# ----------------------------------------------------------------------------
# precompute_inverts.py writes its .pt files to $ScratchDir/invert_face/. The
# EXIT trap above rsyncs them to $FINAL_INVERT_DIR at job end (single batched
# write to /home), which is the only /home touch during the job.
# ============================================================================
python -m inference.precompute_inverts \
    --base_model_path      "$WAN_MODEL" \
    --cache_dir            "$WAN_BETA_CACHE" \
    --output_dir           "$ScratchDir/invert_face" \
    --height               "$HEIGHT" \
    --width                "$WIDTH" \
    --num_frames           "$NUM_FRAMES" \
    --num_inference_steps  "$NUM_INFERENCE_STEPS" \
    --num_faces            "$NUM_FACES" \
    --start_idx            "$START_IDX" \
    --save_dtype           "$SAVE_DTYPE" \
    --prompt               "$PROMPT" \
    $OVERWRITE_FLAG

# Cleanup of $ScratchDir + rsync of invert_face/ -> /home is handled by the
# EXIT trap above.
