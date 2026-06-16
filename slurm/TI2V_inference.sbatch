#!/bin/bash
#SBATCH --job-name=wan-ti2v-100
#SBATCH --partition=dmb
# Pin to hpc-node31 (Blackwell). Requires the wan22-bw env and the native
# attention backend (flash-attn is not available there).
#SBATCH --nodelist=hpc-node31
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=160G
#SBATCH --output=/home/s2710099/logs/wan-ti2v-100-%j.out
#SBATCH --error=/home/s2710099/logs/wan-ti2v-100-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams

set -euo pipefail

# ============================================================================
# Local scratch (per HPC admin: do NOT do heavy I/O on /home)
# ============================================================================
ScratchDir="/local/${SLURM_JOBID}"

cleanup_node() {
    set +e
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
conda activate wan22-bw

# ============================================================================
# Resolve TI2V-5B-Diffusers snapshot on /home (read once during stage-in)
# ----------------------------------------------------------------------------
# This is the diffusers-format snapshot, downloaded with:
#   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers
# NOT the raw Wan-format dir /home/s2710099/models/Wan2.2-TI2V-5B (which
# WanImageToVideoPipeline.from_pretrained cannot load).
# ============================================================================
WAN_CACHE_HOME=$HOME/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers
WAN_SRC=$(ls -d "$WAN_CACHE_HOME"/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
[ -z "$WAN_SRC" ] && {
    echo "ERROR: TI2V-5B-Diffusers not cached at $WAN_CACHE_HOME" >&2
    echo "       Run: huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers" >&2
    exit 1
}

# Faces dir on /home (100 small face_{i}.pt files; read once each, left in
# place like slurm/unseen_inference.sbatch does with its PNGs).
export FACES_DIR=${FACES_DIR:-$HOME/cache/wan-beta/raw_face}
[ -d "$FACES_DIR" ] || { echo "ERROR: faces dir $FACES_DIR not found" >&2; exit 1; }

# ============================================================================
# Stage TI2V snapshot to local scratch (cp -aL resolves HF blob symlinks)
# ============================================================================
mkdir -p "$ScratchDir/wan_model" "$ScratchDir/hf_home" "$ScratchDir/torch"

echo "[stage] copying TI2V snapshot ($WAN_SRC) -> $ScratchDir/wan_model"
cp -aL "$WAN_SRC"/. "$ScratchDir/wan_model"/
echo "[stage] staged size:"
du -sh "$ScratchDir"/* 2>/dev/null || true

# ============================================================================
# Re-bind heavy-I/O paths to scratch
# ============================================================================
export WAN_MODEL="$ScratchDir/wan_model"
export HF_HOME="$ScratchDir/hf_home"
export HF_HUB_CACHE="$ScratchDir/hf_home/hub"
export TRANSFORMERS_CACHE="$ScratchDir/hf_home"
export HF_HUB_OFFLINE=1
export TORCH_HOME="$ScratchDir/torch"

# Diffusers attention backend. hpc-node31 is Blackwell and has no flash-attn,
# so force the native backend (matches slurm/unseen_inference.sbatch).
export DIFFUSERS_ATTN_BACKEND=${DIFFUSERS_ATTN_BACKEND:-native}
export WAN_FORCE_NATIVE_ATTN=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ============================================================================
# Tunables — override on the sbatch command line, e.g.:
#   NUM_INFERENCE_STEPS=50 GUIDANCE_SCALE=5.0 sbatch slurm/TI2V_inference.sbatch
# TI2V-5B VAE is 16x spatial (vs the 8x Wan2.1 VAE in wan_vanilla.sbatch), so
# H/W must be a multiple of 32 -- 528 is invalid and crashes with a token-grid
# mismatch. 512x512 is the nearest valid size; guidance 5.0 is TI2V-5B's
# single-CFG recommended value.
# ============================================================================
export PROJECT_NAME=${PROJECT_NAME:-wan_ti2v_100}
export HEIGHT=${HEIGHT:-512}
export WIDTH=${WIDTH:-512}
export NUM_FRAMES=${NUM_FRAMES:-61}
export NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-40}
export GUIDANCE_SCALE=${GUIDANCE_SCALE:-5.0}
export SEED=${SEED:-0}
export FPS=${FPS:-16}
export MAX_PROMPTS=${MAX_PROMPTS:-}
export START_IDX=${START_IDX:-0}

OUTPUT_DIR="$HOME/outputs/inference/$PROJECT_NAME"
echo "[run] PROJECT_NAME=$PROJECT_NAME -> $OUTPUT_DIR"
echo "[run] FACES_DIR=$FACES_DIR"
echo "[run] ${HEIGHT}x${WIDTH}x${NUM_FRAMES}  steps=$NUM_INFERENCE_STEPS  "\
     "cfg=$GUIDANCE_SCALE  seed=$SEED  fps=$FPS  "\
     "start_idx=$START_IDX  max_prompts=${MAX_PROMPTS:-<all>}"

# ============================================================================
# Run — final mp4s written to /home (small sequential writes, allowed)
# ============================================================================
python -m inference.TI2V_inference \
    --model_path "$WAN_MODEL" \
    --faces_dir "$FACES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num_frames "$NUM_FRAMES" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --seed "$SEED" \
    --fps "$FPS" \
    --start_idx "$START_IDX" \
    ${MAX_PROMPTS:+--max_prompts "$MAX_PROMPTS"}

# Cleanup of $ScratchDir handled by the EXIT trap above.
