#!/bin/bash
#SBATCH --job-name=wan-vanilla-100
#SBATCH --partition=dmb
# Only hpc-node07 / hpc-node09 (L40 48GB) and ctit092 (A40 45GB) are suitable.
# Exclude ctit084/085 (RTX6000 24GB, too little VRAM) and hpc-node31 (Blackwell,
# needs the wan22-bw env). Slurm picks whichever of the three is free.
#SBATCH --exclude=ctit084,ctit085,hpc-node31
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=160G
#SBATCH --output=/home/s2710099/logs/wan-vanilla-100-%j.out
#SBATCH --error=/home/s2710099/logs/wan-vanilla-100-%j.err
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
conda activate wan22

# ============================================================================
# Resolve Wan snapshot on /home (read once during stage-in)
# ============================================================================
WAN_CACHE_HOME=$HOME/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers
WAN_SRC=$(ls -d "$WAN_CACHE_HOME"/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
[ -z "$WAN_SRC" ] && { echo "ERROR: Wan model not cached at $WAN_CACHE_HOME" >&2; exit 1; }

# ============================================================================
# Stage Wan snapshot to local scratch (cp -aL resolves HF blob symlinks)
# ============================================================================
mkdir -p "$ScratchDir/wan_model" "$ScratchDir/hf_home" "$ScratchDir/torch"

echo "[stage] copying Wan snapshot ($WAN_SRC) -> $ScratchDir/wan_model"
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

# Diffusers attention backend. On A40 (wan22 env) flash-attn 2.8.3 is available
# and diffusers needs this env var to pick it up. If flash errors, fall back
# with DIFFUSERS_ATTN_BACKEND=native on the sbatch command line.
export DIFFUSERS_ATTN_BACKEND=${DIFFUSERS_ATTN_BACKEND:-flash}
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ============================================================================
# Tunables — override on the sbatch command line, e.g.:
#   NUM_INFERENCE_STEPS=50 GUIDANCE_SCALE=5.0 sbatch slurm/wan_vanilla.sbatch
# Defaults below = Wan 2.2 A14B model-card recommended dual CFG (neutral
# clean reference, independent of any method).
# ============================================================================
export PROJECT_NAME=${PROJECT_NAME:-wan_vanilla_100}
export HEIGHT=${HEIGHT:-528}
export WIDTH=${WIDTH:-528}
export NUM_FRAMES=${NUM_FRAMES:-61}
export NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-40}
export GUIDANCE_SCALE=${GUIDANCE_SCALE:-4.0}
export GUIDANCE_SCALE_2=${GUIDANCE_SCALE_2:-3.0}
export SEED=${SEED:-0}
export FPS=${FPS:-16}
export MAX_PROMPTS=${MAX_PROMPTS:-}

OUTPUT_DIR="$HOME/outputs/inference/$PROJECT_NAME"
echo "[run] PROJECT_NAME=$PROJECT_NAME -> $OUTPUT_DIR"
echo "[run] ${HEIGHT}x${WIDTH}x${NUM_FRAMES}  steps=$NUM_INFERENCE_STEPS  "\
     "cfg=$GUIDANCE_SCALE/$GUIDANCE_SCALE_2  seed=$SEED  fps=$FPS  "\
     "max_prompts=${MAX_PROMPTS:-<all>}"

# ============================================================================
# Run — final mp4s written to /home (small sequential writes, allowed)
# ============================================================================
python -m inference.run_wan_vanilla \
    --model_path "$WAN_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num_frames "$NUM_FRAMES" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --guidance_scale_2 "$GUIDANCE_SCALE_2" \
    --seed "$SEED" \
    --fps "$FPS" \
    ${MAX_PROMPTS:+--max_prompts "$MAX_PROMPTS"}

# Cleanup of $ScratchDir handled by the EXIT trap above.
