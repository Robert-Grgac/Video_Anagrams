#!/bin/bash
#SBATCH --job-name=ptd-100-fair
#SBATCH --partition=dmb
#SBATCH --nodelist=hpc-node09
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --mem=160G
#SBATCH --output=/home/s2710099/logs/ptd-100-fair-%j.out
#SBATCH --error=/home/s2710099/logs/ptd-100-fair-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams

set -euo pipefail

# ============================================================================
# Local scratch (per HPC admin: do NOT do heavy I/O on /home).
# ============================================================================
ScratchDir="/local/${SLURM_JOBID}"

cleanup_node() {
    set +e
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
# ----------------------------------------------------------------------------
# WanPTDPipeline only needs:
#   - the Wan 2.2 A14B snapshot (no separate ControlNet, no HED config)
#   - the deterministic_invert_faces_528x528x61 dir (101 step_*.pt per face)
# ============================================================================
WAN_CACHE_HOME=$HOME/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers
WAN_SRC=$(ls -d "$WAN_CACHE_HOME"/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
[ -z "$WAN_SRC" ] && { echo "ERROR: Wan model not cached at $WAN_CACHE_HOME" >&2; exit 1; }

REF_LATENTS_HOME=${REF_LATENTS_HOME:-$HOME/cache/wan-beta/deterministic_invert_faces_528x528x61}
[ -d "$REF_LATENTS_HOME" ] \
    || { echo "ERROR: ref latents dir missing at $REF_LATENTS_HOME" >&2; exit 1; }

# ============================================================================
# Stage inputs to local scratch
# ----------------------------------------------------------------------------
# cp -aL follows symlinks so staged copies are real files. The deterministic
# invert dir is ~43GB; the Wan snapshot is ~32GB. Both must fit in /local
# (typically 100s of GB on dmb nodes).
# ============================================================================
mkdir -p "$ScratchDir/wan_model" \
         "$ScratchDir/ref_latents" \
         "$ScratchDir/hf_home" \
         "$ScratchDir/torch" \
         "$ScratchDir/wandb"

echo "[stage] copying Wan snapshot ($WAN_SRC) -> $ScratchDir/wan_model"
cp -aL "$WAN_SRC"/. "$ScratchDir/wan_model"/

echo "[stage] copying ref latents ($REF_LATENTS_HOME) -> $ScratchDir/ref_latents"
cp -aL "$REF_LATENTS_HOME"/. "$ScratchDir/ref_latents"/

echo "[stage] staged size:"
du -sh "$ScratchDir"/* 2>/dev/null || true

# ============================================================================
# Re-bind all heavy-I/O paths to scratch
# ============================================================================
export WAN_MODEL="$ScratchDir/wan_model"
export REF_LATENTS_ROOT="$ScratchDir/ref_latents"

export HF_HOME="$ScratchDir/hf_home"
export HF_HUB_CACHE="$ScratchDir/hf_home/hub"
export TRANSFORMERS_CACHE="$ScratchDir/hf_home"
export HF_HUB_OFFLINE=1
export TORCH_HOME="$ScratchDir/torch"
export WANDB_DIR="$ScratchDir/wandb"
export WANDB_MODE=${WANDB_MODE:-online}

export DIFFUSERS_ATTN_BACKEND="flash"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[paths] WAN_MODEL=$WAN_MODEL"
echo "[paths] REF_LATENTS_ROOT=$REF_LATENTS_ROOT"
echo "[paths] HF_HOME=$HF_HOME  TORCH_HOME=$TORCH_HOME  WANDB_DIR=$WANDB_DIR"

# ============================================================================
# Tunables — override on the sbatch command line, e.g.:
#   OUTPUT_DIR=$HOME/outputs/inference/ptd_baseline_v2 sbatch slurm/ptd_pipeline_100_fair.sbatch
# ============================================================================
export OUTPUT_DIR=${OUTPUT_DIR:-$HOME/outputs/inference/ptd_og_pipeline_100_fair}
export WANDB_PROJECT=${WANDB_PROJECT:-PTD_inference_100_fair}
export MAX_PAIRS=${MAX_PAIRS:-}   # blank = all 100
export START_IDX=${START_IDX:-0}
export SEED=${SEED:-0}

mkdir -p "$OUTPUT_DIR"

echo "[run] OUTPUT_DIR=$OUTPUT_DIR"
echo "[run] WANDB_PROJECT=$WANDB_PROJECT  MAX_PAIRS=${MAX_PAIRS:-<all>}  START_IDX=$START_IDX"

# ============================================================================
# Run
# ============================================================================
CMD=(python -m PTD_Pipeline.run_WanPTDPipeline_100_fair
    --model_path "$WAN_MODEL"
    --ref_latents_root "$REF_LATENTS_ROOT"
    --output_dir "$OUTPUT_DIR"
    --wandb_project "$WANDB_PROJECT"
    --start_idx "$START_IDX"
    --seed "$SEED"
)
if [ -n "$MAX_PAIRS" ]; then
    CMD+=(--max_pairs "$MAX_PAIRS")
fi

"${CMD[@]}"

# Cleanup of $ScratchDir is handled by the EXIT trap above.
