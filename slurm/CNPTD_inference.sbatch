#!/bin/bash
#SBATCH --job-name=cnptd-infer
#SBATCH --partition=dmb
#SBATCH --nodelist=hpc-node07
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --output=/home/s2710099/logs/cnptd-infer-%j.out
#SBATCH --error=/home/s2710099/logs/cnptd-infer-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams

set -euo pipefail

# ============================================================================
# Local scratch (per HPC admin: do NOT do heavy I/O on /home).
# Reading the Wan 2.2 A14B shards over NFS thrashes the page cache and gets
# OOM-killed even with --mem=96G; staging to /local/$JOBID first avoids it.
# ============================================================================
ScratchDir="/local/${SLURM_JOBID}"

cleanup_node() {
    set +e
    # Sync wandb run files back to /home before wiping scratch
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
CKPT_HOME=$HOME/checkpoints/wan-beta/beta-008_final.safetensors

[ -z "$WAN_SRC" ] && { echo "ERROR: Wan model not cached at $WAN_CACHE_HOME" >&2; exit 1; }
[ -z "$HED_SRC" ] && { echo "ERROR: HED config not cached at $HED_CACHE_HOME" >&2; exit 1; }
[ -f "$CKPT_HOME" ] || { echo "ERROR: checkpoint not found at $CKPT_HOME" >&2; exit 1; }

# ============================================================================
# Stage inputs to local scratch (cp -aL follows symlinks so staged copies are
# real files, not pointers back to /home).
# ============================================================================
mkdir -p "$ScratchDir/wan_model" \
         "$ScratchDir/hed_config" \
         "$ScratchDir/wan-beta-ckpt" \
         "$ScratchDir/hf_home" \
         "$ScratchDir/torch" \
         "$ScratchDir/wandb"

echo "[stage] copying Wan snapshot ($WAN_SRC) -> $ScratchDir/wan_model"
cp -aL "$WAN_SRC"/. "$ScratchDir/wan_model"/

echo "[stage] copying HED controlnet config ($HED_SRC) -> $ScratchDir/hed_config"
cp -aL "$HED_SRC"/. "$ScratchDir/hed_config"/

echo "[stage] copying checkpoint -> $ScratchDir/wan-beta-ckpt"
cp -L "$CKPT_HOME" "$ScratchDir/wan-beta-ckpt/beta-008_final.safetensors"

echo "[stage] staged size:"
du -sh "$ScratchDir"/* 2>/dev/null || true

# ============================================================================
# Re-bind all heavy-I/O paths to scratch
# ============================================================================
export WAN_MODEL="$ScratchDir/wan_model"
export HED_CONFIG="$ScratchDir/hed_config"
export CHECKPOINT="$ScratchDir/wan-beta-ckpt/beta-008_final.safetensors"

# HF / Torch / wandb caches on local disk
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
echo "[paths] HED_CONFIG=$HED_CONFIG"
echo "[paths] CHECKPOINT=$CHECKPOINT"
echo "[paths] HF_HOME=$HF_HOME  TORCH_HOME=$TORCH_HOME  WANDB_DIR=$WANDB_DIR"

# ============================================================================
# Run
# ----------------------------------------------------------------------------
# The script reads WAN_MODEL / HED_CONFIG / CHECKPOINT from env (with /home
# fallbacks for interactive use without staging). Final mp4s are written to
# /home/s2710099/outputs/inference/... -- bounded sequential writes, allowed
# per HPC admin guidance.
# ============================================================================
python PTD_Pipeline/run_WanCNPTDPipeline.py

# Cleanup of $ScratchDir is handled by the EXIT trap above.
