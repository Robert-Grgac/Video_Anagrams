#!/bin/bash
#SBATCH --job-name=ptd-ablation
#SBATCH --partition=dmb
# Only hpc-node07 / hpc-node09 (L40 48GB) and ctit092 (A40 45GB) are suitable.
# Exclude ctit084/085 (RTX6000 24GB, too little VRAM) and hpc-node31 (Blackwell,
# needs the wan22-bw env). Slurm picks whichever of the three is free.
#SBATCH --exclude=ctit084,ctit085,hpc-node31
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=160G
#SBATCH --output=/home/s2710099/logs/ptd-ablation-%j.out
#SBATCH --error=/home/s2710099/logs/ptd-ablation-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams

set -euo pipefail

# ============================================================================
# Runs PTD_Pipeline.PTD_ablation_experiments: loads the WanPTDiffusionPipeline
# ONCE and writes 4×100 mp4s (no-heuristic / h1 / h2 / h4) to four fixed dirs
# under $HOME/outputs/inference. Video-output-only (no wandb, no HOG, no CFG,
# no conditional-baseline pass). The script skips any mp4 that already exists,
# so if this 48h job times out before all 400 videos are done, just re-submit
# the SAME sbatch and it resumes.
# ============================================================================

# ----------------------------------------------------------------------------
# Local scratch (per HPC admin: do NOT do heavy I/O on /home).
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# Conda env
# ----------------------------------------------------------------------------
source $(conda info --base)/etc/profile.d/conda.sh
conda activate wan22

# ----------------------------------------------------------------------------
# Resolve source paths on /home. WanPTDPipeline needs only the Wan 2.2 A14B
# snapshot (no ControlNet / HED) and the deterministic-invert reference latents
# (101 step_*.pt per face).
# ----------------------------------------------------------------------------
WAN_CACHE_HOME=$HOME/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers
WAN_SRC=$(ls -d "$WAN_CACHE_HOME"/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
[ -z "$WAN_SRC" ] && { echo "ERROR: Wan model not cached at $WAN_CACHE_HOME" >&2; exit 1; }

REF_LATENTS_HOME=${REF_LATENTS_HOME:-$HOME/cache/wan-beta/deterministic_invert_faces_528x528x61}
[ -d "$REF_LATENTS_HOME" ] \
    || { echo "ERROR: ref latents dir missing at $REF_LATENTS_HOME" >&2; exit 1; }

# ----------------------------------------------------------------------------
# Stage inputs to local scratch (cp -aL: follow HF blob symlinks -> real files).
# Wan snapshot ~32GB + invert latents ~44GB. Both must fit in /local.
# ----------------------------------------------------------------------------
mkdir -p "$ScratchDir/wan_model" \
         "$ScratchDir/ref_latents" \
         "$ScratchDir/hf_home" \
         "$ScratchDir/torch"

echo "[stage] copying Wan snapshot ($WAN_SRC) -> $ScratchDir/wan_model"
cp -aL "$WAN_SRC"/. "$ScratchDir/wan_model"/

echo "[stage] copying ref latents ($REF_LATENTS_HOME) -> $ScratchDir/ref_latents"
cp -aL "$REF_LATENTS_HOME"/. "$ScratchDir/ref_latents"/

echo "[stage] staged size:"
du -sh "$ScratchDir"/* 2>/dev/null || true

# ----------------------------------------------------------------------------
# Re-bind heavy-I/O paths to scratch
# ----------------------------------------------------------------------------
export WAN_MODEL="$ScratchDir/wan_model"
export REF_LATENTS_ROOT="$ScratchDir/ref_latents"

export HF_HOME="$ScratchDir/hf_home"
export HF_HUB_CACHE="$ScratchDir/hf_home/hub"
export TRANSFORMERS_CACHE="$ScratchDir/hf_home"
export HF_HUB_OFFLINE=1
export TORCH_HOME="$ScratchDir/torch"

export DIFFUSERS_ATTN_BACKEND="flash"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[paths] WAN_MODEL=$WAN_MODEL"
echo "[paths] REF_LATENTS_ROOT=$REF_LATENTS_ROOT"

# ----------------------------------------------------------------------------
# Tunables — override on the sbatch command line, e.g.:
#   MAX_PAIRS=2 sbatch slurm/PTD_ablation_experiments.sbatch   # quick smoke
# Output dirs are fixed inside the script:
#   PTD_noHeuirsitc / PTD_heuristic_1 / PTD_heuristic_2 / heuristic_4
# ----------------------------------------------------------------------------
export SEED=${SEED:-0}
export MAX_PAIRS=${MAX_PAIRS:-}   # blank = all 100 per heuristic
# HEURISTIC=all (default) runs all four consecutively. Set
# HEURISTIC=no_heuristic|heuristic_1|heuristic_2|heuristic_4 to run just one
# (e.g. to split across several local nodes, one heuristic per job).
export HEURISTIC=${HEURISTIC:-all}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/home/s2710099/outputs/inference}

CMD=(python -m PTD_Pipeline.PTD_ablation_experiments
    --model_path "$WAN_MODEL"
    --ref_latents_root "$REF_LATENTS_ROOT"
    --output_root "$OUTPUT_ROOT"
    --heuristic "$HEURISTIC"
    --seed "$SEED"
)
if [ -n "$MAX_PAIRS" ]; then
    CMD+=(--max_pairs "$MAX_PAIRS")
fi

echo "[run] ${CMD[*]}"
"${CMD[@]}"

# Cleanup of $ScratchDir is handled by the EXIT trap above.
