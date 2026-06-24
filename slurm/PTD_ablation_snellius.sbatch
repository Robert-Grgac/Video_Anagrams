#!/bin/bash
#SBATCH --job-name=ptd-abl
#SBATCH --account=ttsei15225
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=/home/%u/robert_thesis/logs/%x-%j.out
#SBATCH --error=/home/%u/robert_thesis/logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

# ============================================================================
# Snellius PTD heuristic-ablation runner — ONE heuristic per job.
#
# Submit one job per heuristic (HEURISTIC selects which of the four). Pick the
# GPU at submit time (sbatch flags override the #SBATCH defaults above):
#   gpu_h100 (fast)  : sbatch --partition=gpu_h100 --mem=180G ...
#   gpu_a100 (weaker): sbatch --partition=gpu_a100 ...        (works via offload)
#   gpu_mig is NOT usable (MIG slice VRAM too small for the two-expert model).
#
# Examples (one per heuristic, run them in parallel — they're independent jobs):
#   HEURISTIC=no_heuristic sbatch --job-name=ptd-none slurm/PTD_ablation_snellius.sbatch
#   HEURISTIC=heuristic_1  sbatch --job-name=ptd-h1   slurm/PTD_ablation_snellius.sbatch
#   HEURISTIC=heuristic_2  sbatch --job-name=ptd-h2   slurm/PTD_ablation_snellius.sbatch
#   HEURISTIC=heuristic_4  sbatch --job-name=ptd-h4   slurm/PTD_ablation_snellius.sbatch
#
# PRECONDITIONS (one-time, on a login node):
#   mkdir -p /scratch-shared/$USER/{hf-cache,wan-beta-cache,outputs,logs}
#   mkdir -p $HOME/robert_thesis/logs
#   [ -e $HOME/outputs ] || ln -s /scratch-shared/$USER/outputs $HOME/outputs
#   # The Wan snapshot must be cached under /scratch-shared/$USER/hf-cache/hub,
#   # and the deterministic-invert latents present at $REF_LATENTS_HOME below
#   # (rsync them from the local cluster if not already there).
# ============================================================================

export HEURISTIC=${HEURISTIC:?set HEURISTIC=no_heuristic|heuristic_1|heuristic_2|heuristic_4|all}

# --chdir does not expand %u on Snellius; cd here instead.
cd "$HOME/robert_thesis/Video_Anagrams"

ScratchDir="$TMPDIR"
echo "[scratch] using TMPDIR=$ScratchDir"

# ----------------------------------------------------------------------------
# Modules + conda env (PATH fix is mandatory on Snellius — base python shadows
# the env's otherwise).
# ----------------------------------------------------------------------------
module purge
module load 2024
module load Miniconda3/24.7.1-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate wan22
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r
echo "[env] python=$(which python)  ($(python --version))"

# ----------------------------------------------------------------------------
# Resolve source paths on persistent storage.
# ----------------------------------------------------------------------------
WAN_CACHE_HOME=/scratch-shared/$USER/hf-cache/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers
WAN_SRC=$(ls -d "$WAN_CACHE_HOME"/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
[ -z "$WAN_SRC" ] && { echo "ERROR: Wan model not cached at $WAN_CACHE_HOME" >&2; exit 1; }

REF_LATENTS_HOME=${REF_LATENTS_HOME:-/scratch-shared/$USER/wan-beta-cache/deterministic_invert_faces_528x528x61}
[ -d "$REF_LATENTS_HOME" ] \
    || { echo "ERROR: ref latents dir missing at $REF_LATENTS_HOME (rsync it from the local cluster)" >&2; exit 1; }

# ----------------------------------------------------------------------------
# Stage to node-local $TMPDIR (fast disk).
# ----------------------------------------------------------------------------
mkdir -p "$ScratchDir/wan_model" "$ScratchDir/ref_latents" \
         "$ScratchDir/hf_home" "$ScratchDir/torch"

echo "[stage] Wan snapshot ($WAN_SRC) -> $ScratchDir/wan_model"
cp -aL "$WAN_SRC"/. "$ScratchDir/wan_model"/
echo "[stage] ref latents ($REF_LATENTS_HOME) -> $ScratchDir/ref_latents"
cp -aL "$REF_LATENTS_HOME"/. "$ScratchDir/ref_latents"/
echo "[stage] staged size:"; du -sh "$ScratchDir"/* 2>/dev/null || true

# ----------------------------------------------------------------------------
# Re-bind heavy-I/O paths to scratch.
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

# Final mp4s go to $HOME/outputs/inference (symlinked to /scratch-shared).
export OUTPUT_ROOT=${OUTPUT_ROOT:-$HOME/outputs/inference}
export SEED=${SEED:-0}
export MAX_PAIRS=${MAX_PAIRS:-}   # blank = all 100

echo "[run] HEURISTIC=$HEURISTIC  OUTPUT_ROOT=$OUTPUT_ROOT  SEED=$SEED"

CMD=(python -m PTD_Pipeline.PTD_ablation_experiments
    --model_path "$WAN_MODEL"
    --ref_latents_root "$REF_LATENTS_ROOT"
    --output_root "$OUTPUT_ROOT"
    --heuristic "$HEURISTIC"
    --seed "$SEED"
)
[ -n "$MAX_PAIRS" ] && CMD+=(--max_pairs "$MAX_PAIRS")

echo "[run] ${CMD[*]}"
"${CMD[@]}"
