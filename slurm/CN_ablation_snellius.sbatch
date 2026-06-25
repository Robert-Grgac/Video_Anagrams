#!/bin/bash
#SBATCH --job-name=cn-abl
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
# Snellius ControlNet-ablation runner — ONE setup per job.
#   SETUP=single -> single shared CN (beta-002)      -> single_cn_dual_experts
#   SETUP=dual   -> beta-007 high + beta-004 low      -> dual_cn_dual_experts
#   SETUP=both   -> both, sequentially (single job).
#
# Pick the GPU at submit time (sbatch flags override the #SBATCH defaults):
#   gpu_h100 (fast)  : sbatch --partition=gpu_h100 --mem=180G ...
#   gpu_a100 (weaker): sbatch --partition=gpu_a100 ...        (works via offload)
#   gpu_mig is NOT usable (MIG slice VRAM too small).
#
# Examples (independent jobs, can run in parallel):
#   SETUP=single sbatch --job-name=cn-single slurm/CN_ablation_snellius.sbatch
#   SETUP=dual   sbatch --job-name=cn-dual   slurm/CN_ablation_snellius.sbatch
#
# PRECONDITIONS (one-time, on a login node):
#   mkdir -p /scratch-shared/$USER/{hf-cache,wan-beta-cache,outputs,logs}
#   mkdir -p $HOME/robert_thesis/logs
#   [ -e $HOME/outputs ] || ln -s /scratch-shared/$USER/outputs $HOME/outputs
#   # Wan + HED snapshots cached under /scratch-shared/$USER/hf-cache/hub;
#   # canny/ + manifest.json present under /scratch-shared/$USER/wan-beta-cache;
#   # checkpoints in $HOME/robert_thesis/checkpoints (same filenames as local).
# ============================================================================

export SETUP=${SETUP:?set SETUP=single|dual|both}

cd "$HOME/robert_thesis/Video_Anagrams"

ScratchDir="$TMPDIR"
echo "[scratch] using TMPDIR=$ScratchDir"

# ----------------------------------------------------------------------------
# Modules + conda env (PATH fix mandatory on Snellius).
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
# Resolve source paths.
# ----------------------------------------------------------------------------
WAN_CACHE_HOME=/scratch-shared/$USER/hf-cache/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers
HED_CACHE_HOME=/scratch-shared/$USER/hf-cache/hub/models--TheDenk--wan2.2-t2v-a14b-controlnet-hed-v1
WAN_SRC=$(ls -d "$WAN_CACHE_HOME"/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
HED_SRC=$(ls -d "$HED_CACHE_HOME"/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
[ -z "$WAN_SRC" ] && { echo "ERROR: Wan model not cached at $WAN_CACHE_HOME" >&2; exit 1; }
[ -z "$HED_SRC" ] && { echo "ERROR: HED config not cached at $HED_CACHE_HOME" >&2; exit 1; }

# Fail fast if the snapshots are INCOMPLETE. /scratch-shared auto-purges after
# 14 days and leaves empty stub dirs, so the existence checks above pass but the
# model then dies at load with "Unrecognized model ... model_type". Verify the
# key per-component files actually resolve (-s follows symlinks into blobs/).
check_wan_complete() {
    local d="$1" miss=0 f
    for f in model_index.json tokenizer/tokenizer_config.json text_encoder/config.json \
             vae/config.json transformer/config.json transformer_2/config.json; do
        [ -s "$d/$f" ] || { echo "  MISSING: $d/$f" >&2; miss=1; }
    done
    return $miss
}
if ! check_wan_complete "$WAN_SRC"; then
    echo "ERROR: Wan snapshot at $WAN_SRC is incomplete (scratch-shared purge?)." >&2
    echo "Re-download on a login node, then resubmit:" >&2
    echo "  huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers --cache-dir /scratch-shared/\$USER/hf-cache/hub" >&2
    exit 1
fi
[ -s "$HED_SRC/config.json" ] || {
    echo "ERROR: HED config at $HED_SRC is incomplete (scratch-shared purge?). Re-download:" >&2
    echo "  huggingface-cli download TheDenk/wan2.2-t2v-a14b-controlnet-hed-v1 --cache-dir /scratch-shared/\$USER/hf-cache/hub" >&2
    exit 1
}

BETA_CACHE_HOME=${WAN_BETA_CACHE_HOME:-/scratch-shared/$USER/wan-beta-cache}
# Checkpoints live in $HOME/robert_thesis/checkpoints on Snellius (same names).
BETA_CKPT_HOME=${WAN_BETA_CKPT_HOME:-$HOME/robert_thesis/checkpoints}

export SINGLE_CKPT_NAME=${SINGLE_CKPT_NAME:-beta-002_final.safetensors}
export HIGH_CKPT_NAME=${HIGH_CKPT_NAME:-beta-007_v2_final.safetensors}
export LOW_CKPT_NAME=${LOW_CKPT_NAME:-beta-004_final.safetensors}

# Only stage the checkpoints the chosen setup actually needs.
case "$SETUP" in
    single) CKPTS=("$SINGLE_CKPT_NAME") ;;
    dual)   CKPTS=("$HIGH_CKPT_NAME" "$LOW_CKPT_NAME") ;;
    both)   CKPTS=("$SINGLE_CKPT_NAME" "$HIGH_CKPT_NAME" "$LOW_CKPT_NAME") ;;
esac
for ck in "${CKPTS[@]}"; do
    [ -f "$BETA_CKPT_HOME/$ck" ] \
        || { echo "ERROR: checkpoint $BETA_CKPT_HOME/$ck not found" >&2; exit 1; }
done

# ----------------------------------------------------------------------------
# Stage to node-local $TMPDIR. Only canny/ + manifest.json from the beta cache
# (NOT the big invert latents — the CN runs don't read them).
# ----------------------------------------------------------------------------
mkdir -p "$ScratchDir/wan_model" "$ScratchDir/hed_config" \
         "$ScratchDir/wan-beta-cache/canny" "$ScratchDir/wan-beta-ckpt" \
         "$ScratchDir/hf_home" "$ScratchDir/torch"

echo "[stage] Wan snapshot -> $ScratchDir/wan_model"
cp -aL "$WAN_SRC"/. "$ScratchDir/wan_model"/
echo "[stage] HED config -> $ScratchDir/hed_config"
cp -aL "$HED_SRC"/. "$ScratchDir/hed_config"/
echo "[stage] canny + manifest -> $ScratchDir/wan-beta-cache"
cp -aL "$BETA_CACHE_HOME/canny"/. "$ScratchDir/wan-beta-cache/canny"/
cp -L "$BETA_CACHE_HOME/manifest.json" "$ScratchDir/wan-beta-cache/manifest.json"
echo "[stage] checkpoints: ${CKPTS[*]}"
for ck in "${CKPTS[@]}"; do
    cp -L "$BETA_CKPT_HOME/$ck" "$ScratchDir/wan-beta-ckpt/$ck"
done
echo "[stage] staged size:"; du -sh "$ScratchDir"/* 2>/dev/null || true

# ----------------------------------------------------------------------------
# Re-bind heavy-I/O paths to scratch.
# ----------------------------------------------------------------------------
export WAN_MODEL="$ScratchDir/wan_model"
export HED_CONFIG="$ScratchDir/hed_config"
export WAN_BETA_CACHE="$ScratchDir/wan-beta-cache"
export WAN_BETA_CKPT="$ScratchDir/wan-beta-ckpt"
export HF_HOME="$ScratchDir/hf_home"
export HF_HUB_CACHE="$ScratchDir/hf_home/hub"
export TRANSFORMERS_CACHE="$ScratchDir/hf_home"
export HF_HUB_OFFLINE=1
export TORCH_HOME="$ScratchDir/torch"
export DIFFUSERS_ATTN_BACKEND="flash"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export OUTPUT_ROOT=${OUTPUT_ROOT:-$HOME/outputs/inference}
export SEED=${SEED:-42}
export MAX_PAIRS=${MAX_PAIRS:-}   # blank = all 100

echo "[run] SETUP=$SETUP  OUTPUT_ROOT=$OUTPUT_ROOT  SEED=$SEED"

CMD=(python -m training.CN_ablation_experiments
    --base_model_path "$WAN_MODEL"
    --controlnet_config_repo "$HED_CONFIG"
    --cache_dir "$WAN_BETA_CACHE"
    --output_root "$OUTPUT_ROOT"
    --setup "$SETUP"
    --single_checkpoint "$WAN_BETA_CKPT/$SINGLE_CKPT_NAME"
    --high_checkpoint "$WAN_BETA_CKPT/$HIGH_CKPT_NAME"
    --low_checkpoint "$WAN_BETA_CKPT/$LOW_CKPT_NAME"
    --seed "$SEED"
)
[ -n "$MAX_PAIRS" ] && CMD+=(--max_pairs "$MAX_PAIRS")

echo "[run] ${CMD[*]}"
"${CMD[@]}"
