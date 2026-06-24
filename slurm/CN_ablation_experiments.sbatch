#!/bin/bash
#SBATCH --job-name=cn-ablation
#SBATCH --partition=dmb
# Only hpc-node07 / hpc-node09 (L40 48GB) and ctit092 (A40 45GB) are suitable.
# Exclude ctit084/085 (RTX6000 24GB, too little VRAM) and hpc-node31 (Blackwell,
# needs the wan22-bw env). Slurm picks whichever of the three is free.
#SBATCH --exclude=ctit084,ctit085,hpc-node31
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=160G
#SBATCH --output=/home/s2710099/logs/cn-ablation-%j.out
#SBATCH --error=/home/s2710099/logs/cn-ablation-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams

set -euo pipefail

# ============================================================================
# Runs training.CN_ablation_experiments: two consecutive passes over the same
# canonical 100 face<->slug pairs.
#   Pass A — single shared CN (beta-002) -> single_cn_dual_experts
#   Pass B — dual CN beta-007 high + beta-004 low -> dual_cn_dual_experts
# The two pipelines are never co-resident (pass A is torn down before pass B is
# built); within a pass enable_model_cpu_offload keeps only the active expert on
# the GPU. Video-output-only (no wandb / analytics). The script skips any mp4
# that already exists, so if this 48h job times out, re-submit the SAME sbatch
# to resume.
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
# Resolve source paths on /home (read once during stage-in).
# ----------------------------------------------------------------------------
WAN_CACHE_HOME=$HOME/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers
HED_CACHE_HOME=$HOME/.cache/huggingface/hub/models--TheDenk--wan2.2-t2v-a14b-controlnet-hed-v1
WAN_SRC=$(ls -d "$WAN_CACHE_HOME"/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
HED_SRC=$(ls -d "$HED_CACHE_HOME"/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
[ -z "$WAN_SRC" ] && { echo "ERROR: Wan model not cached at $WAN_CACHE_HOME" >&2; exit 1; }
[ -z "$HED_SRC" ] && { echo "ERROR: HED config not cached at $HED_CACHE_HOME" >&2; exit 1; }

BETA_CKPT_HOME=${WAN_BETA_CKPT_HOME:-$HOME/checkpoints/wan-beta}
BETA_CACHE_HOME=${WAN_BETA_CACHE_HOME:-$HOME/cache/wan-beta}

# Checkpoints (overridable on the command line).
export SINGLE_CKPT_NAME=${SINGLE_CKPT_NAME:-beta-002_final.safetensors}   # single shared CN
export HIGH_CKPT_NAME=${HIGH_CKPT_NAME:-beta-007_v2_final.safetensors}    # dual: high-noise CN
export LOW_CKPT_NAME=${LOW_CKPT_NAME:-beta-004_final.safetensors}         # dual: low-noise CN
for ck in "$SINGLE_CKPT_NAME" "$HIGH_CKPT_NAME" "$LOW_CKPT_NAME"; do
    [ -f "$BETA_CKPT_HOME/$ck" ] \
        || { echo "ERROR: checkpoint $BETA_CKPT_HOME/$ck not found" >&2; exit 1; }
done

# ----------------------------------------------------------------------------
# Stage inputs to local scratch. The CN runs only read canny/ + manifest.json
# from the beta cache (NOT the 44GB invert latents), so stage just those.
# cp -aL: follow HF blob symlinks -> real files.
# ----------------------------------------------------------------------------
mkdir -p "$ScratchDir/wan_model" \
         "$ScratchDir/hed_config" \
         "$ScratchDir/wan-beta-cache/canny" \
         "$ScratchDir/wan-beta-ckpt" \
         "$ScratchDir/hf_home" \
         "$ScratchDir/torch"

echo "[stage] copying Wan snapshot ($WAN_SRC) -> $ScratchDir/wan_model"
cp -aL "$WAN_SRC"/. "$ScratchDir/wan_model"/

echo "[stage] copying HED controlnet config ($HED_SRC) -> $ScratchDir/hed_config"
cp -aL "$HED_SRC"/. "$ScratchDir/hed_config"/

echo "[stage] copying canny + manifest from beta cache"
cp -aL "$BETA_CACHE_HOME/canny"/. "$ScratchDir/wan-beta-cache/canny"/
cp -L "$BETA_CACHE_HOME/manifest.json" "$ScratchDir/wan-beta-cache/manifest.json"

echo "[stage] copying checkpoints"
for ck in "$SINGLE_CKPT_NAME" "$HIGH_CKPT_NAME" "$LOW_CKPT_NAME"; do
    cp -L "$BETA_CKPT_HOME/$ck" "$ScratchDir/wan-beta-ckpt/$ck"
done

echo "[stage] staged size:"
du -sh "$ScratchDir"/* 2>/dev/null || true

# ----------------------------------------------------------------------------
# Re-bind heavy-I/O paths to scratch
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
# Reduce allocator fragmentation; helps the two-transformer offload pattern.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[paths] WAN_MODEL=$WAN_MODEL"
echo "[paths] HED_CONFIG=$HED_CONFIG"
echo "[paths] WAN_BETA_CACHE=$WAN_BETA_CACHE"
echo "[paths] WAN_BETA_CKPT=$WAN_BETA_CKPT"

# ----------------------------------------------------------------------------
# Tunables — override on the sbatch command line, e.g.:
#   MAX_PAIRS=2 sbatch slurm/CN_ablation_experiments.sbatch   # quick smoke
# Output dirs are fixed inside the script:
#   single_cn_dual_experts / dual_cn_dual_experts
# ----------------------------------------------------------------------------
export SEED=${SEED:-42}
export MAX_PAIRS=${MAX_PAIRS:-}   # blank = all 100 per pass
# SETUP=both (default) runs single then dual. Set SETUP=single|dual to run just
# one pass (e.g. to split single/dual across two local nodes).
export SETUP=${SETUP:-both}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/home/s2710099/outputs/inference}

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
if [ -n "$MAX_PAIRS" ]; then
    CMD+=(--max_pairs "$MAX_PAIRS")
fi

echo "[run] ${CMD[*]}"
"${CMD[@]}"

# Cleanup of $ScratchDir is handled by the EXIT trap above.
