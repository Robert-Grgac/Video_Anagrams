#!/bin/bash
#SBATCH --job-name=wan-beta-infer
#SBATCH --partition=dmb
#SBATCH --nodelist=ctit092
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/s2710099/logs/wan-beta-infer-%j.out
#SBATCH --error=/home/s2710099/logs/wan-beta-infer-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams

set -euo pipefail

# Activate conda environment (interpreter: ~/.conda/envs/wan22/bin/python)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate wan22

# Wan-beta paths — flat top-level convention
export WAN_BETA_CACHE=${WAN_BETA_CACHE:-$HOME/cache/wan-beta}
export WAN_BETA_CKPT=${WAN_BETA_CKPT:-$HOME/checkpoints/wan-beta}
export WAN_BETA_OUT=${WAN_BETA_OUT:-$HOME/outputs/wan-beta}

# HF cache (default location, populated by head node) and offline mode for compute nodes
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1

# Diffusers attention backend
export DIFFUSERS_ATTN_BACKEND="flash"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Reduce CUDA allocator fragmentation; helps with the two-transformer offload pattern
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Resolve cached snapshot dirs (avoids HF model_info() network call on offline nodes)
WAN_CACHE=$HOME/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers
HED_CACHE=$HOME/.cache/huggingface/hub/models--TheDenk--wan2.2-t2v-a14b-controlnet-hed-v1
export WAN_MODEL=${WAN_MODEL:-$(ls -d $WAN_CACHE/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')}
export HED_CONFIG=${HED_CONFIG:-$(ls -d $HED_CACHE/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')}
[ -z "$WAN_MODEL" ] && { echo "ERROR: Wan model not cached at $WAN_CACHE" >&2; exit 1; }
[ -z "$HED_CONFIG" ] && { echo "ERROR: HED config not cached at $HED_CACHE" >&2; exit 1; }
echo "[paths] WAN_MODEL=$WAN_MODEL"
echo "[paths] HED_CONFIG=$HED_CONFIG"

# Tunables — override on the sbatch command line, e.g.:
#   CHECKPOINT=$HOME/checkpoints/wan-beta/beta-001_step6000.safetensors sbatch slurm/inference_beta.sbatch
export CHECKPOINT=${CHECKPOINT:-$WAN_BETA_CKPT/beta-001_final.safetensors}
export FACE_IDX=${FACE_IDX:-0}
export SLUG=${SLUG:-}                                # blank → use first available for face_idx
export OUTPUT=${OUTPUT:-$WAN_BETA_OUT/beta-001_infer.mp4}
export STEPS=${STEPS:-100}
export GUIDANCE=${GUIDANCE:-5.0}
export CN_WEIGHT=${CN_WEIGHT:-1.0}
export WEIGHTS=${WEIGHTS:-}                          # comma-sep sweep, e.g. "1.0,1.5,2.5,4.0"
export CN_START=${CN_START:-0.0}                     # ControlNet only on high-noise expert
export CN_END=${CN_END:-0.125}                       # = 1 - boundary_ratio (0.875)
export ENDS=${ENDS:-}                                # comma-sep sweep over controlnet_guidance_end,
                                                     # e.g. "0.125,0.5,0.875,1.0".
                                                     # Combined with WEIGHTS as Cartesian product;
                                                     # one mp4 per (weight, end) pair.
export SEED=${SEED:-42}

mkdir -p "$(dirname "$OUTPUT")"

CMD=(python -m training.run_inference_beta
    --checkpoint_path "$CHECKPOINT"
    --base_model_path "$WAN_MODEL"
    --controlnet_config_repo "$HED_CONFIG"
    --cache_dir "$WAN_BETA_CACHE"
    --output_path "$OUTPUT"
    --face_idx "$FACE_IDX"
    --num_inference_steps "$STEPS"
    --guidance_scale "$GUIDANCE"
    --controlnet_weight "$CN_WEIGHT"
    --controlnet_guidance_start "$CN_START"
    --controlnet_guidance_end "$CN_END"
    --seed "$SEED"
)
if [ -n "$SLUG" ]; then
    CMD+=(--slug "$SLUG")
fi
if [ -n "$WEIGHTS" ]; then
    CMD+=(--weights "$WEIGHTS")
fi
if [ -n "$ENDS" ]; then
    CMD+=(--ends "$ENDS")
fi

"${CMD[@]}"
