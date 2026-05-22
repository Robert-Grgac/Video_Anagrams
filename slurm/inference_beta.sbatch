#!/bin/bash
#SBATCH --job-name=wan-beta-infer
#SBATCH --partition=dmb
#SBATCH --nodelist=ctit092
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=/home/s2710099/logs/wan-beta-infer-%j.out
#SBATCH --error=/home/s2710099/logs/wan-beta-infer-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams

set -euo pipefail

# Activate conda environment (interpreter: ~/.conda/envs/wan22/bin/python).
# ctit092 is an A40 node — flash-attn works, use the regular wan22 env.
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
#
# Defaults below run beta-007_v2 at two weights (0.0 and 1.0) on the same seed,
# so the two outputs differ only in the CN contribution. WEIGHTS="0.0,1.0"
# triggers the in-script Cartesian sweep, producing one mp4 per (face, weight)
# pair with a '_w{weight}' suffix; the seed is shared across the sweep.
# DYNAMIC_CN_END=1 mirrors how v2 was trained (cn_end_fraction≈0.14).
export CHECKPOINT=${CHECKPOINT:-$WAN_BETA_CKPT/beta-007_v2_final.safetensors}
export RUN_TAG=${RUN_TAG:-beta-007_v2_w0_vs_w1}      # used as the output basename
export FACE_IDXS=${FACE_IDXS:-"0 25 50 75 99"}       # space-separated; one mp4 per face_idx
export SLUG=${SLUG:-}                                # blank → first manifest slug per face_idx
export OUTPUT_DIR=${OUTPUT_DIR:-$WAN_BETA_OUT/$RUN_TAG}
export STEPS=${STEPS:-100}
export GUIDANCE=${GUIDANCE:-5.0}
export CN_WEIGHT=${CN_WEIGHT:-1.0}                   # ignored when WEIGHTS is set
export CN_STRIDE=${CN_STRIDE:-3}                     # CN-residual layer stride. 3 matches how
                                                     # beta-007_v2 was trained. Setting 1 packs
                                                     # all 6 CN residuals into the first 6
                                                     # transformer blocks (OOD vs. training —
                                                     # the CN's output projections were
                                                     # calibrated for stride=3 block positions).
export WEIGHTS=${WEIGHTS:-"0.0,1.0"}                 # comma-sep sweep; default = A/B at same seed
export CN_START=${CN_START:-0.0}                     # ControlNet only on high-noise expert
export CN_END=${CN_END:-0.125}                       # = 1 - boundary_ratio (0.875); ignored if DYNAMIC_CN_END=1
export ENDS=${ENDS:-}                                # comma-sep sweep over controlnet_guidance_end,
                                                     # e.g. "0.125,0.5,0.875,1.0".
                                                     # Combined with WEIGHTS as Cartesian product;
                                                     # one mp4 per (weight, end) pair (per face).
export DYNAMIC_CN_END=${DYNAMIC_CN_END:-1}           # 1 → auto-compute cn_end from the sigma
                                                     # trajectory (mirrors train_beta7). Overrides
                                                     # CN_END and ENDS. On by default to match how
                                                     # beta-007_v3 was trained.
export PROMPT_CACHE_PATH=${PROMPT_CACHE_PATH:-}      # .pt file with precomputed positive prompt
                                                     # embedding (mirrors training-time eval).
                                                     # Negative is re-encoded fresh at matching
                                                     # length so both sides line up. Leave unset:
                                                     # v3 cache and pipeline default both use 226.
export PROMPT_EMBED_MAX_LEN=${PROMPT_EMBED_MAX_LEN:-} # pad length for the re-encoded negative.
                                                     # Beta cache uses 226 (post bug fix); pipeline default is 226.
export SEED=${SEED:-42}

mkdir -p "$OUTPUT_DIR"

echo "[run] CHECKPOINT=$CHECKPOINT"
echo "[run] FACE_IDXS=$FACE_IDXS"
echo "[run] CN_WEIGHT=$CN_WEIGHT  DYNAMIC_CN_END=$DYNAMIC_CN_END"
echo "[run] OUTPUT_DIR=$OUTPUT_DIR"

# Single python invocation: pipeline builds ONCE, run_inference_beta loops over
# all face_idxs internally. Output filenames get a '_face{idx}' suffix.
CMD=(python -m training.run_inference_beta
    --checkpoint_path "$CHECKPOINT"
    --base_model_path "$WAN_MODEL"
    --controlnet_config_repo "$HED_CONFIG"
    --cache_dir "$WAN_BETA_CACHE"
    --output_path "$OUTPUT_DIR/${RUN_TAG}.mp4"
    --face_idxs "$FACE_IDXS"
    --num_inference_steps "$STEPS"
    --guidance_scale "$GUIDANCE"
    --controlnet_weight "$CN_WEIGHT"
    --controlnet_stride "$CN_STRIDE"
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
if [ "$DYNAMIC_CN_END" = "1" ]; then
    CMD+=(--dynamic_cn_end)
fi
if [ -n "$PROMPT_CACHE_PATH" ]; then
    CMD+=(--prompt_cache_path "$PROMPT_CACHE_PATH")
fi
if [ -n "$PROMPT_EMBED_MAX_LEN" ]; then
    CMD+=(--prompt_embed_max_len "$PROMPT_EMBED_MAX_LEN")
fi

"${CMD[@]}"
