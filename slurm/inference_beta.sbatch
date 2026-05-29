#!/bin/bash
#SBATCH --job-name=wan-beta-infer
#SBATCH --partition=dmb
# Only hpc-node07 / hpc-node09 (L40 48GB) and ctit092 (A40 45GB) are suitable.
# Exclude ctit084/085 (RTX6000 24GB, too little VRAM) and hpc-node31 (Blackwell,
# needs the wan22-bw env). Slurm picks whichever of the three is free.
#SBATCH --exclude=ctit084,ctit085,hpc-node31
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=160G
#SBATCH --output=/home/s2710099/logs/wan-beta-infer-%j.out
#SBATCH --error=/home/s2710099/logs/wan-beta-infer-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams

set -euo pipefail

# ============================================================================
# Local scratch (per HPC admin: do NOT do heavy I/O on /home).
# Reading the Wan 2.2 A14B shards over NFS takes 30-40 s/shard AND repeatedly
# triggers external SIGKILL even with --mem=160G. Staging the snapshot to
# /local/$SLURM_JOBID first cuts shard load time to ~7 s and avoids the NFS
# watchdog issue entirely.
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

# Activate conda environment (interpreter: ~/.conda/envs/wan22/bin/python).
source $(conda info --base)/etc/profile.d/conda.sh
conda activate wan22

# Wan-beta paths on /home (read once during stage-in)
export WAN_BETA_CACHE_HOME=${WAN_BETA_CACHE_HOME:-$HOME/cache/wan-beta}
export WAN_BETA_CKPT_HOME=${WAN_BETA_CKPT_HOME:-$HOME/checkpoints/wan-beta}
export WAN_BETA_OUT=${WAN_BETA_OUT:-$HOME/outputs/wan-beta}

# Resolve cached snapshot dirs (avoids HF model_info() network call on offline nodes)
WAN_CACHE_HOME=$HOME/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers
HED_CACHE_HOME=$HOME/.cache/huggingface/hub/models--TheDenk--wan2.2-t2v-a14b-controlnet-hed-v1
WAN_SRC=$(ls -d $WAN_CACHE_HOME/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
HED_SRC=$(ls -d $HED_CACHE_HOME/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')
[ -z "$WAN_SRC" ] && { echo "ERROR: Wan model not cached at $WAN_CACHE_HOME" >&2; exit 1; }
[ -z "$HED_SRC" ] && { echo "ERROR: HED config not cached at $HED_CACHE_HOME" >&2; exit 1; }

# Checkpoint name (resolved on /home, staged below). Overridable from the
# sbatch command line via CHECKPOINT=<absolute path>.
export CHECKPOINT_HOME=${CHECKPOINT:-$WAN_BETA_CKPT_HOME/beta-007_v2_final.safetensors}
[ -f "$CHECKPOINT_HOME" ] \
    || { echo "ERROR: checkpoint $CHECKPOINT_HOME not found" >&2; exit 1; }

# ============================================================================
# Stage inputs to local scratch
# ----------------------------------------------------------------------------
# cp -aL follows symlinks (HF hub stores snapshots/<rev>/ as symlinks into
# blobs/<hash>) so the staged copies are real files, not pointers back to /home.
# ============================================================================
mkdir -p "$ScratchDir/wan_model" \
         "$ScratchDir/hed_config" \
         "$ScratchDir/wan-beta-cache" \
         "$ScratchDir/wan-beta-ckpt" \
         "$ScratchDir/hf_home" \
         "$ScratchDir/torch" \
         "$ScratchDir/wandb"

echo "[stage] copying Wan snapshot ($WAN_SRC) -> $ScratchDir/wan_model"
cp -aL "$WAN_SRC"/. "$ScratchDir/wan_model"/

echo "[stage] copying HED controlnet config ($HED_SRC) -> $ScratchDir/hed_config"
cp -aL "$HED_SRC"/. "$ScratchDir/hed_config"/

echo "[stage] copying beta cache ($WAN_BETA_CACHE_HOME) -> $ScratchDir/wan-beta-cache"
cp -aL "$WAN_BETA_CACHE_HOME"/. "$ScratchDir/wan-beta-cache"/

echo "[stage] copying checkpoint $(basename $CHECKPOINT_HOME) -> $ScratchDir/wan-beta-ckpt"
cp -L "$CHECKPOINT_HOME" "$ScratchDir/wan-beta-ckpt/$(basename $CHECKPOINT_HOME)"

echo "[stage] staged size:"
du -sh "$ScratchDir"/* 2>/dev/null || true

# ============================================================================
# Re-bind all heavy-I/O paths to scratch
# ============================================================================
export WAN_MODEL="$ScratchDir/wan_model"
export HED_CONFIG="$ScratchDir/hed_config"
export WAN_BETA_CACHE="$ScratchDir/wan-beta-cache"
export WAN_BETA_CKPT="$ScratchDir/wan-beta-ckpt"
export CHECKPOINT="$WAN_BETA_CKPT/$(basename $CHECKPOINT_HOME)"

# HF / Torch / wandb caches on local disk
export HF_HOME="$ScratchDir/hf_home"
export HF_HUB_CACHE="$ScratchDir/hf_home/hub"
export TRANSFORMERS_CACHE="$ScratchDir/hf_home"
export HF_HUB_OFFLINE=1
export TORCH_HOME="$ScratchDir/torch"
export WANDB_DIR="$ScratchDir/wandb"
export WANDB_MODE=${WANDB_MODE:-online}

# Diffusers attention backend
export DIFFUSERS_ATTN_BACKEND="flash"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Reduce CUDA allocator fragmentation; helps with the two-transformer offload pattern
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[paths] WAN_MODEL=$WAN_MODEL"
echo "[paths] HED_CONFIG=$HED_CONFIG"
echo "[paths] CHECKPOINT=$CHECKPOINT"
echo "[paths] WAN_BETA_CACHE=$WAN_BETA_CACHE"
echo "[paths] HF_HOME=$HF_HOME  TORCH_HOME=$TORCH_HOME  WANDB_DIR=$WANDB_DIR"

# Tunables — override on the sbatch command line, e.g.:
#   CHECKPOINT=$HOME/checkpoints/wan-beta/beta-001_step6000.safetensors sbatch slurm/inference_beta.sbatch
#
# Defaults below run beta-007_v2 at two weights (0.0 and 1.0) on the same seed,
# so the two outputs differ only in the CN contribution. WEIGHTS="0.0,1.0"
# triggers the in-script Cartesian sweep, producing one mp4 per (face, weight)
# pair with a '_w{weight}' suffix; the seed is shared across the sweep.
# DYNAMIC_CN_END=1 mirrors how v2 was trained (cn_end_fraction≈0.14).
# Note: CHECKPOINT (the override target) is captured up top as CHECKPOINT_HOME
# and re-exported below pointing at the staged copy in $ScratchDir. Don't
# re-export it here or it'll clobber the staged path.
export RUN_TAG=${RUN_TAG:-beta-007_v2_w0_vs_w1}      # used as the output basename
export FACE_IDXS=${FACE_IDXS:-"0 25 50 75 99"}       # space-separated; one mp4 per face_idx
export SLUG=${SLUG:-}                                # blank → canonical declaration-order slug per face_idx
                                                     # (face_i ↔ PROMPTS_BATCH_1+BATCH_2[i], e.g.
                                                     # face_0↔snowy_mountain). Matches the pairing in
                                                     # inference/run_inference.py and the WanPTD
                                                     # baseline runner. Set SLUG explicitly to override.
export OUTPUT_DIR=${OUTPUT_DIR:-$WAN_BETA_OUT/$RUN_TAG}
export STEPS=${STEPS:-100}
export NUM_FRAMES=${NUM_FRAMES:-9}                   # clip length. Script default is 9;
                                                     # set 61 to match the vanilla/PTD/CN+PTD
                                                     # sets for a fair quantitative comparison.
export GUIDANCE=${GUIDANCE:-5.0}
export CN_WEIGHT=${CN_WEIGHT:-1.0}                   # ignored when WEIGHTS is set
export CN_STRIDE=${CN_STRIDE:-3}                     # CN-residual layer stride. 3 matches how
                                                     # beta-007_v2 was trained. Setting 1 packs
                                                     # all 6 CN residuals into the first 6
                                                     # transformer blocks (OOD vs. training —
                                                     # the CN's output projections were
                                                     # calibrated for stride=3 block positions).
export CONTROL_SUBDIR=${CONTROL_SUBDIR:-canny}       # which subdir of WAN_BETA_CACHE holds the
                                                     # (3, H, W) uint8 .pt control inputs. 'canny'
                                                     # is the script default. 'raw_face' matches
                                                     # what inference/run_inference.py feeds the
                                                     # CN (i.e. apples-to-apples with the combined
                                                     # CN+PTD pipeline). 'silhouette' matches what
                                                     # beta-008 was actually trained on.
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
echo "[run] CONTROL_SUBDIR=$CONTROL_SUBDIR"
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
    --num_frames "$NUM_FRAMES"
    --num_inference_steps "$STEPS"
    --guidance_scale "$GUIDANCE"
    --controlnet_weight "$CN_WEIGHT"
    --controlnet_stride "$CN_STRIDE"
    --controlnet_guidance_start "$CN_START"
    --controlnet_guidance_end "$CN_END"
    --control_subdir "$CONTROL_SUBDIR"
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
