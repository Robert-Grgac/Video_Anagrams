#!/bin/bash
#SBATCH --job-name=wan-eval
#SBATCH --partition=dmb
# Eval models are tiny (<=2GB), so any non-Blackwell dmb GPU node works.
# Exclude hpc-node31 (Blackwell -> needs wan22-bw env); the 24GB Turing nodes
# ctit084/085 are fine here.
#SBATCH --exclude=hpc-node31
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --output=/home/s2710099/logs/wan-eval-%j.out
#SBATCH --error=/home/s2710099/logs/wan-eval-%j.err
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
# Conda env (non-Blackwell; all eval deps installed here)
# ============================================================================
source $(conda info --base)/etc/profile.d/conda.sh
conda activate wan22

# ============================================================================
# Resolve eval model caches on /home (small files, ~3GB total). Each is staged
# to scratch and the corresponding env var is re-pointed so the run is offline.
# Missing caches are warned about, not fatal: the per-metric loop (set +e) lets
# the metrics that DO have their model still run.
# ============================================================================
VICLIP_SRC=""
for c in "$HOME/checkpoints/viclip/ViClip-InternVid-10M-FLT.pth" \
         "analysis/third_party/viclip/ViClip-InternVid-10M-FLT.pth"; do
    [ -f "$c" ] && { VICLIP_SRC="$c"; break; }
done
INSIGHTFACE_SRC="$HOME/.insightface"
CLIP_SRC="$HOME/.cache/clip"
TORCH_HUB_SRC="$HOME/.cache/torch"
# LAION-AI/aesthetic-predictor V1 head (Linear 768->1 on CLIP ViT-L/14).
AESTHETIC_SRC=""
for c in "$HOME/src/aesthetic-predictor/sa_0_4_vit_l_14_linear.pth" \
         "$HOME/checkpoints/aesthetic/sa_0_4_vit_l_14_linear.pth" \
         "analysis/third_party/aesthetic/sa_0_4_vit_l_14_linear.pth"; do
    [ -f "$c" ] && { AESTHETIC_SRC="$c"; break; }
done

mkdir -p "$ScratchDir/viclip" "$ScratchDir/insightface" "$ScratchDir/clip" \
         "$ScratchDir/torch" "$ScratchDir/aesthetic"

if [ -n "$VICLIP_SRC" ]; then
    echo "[stage] ViCLIP ckpt ($VICLIP_SRC)"
    cp -L "$VICLIP_SRC" "$ScratchDir/viclip/ViClip-InternVid-10M-FLT.pth"
    export VICLIP_CKPT="$ScratchDir/viclip/ViClip-InternVid-10M-FLT.pth"
else
    echo "[warn] ViCLIP ckpt not cached -> viclip metric will fail (or use --backend openclip)"
fi

if [ -d "$INSIGHTFACE_SRC" ]; then
    echo "[stage] insightface ($INSIGHTFACE_SRC)"
    cp -aL "$INSIGHTFACE_SRC"/. "$ScratchDir/insightface"/
    export INSIGHTFACE_HOME="$ScratchDir/insightface"
else
    echo "[warn] ~/.insightface not found -> retinaface metric will fail"
fi

if [ -d "$CLIP_SRC" ]; then
    echo "[stage] openai CLIP cache ($CLIP_SRC)"
    cp -aL "$CLIP_SRC"/. "$ScratchDir/clip"/
    export CLIP_CACHE="$ScratchDir/clip"
else
    echo "[warn] ~/.cache/clip not found -> aesthetic metric will fail"
fi

if [ -d "$TORCH_HUB_SRC" ]; then
    echo "[stage] torch hub cache ($TORCH_HUB_SRC)"
    cp -aL "$TORCH_HUB_SRC"/. "$ScratchDir/torch"/
fi
export TORCH_HOME="$ScratchDir/torch"

if [ -n "$AESTHETIC_SRC" ]; then
    echo "[stage] aesthetic head ($AESTHETIC_SRC)"
    cp -L "$AESTHETIC_SRC" "$ScratchDir/aesthetic/sa_0_4_vit_l_14_linear.pth"
    export AESTHETIC_CKPT="$ScratchDir/aesthetic/sa_0_4_vit_l_14_linear.pth"
else
    echo "[warn] aesthetic head not cached -> aesthetic metric will fail"
fi

echo "[stage] staged size:"
du -sh "$ScratchDir"/* 2>/dev/null || true

# cd-fvd I3D ckpt ships inside the cdfvd package (read once from the env on
# /home); eval_fvd.py resolves it automatically. Set CDFVD_I3D_CKPT to override.
# ViCLIP bpe vocab is vendored in-repo (analysis/third_party/viclip/) and loaded
# from there, so it needs no staging.

export HF_HUB_OFFLINE=1
export DIFFUSERS_ATTN_BACKEND="native"
export WAN_FORCE_NATIVE_ATTN=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Method/vanilla mp4s and the raw-face cache are read directly from /home
# (sequential reads, allowed). Results (small CSV/JSON) written to /home.
echo "[paths] VICLIP_CKPT=${VICLIP_CKPT:-<none>}"
echo "[paths] INSIGHTFACE_HOME=${INSIGHTFACE_HOME:-<none>}  CLIP_CACHE=${CLIP_CACHE:-<none>}"
echo "[paths] TORCH_HOME=$TORCH_HOME  AESTHETIC_CKPT=${AESTHETIC_CKPT:-<none>}"

# ============================================================================
# Metric selection. Default = the GPU metrics. CPU-fine metrics
# (lpips_vanilla, lpips_face, retinaface) are normally run on the login node,
# but can be requested here too:
#   METRICS="viclip fvd aesthetic" sbatch slurm/eval.sbatch
#   METRICS="lpips_face retinaface" sbatch slurm/eval.sbatch
# Optional: EVAL_CONDITIONS="cn_only cn_ptd"  RESULTS_DIR=analysis/results
# ============================================================================
METRICS=${METRICS:-"viclip fvd aesthetic"}
RESULTS_DIR=${RESULTS_DIR:-analysis/results}
COND_ARGS=()
[ -n "${EVAL_CONDITIONS:-}" ] && COND_ARGS=(--conditions ${EVAL_CONDITIONS})

run_one() {
    local m="$1"
    case "$m" in
        viclip)        python -m analysis.eval_viclip       --device cuda --results_dir "$RESULTS_DIR" "${COND_ARGS[@]}" ;;
        fvd)           python -m analysis.eval_fvd           --device cuda --results_dir "$RESULTS_DIR" "${COND_ARGS[@]}" ;;
        aesthetic)     python -m analysis.eval_aesthetic     --device cuda --results_dir "$RESULTS_DIR" "${COND_ARGS[@]}" ;;
        lpips_vanilla) python -m analysis.eval_lpips_vanilla --device cuda --results_dir "$RESULTS_DIR" "${COND_ARGS[@]}" ;;
        lpips_face)    python -m analysis.eval_lpips_face    --device cuda --results_dir "$RESULTS_DIR" "${COND_ARGS[@]}" ;;
        retinaface)    python -m analysis.eval_retinaface    --device cpu  --results_dir "$RESULTS_DIR" "${COND_ARGS[@]}" ;;
        *) echo "[error] unknown metric: $m"; return 2 ;;
    esac
}

echo "[run] METRICS='$METRICS'  RESULTS_DIR='$RESULTS_DIR'  EVAL_CONDITIONS='${EVAL_CONDITIONS:-<all>}'"

declare -A STATUS
set +e
for m in $METRICS; do
    echo "==================== metric: $m ===================="
    run_one "$m"
    STATUS[$m]=$?
done
set -e

echo "==================== summary ===================="
overall=0
for m in $METRICS; do
    if [ "${STATUS[$m]:-1}" -eq 0 ]; then
        echo "PASS  $m"
    else
        echo "FAIL  $m (rc=${STATUS[$m]:-?})"
        overall=1
    fi
done

# Cleanup of $ScratchDir handled by the EXIT trap above.
exit $overall
