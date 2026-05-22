#!/bin/bash
#SBATCH --job-name=wan-encode-faces
#SBATCH --partition=dmb
#SBATCH --nodelist=ctit092
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=/home/s2710099/logs/wan-encode-faces-%j.out
#SBATCH --error=/home/s2710099/logs/wan-encode-faces-%j.err
#SBATCH --chdir=/home/s2710099/projects/Video_Anagrams

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate wan22

export WAN_BETA_CACHE=${WAN_BETA_CACHE:-$HOME/cache/wan-beta}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

WAN_CACHE=$HOME/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers
export WAN_MODEL=${WAN_MODEL:-$(ls -d $WAN_CACHE/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')}
[ -z "$WAN_MODEL" ] && { echo "ERROR: Wan model not cached at $WAN_CACHE" >&2; exit 1; }
echo "[paths] WAN_MODEL=$WAN_MODEL"

export HEIGHT=${HEIGHT:-512}
export WIDTH=${WIDTH:-512}
export SEED=${SEED:-42}

echo "[run] raw_face_dir=$WAN_BETA_CACHE/raw_face"
echo "[run] output_dir =$WAN_BETA_CACHE/face_latent"

python encode_raw_faces.py \
    --base_model_path "$WAN_MODEL" \
    --raw_face_dir   "$WAN_BETA_CACHE/raw_face" \
    --output_dir     "$WAN_BETA_CACHE/face_latent" \
    --height "$HEIGHT" \
    --width  "$WIDTH" \
    --seed   "$SEED"
