#!/bin/bash
# ============================================================
# prestage_models.sh
#
# Downloads model weights to ~/work/hf_models/ (GPFS data
# filesystem at /gpfs/data/s001/ssubram7/).
#
# Run ONCE on login node (vhpcprdssh01) which has internet.
# Compute nodes (g009 etc.) likely cannot reach HuggingFace.
#
# Usage:
#   bash prestage_models.sh
# ============================================================

set -euo pipefail

GEN_MODEL="Qwen/Qwen2.5-7B-Instruct"
VAL_MODEL="Qwen/Qwen2.5-14B-Instruct"

# ~/work symlinks to /gpfs/data/s001/ssubram7
# Larger quota than HOME, same GPFS performance.
MODEL_CACHE="$HOME/work/hf_models"

echo "========================================================"
echo "  Model Pre-staging"
echo "  Cache: $MODEL_CACHE"
echo "  Real path: $(readlink -f "$MODEL_CACHE" 2>/dev/null || echo "$MODEL_CACHE")"
echo "========================================================"

mkdir -p "$MODEL_CACHE/hub"

# Activate the dataset_gen environment
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$HOME/work/conda_envs/dataset_gen"

export HF_HOME="$MODEL_CACHE"
export TRANSFORMERS_CACHE="$MODEL_CACHE/hub"
export HF_HUB_CACHE="$MODEL_CACHE/hub"

echo ""
echo "[$(date +%H:%M:%S)] Downloading $GEN_MODEL..."
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('$GEN_MODEL', cache_dir='$MODEL_CACHE/hub')
print(f'  Stored at: {path}')
"

echo ""
echo "[$(date +%H:%M:%S)] Downloading $VAL_MODEL..."
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('$VAL_MODEL', cache_dir='$MODEL_CACHE/hub')
print(f'  Stored at: {path}')
"

echo ""
echo "========================================================"
echo "  Done. Disk usage:"
du -sh "$MODEL_CACHE/hub/models--"* 2>/dev/null || du -sh "$MODEL_CACHE"
echo ""
echo "  Total:"
du -sh "$MODEL_CACHE"
echo "========================================================"
