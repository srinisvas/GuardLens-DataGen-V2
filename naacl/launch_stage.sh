#!/bin/bash
# Shared environment setup. The Python supervisor owns all model processes.
set -euo pipefail
STAGE="${1:?expected b1, b2 or b4}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:?submit from the repository root}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/dataset_gen}"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_BATCH_INVARIANT=1
export MODEL_CACHE="${MODEL_CACHE:-$HOME/work/hf_models}"
export HF_HOME="$MODEL_CACHE"
export TRANSFORMERS_CACHE="$MODEL_CACHE/hub"
export HF_HUB_CACHE="$MODEL_CACHE/hub"
export HF_HUB_OFFLINE=1
cd "$PROJECT_DIR"
exec python3 -u naacl/launch_job.py "$STAGE"
