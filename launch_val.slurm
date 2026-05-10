#!/bin/bash
#SBATCH --job-name=v11_val
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=36:00:00
#SBATCH --output=logs/val_%j.out
#SBATCH --error=logs/val_%j.err

# =========================================================
# launch_val.slurm — Batch 2: Validation
#
# Runs 100% conversation-level validation with Qwen-14B,
# counterfactual on pivot-turn spans, supervision tiers,
# checkpoint/resume support.
#
# Input:  output/semantic_multiturn_v11_augmented.jsonl
# Output: output/semantic_multiturn_v11_validated.jsonl
#
# Usage:
#   sbatch launch_val.slurm
#
# With custom input:
#   sbatch --export=INPUT_FILE="/path/to/raw.jsonl" launch_val.slurm
#
# Chain after generation:
#   GEN_JOB=$(sbatch --parsable launch_gen.slurm)
#   sbatch --dependency=afterok:${GEN_JOB} launch_val.slurm
# =========================================================

set -euo pipefail

# --- Configuration ---
VAL_MODEL="${VAL_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
BACKEND="${BACKEND:-vllm}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
OUTPUT_NAME="${OUTPUT_NAME:-semantic_multiturn_v11}"
INPUT_FILE="${INPUT_FILE:-${OUTPUT_DIR}/${OUTPUT_NAME}_augmented_dedup.jsonl}"
VLLM_PORT="${VLLM_PORT:-8000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25}"
USE_COUNTERFACTUAL="${USE_COUNTERFACTUAL:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${OUTPUT_DIR}" logs

echo "============================================"
echo "Batch 2: Dataset Validation (v11)"
echo "============================================"
echo "  VAL_MODEL:            ${VAL_MODEL}"
echo "  BACKEND:              ${BACKEND}"
echo "  INPUT_FILE:           ${INPUT_FILE}"
echo "  CHECKPOINT_INTERVAL:  ${CHECKPOINT_INTERVAL}"
echo "  USE_COUNTERFACTUAL:   ${USE_COUNTERFACTUAL}"
echo "  SLURM_JOB_ID:         ${SLURM_JOB_ID:-local}"
echo "============================================"

# --- Verify input exists ---
if [ ! -f "${INPUT_FILE}" ]; then
    # Fall back to augmented (non-dedup), then raw
    FALLBACK_AUG="${OUTPUT_DIR}/${OUTPUT_NAME}_augmented.jsonl"
    FALLBACK_RAW="${OUTPUT_DIR}/${OUTPUT_NAME}_raw.jsonl"
    if [ -f "${FALLBACK_AUG}" ]; then
        echo "WARNING: ${INPUT_FILE} not found, using ${FALLBACK_AUG}"
        INPUT_FILE="${FALLBACK_AUG}"
    elif [ -f "${FALLBACK_RAW}" ]; then
        echo "WARNING: ${INPUT_FILE} not found, using ${FALLBACK_RAW}"
        INPUT_FILE="${FALLBACK_RAW}"
    else
        echo "ERROR: No input file found"
        exit 1
    fi
fi

INPUT_COUNT=$(wc -l < "${INPUT_FILE}")
echo "  Input records: ${INPUT_COUNT}"

# --- Check for existing checkpoint (resume support) ---
VALIDATED_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}_validated.jsonl"
CHECKPOINT_FILE="${VALIDATED_FILE}.checkpoint"
if [ -f "${CHECKPOINT_FILE}" ]; then
    CKPT_COUNT=$(wc -l < "${CHECKPOINT_FILE}")
    echo "  Found checkpoint with ${CKPT_COUNT} validated records — will resume"
fi

# --- Step 1: Start vLLM server with 14B model ---
if [ "${BACKEND}" = "vllm" ]; then
    echo ""
    echo "[$(date)] Starting vLLM server with ${VAL_MODEL}..."

    # 14B needs more memory; use higher utilization
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${VAL_MODEL}" \
        --port "${VLLM_PORT}" \
        --gpu-memory-utilization 0.92 \
        --max-model-len 4096 \
        --dtype auto \
        &
    VLLM_PID=$!
    echo "  vLLM PID: ${VLLM_PID}"

    # Wait for server (14B takes longer to load)
    echo "  Waiting for vLLM to be ready..."
    for i in $(seq 1 300); do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo "  vLLM ready after ${i}s"
            break
        fi
        if [ $i -eq 300 ]; then
            echo "ERROR: vLLM failed to start after 300s"
            kill ${VLLM_PID} 2>/dev/null || true
            exit 1
        fi
        sleep 1
    done
fi

# --- Step 2: Run validation ---
echo ""
echo "[$(date)] Starting validation..."

CF_FLAG=""
if [ "${USE_COUNTERFACTUAL}" = "true" ]; then
    CF_FLAG="--counterfactual"
fi

python3 "${SCRIPT_DIR}/run_hpc.py" \
    --mode val \
    --input "${INPUT_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --output-name "${OUTPUT_NAME}" \
    --backend "${BACKEND}" \
    --model "${VAL_MODEL}" \
    --base-url "http://localhost:${VLLM_PORT}" \
    --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
    ${CF_FLAG}

VAL_STATUS=$?

# --- Step 3: Shutdown vLLM ---
if [ "${BACKEND}" = "vllm" ]; then
    echo "[$(date)] Shutting down vLLM..."
    kill ${VLLM_PID} 2>/dev/null || true
    wait ${VLLM_PID} 2>/dev/null || true
fi

if [ ${VAL_STATUS} -ne 0 ]; then
    echo "ERROR: Validation failed (exit ${VAL_STATUS})"
    echo "  Checkpoint at: ${CHECKPOINT_FILE}"
    echo "  Re-submit this job to resume from checkpoint."
    exit 1
fi

# --- Step 4: Print validation statistics ---
echo ""
echo "[$(date)] Computing validation statistics..."

python3 -c "
import json, sys
from collections import Counter

records = []
with open('${VALIDATED_FILE}') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

total = len(records)
labels = Counter(r.get('label', -1) for r in records)
statuses = Counter(r.get('validation_status', 'unknown') for r in records)
tiers = Counter(r.get('supervision_tier', 'unknown') for r in records)
pivots = Counter(r.get('pivot_kind', 'none') for r in records)
families = Counter(r.get('family', 'unknown') for r in records)
eligible = sum(1 for r in records if r.get('training_eligible', True))

print(f'Total records: {total}')
print(f'Labels: {dict(labels)}')
print(f'Validation status: {dict(statuses)}')
print(f'Supervision tiers: {dict(tiers)}')
print(f'Pivot kinds: {dict(pivots)}')
print(f'Training eligible: {eligible}/{total}')
print(f'Families: {dict(families)}')

# Loss weight distribution
weights = [r.get('loss_weight', 0.5) for r in records if r.get('training_eligible', True)]
if weights:
    avg_w = sum(weights) / len(weights)
    print(f'Avg loss weight (eligible): {avg_w:.3f}')
"

# --- Summary ---
echo ""
echo "============================================"
echo "Batch 2 Complete"
echo "============================================"
echo "  Validated file: ${VALIDATED_FILE}"
echo "  Next steps:"
echo "    1. python3 mhj_loader.py --input mhj.jsonl --output ${OUTPUT_DIR}/${OUTPUT_NAME}_mhj.jsonl"
echo "    2. python3 split_dataset.py --input ${VALIDATED_FILE} --output-dir ${OUTPUT_DIR}/splits/"
echo "============================================"
echo "[$(date)] Done."
