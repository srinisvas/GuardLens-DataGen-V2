#!/bin/bash
#SBATCH --job-name=v11_gen
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gen_%j.out
#SBATCH --error=logs/gen_%j.err

# =========================================================
# launch_gen.slurm — Batch 1: Generation
#
# Generates ~1,000 base pairs with Qwen-7B via vLLM,
# 1 paraphrase variant, builtin seeds, augmentation,
# pre-validation dedup. No validation (that's batch 2).
#
# Output: output/semantic_multiturn_v11_raw.jsonl
#
# Usage:
#   sbatch launch_gen.slurm
#
# With external seeds:
#   sbatch --export=SEED_FILES="/path/to/seeds.jsonl" launch_gen.slurm
# =========================================================

set -euo pipefail

# --- Configuration ---
N_PAIRS="${N_PAIRS:-1000}"
PARAPHRASE_VARIANTS="${PARAPHRASE_VARIANTS:-1}"
RANDOM_SEED="${RANDOM_SEED:-42}"
GEN_MODEL="${GEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
BACKEND="${BACKEND:-vllm}"
OUTPUT_NAME="${OUTPUT_NAME:-semantic_multiturn_v11}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
MISLEADING_FRACTION="${MISLEADING_FRACTION:-0.10}"
VLLM_PORT="${VLLM_PORT:-8000}"
SEED_FILES="${SEED_FILES:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${OUTPUT_DIR}" logs

echo "============================================"
echo "Batch 1: Dataset Generation (v11)"
echo "============================================"
echo "  N_PAIRS:              ${N_PAIRS}"
echo "  PARAPHRASE_VARIANTS:  ${PARAPHRASE_VARIANTS}"
echo "  GEN_MODEL:            ${GEN_MODEL}"
echo "  BACKEND:              ${BACKEND}"
echo "  MISLEADING_FRACTION:  ${MISLEADING_FRACTION}"
echo "  RANDOM_SEED:          ${RANDOM_SEED}"
echo "  OUTPUT_DIR:           ${OUTPUT_DIR}"
echo "  SEED_FILES:           ${SEED_FILES:-none}"
echo "  SLURM_JOB_ID:         ${SLURM_JOB_ID:-local}"
echo "============================================"

# --- Step 1: Start vLLM server ---
if [ "${BACKEND}" = "vllm" ]; then
    echo "[$(date)] Starting vLLM server on port ${VLLM_PORT}..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${GEN_MODEL}" \
        --port "${VLLM_PORT}" \
        --gpu-memory-utilization 0.90 \
        --max-model-len 4096 \
        --dtype auto \
        &
    VLLM_PID=$!
    echo "  vLLM PID: ${VLLM_PID}"

    # Wait for server
    echo "  Waiting for vLLM to be ready..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo "  vLLM ready after ${i}s"
            break
        fi
        if [ $i -eq 120 ]; then
            echo "ERROR: vLLM failed to start after 120s"
            kill ${VLLM_PID} 2>/dev/null || true
            exit 1
        fi
        sleep 1
    done
fi

# --- Step 2: Run generation ---
echo ""
echo "[$(date)] Starting generation..."

SEED_ARGS=""
if [ -n "${SEED_FILES}" ]; then
    SEED_ARGS="--seed-files ${SEED_FILES}"
fi

python3 "${SCRIPT_DIR}/run_hpc.py" \
    --mode gen \
    --n-pairs "${N_PAIRS}" \
    --paraphrase-variants "${PARAPHRASE_VARIANTS}" \
    --seed "${RANDOM_SEED}" \
    --output-dir "${OUTPUT_DIR}" \
    --output-name "${OUTPUT_NAME}" \
    --backend "${BACKEND}" \
    --model "${GEN_MODEL}" \
    --base-url "http://localhost:${VLLM_PORT}" \
    --misleading-fraction "${MISLEADING_FRACTION}" \
    --llm-annotator \
    ${SEED_ARGS}

GEN_STATUS=$?
RAW_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}_raw.jsonl"

if [ ${GEN_STATUS} -ne 0 ] || [ ! -f "${RAW_FILE}" ]; then
    echo "ERROR: Generation failed (exit ${GEN_STATUS})"
    [ "${BACKEND}" = "vllm" ] && kill ${VLLM_PID} 2>/dev/null || true
    exit 1
fi

RAW_COUNT=$(wc -l < "${RAW_FILE}")
echo "[$(date)] Generation complete: ${RAW_COUNT} records in ${RAW_FILE}"

# --- Step 3: Run augmentation ---
echo ""
echo "[$(date)] Running augmentation..."

AUGMENTED_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}_augmented.jsonl"

python3 "${SCRIPT_DIR}/augment_dataset.py" \
    --input "${RAW_FILE}" \
    --output "${AUGMENTED_FILE}" \
    --n-hard-negatives 150 \
    --n-borderline 80 \
    --noise-fraction 0.30 \
    --blur-fraction 0.30 \
    --false-lead-fraction 0.20

AUG_STATUS=$?
if [ ${AUG_STATUS} -ne 0 ]; then
    echo "WARNING: Augmentation failed (exit ${AUG_STATUS}), using raw file"
    cp "${RAW_FILE}" "${AUGMENTED_FILE}"
fi

AUG_COUNT=$(wc -l < "${AUGMENTED_FILE}")
echo "[$(date)] Augmentation complete: ${AUG_COUNT} records"

# --- Step 3b: Post-augmentation dedup ---
echo ""
echo "[$(date)] Running post-augmentation dedup..."

DEDUP_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}_augmented_dedup.jsonl"

python3 -c "
import sys, json
sys.path.insert(0, '${SCRIPT_DIR}')
import build_semantic_datasetv11 as gen

records = []
with open('${AUGMENTED_FILE}') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

before = len(records)
records = gen.deduplicate_dataset(records, threshold=0.70)
gen.write_jsonl(records, '${DEDUP_FILE}')
print(f'Post-augmentation dedup: {before} -> {len(records)} ({before - len(records)} removed)')
"

DEDUP_STATUS=$?
if [ ${DEDUP_STATUS} -ne 0 ]; then
    echo "WARNING: Post-augmentation dedup failed, using augmented file as-is"
    cp "${AUGMENTED_FILE}" "${DEDUP_FILE}"
fi

FINAL_COUNT=$(wc -l < "${DEDUP_FILE}")
echo "[$(date)] Final output: ${FINAL_COUNT} records"

# --- Step 4: Shutdown vLLM ---
if [ "${BACKEND}" = "vllm" ]; then
    echo "[$(date)] Shutting down vLLM..."
    kill ${VLLM_PID} 2>/dev/null || true
    wait ${VLLM_PID} 2>/dev/null || true
fi

# --- Summary ---
echo ""
echo "============================================"
echo "Batch 1 Complete"
echo "============================================"
echo "  Raw records:       ${RAW_COUNT}"
echo "  After augmentation: ${AUG_COUNT}"
echo "  After dedup:       ${FINAL_COUNT}"
echo "  Output file:       ${DEDUP_FILE}"
echo "  Next step:         sbatch launch_val.slurm"
echo "============================================"
echo "[$(date)] Done."
