#!/bin/bash
#SBATCH --job-name=v11_gen
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gen_%j.out
#SBATCH --error=logs/gen_%j.err

# =========================================================
# launch_gen.slurm — Batch 1: Generation (v11)
#
# Generates ~1,000 base pairs with Qwen-7B via vLLM,
# 1 paraphrase variant, builtin seeds, augmentation,
# pre-validation dedup. No validation (that's batch 2).
#
# Usage:
#   sbatch launch_gen.slurm
#   N_PAIRS=20 sbatch launch_gen.slurm          # smoke test
#   sbatch --export=SEED_FILES="seeds.jsonl" launch_gen.slurm
# =========================================================

set -euo pipefail

# ==================== CONFIGURATION =======================

# --- Generation ---
N_PAIRS="${N_PAIRS:-1000}"
PARAPHRASE_VARIANTS="${PARAPHRASE_VARIANTS:-1}"
RANDOM_SEED="${RANDOM_SEED:-42}"
MISLEADING_FRACTION="${MISLEADING_FRACTION:-0.10}"
SEED_FILES="${SEED_FILES:-}"

# --- Model ---
GEN_MODEL="${GEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
BACKEND="${BACKEND:-vllm}"
PORT="${VLLM_PORT:-8000}"

# --- Storage ---
MODEL_CACHE="${MODEL_CACHE:-$HOME/work/hf_models}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/staging/dataset_gen_output}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/work/results/dataset_gen}"
OUTPUT_NAME="${OUTPUT_NAME:-semantic_multiturn_v11}"

# --- Environment ---
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/dataset_gen}"

# --- Paths ---
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SCRIPT_DIR="${PROJECT_DIR}"
LOG_DIR="$PROJECT_DIR/logs"

# ==================== END CONFIGURATION ===================

VLLM_PID=""

echo "========================================================"
echo "  Batch 1: Dataset Generation (v11)"
echo "  Job:          ${SLURM_JOB_ID:-interactive}"
echo "  Node:         $(hostname)"
echo "  Gen model:    $GEN_MODEL"
echo "  N pairs:      $N_PAIRS"
echo "  Model cache:  $MODEL_CACHE"
echo "  Output:       $OUTPUT_DIR"
echo "  Results:      $RESULTS_DIR"
echo "========================================================"


# ==================== ENVIRONMENT =========================

# Try loading CUDA module
if command -v module &>/dev/null; then
    for mod in cuda/12.1 cuda/12.2 cuda/12.4 cuda/12 cuda; do
        module load "$mod" 2>/dev/null && break
    done
fi

# Find CUDA_HOME if not set by module
if [[ -z "${CUDA_HOME:-}" ]]; then
    for p in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.1 /opt/cuda; do
        if [[ -f "$p/bin/nvcc" ]]; then
            export CUDA_HOME="$p"
            export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
            break
        fi
    done
fi

# Initialize conda and activate
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/.conda")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Redirect temp/cache off HOME
export TMPDIR="$HOME/work/.tmp"
export PIP_CACHE_DIR="$HOME/work/.pip_cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" 2>/dev/null || true

# Lock HF_HOME to model cache BEFORE any Python import
export HF_HOME="$MODEL_CACHE"
export TRANSFORMERS_CACHE="$MODEL_CACHE/hub"
export HF_HUB_CACHE="$MODEL_CACHE/hub"

# cd to project directory (SLURM may start in spool dir)
cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR" "$LOG_DIR" 2>/dev/null || true

echo "HF_HOME locked to: $HF_HOME"
echo "CUDA_HOME: ${CUDA_HOME:-not set}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "GPU: not detected (will fail at vLLM start)"
echo "System RAM: $(free -h | awk '/^Mem:/{print $2}') total, $(free -h | awk '/^Mem:/{print $7}') available"
echo ""


# ==================== CLEANUP TRAP ========================

stop_vllm() {
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] Stopping vLLM (pid $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
        sleep 5
    fi
}

cleanup() {
    local exit_code=$?
    echo ""
    echo "[$(date +%H:%M:%S)] Cleanup (exit code: $exit_code)..."
    stop_vllm

    # Save partial results on unexpected exit
    if [[ $exit_code -ne 0 ]]; then
        local partial="${OUTPUT_DIR}/${OUTPUT_NAME}_raw.jsonl"
        if [[ -f "$partial" ]]; then
            echo "  Saving partial results to $RESULTS_DIR..."
            cp "$partial" "$RESULTS_DIR/${OUTPUT_NAME}_PARTIAL.jsonl" 2>/dev/null || true
        fi
    fi
    echo "  Done."
}
trap cleanup EXIT


# ==================== PRE-FLIGHT CHECKS ===================

echo "--- Storage pre-flight checks ---"

# Check model cache exists
if [[ ! -d "$MODEL_CACHE" ]]; then
    echo "ERROR: Model cache not found: $MODEL_CACHE"
    echo "Run: bash prestage_models.sh (on login node)"
    exit 1
fi
echo "  Model cache: $(du -sh "$MODEL_CACHE" 2>/dev/null | cut -f1) at $MODEL_CACHE"

# Check model directory exists
GEN_DIR_NAME="models--$(echo "$GEN_MODEL" | sed 's/\//--/g')"
if [[ ! -d "$MODEL_CACHE/hub/$GEN_DIR_NAME" ]]; then
    echo "ERROR: $GEN_MODEL not found in cache."
    echo "  Expected: $MODEL_CACHE/hub/$GEN_DIR_NAME"
    echo "  Run: bash prestage_models.sh"
    exit 1
fi
echo "  $GEN_MODEL: $(du -sh "$MODEL_CACHE/hub/$GEN_DIR_NAME" 2>/dev/null | cut -f1)"

# Check disk space
for dir_label_path in "Output:$OUTPUT_DIR" "Results:$RESULTS_DIR"; do
    label="${dir_label_path%%:*}"
    dir="${dir_label_path##*:}"
    avail=$(df --output=avail -BG "$dir" 2>/dev/null | tail -1 | tr -d ' G')
    echo "  $label dir ($dir): ${avail:-?}G available"
    if [[ "${avail:-0}" -lt 5 ]]; then
        echo "  WARNING: Less than 5GB free."
    fi
done
echo ""


# ==================== STAGE 1: GENERATE ===================

echo "========================================================"
echo "  Stage 1: Generation ($N_PAIRS pairs, $GEN_MODEL)"
echo "========================================================"

# --- Start vLLM ---
if [ "${BACKEND}" = "vllm" ]; then
    echo "[$(date +%H:%M:%S)] Starting vLLM: $GEN_MODEL"
    echo "  HF_HOME=$HF_HOME"

    python3 -m vllm.entrypoints.openai.api_server \
        --model "$GEN_MODEL" \
        --port "$PORT" \
        --dtype bfloat16 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.90 \
        --download-dir "$HF_HOME/hub" \
        --disable-log-requests \
        --disable-log-stats \
        2>"$LOG_DIR/vllm_gen_${SLURM_JOB_ID:-$$}.log" &
    VLLM_PID=$!

    # Wait for server, checking if process died
    for i in $(seq 1 180); do
        if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
            echo "[$(date +%H:%M:%S)] vLLM ready in $((i * 10))s"
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "FATAL: vLLM died during startup. Last 30 lines:"
            tail -30 "$LOG_DIR/vllm_gen_${SLURM_JOB_ID:-$$}.log"
            exit 1
        fi
        if [ $i -eq 180 ]; then
            echo "FATAL: vLLM not ready after 30 min."
            tail -30 "$LOG_DIR/vllm_gen_${SLURM_JOB_ID:-$$}.log"
            exit 1
        fi
        sleep 10
    done
fi

# --- Run generation ---
echo ""
echo "[$(date +%H:%M:%S)] Starting generation..."

SEED_ARGS=""
if [ -n "${SEED_FILES}" ]; then
    SEED_ARGS="--seed-files ${SEED_FILES}"
fi

STAGE1_START=$(date +%s)

python3 "${SCRIPT_DIR}/run_hpc.py" \
    --mode gen \
    --n-pairs "${N_PAIRS}" \
    --paraphrase-variants "${PARAPHRASE_VARIANTS}" \
    --seed "${RANDOM_SEED}" \
    --output-dir "${OUTPUT_DIR}" \
    --output-name "${OUTPUT_NAME}" \
    --backend "${BACKEND}" \
    --model "${GEN_MODEL}" \
    --base-url "http://localhost:${PORT}" \
    --misleading-fraction "${MISLEADING_FRACTION}" \
    --llm-annotator \
    ${SEED_ARGS}

GEN_STATUS=$?
RAW_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}_raw.jsonl"

if [ ${GEN_STATUS} -ne 0 ] || [ ! -f "${RAW_FILE}" ]; then
    echo "ERROR: Generation failed (exit ${GEN_STATUS})"
    exit 1
fi

STAGE1_END=$(date +%s)
STAGE1_ELAPSED=$(( (STAGE1_END - STAGE1_START) / 60 ))
RAW_COUNT=$(wc -l < "${RAW_FILE}")
echo "[$(date +%H:%M:%S)] Generation done: ${RAW_COUNT} records, ${STAGE1_ELAPSED} min"


# ==================== STAGE 2: AUGMENTATION ===============

echo ""
echo "========================================================"
echo "  Stage 2: Augmentation"
echo "========================================================"

STAGE2_START=$(date +%s)
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
echo "[$(date +%H:%M:%S)] Augmentation done: ${AUG_COUNT} records"


# ==================== STAGE 3: POST-AUG DEDUP =============

echo ""
echo "[$(date +%H:%M:%S)] Post-augmentation dedup..."
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
    echo "WARNING: Post-augmentation dedup failed, using augmented file"
    cp "${AUGMENTED_FILE}" "${DEDUP_FILE}"
fi

FINAL_COUNT=$(wc -l < "${DEDUP_FILE}")
STAGE2_END=$(date +%s)
STAGE2_ELAPSED=$(( (STAGE2_END - STAGE2_START) / 60 ))
echo "[$(date +%H:%M:%S)] Augmentation + dedup done: ${FINAL_COUNT} records, ${STAGE2_ELAPSED} min"


# ==================== SHUTDOWN VLLM ======================

stop_vllm


# ==================== COPY TO PERSISTENT STORAGE ==========

echo ""
echo "--- Copying results to persistent storage ---"
echo "  From: ${DEDUP_FILE}"
echo "  To:   ${RESULTS_DIR}/"

cp "${DEDUP_FILE}" "${RESULTS_DIR}/${OUTPUT_NAME}_augmented_dedup.jsonl"

# Also copy raw for reference
cp "${RAW_FILE}" "${RESULTS_DIR}/${OUTPUT_NAME}_raw.jsonl" 2>/dev/null || true

# Verify
if [[ -f "${RESULTS_DIR}/${OUTPUT_NAME}_augmented_dedup.jsonl" ]]; then
    local_lines=$(wc -l < "${DEDUP_FILE}")
    result_lines=$(wc -l < "${RESULTS_DIR}/${OUTPUT_NAME}_augmented_dedup.jsonl")
    if [[ "$local_lines" -eq "$result_lines" ]]; then
        echo "  Verified: $result_lines records ($(du -sh "${RESULTS_DIR}/${OUTPUT_NAME}_augmented_dedup.jsonl" | cut -f1))"
    else
        echo "  WARNING: Line count mismatch! Source=$local_lines, Dest=$result_lines"
    fi
else
    echo "  WARNING: Copy failed!"
fi


# ==================== JOB METADATA ========================

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( (TOTAL_END - STAGE1_START) / 60 ))

cat > "${RESULTS_DIR}/job_gen_${SLURM_JOB_ID:-$$}_meta.json" <<METAEOF
{
    "job_id": "${SLURM_JOB_ID:-interactive}",
    "node": "$(hostname)",
    "pipeline_version": "v11",
    "stage": "generation",
    "gen_model": "$GEN_MODEL",
    "n_pairs": $N_PAIRS,
    "paraphrase_variants": $PARAPHRASE_VARIANTS,
    "misleading_fraction": $MISLEADING_FRACTION,
    "random_seed": $RANDOM_SEED,
    "raw_records": $RAW_COUNT,
    "augmented_records": $AUG_COUNT,
    "final_records": $FINAL_COUNT,
    "gen_elapsed_min": $STAGE1_ELAPSED,
    "total_elapsed_min": $TOTAL_ELAPSED,
    "completed": "$(date -Iseconds)"
}
METAEOF


# ==================== SUMMARY =============================

echo ""
echo "========================================================"
echo "  Batch 1 Complete (${TOTAL_ELAPSED} min)"
echo "  Raw records:       ${RAW_COUNT}"
echo "  After augmentation: ${AUG_COUNT}"
echo "  After dedup:       ${FINAL_COUNT}"
echo "  Working:           ${DEDUP_FILE}"
echo "  Persistent:        ${RESULTS_DIR}/${OUTPUT_NAME}_augmented_dedup.jsonl"
echo "  Next step:         sbatch launch_val.slurm"
echo "========================================================"
echo "[$(date +%H:%M:%S)] Done."