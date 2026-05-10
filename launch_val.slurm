#!/bin/bash
#SBATCH --job-name=v11_val
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=36:00:00
#SBATCH --output=logs/val_%j.out
#SBATCH --error=logs/val_%j.err

# =========================================================
# launch_val.slurm — Batch 2: Validation (v11)
#
# Runs 100% conversation-level validation with Qwen-14B,
# structured LLM judge, pivot-kind classification,
# supervision tier assignment. Counterfactual off by default.
#
# Usage:
#   sbatch launch_val.slurm
#   USE_COUNTERFACTUAL=true sbatch launch_val.slurm
#
# Chain after generation:
#   GEN_JOB=$(sbatch --parsable launch_gen.slurm)
#   sbatch --dependency=afterok:${GEN_JOB} launch_val.slurm
# =========================================================

set -euo pipefail

# ==================== CONFIGURATION =======================

# --- Validation ---
VAL_MODEL="${VAL_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
BACKEND="${BACKEND:-vllm}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25}"
USE_COUNTERFACTUAL="${USE_COUNTERFACTUAL:-false}"

# --- Model ---
PORT="${VLLM_PORT:-8000}"

# --- Storage ---
MODEL_CACHE="${MODEL_CACHE:-$HOME/work/hf_models}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/staging/dataset_gen_output}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/work/results/dataset_gen}"
OUTPUT_NAME="${OUTPUT_NAME:-semantic_multiturn_v11}"
INPUT_FILE="${INPUT_FILE:-${OUTPUT_DIR}/${OUTPUT_NAME}_augmented_dedup.jsonl}"

# --- Environment ---
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/dataset_gen}"

# --- Paths ---
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SCRIPT_DIR="${PROJECT_DIR}"
LOG_DIR="$PROJECT_DIR/logs"

# ==================== END CONFIGURATION ===================

VLLM_PID=""

echo "========================================================"
echo "  Batch 2: Dataset Validation (v11)"
echo "  Job:              ${SLURM_JOB_ID:-interactive}"
echo "  Node:             $(hostname)"
echo "  Val model:        $VAL_MODEL"
echo "  Input:            $INPUT_FILE"
echo "  Counterfactual:   $USE_COUNTERFACTUAL"
echo "  Checkpoint every: $CHECKPOINT_INTERVAL records"
echo "  Model cache:      $MODEL_CACHE"
echo "  Output:           $OUTPUT_DIR"
echo "  Results:          $RESULTS_DIR"
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

# cd to project directory
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

    if [[ $exit_code -ne 0 ]]; then
        echo "  Job ended with error. Checkpoint preserved for resume."
        local ckpt="${OUTPUT_DIR}/${OUTPUT_NAME}_validated.jsonl.checkpoint"
        if [[ -f "$ckpt" ]]; then
            echo "  Checkpoint: $(wc -l < "$ckpt") records at $ckpt"
            echo "  Re-submit this job to resume."
        fi
    fi
    echo "  Done."
}
trap cleanup EXIT


# ==================== PRE-FLIGHT CHECKS ===================

echo "--- Storage pre-flight checks ---"

# Check model cache
if [[ ! -d "$MODEL_CACHE" ]]; then
    echo "ERROR: Model cache not found: $MODEL_CACHE"
    echo "Run: bash prestage_models.sh (on login node)"
    exit 1
fi
echo "  Model cache: $(du -sh "$MODEL_CACHE" 2>/dev/null | cut -f1) at $MODEL_CACHE"

# Check val model directory
VAL_DIR_NAME="models--$(echo "$VAL_MODEL" | sed 's/\//--/g')"
if [[ ! -d "$MODEL_CACHE/hub/$VAL_DIR_NAME" ]]; then
    echo "ERROR: $VAL_MODEL not found in cache."
    echo "  Expected: $MODEL_CACHE/hub/$VAL_DIR_NAME"
    echo "  Run: bash prestage_models.sh"
    exit 1
fi
echo "  $VAL_MODEL: $(du -sh "$MODEL_CACHE/hub/$VAL_DIR_NAME" 2>/dev/null | cut -f1)"

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

# Verify input file exists
if [ ! -f "${INPUT_FILE}" ]; then
    # Cascading fallback
    FALLBACK_AUG="${OUTPUT_DIR}/${OUTPUT_NAME}_augmented.jsonl"
    FALLBACK_RAW="${OUTPUT_DIR}/${OUTPUT_NAME}_raw.jsonl"
    # Also check persistent storage
    FALLBACK_RESULTS="${RESULTS_DIR}/${OUTPUT_NAME}_augmented_dedup.jsonl"
    if [ -f "${FALLBACK_RESULTS}" ]; then
        echo "  Input not at ${INPUT_FILE}, using persistent copy: ${FALLBACK_RESULTS}"
        INPUT_FILE="${FALLBACK_RESULTS}"
    elif [ -f "${FALLBACK_AUG}" ]; then
        echo "  WARNING: Using ${FALLBACK_AUG} (no dedup version found)"
        INPUT_FILE="${FALLBACK_AUG}"
    elif [ -f "${FALLBACK_RAW}" ]; then
        echo "  WARNING: Using ${FALLBACK_RAW} (no augmented version found)"
        INPUT_FILE="${FALLBACK_RAW}"
    else
        echo "ERROR: No input file found. Run launch_gen.slurm first."
        exit 1
    fi
fi

INPUT_COUNT=$(wc -l < "${INPUT_FILE}")
echo "  Input: ${INPUT_COUNT} records from ${INPUT_FILE}"

# Check for existing checkpoint
VALIDATED_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}_validated.jsonl"
CHECKPOINT_FILE="${VALIDATED_FILE}.checkpoint"
if [ -f "${CHECKPOINT_FILE}" ]; then
    CKPT_COUNT=$(wc -l < "${CHECKPOINT_FILE}")
    echo "  Found checkpoint: ${CKPT_COUNT} records validated — will resume"
fi

echo ""


# ==================== START VLLM ==========================

if [ "${BACKEND}" = "vllm" ]; then
    echo "[$(date +%H:%M:%S)] Starting vLLM: $VAL_MODEL"
    echo "  HF_HOME=$HF_HOME"

    # 14B needs more memory; use higher utilization
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$VAL_MODEL" \
        --port "$PORT" \
        --dtype bfloat16 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.92 \
        --download-dir "$HF_HOME/hub" \
        --disable-log-requests \
        --disable-log-stats \
        2>"$LOG_DIR/vllm_val_${SLURM_JOB_ID:-$$}.log" &
    VLLM_PID=$!

    # Wait for server (14B takes longer to load)
    for i in $(seq 1 180); do
        if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
            echo "[$(date +%H:%M:%S)] vLLM ready in $((i * 10))s"
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "FATAL: vLLM died during startup. Last 30 lines:"
            tail -30 "$LOG_DIR/vllm_val_${SLURM_JOB_ID:-$$}.log"
            exit 1
        fi
        if [ $i -eq 180 ]; then
            echo "FATAL: vLLM not ready after 30 min."
            tail -30 "$LOG_DIR/vllm_val_${SLURM_JOB_ID:-$$}.log"
            exit 1
        fi
        sleep 10
    done
fi


# ==================== VALIDATION ==========================

echo ""
echo "========================================================"
echo "  Validation ($VAL_MODEL, counterfactual=$USE_COUNTERFACTUAL)"
echo "========================================================"

CF_FLAG=""
if [ "${USE_COUNTERFACTUAL}" = "true" ]; then
    CF_FLAG="--counterfactual"
fi

STAGE_START=$(date +%s)

python3 "${SCRIPT_DIR}/run_hpc.py" \
    --mode val \
    --input "${INPUT_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --output-name "${OUTPUT_NAME}" \
    --backend "${BACKEND}" \
    --model "${VAL_MODEL}" \
    --base-url "http://localhost:${PORT}" \
    --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
    ${CF_FLAG}

VAL_STATUS=$?

STAGE_END=$(date +%s)
STAGE_ELAPSED=$(( (STAGE_END - STAGE_START) / 60 ))


# ==================== SHUTDOWN VLLM ======================

stop_vllm

if [ ${VAL_STATUS} -ne 0 ]; then
    echo "ERROR: Validation failed (exit ${VAL_STATUS})"
    echo "  Checkpoint preserved. Re-submit to resume."
    exit 1
fi

echo "[$(date +%H:%M:%S)] Validation done: ${STAGE_ELAPSED} min"


# ==================== STATISTICS ==========================

echo ""
echo "========================================================"
echo "  Validation Statistics"
echo "========================================================"

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
print(f'Families: {dict(families.most_common(10))}')

# Jailbreak rate
mal = [r for r in records if r.get('label') == 1]
jb = sum(1 for r in mal if r.get('causal_validation', {}).get('jailbreak_detected'))
print(f'Jailbreak rate: {jb}/{len(mal)} ({jb/max(len(mal),1)*100:.1f}%)')

# False alarm rate
ben = [r for r in records if r.get('label') == 0]
fa = sum(1 for r in ben if r.get('validation_status') == 'rejected')
print(f'False alarm rate: {fa}/{len(ben)} ({fa/max(len(ben),1)*100:.1f}%)')

# Judge confidence
confs = [r.get('judge_confidence', 0) for r in records if r.get('validation_status') == 'validated']
if confs:
    print(f'Judge confidence: mean={sum(confs)/len(confs):.3f}, min={min(confs):.3f}, max={max(confs):.3f}')

# Loss weight distribution
weights = [r.get('loss_weight', 0.5) for r in records if r.get('training_eligible', True)]
if weights:
    print(f'Avg loss weight (eligible): {sum(weights)/len(weights):.3f}')

# Span stats
causal = incidental = construction = 0
for r in records:
    for t in r.get('turns', []):
        for s in t.get('span_annotations', []):
            ct = s.get('causal_type', 'unvalidated')
            if ct == 'causal': causal += 1
            elif ct == 'incidental': incidental += 1
            else: construction += 1
if causal or incidental:
    print(f'Spans: {causal} causal, {incidental} incidental, {construction} construction')
"


# ==================== COPY TO PERSISTENT STORAGE ==========

echo ""
echo "--- Copying results to persistent storage ---"
echo "  From: ${VALIDATED_FILE}"
echo "  To:   ${RESULTS_DIR}/"

cp "${VALIDATED_FILE}" "${RESULTS_DIR}/${OUTPUT_NAME}_validated.jsonl"

# Verify
if [[ -f "${RESULTS_DIR}/${OUTPUT_NAME}_validated.jsonl" ]]; then
    local_lines=$(wc -l < "${VALIDATED_FILE}")
    result_lines=$(wc -l < "${RESULTS_DIR}/${OUTPUT_NAME}_validated.jsonl")
    if [[ "$local_lines" -eq "$result_lines" ]]; then
        echo "  Verified: $result_lines records ($(du -sh "${RESULTS_DIR}/${OUTPUT_NAME}_validated.jsonl" | cut -f1))"
    else
        echo "  WARNING: Line count mismatch! Source=$local_lines, Dest=$result_lines"
    fi
else
    echo "  WARNING: Copy failed!"
fi


# ==================== JOB METADATA ========================

TOTAL_ELAPSED=$(( (STAGE_END - STAGE_START) / 60 ))
VAL_COUNT=$(wc -l < "${VALIDATED_FILE}")

cat > "${RESULTS_DIR}/job_val_${SLURM_JOB_ID:-$$}_meta.json" <<METAEOF
{
    "job_id": "${SLURM_JOB_ID:-interactive}",
    "node": "$(hostname)",
    "pipeline_version": "v11",
    "stage": "validation",
    "val_model": "$VAL_MODEL",
    "input_file": "$INPUT_FILE",
    "input_records": $INPUT_COUNT,
    "validated_records": $VAL_COUNT,
    "counterfactual": $( [ "${USE_COUNTERFACTUAL}" = "true" ] && echo "true" || echo "false" ),
    "checkpoint_interval": $CHECKPOINT_INTERVAL,
    "elapsed_min": $TOTAL_ELAPSED,
    "completed": "$(date -Iseconds)"
}
METAEOF


# ==================== SUMMARY =============================

echo ""
echo "========================================================"
echo "  Batch 2 Complete (${TOTAL_ELAPSED} min)"
echo "  Input:      ${INPUT_COUNT} records"
echo "  Validated:  ${VAL_COUNT} records"
echo "  Working:    ${VALIDATED_FILE}"
echo "  Persistent: ${RESULTS_DIR}/${OUTPUT_NAME}_validated.jsonl"
echo "  Next steps:"
echo "    python3 mhj_loader.py --input mhj.jsonl --output ${OUTPUT_DIR}/${OUTPUT_NAME}_mhj.jsonl"
echo "    python3 split_dataset.py --input ${VALIDATED_FILE} --output-dir ${OUTPUT_DIR}/splits/"
if [ "${USE_COUNTERFACTUAL}" = "false" ]; then
    echo "    USE_COUNTERFACTUAL=true sbatch launch_val.slurm  # counterfactual pass"
fi
echo "========================================================"
echo "[$(date +%H:%M:%S)] Done."