#!/bin/bash
#SBATCH --job-name=v11_val
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/val_%j.out
#SBATCH --error=logs/val_%j.err

# =========================================================
# launch_val.slurm — Parallel Sharded Validation (v11)
#
# Splits the input dataset into N_VAL_SHARDS shards and
# validates each shard in parallel on a separate GPU.
# Merges results into a single output file.
#
# Usage:
#   # 2-GPU validation with Qwen-14B (default)
#   INPUT_FILE=output/interactive_raw.jsonl \
#   OUTPUT_NAME=semantic_multiturn_v11_interactive \
#   sbatch launch_val.slurm
#
#   # 1-GPU validation (e.g. when other GPUs are busy)
#   N_VAL_SHARDS=1 sbatch launch_val.slurm
#
#   # Different validation model
#   VAL_MODEL=Qwen/Qwen2.5-7B-Instruct sbatch launch_val.slurm
# =========================================================

set -euo pipefail

# ==================== CONFIGURATION =======================

VAL_MODEL="${VAL_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
BACKEND="${BACKEND:-vllm}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25}"
USE_COUNTERFACTUAL="${USE_COUNTERFACTUAL:-false}"

# Parallel shards (1 GPU each)
N_VAL_SHARDS="${N_VAL_SHARDS:-2}"

# Port base — each shard uses PORT_BASE + shard_index
PORT_BASE="${VLLM_PORT:-8000}"

# Storage
MODEL_CACHE="${MODEL_CACHE:-$HOME/work/hf_models}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/staging/dataset_gen_output}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/work/results/dataset_gen}"
OUTPUT_NAME="${OUTPUT_NAME:-semantic_multiturn_v11}"
INPUT_FILE="${INPUT_FILE:-${OUTPUT_DIR}/${OUTPUT_NAME}_augmented_dedup.jsonl}"

# Environment
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/dataset_gen}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SCRIPT_DIR="${PROJECT_DIR}"
LOG_DIR="$PROJECT_DIR/logs"

# ==================== END CONFIGURATION ===================

declare -a ALL_PIDS=()
declare -a VAL_PIDS=()

echo "========================================================"
echo "  Parallel Validation (v11)"
echo "  Job:              ${SLURM_JOB_ID:-interactive}"
echo "  Node:             $(hostname)"
echo "  Val model:        $VAL_MODEL"
echo "  Input:            $INPUT_FILE"
echo "  Shards:           $N_VAL_SHARDS (parallel)"
echo "  Counterfactual:   $USE_COUNTERFACTUAL"
echo "  Checkpoint every: $CHECKPOINT_INTERVAL records"
echo "  Model cache:      $MODEL_CACHE"
echo "  Output:           $OUTPUT_DIR"
echo "========================================================"


# ==================== ENVIRONMENT =========================

if command -v module &>/dev/null; then
    for mod in cuda/12.1 cuda/12.2 cuda/12.4 cuda/12 cuda; do
        module load "$mod" 2>/dev/null && break
    done
fi

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

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/.conda")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export TMPDIR="$HOME/work/.tmp"
export PIP_CACHE_DIR="$HOME/work/.pip_cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" 2>/dev/null || true

export HF_HOME="$MODEL_CACHE"
export TRANSFORMERS_CACHE="$MODEL_CACHE/hub"
export HF_HUB_CACHE="$MODEL_CACHE/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR" "$LOG_DIR" 2>/dev/null || true

echo "HF_HOME: $HF_HOME"
echo "HF_HUB_OFFLINE: $HF_HUB_OFFLINE"
echo "CUDA_HOME: ${CUDA_HOME:-not set}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "GPU: not detected"
echo ""


# ==================== CLEANUP TRAP ========================

cleanup() {
    local exit_code=$?
    echo ""
    echo "[$(date +%H:%M:%S)] Cleanup (exit code: $exit_code)..."
    for pid in "${ALL_PIDS[@]}"; do
        if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true

    if [[ $exit_code -ne 0 ]]; then
        echo "  Job failed. Shard checkpoints preserved for resume."
        for i in $(seq 0 $((N_VAL_SHARDS - 1))); do
            local ckpt="${OUTPUT_DIR}/${OUTPUT_NAME}_valshard${i}.jsonl.checkpoint"
            if [[ -f "$ckpt" ]]; then
                echo "  Shard $i checkpoint: $(wc -l < "$ckpt") records"
            fi
        done
    fi
    echo "  Done."
}
trap cleanup EXIT


# ==================== PRE-FLIGHT CHECKS ===================

echo "--- Pre-flight checks ---"

if [[ ! -d "$MODEL_CACHE" ]]; then
    echo "ERROR: Model cache not found: $MODEL_CACHE"
    exit 1
fi

VAL_DIR_NAME="models--$(echo "$VAL_MODEL" | sed 's/\//--/g')"
if [[ ! -d "$MODEL_CACHE/hub/$VAL_DIR_NAME" ]]; then
    echo "ERROR: $VAL_MODEL not found in cache"
    exit 1
fi
echo "  $VAL_MODEL: $(du -sh "$MODEL_CACHE/hub/$VAL_DIR_NAME" 2>/dev/null | cut -f1)"

# Verify input
if [ ! -f "${INPUT_FILE}" ]; then
    # Fallback cascade
    for fallback in \
        "${OUTPUT_DIR}/${OUTPUT_NAME}_raw.jsonl" \
        "${RESULTS_DIR}/${OUTPUT_NAME}_raw.jsonl" \
        "${OUTPUT_DIR}/${OUTPUT_NAME}_augmented_dedup.jsonl" \
        "${RESULTS_DIR}/${OUTPUT_NAME}_augmented_dedup.jsonl"; do
        if [ -f "$fallback" ]; then
            echo "  Using fallback: $fallback"
            INPUT_FILE="$fallback"
            break
        fi
    done
    if [ ! -f "${INPUT_FILE}" ]; then
        echo "ERROR: No input file found"
        exit 1
    fi
fi

INPUT_COUNT=$(wc -l < "${INPUT_FILE}")
echo "  Input: $INPUT_COUNT records from $INPUT_FILE"
echo ""


# ==================== SPLIT INPUT INTO SHARDS =============

echo "[$(date +%H:%M:%S)] Splitting input into $N_VAL_SHARDS shards..."

python3 -c "
import json, sys, math

records = []
with open('${INPUT_FILE}') as f:
    for line in f:
        if line.strip():
            records.append(line)

n = len(records)
shards = ${N_VAL_SHARDS}
shard_size = math.ceil(n / shards)

for i in range(shards):
    start = i * shard_size
    end = min(start + shard_size, n)
    shard_file = '${OUTPUT_DIR}/${OUTPUT_NAME}_valshard' + str(i) + '_input.jsonl'
    with open(shard_file, 'w') as f:
        for line in records[start:end]:
            f.write(line)
    print(f'  Shard {i}: records {start}-{end-1} ({end-start} records) -> {shard_file}')
"


# ==================== START VLLM SERVERS ==================

echo ""
echo "[$(date +%H:%M:%S)] Starting $N_VAL_SHARDS vLLM servers..."

for i in $(seq 0 $((N_VAL_SHARDS - 1))); do
    port=$((PORT_BASE + i))
    echo "  Shard $i: GPU $i, port $port"

    CUDA_VISIBLE_DEVICES=$i python3 -m vllm.entrypoints.openai.api_server \
        --model "$VAL_MODEL" \
        --port "$port" \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.92 \
        --download-dir "$HF_HOME/hub" \
        --disable-log-requests \
        --disable-log-stats \
        2>"$LOG_DIR/vllm_val${i}_${SLURM_JOB_ID:-$$}.log" &
    ALL_PIDS+=($!)
done

# Wait for all servers
echo ""
for i in $(seq 0 $((N_VAL_SHARDS - 1))); do
    port=$((PORT_BASE + i))
    for attempt in $(seq 1 180); do
        if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
            echo "  Shard $i (port $port) ready in $((attempt * 10))s"
            break
        fi
        if [ $attempt -eq 180 ]; then
            echo "FATAL: Shard $i (port $port) not ready after 30 min"
            tail -20 "$LOG_DIR/vllm_val${i}_${SLURM_JOB_ID:-$$}.log" 2>/dev/null
            exit 1
        fi
        sleep 10
    done
done

echo "[$(date +%H:%M:%S)] All servers ready."
echo ""


# ==================== PARALLEL VALIDATION =================

echo "========================================================"
echo "  Validating $INPUT_COUNT records across $N_VAL_SHARDS shards"
echo "========================================================"

STAGE_START=$(date +%s)

CF_FLAG=""
if [ "${USE_COUNTERFACTUAL}" = "true" ]; then
    CF_FLAG="--counterfactual"
fi

for i in $(seq 0 $((N_VAL_SHARDS - 1))); do
    port=$((PORT_BASE + i))
    shard_input="${OUTPUT_DIR}/${OUTPUT_NAME}_valshard${i}_input.jsonl"
    shard_output_name="${OUTPUT_NAME}_valshard${i}"

    echo "[$(date +%H:%M:%S)] Shard $i: starting validation on port $port"

    python3 "${SCRIPT_DIR}/run_hpc.py" \
        --mode val \
        --input "$shard_input" \
        --output-dir "${OUTPUT_DIR}" \
        --output-name "$shard_output_name" \
        --backend "${BACKEND}" \
        --model "${VAL_MODEL}" \
        --base-url "http://localhost:$port" \
        --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
        ${CF_FLAG} \
        2>"$LOG_DIR/valshard${i}_${SLURM_JOB_ID:-$$}.err" &
    VAL_PIDS+=($!)
    ALL_PIDS+=($!)
done

# Wait for all validation shards
echo ""
echo "[$(date +%H:%M:%S)] Waiting for $N_VAL_SHARDS validation shards..."
FAILED=0
for i in $(seq 0 $((N_VAL_SHARDS - 1))); do
    pid=${VAL_PIDS[$i]}
    if wait "$pid"; then
        shard_out="${OUTPUT_DIR}/${OUTPUT_NAME}_valshard${i}_validated.jsonl"
        if [ -f "$shard_out" ]; then
            shard_count=$(wc -l < "$shard_out")
            echo "  Shard $i: completed ($shard_count records)"
        else
            echo "  Shard $i: completed but output missing"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "  Shard $i: FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

STAGE_END=$(date +%s)
STAGE_ELAPSED=$(( (STAGE_END - STAGE_START) / 60 ))

if [ $FAILED -eq $N_VAL_SHARDS ]; then
    echo "FATAL: All shards failed"
    exit 1
fi


# ==================== MERGE SHARDS ========================

echo ""
echo "[$(date +%H:%M:%S)] Merging $N_VAL_SHARDS validation shards..."

VALIDATED_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}_validated.jsonl"

python3 -c "
import json, os

records = []
for i in range(${N_VAL_SHARDS}):
    shard = '${OUTPUT_DIR}/${OUTPUT_NAME}_valshard' + str(i) + '_validated.jsonl'
    if not os.path.exists(shard):
        print(f'  Shard {i}: MISSING (skipped)')
        continue
    count = 0
    with open(shard) as f:
        for line in f:
            if line.strip():
                records.append(line)
                count += 1
    print(f'  Shard {i}: {count} records')

print(f'Total merged: {len(records)} records')

with open('${VALIDATED_FILE}', 'w') as f:
    for line in records:
        f.write(line)
"

VAL_COUNT=$(wc -l < "$VALIDATED_FILE")
echo "[$(date +%H:%M:%S)] Merged: $VAL_COUNT records"


# ==================== SHUTDOWN SERVERS ====================

echo ""
echo "[$(date +%H:%M:%S)] Shutting down vLLM servers..."
for pid in "${ALL_PIDS[@]}"; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
    fi
done
wait 2>/dev/null || true
ALL_PIDS=()
sleep 3


# ==================== STATISTICS ==========================

echo ""
echo "========================================================"
echo "  Validation Statistics"
echo "========================================================"

python3 -c "
import json
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
eligible = sum(1 for r in records if r.get('training_eligible', True))

print(f'Total records: {total}')
print(f'Labels: {dict(labels)}')
print(f'Validation status: {dict(statuses)}')
print(f'Supervision tiers: {dict(tiers)}')
print(f'Pivot kinds: {dict(pivots)}')
print(f'Training eligible: {eligible}/{total}')

mal = [r for r in records if r.get('label') == 1]
jb = sum(1 for r in mal if r.get('causal_validation', {}).get('jailbreak_detected'))
print(f'Jailbreak rate ({\"$VAL_MODEL\".split(\"/\")[-1]}): {jb}/{len(mal)} ({jb/max(len(mal),1)*100:.1f}%)')

ben = [r for r in records if r.get('label') == 0]
fa = sum(1 for r in ben if r.get('validation_status') == 'rejected')
print(f'False alarm rate: {fa}/{len(ben)} ({fa/max(len(ben),1)*100:.1f}%)')

confs = [r.get('judge_confidence', 0) for r in records if r.get('validation_status') == 'validated']
if confs:
    print(f'Judge confidence: mean={sum(confs)/len(confs):.3f}')

# Cross-reference with llama_validation if present
llama_jb = sum(1 for r in mal if r.get('llama_validation', {}).get('jailbreak_detected'))
if llama_jb > 0:
    val_jb_set = set(r.get('conversation_id') for r in mal if r.get('causal_validation', {}).get('jailbreak_detected'))
    llama_jb_set = set(r.get('conversation_id') for r in mal if r.get('llama_validation', {}).get('jailbreak_detected'))
    both = len(val_jb_set & llama_jb_set)
    print(f'')
    print(f'Cross-model comparison:')
    print(f'  Llama-8B jailbreaks (from generation): {len(llama_jb_set)}')
    print(f'  {\"$VAL_MODEL\".split(\"/\")[-1]} jailbreaks (this validation): {len(val_jb_set)}')
    print(f'  Both models: {both}')
    print(f'  Transfer rate: {both}/{len(llama_jb_set)} ({both/max(len(llama_jb_set),1)*100:.1f}%)')
"


# ==================== COPY TO PERSISTENT ==================

echo ""
echo "--- Copying to persistent storage ---"
cp "$VALIDATED_FILE" "$RESULTS_DIR/${OUTPUT_NAME}_validated.jsonl"

if [[ -f "$RESULTS_DIR/${OUTPUT_NAME}_validated.jsonl" ]]; then
    result_lines=$(wc -l < "$RESULTS_DIR/${OUTPUT_NAME}_validated.jsonl")
    echo "  Verified: $result_lines records ($(du -sh "$RESULTS_DIR/${OUTPUT_NAME}_validated.jsonl" | cut -f1))"
fi

# Keep shard files for inspection (remove manually after verifying merge)
echo "  Shard files preserved for inspection at:"
for i in $(seq 0 $((N_VAL_SHARDS - 1))); do
    local_shard="${OUTPUT_DIR}/${OUTPUT_NAME}_valshard${i}_validated.jsonl"
    if [[ -f "$local_shard" ]]; then
        echo "    $local_shard ($(wc -l < "$local_shard") records)"
    fi
    local_ckpt="${local_shard}.checkpoint.done"
    if [[ -f "$local_ckpt" ]]; then
        echo "    $local_ckpt"
    fi
done
echo "  To clean up: rm ${OUTPUT_DIR}/${OUTPUT_NAME}_valshard*"


# ==================== JOB METADATA ========================

TOTAL_ELAPSED=$(( (STAGE_END - STAGE_START) / 60 ))

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
    "n_shards": $N_VAL_SHARDS,
    "counterfactual": $( [ "${USE_COUNTERFACTUAL}" = "true" ] && echo "true" || echo "false" ),
    "elapsed_min": $TOTAL_ELAPSED,
    "completed": "$(date -Iseconds)"
}
METAEOF


# ==================== SUMMARY =============================

echo ""
echo "========================================================"
echo "  Validation Complete (${TOTAL_ELAPSED} min)"
echo "  Model:      $VAL_MODEL"
echo "  Shards:     $N_VAL_SHARDS parallel"
echo "  Input:      $INPUT_COUNT records"
echo "  Validated:  $VAL_COUNT records"
echo "  Persistent: $RESULTS_DIR/${OUTPUT_NAME}_validated.jsonl"
echo ""
echo "  Next: merge validations"
echo "    python3 merge_validations.py \\"
echo "      --interactive $VALIDATED_FILE \\"
echo "      --output ${OUTPUT_DIR}/${OUTPUT_NAME}_merged.jsonl"
echo "========================================================"
echo "[$(date +%H:%M:%S)] Done."