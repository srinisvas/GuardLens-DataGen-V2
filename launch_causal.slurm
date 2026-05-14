#!/bin/bash
#SBATCH --job-name=v11_causal
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=logs/causal_%j.out
#SBATCH --error=logs/causal_%j.err

# =========================================================
# launch_causal.slurm — 4-GPU Parallel Causal Analysis
#
# Splits input into N_SHARDS, runs causal_analysis.py on
# each shard in parallel with its own vLLM server.
#
# Usage:
#   INPUT_FILE=output/combined_dedup.jsonl sbatch launch_causal.slurm
#   N_SHARDS=2 sbatch launch_causal.slurm   # use 2 GPUs only
# =========================================================

set -euo pipefail

VAL_MODEL="${VAL_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
N_SHARDS="${N_SHARDS:-4}"
PORT_BASE="${VLLM_PORT:-8000}"
MODEL_CACHE="${MODEL_CACHE:-$HOME/work/hf_models}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/staging/dataset_gen_output}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/work/results/dataset_gen}"
INPUT_FILE="${INPUT_FILE:-${OUTPUT_DIR}/combined_dedup.jsonl}"
OUTPUT_NAME="${OUTPUT_NAME:-causal_analyzed_all}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/dataset_gen}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
LOG_DIR="$PROJECT_DIR/logs"

declare -a ALL_PIDS=()
declare -a ANALYSIS_PIDS=()

echo "========================================================"
echo "  4-Pass Causal Analysis (${N_SHARDS}-GPU Parallel)"
echo "  Job:       ${SLURM_JOB_ID:-interactive}"
echo "  Node:      $(hostname)"
echo "  Model:     $VAL_MODEL"
echo "  Input:     $INPUT_FILE"
echo "  Shards:    $N_SHARDS"
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
            export CUDA_HOME="$p"; export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"; break
        fi
    done
fi

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/.conda")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export TMPDIR="$HOME/work/.tmp"; export PIP_CACHE_DIR="$HOME/work/.pip_cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" 2>/dev/null || true
export HF_HOME="$MODEL_CACHE"; export TRANSFORMERS_CACHE="$MODEL_CACHE/hub"
export HF_HUB_CACHE="$MODEL_CACHE/hub"; export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR" "$LOG_DIR" 2>/dev/null || true

nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null
echo ""

# ==================== CLEANUP TRAP ========================

cleanup() {
    local exit_code=$?
    echo ""
    echo "[$(date +%H:%M:%S)] Cleanup (exit $exit_code)..."
    for pid in "${ALL_PIDS[@]}"; do
        if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true

    if [[ $exit_code -ne 0 ]]; then
        echo "  Intermediate files preserved:"
        for i in $(seq 0 $((N_SHARDS - 1))); do
            for suffix in "_after_pass1.jsonl" "_after_pass2.jsonl" "_after_pass3.jsonl" ".jsonl"; do
                local f="${OUTPUT_DIR}/${OUTPUT_NAME}_shard${i}${suffix}"
                if [[ -f "$f" ]]; then
                    echo "    $f ($(wc -l < "$f") records)"
                fi
            done
        done
    fi
    echo "  Done."
}
trap cleanup EXIT

# ==================== PRE-FLIGHT =========================

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input not found: $INPUT_FILE"; exit 1
fi
INPUT_COUNT=$(wc -l < "$INPUT_FILE")
echo "Input: $INPUT_COUNT records"

# ==================== SPLIT INTO SHARDS ===================

echo "[$(date +%H:%M:%S)] Splitting into $N_SHARDS shards..."

python3 -c "
import json, math

records = []
with open('${INPUT_FILE}') as f:
    for line in f:
        if line.strip():
            records.append(line)

n = len(records)
shards = ${N_SHARDS}
shard_size = math.ceil(n / shards)

for i in range(shards):
    start = i * shard_size
    end = min(start + shard_size, n)
    shard_file = '${OUTPUT_DIR}/${OUTPUT_NAME}_shard' + str(i) + '_input.jsonl'
    with open(shard_file, 'w') as f:
        for line in records[start:end]:
            f.write(line)
    print(f'  Shard {i}: {end - start} records -> {shard_file}')
"

# ==================== START VLLM SERVERS ==================

echo ""
echo "[$(date +%H:%M:%S)] Starting $N_SHARDS vLLM servers..."

for i in $(seq 0 $((N_SHARDS - 1))); do
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
        2>"$LOG_DIR/vllm_causal${i}_${SLURM_JOB_ID:-$$}.log" &
    ALL_PIDS+=($!)
done

echo ""
for i in $(seq 0 $((N_SHARDS - 1))); do
    port=$((PORT_BASE + i))
    for attempt in $(seq 1 180); do
        if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
            echo "  Shard $i (port $port) ready in $((attempt * 10))s"
            break
        fi
        if [ $attempt -eq 180 ]; then
            echo "FATAL: Shard $i timeout"; exit 1
        fi
        sleep 10
    done
done

echo "[$(date +%H:%M:%S)] All servers ready."

# ==================== PARALLEL CAUSAL ANALYSIS ============

echo ""
echo "========================================================"
echo "  Running 4-pass causal analysis on $N_SHARDS shards"
echo "========================================================"

STAGE_START=$(date +%s)

for i in $(seq 0 $((N_SHARDS - 1))); do
    port=$((PORT_BASE + i))
    shard_input="${OUTPUT_DIR}/${OUTPUT_NAME}_shard${i}_input.jsonl"
    shard_output="${OUTPUT_DIR}/${OUTPUT_NAME}_shard${i}.jsonl"

    echo "[$(date +%H:%M:%S)] Shard $i: starting on port $port"

    python3 "${PROJECT_DIR}/causal_analysis.py" \
        --input "$shard_input" \
        --output "$shard_output" \
        --validator-backend vllm \
        --validator-model "$VAL_MODEL" \
        --validator-url "http://localhost:$port" \
        --checkpoint-dir "$OUTPUT_DIR" \
        2>"$LOG_DIR/causal_shard${i}_${SLURM_JOB_ID:-$$}.err" &
    ANALYSIS_PIDS+=($!)
    ALL_PIDS+=($!)
done

echo ""
echo "[$(date +%H:%M:%S)] Waiting for $N_SHARDS analysis shards..."
FAILED=0
for i in $(seq 0 $((N_SHARDS - 1))); do
    pid=${ANALYSIS_PIDS[$i]}
    if wait "$pid"; then
        shard_out="${OUTPUT_DIR}/${OUTPUT_NAME}_shard${i}.jsonl"
        if [ -f "$shard_out" ]; then
            echo "  Shard $i: completed ($(wc -l < "$shard_out") records)"
        else
            echo "  Shard $i: completed but output missing"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "  Shard $i: FAILED (exit $?)"
        echo "  stderr: $(tail -5 "$LOG_DIR/causal_shard${i}_${SLURM_JOB_ID:-$$}.err" 2>/dev/null)"
        FAILED=$((FAILED + 1))
    fi
done

STAGE_END=$(date +%s)
STAGE_ELAPSED=$(( (STAGE_END - STAGE_START) / 60 ))

if [ $FAILED -eq $N_SHARDS ]; then
    echo "FATAL: All shards failed"; exit 1
fi

# ==================== MERGE SHARDS ========================

echo ""
echo "[$(date +%H:%M:%S)] Merging $N_SHARDS shards..."
FINAL_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}.jsonl"

python3 -c "
import json, os

records = []
for i in range(${N_SHARDS}):
    shard = '${OUTPUT_DIR}/${OUTPUT_NAME}_shard' + str(i) + '.jsonl'
    if not os.path.exists(shard):
        print(f'  Shard {i}: MISSING')
        continue
    count = 0
    with open(shard) as f:
        for line in f:
            if line.strip():
                records.append(line)
                count += 1
    print(f'  Shard {i}: {count} records')

print(f'Total: {len(records)} records')
with open('${FINAL_FILE}', 'w') as f:
    for line in records:
        f.write(line)
"

FINAL_COUNT=$(wc -l < "$FINAL_FILE")

# ==================== SHUTDOWN ============================

echo ""
echo "[$(date +%H:%M:%S)] Shutting down vLLM servers..."
for pid in "${ALL_PIDS[@]}"; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
    fi
done
wait 2>/dev/null || true
ALL_PIDS=()

# ==================== STATISTICS ==========================

echo ""
echo "========================================================"
echo "  Causal Analysis Statistics"
echo "========================================================"

python3 -c "
import json
from collections import Counter

records = []
with open('${FINAL_FILE}') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

mal = [r for r in records if r.get('label') == 1]

# Supervision tiers
tiers = Counter(r.get('supervision_tier', 'unknown') for r in records)
print(f'Supervision tiers:')
for t, c in tiers.most_common():
    print(f'  {t}: {c}')

# Pivot-turn CF
pivot_labels = Counter(
    r.get('causal_analysis', {}).get('pivot_turn_cf', {}).get('label', 'none')
    for r in mal
)
print(f'')
print(f'Pivot-turn CF (malicious):')
for l, c in pivot_labels.most_common():
    print(f'  {l}: {c}')

# Span stats
causal = incidental = unvalidated = 0
by_label = Counter()
for r in records:
    for t in r.get('turns', []):
        for s in t.get('span_annotations', []):
            ct = s.get('causal_type', 'unvalidated')
            if ct == 'causal':
                causal += 1
                by_label[s.get('label', '?')] += 1
            elif ct == 'incidental':
                incidental += 1
            else:
                unvalidated += 1
print(f'')
print(f'Spans: {causal} causal, {incidental} incidental, {unvalidated} unvalidated')
if by_label:
    print(f'Causal spans by label: {dict(by_label.most_common())}')

# Loss weights
weights = [r.get('loss_weight', 0.5) for r in records if r.get('training_eligible', True)]
if weights:
    print(f'')
    print(f'Training eligible: {len(weights)}/{len(records)}')
    print(f'Avg loss weight: {sum(weights)/len(weights):.3f}')
    w_dist = Counter()
    for w in weights:
        if w >= 0.9: w_dist['>=0.9'] += 1
        elif w >= 0.6: w_dist['0.6-0.9'] += 1
        elif w >= 0.3: w_dist['0.3-0.6'] += 1
        else: w_dist['<0.3'] += 1
    print(f'Loss weight distribution: {dict(w_dist)}')
"

# ==================== COPY TO PERSISTENT ==================

echo ""
cp "$FINAL_FILE" "$RESULTS_DIR/${OUTPUT_NAME}.jsonl"
echo "Persistent: $RESULTS_DIR/${OUTPUT_NAME}.jsonl ($FINAL_COUNT records)"

# Preserve shard intermediates
echo "Shard files preserved at $OUTPUT_DIR/${OUTPUT_NAME}_shard*"

# ==================== METADATA ============================

cat > "${RESULTS_DIR}/job_causal_${SLURM_JOB_ID:-$$}_meta.json" <<METAEOF
{
    "job_id": "${SLURM_JOB_ID:-interactive}",
    "node": "$(hostname)",
    "stage": "causal_analysis_4pass",
    "val_model": "$VAL_MODEL",
    "input_records": $INPUT_COUNT,
    "final_records": $FINAL_COUNT,
    "n_shards": $N_SHARDS,
    "elapsed_min": $STAGE_ELAPSED,
    "completed": "$(date -Iseconds)"
}
METAEOF

echo ""
echo "========================================================"
echo "  Causal Analysis Complete (${STAGE_ELAPSED} min)"
echo "  Shards:    $N_SHARDS parallel"
echo "  Input:     $INPUT_COUNT records"
echo "  Output:    $FINAL_COUNT records"
echo "  Next:      python3 split_dataset.py --input ${FINAL_FILE} --output-dir ${OUTPUT_DIR}/splits/"
echo "========================================================"
