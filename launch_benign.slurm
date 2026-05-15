#!/bin/bash
#SBATCH --job-name=v11_benign
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=8:00:00
#SBATCH --output=logs/benign_%j.out
#SBATCH --error=logs/benign_%j.err

# =========================================================
# launch_benign.slurm — 4-GPU Parallel Benign Generation
#
# Runs 2 independent benign generation pipelines:
#   Pipeline A: GPU 0 (Qwen-7B gen) + GPU 1 (Llama-8B resp)
#   Pipeline B: GPU 2 (Qwen-7B gen) + GPU 3 (Llama-8B resp)
#
# No adversarial feedback — just natural multi-turn conversations.
# Much faster than adversarial generation (~1 min/record vs ~3 min).
#
# Usage:
#   sbatch launch_benign.slurm
#   N_RECORDS=100 sbatch launch_benign.slurm   # smoke test
# =========================================================

set -euo pipefail

# ==================== CONFIGURATION =======================

N_RECORDS="${N_RECORDS:-1000}"
RANDOM_SEED="${RANDOM_SEED:-99}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25}"
N_PIPELINES="${N_PIPELINES:-2}"

GEN_MODEL="${GEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TARGET_MODEL="${TARGET_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
GEN_PORT_BASE="${GEN_PORT_BASE:-8000}"
TARGET_PORT_BASE="${TARGET_PORT_BASE:-9000}"

MODEL_CACHE="${MODEL_CACHE:-$HOME/work/hf_models}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/staging/dataset_gen_output}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/work/results/dataset_gen}"
OUTPUT_NAME="${OUTPUT_NAME:-benign_pool_v11}"

CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/dataset_gen}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SCRIPT_DIR="${PROJECT_DIR}"
LOG_DIR="$PROJECT_DIR/logs"

# ==================== END CONFIGURATION ===================

declare -a ALL_PIDS=()
declare -a PIPELINE_PIDS=()

echo "========================================================"
echo "  Benign Conversation Generation (v11)"
echo "  Job:        ${SLURM_JOB_ID:-interactive}"
echo "  Node:       $(hostname)"
echo "  Generator:  $GEN_MODEL"
echo "  Target:     $TARGET_MODEL"
echo "  Records:    $N_RECORDS"
echo "  Pipelines:  $N_PIPELINES parallel"
echo "  Per pipe:   $(( N_RECORDS / N_PIPELINES ))"
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
    echo "  Done."
}
trap cleanup EXIT


# ==================== PRE-FLIGHT =========================

echo "--- Pre-flight checks ---"
for model_name in "$GEN_MODEL" "$TARGET_MODEL"; do
    dir_name="models--$(echo "$model_name" | sed 's/\//--/g')"
    if [[ ! -d "$MODEL_CACHE/hub/$dir_name" ]]; then
        echo "ERROR: $model_name not found in cache"; exit 1
    fi
    echo "  $model_name: $(du -sh "$MODEL_CACHE/hub/$dir_name" 2>/dev/null | cut -f1)"
done
echo ""


# ==================== START VLLM SERVERS ==================

echo "[$(date +%H:%M:%S)] Starting $((N_PIPELINES * 2)) vLLM servers..."

for i in $(seq 0 $((N_PIPELINES - 1))); do
    gen_gpu=$((i * 2))
    target_gpu=$((i * 2 + 1))
    gen_port=$((GEN_PORT_BASE + i))
    target_port=$((TARGET_PORT_BASE + i))

    echo "  Pipeline $i: gen GPU $gen_gpu (port $gen_port), target GPU $target_gpu (port $target_port)"

    CUDA_VISIBLE_DEVICES=$gen_gpu python3 -m vllm.entrypoints.openai.api_server \
        --model "$GEN_MODEL" --port "$gen_port" --dtype bfloat16 \
        --max-model-len 8192 --gpu-memory-utilization 0.90 \
        --download-dir "$HF_HOME/hub" \
        --disable-log-requests --disable-log-stats \
        2>"$LOG_DIR/vllm_benign_gen${i}_${SLURM_JOB_ID:-$$}.log" &
    ALL_PIDS+=($!)

    CUDA_VISIBLE_DEVICES=$target_gpu python3 -m vllm.entrypoints.openai.api_server \
        --model "$TARGET_MODEL" --port "$target_port" --dtype bfloat16 \
        --max-model-len 8192 --gpu-memory-utilization 0.90 \
        --download-dir "$HF_HOME/hub" \
        --disable-log-requests --disable-log-stats \
        2>"$LOG_DIR/vllm_benign_target${i}_${SLURM_JOB_ID:-$$}.log" &
    ALL_PIDS+=($!)
done

# Wait for all servers
echo ""
for i in $(seq 0 $((N_PIPELINES - 1))); do
    for port in $((GEN_PORT_BASE + i)) $((TARGET_PORT_BASE + i)); do
        for attempt in $(seq 1 180); do
            if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
                echo "  Port $port ready in $((attempt * 10))s"
                break
            fi
            if [ $attempt -eq 180 ]; then
                echo "FATAL: Port $port timeout"; exit 1
            fi
            sleep 10
        done
    done
done

echo "[$(date +%H:%M:%S)] All servers ready."
echo ""


# ==================== PARALLEL GENERATION =================

echo "========================================================"
echo "  Generating $N_RECORDS benign records ($N_PIPELINES pipelines)"
echo "========================================================"

STAGE_START=$(date +%s)
RECORDS_PER_PIPELINE=$((N_RECORDS / N_PIPELINES))
REMAINDER=$((N_RECORDS % N_PIPELINES))

for i in $(seq 0 $((N_PIPELINES - 1))); do
    gen_port=$((GEN_PORT_BASE + i))
    target_port=$((TARGET_PORT_BASE + i))
    shard_seed=$((RANDOM_SEED + i * 10000))
    shard_file="${OUTPUT_DIR}/${OUTPUT_NAME}_shard${i}.jsonl"

    shard_records=$RECORDS_PER_PIPELINE
    if [ $i -eq $((N_PIPELINES - 1)) ]; then
        shard_records=$((RECORDS_PER_PIPELINE + REMAINDER))
    fi

    echo "[$(date +%H:%M:%S)] Pipeline $i: $shard_records records, seed $shard_seed"

    python3 "${SCRIPT_DIR}/benign_generator.py" \
        --n-records "$shard_records" \
        --seed "$shard_seed" \
        --output "$shard_file" \
        --generator-backend vllm \
        --generator-model "$GEN_MODEL" \
        --generator-url "http://localhost:$gen_port" \
        --target-backend vllm \
        --target-model "$TARGET_MODEL" \
        --target-url "http://localhost:$target_port" \
        --checkpoint-interval "$CHECKPOINT_INTERVAL" \
        2>"$LOG_DIR/benign_pipe${i}_${SLURM_JOB_ID:-$$}.err" &
    PIPELINE_PIDS+=($!)
    ALL_PIDS+=($!)
done

# Wait for all pipelines
echo ""
echo "[$(date +%H:%M:%S)] Waiting for $N_PIPELINES pipelines..."
FAILED=0
for i in $(seq 0 $((N_PIPELINES - 1))); do
    pid=${PIPELINE_PIDS[$i]}
    if wait "$pid"; then
        echo "  Pipeline $i: completed"
    else
        echo "  Pipeline $i: FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

STAGE_END=$(date +%s)
STAGE_ELAPSED=$(( (STAGE_END - STAGE_START) / 60 ))

if [ $FAILED -eq $N_PIPELINES ]; then
    echo "FATAL: All pipelines failed"; exit 1
fi


# ==================== MERGE SHARDS ========================

echo ""
echo "[$(date +%H:%M:%S)] Merging shards..."
RAW_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}_raw.jsonl"

python3 -c "
import json, os
sys_path = '${SCRIPT_DIR}'
import sys; sys.path.insert(0, sys_path)
import build_semantic_datasetv11 as gen

records = []
for i in range(${N_PIPELINES}):
    shard = '${OUTPUT_DIR}/${OUTPUT_NAME}_shard' + str(i) + '.jsonl'
    if not os.path.exists(shard):
        print(f'  Shard {i}: MISSING')
        continue
    count = 0
    with open(shard) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                count += 1
    print(f'  Shard {i}: {count} records')

print(f'Total: {len(records)} records')

# No semantic dedup for benign pool — similar conversations are expected
# and useful. Only remove exact duplicates by conversation_id.
seen_ids = set()
unique = []
for r in records:
    cid = r.get('conversation_id', '')
    if cid not in seen_ids:
        seen_ids.add(cid)
        unique.append(r)
if len(unique) < len(records):
    print(f'Removed {len(records) - len(unique)} exact duplicates')
records = unique

gen.write_jsonl(records, '${RAW_FILE}')
"

RAW_COUNT=$(wc -l < "$RAW_FILE")
echo "[$(date +%H:%M:%S)] Merged: $RAW_COUNT records, ${STAGE_ELAPSED} min"


# ==================== SHUTDOWN ============================

echo ""
echo "[$(date +%H:%M:%S)] Shutting down servers..."
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
echo "  Benign Generation Statistics"
echo "========================================================"

python3 -c "
import json
from collections import Counter

records = []
with open('${RAW_FILE}') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

cats = Counter(r.get('family', '?') for r in records)
lengths = [r.get('conversation_length', 0) for r in records]
avg_len = sum(lengths) / max(len(lengths), 1)
false_leads = sum(1 for r in records if r.get('has_false_lead'))

print(f'Total: {len(records)}')
print(f'Categories:')
for c, n in cats.most_common():
    print(f'  {c}: {n}')
print(f'Avg conversation length: {avg_len:.1f} turns')
print(f'False-lead records: {false_leads}')

# Length distribution
short = sum(1 for l in lengths if l <= 24)
med = sum(1 for l in lengths if 25 <= l <= 40)
long_ = sum(1 for l in lengths if l > 40)
print(f'Length distribution: short(<25)={short}, med(25-40)={med}, long(>40)={long_}')

# Surface risk
high_sr = 0
for r in records:
    for t in r.get('turns', []):
        if t.get('surface_risk', 0) > 0.3 and t.get('role') == 'user':
            high_sr += 1
print(f'User turns with surface_risk > 0.3: {high_sr}')

# Decoy spans
decoys = 0
for r in records:
    for t in r.get('turns', []):
        for s in t.get('span_annotations', []):
            if s.get('label') == 'DECOY':
                decoys += 1
print(f'DECOY spans (false-lead): {decoys}')
"


# ==================== COPY TO PERSISTENT ==================

echo ""
cp "$RAW_FILE" "$RESULTS_DIR/${OUTPUT_NAME}_raw.jsonl"
echo "Persistent: $RESULTS_DIR/${OUTPUT_NAME}_raw.jsonl ($RAW_COUNT records)"


# ==================== METADATA ============================

cat > "${RESULTS_DIR}/job_benign_${SLURM_JOB_ID:-$$}_meta.json" <<METAEOF
{
    "job_id": "${SLURM_JOB_ID:-interactive}",
    "node": "$(hostname)",
    "stage": "benign_generation",
    "gen_model": "$GEN_MODEL",
    "target_model": "$TARGET_MODEL",
    "n_records": $N_RECORDS,
    "n_pipelines": $N_PIPELINES,
    "final_records": $RAW_COUNT,
    "elapsed_min": $STAGE_ELAPSED,
    "completed": "$(date -Iseconds)"
}
METAEOF


# ==================== SUMMARY =============================

echo ""
echo "========================================================"
echo "  Benign Generation Complete (${STAGE_ELAPSED} min)"
echo "  Records:    $RAW_COUNT"
echo "  Persistent: $RESULTS_DIR/${OUTPUT_NAME}_raw.jsonl"
echo ""
echo "  Next: validate with BOTH Llama and Mistral"
echo "    # Mistral validation"
echo "    VAL_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \\"
echo "    INPUT_FILE=${RAW_FILE} \\"
echo "    OUTPUT_NAME=${OUTPUT_NAME}_mistral \\"
echo "    HF_HUB_OFFLINE=1 sbatch launch_val.slurm"
echo ""
echo "    # Llama validation"
echo "    VAL_MODEL=meta-llama/Meta-Llama-3-8B-Instruct \\"
echo "    INPUT_FILE=${RAW_FILE} \\"
echo "    OUTPUT_NAME=${OUTPUT_NAME}_llama \\"
echo "    HF_HUB_OFFLINE=1 sbatch launch_val.slurm"
echo ""
echo "    # Keep only records accepted by BOTH"
echo "========================================================"
