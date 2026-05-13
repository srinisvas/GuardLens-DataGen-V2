#!/bin/bash
#SBATCH --job-name=v11_igen
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=18:00:00
#SBATCH --output=logs/igen_%j.out
#SBATCH --error=logs/igen_%j.err

# =========================================================
# launch_gen_interactive.slurm — 4-GPU Parallel Generation
#
# Runs 2 independent generation pipelines in parallel:
#   Pipeline A: GPU 0 (generator) + GPU 1 (target)
#   Pipeline B: GPU 2 (generator) + GPU 3 (target)
#
# Each pipeline generates N_PAIRS/2 conversations.
# No augmentation — interactive data is already diverse.
# Output is merged and deduped into a single file.
#
# Usage:
#   sbatch launch_gen_interactive.slurm
#   N_PAIRS=20 sbatch launch_gen_interactive.slurm       # smoke test
#   GEN_MODEL=Qwen/Qwen2.5-14B-Instruct sbatch ...      # use 14B
#
# Two-stage strategy (quantity + quality):
#   N_PAIRS=1000 GEN_MODEL=Qwen/Qwen2.5-7B-Instruct sbatch launch_gen_interactive.slurm
#   N_PAIRS=500  GEN_MODEL=Qwen/Qwen2.5-14B-Instruct GRES=gpu:2 sbatch launch_gen_interactive.slurm
# =========================================================

set -euo pipefail

# ==================== CONFIGURATION =======================

N_PAIRS="${N_PAIRS:-1000}"
RANDOM_SEED="${RANDOM_SEED:-42}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"

# Generator model — switch between 7B and 14B
GEN_MODEL="${GEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TARGET_MODEL="${TARGET_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"

# Number of parallel pipelines (2 GPUs each)
N_PIPELINES="${N_PIPELINES:-2}"

# Port ranges
GEN_PORT_BASE="${GEN_PORT_BASE:-8000}"
TARGET_PORT_BASE="${TARGET_PORT_BASE:-9000}"

# Storage
MODEL_CACHE="${MODEL_CACHE:-$HOME/work/hf_models}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/staging/dataset_gen_output}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/work/results/dataset_gen}"
OUTPUT_NAME="${OUTPUT_NAME:-semantic_multiturn_v11_interactive}"

# Environment
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/dataset_gen}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SCRIPT_DIR="${PROJECT_DIR}"
LOG_DIR="$PROJECT_DIR/logs"

# ==================== END CONFIGURATION ===================

# Track all PIDs for cleanup
declare -a ALL_PIDS=()
declare -a PIPELINE_PIDS=()

echo "========================================================"
echo "  Interactive Adversarial Generation (v11)"
echo "  Job:           ${SLURM_JOB_ID:-interactive}"
echo "  Node:          $(hostname)"
echo "  Generator:     $GEN_MODEL"
echo "  Target:        $TARGET_MODEL"
echo "  N pairs:       $N_PAIRS"
echo "  Pipelines:     $N_PIPELINES (parallel)"
echo "  Pairs/pipe:    $(( N_PAIRS / N_PIPELINES ))"
echo "  Model cache:   $MODEL_CACHE"
echo "  Output:        $OUTPUT_DIR"
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

    # Kill all tracked PIDs
    for pid in "${ALL_PIDS[@]}"; do
        if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "  Stopping pid $pid..."
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true

    if [[ $exit_code -ne 0 ]]; then
        echo "  Job failed. Checking for partial outputs..."
        for i in $(seq 0 $((N_PIPELINES - 1))); do
            local shard="${OUTPUT_DIR}/${OUTPUT_NAME}_shard${i}.jsonl"
            if [[ -f "$shard" ]]; then
                echo "  Shard $i: $(wc -l < "$shard") records"
            fi
            local ckpt="${shard}.checkpoint"
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

for model_name in "$GEN_MODEL" "$TARGET_MODEL"; do
    dir_name="models--$(echo "$model_name" | sed 's/\//--/g')"
    if [[ ! -d "$MODEL_CACHE/hub/$dir_name" ]]; then
        echo "ERROR: $model_name not found in cache"
        echo "  Expected: $MODEL_CACHE/hub/$dir_name"
        exit 1
    fi
    echo "  $model_name: $(du -sh "$MODEL_CACHE/hub/$dir_name" 2>/dev/null | cut -f1)"
done

avail=$(df --output=avail -BG "$OUTPUT_DIR" 2>/dev/null | tail -1 | tr -d ' G')
echo "  Output disk: ${avail:-?}G available"
echo ""


# ==================== START VLLM SERVERS ==================

echo "========================================================"
echo "  Starting $((N_PIPELINES * 2)) vLLM servers"
echo "========================================================"

for i in $(seq 0 $((N_PIPELINES - 1))); do
    gen_gpu=$((i * 2))
    target_gpu=$((i * 2 + 1))
    gen_port=$((GEN_PORT_BASE + i))
    target_port=$((TARGET_PORT_BASE + i))

    echo "[$(date +%H:%M:%S)] Pipeline $i: generator GPU $gen_gpu (port $gen_port), target GPU $target_gpu (port $target_port)"

    # Start generator
    CUDA_VISIBLE_DEVICES=$gen_gpu python3 -m vllm.entrypoints.openai.api_server \
        --model "$GEN_MODEL" \
        --port "$gen_port" \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.90 \
        --download-dir "$HF_HOME/hub" \
        --disable-log-requests \
        --disable-log-stats \
        2>"$LOG_DIR/vllm_gen${i}_${SLURM_JOB_ID:-$$}.log" &
    ALL_PIDS+=($!)

    # Start target
    CUDA_VISIBLE_DEVICES=$target_gpu python3 -m vllm.entrypoints.openai.api_server \
        --model "$TARGET_MODEL" \
        --port "$target_port" \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.90 \
        --download-dir "$HF_HOME/hub" \
        --disable-log-requests \
        --disable-log-stats \
        2>"$LOG_DIR/vllm_target${i}_${SLURM_JOB_ID:-$$}.log" &
    ALL_PIDS+=($!)
done

# Wait for all servers
echo ""
echo "[$(date +%H:%M:%S)] Waiting for all servers..."
ALL_READY=true
for i in $(seq 0 $((N_PIPELINES - 1))); do
    gen_port=$((GEN_PORT_BASE + i))
    target_port=$((TARGET_PORT_BASE + i))

    for server_info in "gen${i}:${gen_port}" "target${i}:${target_port}"; do
        IFS=: read -r name port <<< "$server_info"
        ready=false
        for attempt in $(seq 1 180); do
            if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
                echo "  $name (port $port) ready in $((attempt * 10))s"
                ready=true
                break
            fi
            sleep 10
        done
        if [ "$ready" = false ]; then
            echo "FATAL: $name (port $port) not ready after 30 min"
            tail -20 "$LOG_DIR/vllm_${name}_${SLURM_JOB_ID:-$$}.log" 2>/dev/null
            ALL_READY=false
        fi
    done
done

if [ "$ALL_READY" = false ]; then
    echo "FATAL: Not all servers started. Aborting."
    exit 1
fi

echo "[$(date +%H:%M:%S)] All $((N_PIPELINES * 2)) servers ready."
echo ""


# ==================== PARALLEL GENERATION =================

echo "========================================================"
echo "  Generating $N_PAIRS pairs across $N_PIPELINES pipelines"
echo "========================================================"

STAGE_START=$(date +%s)
PAIRS_PER_PIPELINE=$((N_PAIRS / N_PIPELINES))
REMAINDER=$((N_PAIRS % N_PIPELINES))

for i in $(seq 0 $((N_PIPELINES - 1))); do
    gen_port=$((GEN_PORT_BASE + i))
    target_port=$((TARGET_PORT_BASE + i))
    shard_seed=$((RANDOM_SEED + i * 10000))
    shard_file="${OUTPUT_DIR}/${OUTPUT_NAME}_shard${i}.jsonl"

    # Last pipeline gets the remainder
    shard_pairs=$PAIRS_PER_PIPELINE
    if [ $i -eq $((N_PIPELINES - 1)) ]; then
        shard_pairs=$((PAIRS_PER_PIPELINE + REMAINDER))
    fi

    echo "[$(date +%H:%M:%S)] Pipeline $i: $shard_pairs pairs, seed $shard_seed -> $shard_file"

    python3 "${SCRIPT_DIR}/interactive_generator.py" \
        --n-pairs "$shard_pairs" \
        --seed "$shard_seed" \
        --output "$shard_file" \
        --generator-backend vllm \
        --generator-model "$GEN_MODEL" \
        --generator-url "http://localhost:$gen_port" \
        --target-backend vllm \
        --target-model "$TARGET_MODEL" \
        --target-url "http://localhost:$target_port" \
        --checkpoint-interval "$CHECKPOINT_INTERVAL" \
        2>"$LOG_DIR/pipeline${i}_${SLURM_JOB_ID:-$$}.err" &
    PIPELINE_PIDS+=($!)
    ALL_PIDS+=($!)
done

# Wait for all pipelines, track failures
echo ""
echo "[$(date +%H:%M:%S)] Waiting for $N_PIPELINES pipelines..."
FAILED=0
for i in $(seq 0 $((N_PIPELINES - 1))); do
    pid=${PIPELINE_PIDS[$i]}
    if wait "$pid"; then
        echo "  Pipeline $i (pid $pid): completed"
    else
        echo "  Pipeline $i (pid $pid): FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

STAGE_END=$(date +%s)
STAGE_ELAPSED=$(( (STAGE_END - STAGE_START) / 60 ))

if [ $FAILED -eq $N_PIPELINES ]; then
    echo "FATAL: All pipelines failed"
    exit 1
fi
if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED/$N_PIPELINES pipelines failed. Merging available shards."
fi


# ==================== MERGE SHARDS ========================

echo ""
echo "[$(date +%H:%M:%S)] Merging shards..."
RAW_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}_raw.jsonl"

python3 -c "
import json, sys, os
sys.path.insert(0, '${SCRIPT_DIR}')
import build_semantic_datasetv11 as gen

records = []
for i in range(${N_PIPELINES}):
    shard = '${OUTPUT_DIR}/${OUTPUT_NAME}_shard' + str(i) + '.jsonl'
    if not os.path.exists(shard):
        print(f'  Shard {i}: MISSING (skipped)')
        continue
    count = 0
    with open(shard) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                count += 1
    print(f'  Shard {i}: {count} records')

print(f'Total raw: {len(records)} records')

# Dedup across shards (different seeds should produce different records,
# but strategy overlap may create near-duplicates)
before = len(records)
records = gen.deduplicate_dataset(records, threshold=0.70)
print(f'Cross-shard dedup: {before} -> {len(records)} ({before - len(records)} removed)')

gen.write_jsonl(records, '${RAW_FILE}')
print(f'Wrote {len(records)} records to ${RAW_FILE}')
"

if [ ! -f "$RAW_FILE" ]; then
    echo "ERROR: Merge failed"
    exit 1
fi

RAW_COUNT=$(wc -l < "$RAW_FILE")
echo "[$(date +%H:%M:%S)] Merged: $RAW_COUNT records, ${STAGE_ELAPSED} min generation"


# ==================== SHUTDOWN SERVERS ====================

echo ""
echo "[$(date +%H:%M:%S)] Shutting down vLLM servers..."
for pid in "${ALL_PIDS[@]}"; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
    fi
done
wait 2>/dev/null || true
ALL_PIDS=()  # Clear so cleanup doesn't double-kill
sleep 3


# ==================== COPY TO PERSISTENT ==================

echo ""
echo "--- Copying to persistent storage ---"
cp "$RAW_FILE" "$RESULTS_DIR/${OUTPUT_NAME}_raw.jsonl"

if [[ -f "$RESULTS_DIR/${OUTPUT_NAME}_raw.jsonl" ]]; then
    result_lines=$(wc -l < "$RESULTS_DIR/${OUTPUT_NAME}_raw.jsonl")
    echo "  Verified: $result_lines records ($(du -sh "$RESULTS_DIR/${OUTPUT_NAME}_raw.jsonl" | cut -f1))"
fi


# ==================== STATISTICS ==========================

echo ""
python3 -c "
import json
from collections import Counter

records = []
with open('${RAW_FILE}') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

mal = [r for r in records if r.get('label') == 1]
ben = [r for r in records if r.get('label') == 0]

jb = sum(1 for r in mal if r.get('llama_validation', {}).get('jailbreak_detected'))

print(f'Total: {len(records)} ({len(mal)} malicious, {len(ben)} benign)')
print(f'Llama jailbreak rate: {jb}/{len(mal)} ({jb/max(len(mal),1)*100:.1f}%)')

strategies = Counter(
    r.get('metadata', r.get('attack_metadata', {})).get('strategy', '?')
    for r in mal
)
print(f'Strategies: {dict(strategies.most_common())}')

pivots = Counter(
    r.get('llama_validation', {}).get('pivot_kind', 'none')
    for r in mal
)
print(f'Pivot kinds (Llama): {dict(pivots)}')

avg_turns = sum(len(r.get('turns', [])) for r in mal) / max(len(mal), 1)
print(f'Avg turns (malicious): {avg_turns:.1f}')

adaptations = sum(
    r.get('metadata', r.get('attack_metadata', {})).get('adaptations_used', 0)
    for r in mal
)
print(f'Total adaptations: {adaptations}')

# Per-strategy jailbreak rate
print(f'')
print(f'Per-strategy jailbreak rate:')
for strat, count in strategies.most_common():
    strat_jb = sum(
        1 for r in mal
        if r.get('metadata', r.get('attack_metadata', {})).get('strategy') == strat
        and r.get('llama_validation', {}).get('jailbreak_detected')
    )
    print(f'  {strat}: {strat_jb}/{count} ({strat_jb/max(count,1)*100:.0f}%)')

# Max unsafe distribution
max_unsafes = [
    r.get('llama_validation', {}).get('max_unsafe_score', 0)
    for r in mal
]
if max_unsafes:
    above_03 = sum(1 for u in max_unsafes if u >= 0.3)
    above_05 = sum(1 for u in max_unsafes if u >= 0.5)
    print(f'')
    print(f'Max unsafe score distribution:')
    print(f'  >= 0.3: {above_03}/{len(mal)} ({above_03/len(mal)*100:.0f}%)')
    print(f'  >= 0.5: {above_05}/{len(mal)} ({above_05/len(mal)*100:.0f}%)')
    print(f'  mean: {sum(max_unsafes)/len(max_unsafes):.3f}')

confs = [
    r.get('llama_validation', {}).get('avg_judge_confidence', 0)
    for r in mal if r.get('llama_validation', {}).get('validated')
]
if confs:
    print(f'Judge confidence: mean={sum(confs)/len(confs):.3f}')
"


# ==================== JOB METADATA ========================

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( (TOTAL_END - STAGE_START) / 60 ))

cat > "${RESULTS_DIR}/job_igen_${SLURM_JOB_ID:-$$}_meta.json" <<METAEOF
{
    "job_id": "${SLURM_JOB_ID:-interactive}",
    "node": "$(hostname)",
    "pipeline_version": "v11_interactive",
    "stage": "interactive_generation",
    "generator_model": "$GEN_MODEL",
    "target_model": "$TARGET_MODEL",
    "n_pairs": $N_PAIRS,
    "n_pipelines": $N_PIPELINES,
    "random_seed": $RANDOM_SEED,
    "final_records": $RAW_COUNT,
    "gen_elapsed_min": $STAGE_ELAPSED,
    "total_elapsed_min": $TOTAL_ELAPSED,
    "completed": "$(date -Iseconds)"
}
METAEOF


# ==================== SUMMARY =============================

echo ""
echo "========================================================"
echo "  Interactive Generation Complete (${TOTAL_ELAPSED} min)"
echo "  Pipelines:  $N_PIPELINES parallel"
echo "  Generator:  $GEN_MODEL"
echo "  Target:     $TARGET_MODEL"
echo "  Final:      $RAW_COUNT records"
echo "  Persistent: $RESULTS_DIR/${OUTPUT_NAME}_raw.jsonl"
echo ""
echo "  Next: validate with Qwen-7B"
echo "    VAL_MODEL=Qwen/Qwen2.5-7B-Instruct \\"
echo "    INPUT_FILE=${RAW_FILE} \\"
echo "    OUTPUT_NAME=${OUTPUT_NAME} \\"
echo "    HF_HUB_OFFLINE=1 \\"
echo "    sbatch launch_val.slurm"
echo "========================================================"
echo "[$(date +%H:%M:%S)] Done."