#!/bin/bash
#SBATCH --job-name=v11_igen
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=18:00:00
#SBATCH --output=logs/igen_%j.out
#SBATCH --error=logs/igen_%j.err

# =========================================================
# launch_gen_interactive.slurm
#
# Interactive adversarial generation with two models:
#   GPU 0: Qwen-14B (generator — crafts user turns)
#   GPU 1: Llama-8B  (target — responds, gets fooled)
#
# The generator adapts user turns based on the target's
# actual responses, producing conversations that actively
# exploit the target's weaknesses.
#
# Usage:
#   sbatch launch_gen_interactive.slurm
#   N_PAIRS=20 sbatch launch_gen_interactive.slurm   # smoke test
# =========================================================

set -euo pipefail

# ==================== CONFIGURATION =======================

N_PAIRS="${N_PAIRS:-1000}"
RANDOM_SEED="${RANDOM_SEED:-42}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"

GEN_MODEL="${GEN_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
TARGET_MODEL="${TARGET_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
GEN_PORT="${GEN_PORT:-8000}"
TARGET_PORT="${TARGET_PORT:-9000}"

MODEL_CACHE="${MODEL_CACHE:-$HOME/work/hf_models}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/staging/dataset_gen_output}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/work/results/dataset_gen}"
OUTPUT_NAME="${OUTPUT_NAME:-semantic_multiturn_v11_interactive}"

CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/dataset_gen}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SCRIPT_DIR="${PROJECT_DIR}"
LOG_DIR="$PROJECT_DIR/logs"

# ==================== END CONFIGURATION ===================

GEN_PID=""
TARGET_PID=""

echo "========================================================"
echo "  Interactive Adversarial Generation (v11)"
echo "  Job:           ${SLURM_JOB_ID:-interactive}"
echo "  Node:          $(hostname)"
echo "  Generator:     $GEN_MODEL (GPU 0, port $GEN_PORT)"
echo "  Target:        $TARGET_MODEL (GPU 1, port $TARGET_PORT)"
echo "  N pairs:       $N_PAIRS"
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

stop_servers() {
    for pid_var in GEN_PID TARGET_PID; do
        pid="${!pid_var}"
        if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "[$(date +%H:%M:%S)] Stopping $pid_var (pid $pid)..."
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    sleep 3
}

cleanup() {
    local exit_code=$?
    echo ""
    echo "[$(date +%H:%M:%S)] Cleanup (exit code: $exit_code)..."
    stop_servers

    if [[ $exit_code -ne 0 ]]; then
        local ckpt="${OUTPUT_DIR}/${OUTPUT_NAME}_raw.jsonl.checkpoint"
        if [[ -f "$ckpt" ]]; then
            local ckpt_lines=$(wc -l < "$ckpt")
            echo "  Checkpoint: $ckpt_lines records at $ckpt"
            echo "  Re-submit to resume."
            cp "$ckpt" "$RESULTS_DIR/${OUTPUT_NAME}_PARTIAL.jsonl.checkpoint" 2>/dev/null || true
        fi
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
echo "  Starting vLLM servers"
echo "========================================================"

# --- GPU 0: Generator (Qwen-14B) ---
echo "[$(date +%H:%M:%S)] Starting generator: $GEN_MODEL on GPU 0, port $GEN_PORT"

CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model "$GEN_MODEL" \
    --port "$GEN_PORT" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --download-dir "$HF_HOME/hub" \
    --disable-log-requests \
    --disable-log-stats \
    2>"$LOG_DIR/vllm_gen14b_${SLURM_JOB_ID:-$$}.log" &
GEN_PID=$!

# --- GPU 1: Target (Llama-8B) ---
echo "[$(date +%H:%M:%S)] Starting target: $TARGET_MODEL on GPU 1, port $TARGET_PORT"

CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
    --model "$TARGET_MODEL" \
    --port "$TARGET_PORT" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --download-dir "$HF_HOME/hub" \
    --disable-log-requests \
    --disable-log-stats \
    2>"$LOG_DIR/vllm_target_${SLURM_JOB_ID:-$$}.log" &
TARGET_PID=$!

# --- Wait for both servers ---
for server_info in "generator:$GEN_PORT:$GEN_PID:vllm_gen14b" "target:$TARGET_PORT:$TARGET_PID:vllm_target"; do
    IFS=: read -r name port pid logprefix <<< "$server_info"
    echo "  Waiting for $name (port $port)..."
    for i in $(seq 1 180); do
        if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
            echo "  $name ready in $((i * 10))s"
            break
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "FATAL: $name died during startup. Last 30 lines:"
            tail -30 "$LOG_DIR/${logprefix}_${SLURM_JOB_ID:-$$}.log"
            exit 1
        fi
        if [ $i -eq 180 ]; then
            echo "FATAL: $name not ready after 30 min"
            tail -30 "$LOG_DIR/${logprefix}_${SLURM_JOB_ID:-$$}.log"
            exit 1
        fi
        sleep 10
    done
done

echo ""


# ==================== GENERATION ==========================

echo "========================================================"
echo "  Interactive Generation ($N_PAIRS pairs)"
echo "========================================================"

STAGE_START=$(date +%s)
RAW_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}_raw.jsonl"

python3 "${SCRIPT_DIR}/interactive_generator.py" \
    --n-pairs "$N_PAIRS" \
    --seed "$RANDOM_SEED" \
    --output "$RAW_FILE" \
    --generator-backend vllm \
    --generator-model "$GEN_MODEL" \
    --generator-url "http://localhost:$GEN_PORT" \
    --target-backend vllm \
    --target-model "$TARGET_MODEL" \
    --target-url "http://localhost:$TARGET_PORT" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL"

GEN_STATUS=$?

STAGE_END=$(date +%s)
STAGE_ELAPSED=$(( (STAGE_END - STAGE_START) / 60 ))

if [ ${GEN_STATUS} -ne 0 ] || [ ! -f "$RAW_FILE" ]; then
    echo "ERROR: Generation failed (exit $GEN_STATUS)"
    exit 1
fi

RAW_COUNT=$(wc -l < "$RAW_FILE")
echo "[$(date +%H:%M:%S)] Generation done: $RAW_COUNT records, ${STAGE_ELAPSED} min"


# ==================== AUGMENTATION ========================

echo ""
echo "[$(date +%H:%M:%S)] Augmentation..."

AUGMENTED_FILE="${OUTPUT_DIR}/${OUTPUT_NAME}_augmented.jsonl"

python3 "${SCRIPT_DIR}/augment_dataset.py" \
    --input "$RAW_FILE" \
    --output "$AUGMENTED_FILE" \
    --n-hard-negatives 150 \
    --n-borderline 80 \
    --noise-fraction 0.10 \
    --blur-fraction 0.10 \
    --false-lead-fraction 0.15

AUG_STATUS=$?
if [ ${AUG_STATUS} -ne 0 ]; then
    echo "WARNING: Augmentation failed, using raw file"
    cp "$RAW_FILE" "$AUGMENTED_FILE"
fi

AUG_COUNT=$(wc -l < "$AUGMENTED_FILE")
echo "[$(date +%H:%M:%S)] Augmented: $AUG_COUNT records"


# ==================== DEDUP ===============================

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

if [ $? -ne 0 ]; then
    echo "WARNING: Dedup failed, using augmented file"
    cp "$AUGMENTED_FILE" "$DEDUP_FILE"
fi

FINAL_COUNT=$(wc -l < "$DEDUP_FILE")


# ==================== SHUTDOWN SERVERS ====================

stop_servers


# ==================== COPY TO PERSISTENT =================

echo ""
echo "--- Copying to persistent storage ---"
cp "$DEDUP_FILE" "$RESULTS_DIR/${OUTPUT_NAME}_augmented_dedup.jsonl"
cp "$RAW_FILE" "$RESULTS_DIR/${OUTPUT_NAME}_raw.jsonl" 2>/dev/null || true

if [[ -f "$RESULTS_DIR/${OUTPUT_NAME}_augmented_dedup.jsonl" ]]; then
    result_lines=$(wc -l < "$RESULTS_DIR/${OUTPUT_NAME}_augmented_dedup.jsonl")
    echo "  Verified: $result_lines records ($(du -sh "$RESULTS_DIR/${OUTPUT_NAME}_augmented_dedup.jsonl" | cut -f1))"
fi


# ==================== STATISTICS ==========================

echo ""
python3 -c "
import json
from collections import Counter

records = []
with open('${DEDUP_FILE}') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

mal = [r for r in records if r.get('label') == 1]
ben = [r for r in records if r.get('label') == 0]

# Llama jailbreak rate from generation
jb = sum(1 for r in mal if r.get('llama_validation', {}).get('jailbreak_detected'))

print(f'Total: {len(records)} ({len(mal)} malicious, {len(ben)} benign)')
print(f'Llama jailbreak rate: {jb}/{len(mal)} ({jb/max(len(mal),1)*100:.1f}%)')

# Strategy breakdown
strategies = Counter(
    r.get('metadata', r.get('attack_metadata', {})).get('strategy', '?')
    for r in mal
)
print(f'Strategies: {dict(strategies.most_common())}')

# Pivot kinds from llama validation
pivots = Counter(
    r.get('llama_validation', {}).get('pivot_kind', 'none')
    for r in mal
)
print(f'Pivot kinds (Llama): {dict(pivots)}')

# Avg conversation length
avg_turns = sum(len(r.get('turns', [])) for r in mal) / max(len(mal), 1)
print(f'Avg turns (malicious): {avg_turns:.1f}')

# Adaptations used
adaptations = sum(
    r.get('metadata', r.get('attack_metadata', {})).get('adaptations_used', 0)
    for r in mal
)
print(f'Total adaptations used: {adaptations}')

# Judge confidence
confs = [
    r.get('llama_validation', {}).get('avg_judge_confidence', 0)
    for r in mal
    if r.get('llama_validation', {}).get('validated')
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
    "random_seed": $RANDOM_SEED,
    "raw_records": $RAW_COUNT,
    "augmented_records": $AUG_COUNT,
    "final_records": $FINAL_COUNT,
    "gen_elapsed_min": $STAGE_ELAPSED,
    "total_elapsed_min": $TOTAL_ELAPSED,
    "completed": "$(date -Iseconds)"
}
METAEOF


# ==================== SUMMARY =============================

echo ""
echo "========================================================"
echo "  Interactive Generation Complete (${TOTAL_ELAPSED} min)"
echo "  Raw:        $RAW_COUNT records"
echo "  Augmented:  $AUG_COUNT records"
echo "  Final:      $FINAL_COUNT records"
echo "  Persistent: $RESULTS_DIR/${OUTPUT_NAME}_augmented_dedup.jsonl"
echo ""
echo "  Next: validate with Qwen-7B"
echo "  INPUT_FILE=$DEDUP_FILE OUTPUT_NAME=${OUTPUT_NAME} sbatch launch_val.slurm"
echo "========================================================"
echo "[$(date +%H:%M:%S)] Done."
