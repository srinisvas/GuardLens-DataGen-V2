#!/bin/bash
#SBATCH --job-name=gl_naacl_ev
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/naacl_ev_%j.out
#SBATCH --error=logs/naacl_ev_%j.err

# Paired NAACL evidence replay.
# GPU 0: Llama target replay
# GPU 1: Mistral outcome judge
#
# No attack generation occurs in this job.

set -euo pipefail

TARGET_MODEL="${TARGET_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
TARGET_PORT="${TARGET_PORT:-8000}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

MODEL_CACHE="${MODEL_CACHE:-$HOME/work/hf_models}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/staging/dataset_gen_output}"
INPUT_FILE="${INPUT_FILE:-$OUTPUT_DIR/naacl_independent_merged.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_DIR/naacl_evidence.jsonl}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/dataset_gen}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LOG_DIR="$PROJECT_DIR/logs"

MAX_POSITIVE_SPANS="${MAX_POSITIVE_SPANS:-6}"
MAX_NEGATIVE_SPANS="${MAX_NEGATIVE_SPANS:-2}"
SEED="${SEED:-42}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if command -v module &>/dev/null; then
    for mod in cuda/12.1 cuda/12.2 cuda/12.4 cuda/12 cuda; do
        module load "$mod" 2>/dev/null && break
    done
fi

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/.conda")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HOME="$MODEL_CACHE"
export TRANSFORMERS_CACHE="$MODEL_CACHE/hub"
export HF_HUB_CACHE="$MODEL_CACHE/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

cd "$PROJECT_DIR"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: input not found: $INPUT_FILE"
    exit 2
fi

PIDS=()
cleanup() {
    local code=$?
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit "$code"
}
trap cleanup EXIT INT TERM

echo "Target replay: $TARGET_MODEL on GPU 0, port $TARGET_PORT"
echo "Judge:         $JUDGE_MODEL on GPU 1, port $JUDGE_PORT"
echo "Input:         $INPUT_FILE"
echo "Output:        $OUTPUT_FILE"
echo "Positive spans per record: $MAX_POSITIVE_SPANS"
echo "Negative spans per record: $MAX_NEGATIVE_SPANS"

CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model "$TARGET_MODEL" \
    --port "$TARGET_PORT" \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.90 \
    --download-dir "$HF_HOME/hub" \
    --disable-log-requests \
    --disable-log-stats \
    2>"$LOG_DIR/naacl_target_${SLURM_JOB_ID}.log" &
PIDS+=($!)

CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" \
    --port "$JUDGE_PORT" \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.90 \
    --download-dir "$HF_HOME/hub" \
    --disable-log-requests \
    --disable-log-stats \
    2>"$LOG_DIR/naacl_judge_${SLURM_JOB_ID}.log" &
PIDS+=($!)

wait_ready() {
    local port="$1"
    local name="$2"
    for attempt in $(seq 1 180); do
        if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
            echo "$name ready after $((attempt * 10)) seconds"
            return 0
        fi
        sleep 10
    done
    echo "ERROR: $name did not become ready"
    return 1
}

wait_ready "$TARGET_PORT" "target"
wait_ready "$JUDGE_PORT" "judge"

python3 naacl/evidence_analysis.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --target-model "$TARGET_MODEL" \
    --target-base-url "http://localhost:$TARGET_PORT" \
    --judge-model "$JUDGE_MODEL" \
    --judge-base-url "http://localhost:$JUDGE_PORT" \
    --seed "$SEED" \
    --max-positive-spans "$MAX_POSITIVE_SPANS" \
    --max-negative-spans "$MAX_NEGATIVE_SPANS"

python3 naacl/audit_repaired_dataset.py --input "$OUTPUT_FILE"

echo "NAACL paired evidence run complete: $OUTPUT_FILE"
