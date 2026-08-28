#!/bin/bash
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --job-name=v12_bench_phase2
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Phase 2: Config B benchmark
#   Servers: Llama-3.3-70B (GPUs 0,1 TP=2), Qwen3.5-27B (GPU 2), Mistral-Small-3.2 (GPU 3)
#   Runs:    model_bench.py --config-name config_b

set -euo pipefail

REPO_ROOT="$HOME/projects/GuardLens-DataGen-V2"
RESULTS_DIR="$HOME/work/results/model_bench_2026"

source ~/.bashrc
conda activate dataset_gen

export HF_TOKEN=$(cat "$HOME/.hf_token")
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME="$HOME/work/hf_models"
export HF_HUB_CACHE="$HOME/work/hf_models/hub"

mkdir -p "$RESULTS_DIR" "$REPO_ROOT/logs"
cd "$REPO_ROOT"

# -----------------------------------------------------------
# Launch servers
# -----------------------------------------------------------
echo "[$(date +%T)] Launching Llama-3.3-70B on GPUs 0,1 (TP=2), port 8001"
CUDA_VISIBLE_DEVICES=0,1 \
    vllm serve meta-llama/Llama-3.3-70B-Instruct \
        --host 0.0.0.0 --port 8001 \
        --tensor-parallel-size 2 \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.95 \
        --enable-prefix-caching \
        > logs/llama70b_phase2.log 2>&1 &
LLAMA_PID=$!

echo "[$(date +%T)] Launching Qwen3.5-27B on GPU 2, port 8002"
CUDA_VISIBLE_DEVICES=2 \
    vllm serve Qwen/Qwen3.5-27B \
        --host 0.0.0.0 --port 8002 \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.95 \
        --enable-prefix-caching \
        > logs/qwen27b_phase2.log 2>&1 &
QWEN_PID=$!

echo "[$(date +%T)] Launching Mistral-Small-3.2-24B on GPU 3, port 8004"
CUDA_VISIBLE_DEVICES=3 \
    vllm serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
        --host 0.0.0.0 --port 8004 \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.95 \
        --enable-prefix-caching \
        > logs/mistral24b_phase2.log 2>&1 &
MISTRAL_PID=$!

cleanup() {
    echo "[$(date +%T)] Cleaning up servers"
    kill $LLAMA_PID $QWEN_PID $MISTRAL_PID 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

# -----------------------------------------------------------
# Wait for readiness
# -----------------------------------------------------------
wait_for() {
    local port=$1 name=$2 max_wait=1500 elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "[$(date +%T)] $name ready on port $port"
            return 0
        fi
        sleep 15
        elapsed=$((elapsed + 15))
    done
    echo "[$(date +%T)] TIMEOUT waiting for $name on port $port"
    return 1
}

wait_for 8001 "Llama-3.3-70B"
wait_for 8002 "Qwen3.5-27B"
wait_for 8004 "Mistral-Small-3.2"

# -----------------------------------------------------------
# Smoke test
# -----------------------------------------------------------
echo ""
echo "[$(date +%T)] Running smoke test on Config B servers"
python smoke_test_backends.py \
    --config-file model_bench_configs.yaml \
    --config-name config_b \
    --output-json "$RESULTS_DIR/config_b_preflight.json" \
    || { echo "FATAL: smoke test failed. Aborting."; exit 1; }

# -----------------------------------------------------------
# Run benchmark
# -----------------------------------------------------------
echo ""
echo "[$(date +%T)] Starting Config B benchmark (12 conversations per strategy)"
python model_bench.py \
    --config-file model_bench_configs.yaml \
    --config-name config_b \
    --objectives bench_objectives.jsonl \
    --n-per-strategy 12 \
    --strategies A,C,F,K \
    --output-dir "$RESULTS_DIR"

echo ""
echo "[$(date +%T)] Config B benchmark complete."
echo "Report at: $RESULTS_DIR/bench_report_config_b.md"
echo "Metrics at: $RESULTS_DIR/config_b/metrics.json"
echo ""
echo "Next: submit Phase 3 (cross-judge, no reload needed):"
echo "  sbatch slurm/03_bench_phase3.slurm"
