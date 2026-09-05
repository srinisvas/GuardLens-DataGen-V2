#!/bin/bash
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --job-name=v12_bench_phase3
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Phase 3: Cross-judge agreement
#   Servers: Qwen3.5-27B (GPU 0, config_a judge), Mistral-Small-3.2 (GPU 1, config_b judge)
#   Runs:    model_bench.py --cross-judge-only
#
# CRITICAL: launch flags MUST match the corresponding Phase 1/2 judge
# launches exactly so cross-judge scores come from the same inference
# setup. That includes: Qwen prefix cache OFF, thinking OFF; Mistral with
# native tokenizer/config/load.

set -euo pipefail

REPO_ROOT="$HOME/projects/GuardLens-DataGen-V2"
RESULTS_DIR="$HOME/work/results/model_bench_2026"

source ~/.bashrc
conda activate dataset_gen

export HF_HUB_OFFLINE=1
export HF_HOME="$HOME/work/hf_models"
export HF_HUB_CACHE="$HOME/work/hf_models/hub"

mkdir -p "$REPO_ROOT/logs"
cd "$REPO_ROOT"

if [ ! -f "$RESULTS_DIR/config_a/raw_conversations.jsonl" ]; then
    echo "ERROR: $RESULTS_DIR/config_a/raw_conversations.jsonl missing. Run Phase 1 first."
    exit 1
fi
if [ ! -f "$RESULTS_DIR/config_b/raw_conversations.jsonl" ]; then
    echo "ERROR: $RESULTS_DIR/config_b/raw_conversations.jsonl missing. Run Phase 2 first."
    exit 1
fi

# Qwen prefix caching intentionally OFF to match Phase 1/2 judge deployment
echo "[$(date +%T)] Launching Qwen3.5-27B on GPU 0, port 8002 (thinking OFF, prefix cache OFF)"
CUDA_VISIBLE_DEVICES=0 \
    vllm serve Qwen/Qwen3.5-27B \
        --host 127.0.0.1 --port 8002 \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.95 \
        --reasoning-parser qwen3 \
        --language-model-only \
        --default-chat-template-kwargs '{"enable_thinking": false}' \
        > logs/qwen27b_phase3.log 2>&1 &
QWEN_PID=$!

echo "[$(date +%T)] Launching Mistral-Small-3.2 on GPU 1, port 8004 (mistral tokenizer)"
CUDA_VISIBLE_DEVICES=1 \
    vllm serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
        --host 127.0.0.1 --port 8004 \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.95 \
        --enable-prefix-caching \
        --language-model-only \
        --tokenizer_mode mistral \
        --config_format mistral \
        --load_format mistral \
        > logs/mistral24b_phase3.log 2>&1 &
MISTRAL_PID=$!

cleanup() {
    echo "[$(date +%T)] Cleaning up servers"
    kill $QWEN_PID $MISTRAL_PID 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

wait_for() {
    local port=$1 name=$2 max_wait=1500 elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
            echo "[$(date +%T)] $name ready on port $port"
            return 0
        fi
        sleep 15
        elapsed=$((elapsed + 15))
    done
    echo "[$(date +%T)] TIMEOUT waiting for $name on port $port"
    return 1
}

wait_for 8002 "Qwen3.5-27B"
wait_for 8004 "Mistral-Small-3.2"

echo ""
echo "[$(date +%T)] Running cross-judge agreement"
python model_bench.py \
    --config-file model_bench_configs.yaml \
    --cross-judge-only \
    --output-dir "$RESULTS_DIR"

echo ""
echo "[$(date +%T)] Phase 3 complete."
echo "Combined report: $RESULTS_DIR/bench_report.md"
echo "Cross-judge data: $RESULTS_DIR/cross_judge_agreement.json"
echo ""
echo "Read the combined report and decide on model lock:"
echo "  cat $RESULTS_DIR/bench_report.md"
