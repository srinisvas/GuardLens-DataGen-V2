#!/bin/bash
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --job-name=v12_bench_phase1
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Phase 1: Config A benchmark
#   Servers: Llama-3.3-70B (GPUs 0,1 TP=2), Qwen3.5-27B (GPU 2), Gemma-4-31B (GPU 3)
#   Runs:    model_bench.py --config-name config_a
#
# v2 corrections:
#   - --host 127.0.0.1 (no need to expose these to compute-node network)
#   - Qwen3.5 launched with --reasoning-parser qwen3 --language-model-only
#     --default-chat-template-kwargs '{"enable_thinking": false}' to
#     suppress thinking-mode reasoning traces from contaminating outputs.
#   - HF_HUB_OFFLINE=1 so any missing model fails fast instead of hanging
#     while trying to reach hf.co from a compute node.

set -euo pipefail

REPO_ROOT="$HOME/projects/GuardLens-DataGen-V2"
RESULTS_DIR="$HOME/work/results/model_bench_2026"

source ~/.bashrc
conda activate dataset_gen

# Offline HF mode: models must already be in the cache
export HF_HUB_OFFLINE=1
export HF_HOME="$HOME/work/hf_models"
export HF_HUB_CACHE="$HOME/work/hf_models/hub"

mkdir -p "$RESULTS_DIR" "$REPO_ROOT/logs"
cd "$REPO_ROOT"

# Pre-flight: verify staged models exist
for m in "models--meta-llama--Llama-3.3-70B-Instruct" \
         "models--Qwen--Qwen3.5-27B" \
         "models--google--gemma-4-31B-it"; do
    if ! find "$HF_HUB_CACHE" -maxdepth 2 -name "$m" -type d 2>/dev/null | grep -q .; then
        echo "FATAL: model $m not found in $HF_HUB_CACHE"
        echo "Run: python prestage_v12.py --config-file model_bench_configs.yaml"
        exit 1
    fi
done

# -----------------------------------------------------------
# Launch servers (localhost only)
# -----------------------------------------------------------
echo "[$(date +%T)] Launching Llama-3.3-70B on GPUs 0,1 (TP=2), port 8001"
CUDA_VISIBLE_DEVICES=0,1 \
    vllm serve meta-llama/Llama-3.3-70B-Instruct \
        --host 127.0.0.1 --port 8001 \
        --tensor-parallel-size 2 \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.95 \
        --enable-prefix-caching \
        > logs/llama70b_phase1.log 2>&1 &
LLAMA_PID=$!

echo "[$(date +%T)] Launching Qwen3.5-27B on GPU 2, port 8002 (thinking OFF)"
CUDA_VISIBLE_DEVICES=2 \
    vllm serve Qwen/Qwen3.5-27B \
        --host 127.0.0.1 --port 8002 \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.95 \
        --enable-prefix-caching \
        --reasoning-parser qwen3 \
        --language-model-only \
        --default-chat-template-kwargs '{"enable_thinking": false}' \
        > logs/qwen27b_phase1.log 2>&1 &
QWEN_PID=$!

echo "[$(date +%T)] Launching Gemma-4-31B on GPU 3, port 8003"
CUDA_VISIBLE_DEVICES=3 \
    vllm serve google/gemma-4-31B-it \
        --host 127.0.0.1 --port 8003 \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.95 \
        --enable-prefix-caching \
        > logs/gemma31b_phase1.log 2>&1 &
GEMMA_PID=$!

cleanup() {
    echo "[$(date +%T)] Cleaning up servers"
    kill $LLAMA_PID $QWEN_PID $GEMMA_PID 2>/dev/null || true
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

wait_for 8001 "Llama-3.3-70B"
wait_for 8002 "Qwen3.5-27B"
wait_for 8003 "Gemma-4-31B"

# -----------------------------------------------------------
# Smoke test (schema + judge contract + seed determinism)
# -----------------------------------------------------------
echo ""
echo "[$(date +%T)] Running smoke test on Config A servers"
python smoke_test_backends.py \
    --config-file model_bench_configs.yaml \
    --config-name config_a \
    --output-json "$RESULTS_DIR/config_a_preflight.json" \
    || { echo "FATAL: smoke test failed. See output above. Aborting."; exit 1; }

# -----------------------------------------------------------
# Run benchmark
# -----------------------------------------------------------
echo ""
echo "[$(date +%T)] Starting Config A benchmark (12 conversations per strategy)"
python model_bench.py \
    --config-file model_bench_configs.yaml \
    --config-name config_a \
    --objectives bench_objectives.jsonl \
    --n-per-strategy 12 \
    --strategies A,C,F,K \
    --output-dir "$RESULTS_DIR"

echo ""
echo "[$(date +%T)] Config A benchmark complete."
echo "Report at: $RESULTS_DIR/bench_report_config_a.md"
echo "Metrics at: $RESULTS_DIR/config_a/metrics.json"
echo ""
echo "Next: read the report, apply pass criteria, then submit Phase 2:"
echo "  sbatch slurm/02_bench_phase2.slurm"
