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
# v4 runtime corrections:
#   - Prefer the conda C/C++ runtime over the cluster GCC 11.2 runtime.
#   - Force vLLM's native sampler for every model. FlashInfer sampling JITs
#     CUDA kernels and requires nvcc/CUDA_HOME, which is not available on the
#     compute image. Native sampling is supported by vLLM and keeps the
#     benchmark backend identical across model families.
#   - Explicitly disable FlashInfer all-reduce. The pinned vLLM 0.28 default
#     is already off, but locking it prevents environment/default drift.
#   - Run runtime_compat_v12.py before loading weights. On Python 3.11 it
#     applies the exact upstream FlashInfer postponed-annotations fix needed
#     for TP communicator imports and then validates sqlite/CUDA/vLLM.
#   - Qwen3.5 uses explicit --no-enable-prefix-caching. vLLM 0.28 defaults
#     prefix caching ON, so merely omitting --enable-prefix-caching is not
#     sufficient for the hybrid GDN/Mamba model.

set -eo pipefail

REPO_ROOT="$HOME/projects/GuardLens-DataGen-V2"
RESULTS_DIR="${BENCH_RESULTS_DIR:-$HOME/work/results/model_bench_measurement_v1}"
LOG_DIR="$RESULTS_DIR/server_logs"

set +u
source ~/.bashrc
conda activate "$HOME/work/conda_envs/dataset_gen"
set -u

# The cluster injects GCC 11.2 libstdc++ into LD_LIBRARY_PATH. Conda's ICU
# requires CXXABI_1.3.15, so make the environment's matching C/C++ runtime
# take precedence.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Universal vLLM runtime policy for this benchmark. Do not let optional
# FlashInfer JIT paths introduce an nvcc dependency or model-specific sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0

export HF_HUB_OFFLINE=1
export HF_HOME="$HOME/work/hf_models"
export HF_HUB_CACHE="$HOME/work/hf_models/hub"

# Fail before loading models or overwriting the previous preflight.
if [ -e "$RESULTS_DIR/config_a" ] || [ -e "$RESULTS_DIR/config_a_preflight.json" ]; then
    echo "ERROR: Existing config_a artifacts in $RESULTS_DIR. Choose a fresh BENCH_RESULTS_DIR."
    exit 1
fi
mkdir -p "$RESULTS_DIR" "$REPO_ROOT/logs" "$LOG_DIR"
cd "$REPO_ROOT"

# Fail before model loading if the Python/C++ runtime, FlashInfer compatibility,
# CUDA visibility, or Slurm GPU exposure regresses.
echo "[$(date +%T)] Runtime/GPU compatibility preflight"
python runtime_compat_v12.py --required-gpus 4

# Pre-flight: verify staged models exist offline
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
        > "$LOG_DIR/llama70b_phase1.log" 2>&1 &
LLAMA_PID=$!

# Qwen3.5 prefix caching is explicitly OFF. vLLM 0.28 defaults it ON.
echo "[$(date +%T)] Launching Qwen3.5-27B on GPU 2, port 8002 (thinking OFF, prefix cache OFF)"
CUDA_VISIBLE_DEVICES=2 \
    vllm serve Qwen/Qwen3.5-27B \
        --host 127.0.0.1 --port 8002 \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.95 \
        --no-enable-prefix-caching \
        --reasoning-parser qwen3 \
        --language-model-only \
        --default-chat-template-kwargs '{"enable_thinking": false}' \
        > "$LOG_DIR/qwen27b_phase1.log" 2>&1 &
QWEN_PID=$!

echo "[$(date +%T)] Launching Gemma-4-31B on GPU 3, port 8003"
CUDA_VISIBLE_DEVICES=3 \
    vllm serve google/gemma-4-31B-it \
        --host 127.0.0.1 --port 8003 \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.95 \
        --enable-prefix-caching \
        > "$LOG_DIR/gemma31b_phase1.log" 2>&1 &
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

echo ""
echo "[$(date +%T)] Running smoke test on Config A servers"
python smoke_test_backends.py \
    --config-file model_bench_configs.yaml \
    --config-name config_a \
    --output-json "$RESULTS_DIR/config_a_preflight.json" \
    || { echo "FATAL: smoke test failed. See output above. Aborting."; exit 1; }

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
