#!/bin/bash
#SBATCH --job-name=gl_front_roll3
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
#SBATCH --time=24:00:00
#SBATCH --output=logs/front_roll3_%j.out
#SBATCH --error=logs/front_roll3_%j.err

# Stage B1 v3. Deterministic production runtime on A100: vLLM batch invariance
# plus eager execution. This removes accidental scheduler/batch numerical variance
# from later factual/counterfactual comparisons.
set -euo pipefail

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
GPU_COUNT="${GPU_COUNT:-4}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
N_SHARDS="${N_SHARDS:-$((GPU_COUNT * WORKERS_PER_GPU))}"
PORT_BASE="${PORT_BASE:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
SEED="${SEED:-42}"
MODEL_CACHE="${MODEL_CACHE:-$HOME/work/hf_models}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/staging/dataset_gen_output}"
INPUT_FILE="${INPUT_FILE:-$OUTPUT_DIR/guardlens_source_merged_v3_3000.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_DIR/frontier_multi_author_qwen32_rollout_v3_bi_eager_2048_16k.jsonl}"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-$OUTPUT_DIR/frontier_multi_author_qwen32_rollout_v3_bi_eager_2048_16k}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/dataset_gen}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LOG_DIR="$PROJECT_DIR/logs"
SOURCE_PREFLIGHT_MODE="${SOURCE_PREFLIGHT_MODE:-auto}"

if [[ "$GPU_COUNT" -lt 1 || "$GPU_COUNT" -gt 4 ]]; then echo "ERROR: require 1 <= GPU_COUNT <= 4"; exit 2; fi
if [[ "$WORKERS_PER_GPU" -lt 1 || "$WORKERS_PER_GPU" -gt 4 ]]; then echo "ERROR: require 1 <= WORKERS_PER_GPU <= 4"; exit 2; fi
MAX_SHARDS=$((GPU_COUNT * WORKERS_PER_GPU))
if [[ "$N_SHARDS" -lt 1 || "$N_SHARDS" -gt "$MAX_SHARDS" ]]; then echo "ERROR: invalid N_SHARDS=$N_SHARDS"; exit 2; fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/.conda")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_BATCH_INVARIANT=1
export HF_HOME="$MODEL_CACHE"
export TRANSFORMERS_CACHE="$MODEL_CACHE/hub"
export HF_HUB_CACHE="$MODEL_CACHE/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
cd "$PROJECT_DIR"

python3 - <<'PY'
import os, vllm
import vllm.envs as envs
assert os.environ.get("VLLM_BATCH_INVARIANT") == "1"
assert "VLLM_BATCH_INVARIANT" in getattr(envs, "environment_variables", {})
print(f"Deterministic runtime preflight: vLLM={vllm.__version__} batch_invariant=1 enforce_eager=1")
PY

[[ -f "$INPUT_FILE" ]] || { echo "ERROR: input not found: $INPUT_FILE"; exit 2; }
RECORDS=$(grep -cve '^[[:space:]]*$' "$INPUT_FILE")
python3 naacl/check_frontier_environment.py --model-cache "$MODEL_CACHE" --models "$TARGET_MODEL"
case "$SOURCE_PREFLIGHT_MODE" in
  strict)
    if [[ "$RECORDS" -eq 3000 ]]; then
      python3 naacl/audit_frontier_source.py --input "$INPUT_FILE" --expected-records 3000 --expected-pairs 1200 --expected-standalone 600 --expected-scenarios 600
    elif [[ "$RECORDS" -eq 1500 ]]; then
      python3 naacl/audit_frontier_source.py --input "$INPUT_FILE" --expected-records 1500 --expected-pairs 600 --expected-standalone 300 --expected-scenarios 300
    else
      echo "ERROR: strict source audit only knows frozen 1500/3000 constructions"; exit 2
    fi
    ;;
  schema) python3 naacl/audit_frontier_source.py --input "$INPUT_FILE" --schema-only ;;
  auto)
    if [[ "$RECORDS" -eq 3000 ]]; then
      python3 naacl/audit_frontier_source.py --input "$INPUT_FILE" --expected-records 3000 --expected-pairs 1200 --expected-standalone 600 --expected-scenarios 600
    elif [[ "$RECORDS" -eq 1500 ]]; then
      python3 naacl/audit_frontier_source.py --input "$INPUT_FILE" --expected-records 1500 --expected-pairs 600 --expected-standalone 300 --expected-scenarios 300
    else
      python3 naacl/audit_frontier_source.py --input "$INPUT_FILE" --schema-only
    fi
    ;;
  skip) echo "WARN: source preflight explicitly skipped" ;;
  *) echo "ERROR: SOURCE_PREFLIGHT_MODE must be auto|strict|schema|skip"; exit 2 ;;
esac

echo "B1-v3 deterministic topology: GPUs=$GPU_COUNT workers_per_gpu=$WORKERS_PER_GPU shards=$N_SHARDS"
echo "Runtime: VLLM_BATCH_INVARIANT=1 --enforce-eager"
echo "Target: $TARGET_MODEL max_tokens=$MAX_TOKENS context=$MAX_MODEL_LEN"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

PIDS=()
cleanup(){ code=$?; trap - EXIT INT TERM; for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; exit "$code"; }
trap cleanup EXIT INT TERM

for i in $(seq 0 $((GPU_COUNT-1))); do
  port=$((PORT_BASE+i))
  CUDA_VISIBLE_DEVICES=$i VLLM_BATCH_INVARIANT=1 python3 -m vllm.entrypoints.openai.api_server \
    --model "$TARGET_MODEL" --port "$port" --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.92 \
    --enable-prefix-caching --enforce-eager \
    --download-dir "$HF_HOME/hub" --no-enable-log-requests --disable-log-stats \
    2>"$LOG_DIR/front_target_v3_${i}_${SLURM_JOB_ID:-$$}.log" &
  PIDS+=($!)
  sleep 3
done

for i in $(seq 0 $((GPU_COUNT-1))); do
  port=$((PORT_BASE+i))
  for attempt in $(seq 1 180); do
    curl -sf "http://localhost:$port/health" >/dev/null 2>&1 && break
    [[ "$attempt" -eq 180 ]] && { echo "ERROR: target server $i not ready"; exit 1; }
    sleep 10
  done
done

WORKERS=()
for i in $(seq 0 $((N_SHARDS-1))); do
  server_idx=$((i % GPU_COUNT))
  python3 naacl/rollout_frontier_source_v3.py \
    --input "$INPUT_FILE" \
    --output "${OUTPUT_FILE%.jsonl}.shard${i}.jsonl" \
    --checkpoint "${CHECKPOINT_PREFIX}.shard${i}.checkpoint.jsonl" \
    --model "$TARGET_MODEL" --base-url "http://localhost:$((PORT_BASE+server_idx))" \
    --seed "$SEED" --max-tokens "$MAX_TOKENS" --max-model-len "$MAX_MODEL_LEN" \
    --shard-index "$i" --num-shards "$N_SHARDS" &
  WORKERS+=($!)
done
for p in "${WORKERS[@]}"; do wait "$p"; done

python3 naacl/merge_stage_shards.py \
  --source "$INPUT_FILE" --output "$OUTPUT_FILE" --num-shards "$N_SHARDS" \
  --shard-pattern "${OUTPUT_FILE%.jsonl}.shard{i}.jsonl"

python3 naacl/audit_frontier_rollout_v3.py \
  --input "$OUTPUT_FILE" --expected-records "$RECORDS" \
  --expected-model "$TARGET_MODEL" --expected-max-tokens "$MAX_TOKENS" \
  --expected-max-model-len "$MAX_MODEL_LEN"

echo "Frontier rollout v3 deterministic complete and audited: $OUTPUT_FILE"
