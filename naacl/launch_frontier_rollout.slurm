#!/bin/bash
#SBATCH --job-name=gl_front_roll
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
#SBATCH --time=24:00:00
#SBATCH --output=logs/front_roll_%j.out
#SBATCH --error=logs/front_roll_%j.err

# Stage B1. Full-throttle default: four independent Qwen2.5-32B servers, one per
# A100, with two logical rollout workers per server so vLLM can continuously batch
# independent trajectories. Checkpoints are append-only and safe to resume across
# 24-hour allocations.
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
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_DIR/frontier_multi_author_qwen32_rollout_v2_2048_16k.jsonl}"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-$OUTPUT_DIR/frontier_multi_author_qwen32_rollout_v2_2048_16k}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/dataset_gen}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LOG_DIR="$PROJECT_DIR/logs"
SOURCE_PREFLIGHT_MODE="${SOURCE_PREFLIGHT_MODE:-auto}"  # auto|strict|schema|skip
SOURCE_EXPECTED_RECORDS="${SOURCE_EXPECTED_RECORDS:-}"
SOURCE_EXPECTED_PAIRS="${SOURCE_EXPECTED_PAIRS:-}"
SOURCE_EXPECTED_STANDALONE="${SOURCE_EXPECTED_STANDALONE:-}"
SOURCE_EXPECTED_SCENARIOS="${SOURCE_EXPECTED_SCENARIOS:-}"

if [[ "$GPU_COUNT" -lt 1 || "$GPU_COUNT" -gt 4 ]]; then
  echo "ERROR: require 1 <= GPU_COUNT <= 4"
  exit 2
fi
if [[ "$WORKERS_PER_GPU" -lt 1 || "$WORKERS_PER_GPU" -gt 4 ]]; then
  echo "ERROR: require 1 <= WORKERS_PER_GPU <= 4"
  exit 2
fi
MAX_SHARDS=$((GPU_COUNT * WORKERS_PER_GPU))
if [[ "$N_SHARDS" -lt 1 || "$N_SHARDS" -gt "$MAX_SHARDS" ]]; then
  echo "ERROR: N_SHARDS=$N_SHARDS exceeds available logical workers=$MAX_SHARDS"
  exit 2
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/.conda")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HOME="$MODEL_CACHE"
export TRANSFORMERS_CACHE="$MODEL_CACHE/hub"
export HF_HUB_CACHE="$MODEL_CACHE/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
cd "$PROJECT_DIR"

[[ -f "$INPUT_FILE" ]] || { echo "ERROR: input not found: $INPUT_FILE"; exit 2; }
python3 naacl/check_frontier_environment.py \
  --model-cache "$MODEL_CACHE" --models "$TARGET_MODEL"

run_strict_source_audit() {
  local records="$1"
  local expected_records="$SOURCE_EXPECTED_RECORDS"
  local expected_pairs="$SOURCE_EXPECTED_PAIRS"
  local expected_standalone="$SOURCE_EXPECTED_STANDALONE"
  local expected_scenarios="$SOURCE_EXPECTED_SCENARIOS"

  if [[ -z "$expected_records" ]]; then
    if [[ "$records" -eq 1500 ]]; then
      expected_records=1500; expected_pairs=600; expected_standalone=300; expected_scenarios=300
    elif [[ "$records" -eq 3000 ]]; then
      expected_records=3000; expected_pairs=1200; expected_standalone=600; expected_scenarios=600
    else
      echo "ERROR: strict source audit does not know construction counts for $records records."
      echo "Set SOURCE_EXPECTED_RECORDS/PAIRS/STANDALONE/SCENARIOS explicitly."
      exit 2
    fi
  fi

  : "${expected_pairs:?SOURCE_EXPECTED_PAIRS required for explicit strict audit}"
  : "${expected_standalone:?SOURCE_EXPECTED_STANDALONE required for explicit strict audit}"
  : "${expected_scenarios:?SOURCE_EXPECTED_SCENARIOS required for explicit strict audit}"
  python3 naacl/audit_frontier_source.py \
    --input "$INPUT_FILE" \
    --expected-records "$expected_records" \
    --expected-pairs "$expected_pairs" \
    --expected-standalone "$expected_standalone" \
    --expected-scenarios "$expected_scenarios"
}

RECORDS=$(grep -cve '^[[:space:]]*$' "$INPUT_FILE")
case "$SOURCE_PREFLIGHT_MODE" in
  strict)
    run_strict_source_audit "$RECORDS"
    ;;
  schema)
    python3 naacl/audit_frontier_source.py --input "$INPUT_FILE" --schema-only
    ;;
  auto)
    if [[ "$RECORDS" -eq 1500 || "$RECORDS" -eq 3000 ]]; then
      run_strict_source_audit "$RECORDS"
    else
      python3 naacl/audit_frontier_source.py --input "$INPUT_FILE" --schema-only
    fi
    ;;
  skip)
    echo "WARN: SOURCE_PREFLIGHT_MODE=skip; source audit explicitly bypassed"
    ;;
  *)
    echo "ERROR: SOURCE_PREFLIGHT_MODE must be auto|strict|schema|skip"
    exit 2
    ;;
esac

echo "B1 topology: GPUs=$GPU_COUNT workers_per_gpu=$WORKERS_PER_GPU logical_shards=$N_SHARDS"
echo "Target generation contract: max_tokens=$MAX_TOKENS max_model_len=$MAX_MODEL_LEN; require finish_reason=stop + completion_tokens"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

PIDS=()
cleanup(){ code=$?; trap - EXIT INT TERM; for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; exit "$code"; }
trap cleanup EXIT INT TERM

for i in $(seq 0 $((GPU_COUNT-1))); do
  port=$((PORT_BASE+i))
  CUDA_VISIBLE_DEVICES=$i python3 -m vllm.entrypoints.openai.api_server \
    --model "$TARGET_MODEL" --port "$port" --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --download-dir "$HF_HOME/hub" --no-enable-log-requests --disable-log-stats \
    2>"$LOG_DIR/front_target_${i}_${SLURM_JOB_ID:-$$}.log" &
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
  python3 naacl/rollout_frontier_source.py \
    --input "$INPUT_FILE" \
    --output "${OUTPUT_FILE%.jsonl}.shard${i}.jsonl" \
    --checkpoint "${CHECKPOINT_PREFIX}.shard${i}.checkpoint.jsonl" \
    --model "$TARGET_MODEL" --base-url "http://localhost:$((PORT_BASE+server_idx))" \
    --seed "$SEED" --max-tokens "$MAX_TOKENS" --max-model-len "$MAX_MODEL_LEN" \
    --shard-index "$i" --num-shards "$N_SHARDS" &
  WORKERS+=($!)
done
for p in "${WORKERS[@]}"; do wait "$p"; done

python3 - "$INPUT_FILE" "$OUTPUT_FILE" "$N_SHARDS" <<'PY'
import json, sys
src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
merged={}
for i in range(n):
    p=out[:-6]+f'.shard{i}.jsonl'
    for line in open(p, encoding='utf-8'):
        if line.strip():
            r=json.loads(line); merged[str(r['conversation_id'])]=r
source=[json.loads(x) for x in open(src,encoding='utf-8') if x.strip()]
missing=[r['conversation_id'] for r in source if str(r['conversation_id']) not in merged]
if missing: raise SystemExit(f'missing {len(missing)} rollout records')
with open(out,'w',encoding='utf-8') as f:
    for r in source: f.write(json.dumps(merged[str(r['conversation_id'])],ensure_ascii=False)+'\n')
print('merged',len(source),'records ->',out)
PY

python3 naacl/audit_frontier_rollout.py \
  --input "$OUTPUT_FILE" \
  --expected-records "$RECORDS" \
  --expected-model "$TARGET_MODEL" \
  --expected-max-tokens "$MAX_TOKENS" \
  --expected-max-model-len "$MAX_MODEL_LEN"

echo "Frontier rollout complete and completion-audited: $OUTPUT_FILE"
