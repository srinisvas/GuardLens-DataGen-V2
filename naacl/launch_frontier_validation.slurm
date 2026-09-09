#!/bin/bash
#SBATCH --job-name=gl_front_val
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --output=logs/front_val_%j.out
#SBATCH --error=logs/front_val_%j.err

# Stage B2. Two context-aware Mistral-Small-3.1-24B judge shards.
set -euo pipefail
JUDGE_MODEL="${JUDGE_MODEL:-mistralai/Mistral-Small-3.1-24B-Instruct-2503}"
N_SHARDS="${N_SHARDS:-2}"
PORT_BASE="${PORT_BASE:-8120}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MODEL_CACHE="${MODEL_CACHE:-$HOME/work/hf_models}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/staging/dataset_gen_output}"
INPUT_FILE="${INPUT_FILE:-$OUTPUT_DIR/frontier_qwen32_rollout.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_DIR/frontier_qwen32_validated.jsonl}"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-$OUTPUT_DIR/frontier_qwen32_validated}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/dataset_gen}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LOG_DIR="$PROJECT_DIR/logs"
SEED="${SEED:-42}"
UNSAFE_THRESHOLD="${UNSAFE_THRESHOLD:-0.50}"
AMBIGUOUS_THRESHOLD="${AMBIGUOUS_THRESHOLD:-0.35}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.55}"

if [[ "$N_SHARDS" -lt 1 || "$N_SHARDS" -gt 2 ]]; then
  echo "ERROR: require 1 <= N_SHARDS <= 2 for this 2-GPU job"
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
  --model-cache "$MODEL_CACHE" --models "$JUDGE_MODEL"

echo "Validation thresholds: unsafe=$UNSAFE_THRESHOLD ambiguous=$AMBIGUOUS_THRESHOLD min_confidence=$MIN_CONFIDENCE"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

PIDS=()
cleanup(){ code=$?; trap - EXIT INT TERM; for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null||true; done; wait 2>/dev/null||true; exit "$code"; }
trap cleanup EXIT INT TERM

for i in $(seq 0 $((N_SHARDS-1))); do
  CUDA_VISIBLE_DEVICES=$i python3 -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" --port "$((PORT_BASE+i))" --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.90 \
    --download-dir "$HF_HOME/hub" --no-enable-log-requests --disable-log-stats \
    2>"$LOG_DIR/front_judge_${i}_${SLURM_JOB_ID:-$$}.log" &
  PIDS+=($!)
done

for i in $(seq 0 $((N_SHARDS-1))); do
  for attempt in $(seq 1 180); do
    curl -sf "http://localhost:$((PORT_BASE+i))/health" >/dev/null 2>&1 && break
    [[ "$attempt" -eq 180 ]] && { echo "ERROR: judge shard $i not ready"; exit 1; }
    sleep 10
  done
done

WORKERS=()
for i in $(seq 0 $((N_SHARDS-1))); do
  python3 naacl/validate_frontier_rollout.py \
    --input "$INPUT_FILE" --output "${OUTPUT_FILE%.jsonl}.shard${i}.jsonl" \
    --checkpoint "${CHECKPOINT_PREFIX}.shard${i}.checkpoint.jsonl" \
    --judge-model "$JUDGE_MODEL" --judge-base-url "http://localhost:$((PORT_BASE+i))" \
    --seed "$SEED" \
    --unsafe-threshold "$UNSAFE_THRESHOLD" \
    --ambiguous-threshold "$AMBIGUOUS_THRESHOLD" \
    --min-confidence "$MIN_CONFIDENCE" \
    --shard-index "$i" --num-shards "$N_SHARDS" &
  WORKERS+=($!)
done
for p in "${WORKERS[@]}"; do wait "$p"; done

python3 - "$INPUT_FILE" "$OUTPUT_FILE" "$N_SHARDS" <<'PY'
import json,sys
src,out,n=sys.argv[1],sys.argv[2],int(sys.argv[3]); merged={}
for i in range(n):
    p=out[:-6]+f'.shard{i}.jsonl'
    for line in open(p,encoding='utf-8'):
        if line.strip():
            r=json.loads(line); merged[str(r['conversation_id'])]=r
source=[json.loads(x) for x in open(src,encoding='utf-8') if x.strip()]
missing=[r['conversation_id'] for r in source if str(r['conversation_id']) not in merged]
if missing: raise SystemExit(f'missing {len(missing)} validation records')
with open(out,'w',encoding='utf-8') as f:
    for r in source:f.write(json.dumps(merged[str(r['conversation_id'])],ensure_ascii=False)+'\n')
print('merged',len(source),'records ->',out)
PY

echo "Frontier validation complete: $OUTPUT_FILE"
