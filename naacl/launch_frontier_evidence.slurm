#!/bin/bash
#SBATCH --job-name=gl_front_ev
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=logs/front_ev_%j.out
#SBATCH --error=logs/front_ev_%j.err

# Stage B4. Two parallel evidence shards. Each shard uses one primary target GPU
# and one independent judge GPU. Override TARGET_MODEL only for a separate
# robustness experiment; canonical preparation rejects non-Qwen target outputs.
set -euo pipefail
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-mistralai/Mistral-Small-3.1-24B-Instruct-2503}"
N_SHARDS="${N_SHARDS:-2}"
TARGET_MAX_MODEL_LEN="${TARGET_MAX_MODEL_LEN:-16384}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-16384}"
JUDGE_MAX_CONTEXT_CHARS="${JUDGE_MAX_CONTEXT_CHARS:-40000}"
MAX_TOKENS="${MAX_TOKENS:-1280}"
MODEL_CACHE="${MODEL_CACHE:-$HOME/work/hf_models}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/staging/dataset_gen_output}"
INPUT_FILE="${INPUT_FILE:-$OUTPUT_DIR/frontier_qwen32_candidates.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_DIR/frontier_qwen32_evidence.jsonl}"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-$OUTPUT_DIR/frontier_qwen32_evidence}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/dataset_gen}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LOG_DIR="$PROJECT_DIR/logs"
SEED="${SEED:-42}"
UNSAFE_THRESHOLD="${UNSAFE_THRESHOLD:-0.50}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.55}"
WEAK_DELTA="${WEAK_DELTA:-0.25}"
STRONG_DELTA="${STRONG_DELTA:-0.40}"
NEGATIVE_CONTROL_TOLERANCE="${NEGATIVE_CONTROL_TOLERANCE:-0.15}"
MAX_TURN_INTERVENTIONS="${MAX_TURN_INTERVENTIONS:-4}"
MAX_POSITIVE_SPANS="${MAX_POSITIVE_SPANS:-6}"
MAX_NEGATIVE_SPANS="${MAX_NEGATIVE_SPANS:-2}"

if [[ "$N_SHARDS" -lt 1 || "$N_SHARDS" -gt 2 ]]; then
  echo "ERROR: require 1 <= N_SHARDS <= 2 for this 4-GPU job"
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
EXPECTED_RECORDS=$(grep -cve '^[[:space:]]*$' "$INPUT_FILE")
# Candidate materialization preserves Stage-B1 rollout provenance. Recheck it
# before allocating four GPUs so evidence cannot run with a mismatched target envelope.
python3 naacl/audit_frontier_rollout.py \
  --input "$INPUT_FILE" \
  --expected-records "$EXPECTED_RECORDS" \
  --expected-model "$TARGET_MODEL" \
  --expected-max-tokens "$MAX_TOKENS" \
  --expected-max-model-len "$TARGET_MAX_MODEL_LEN" \
  --max-transcript-chars "$JUDGE_MAX_CONTEXT_CHARS"

python3 naacl/check_frontier_environment.py \
  --model-cache "$MODEL_CACHE" --models "$TARGET_MODEL" "$JUDGE_MODEL"

echo "Evidence config: target=$TARGET_MODEL judge=$JUDGE_MODEL max_tokens=$MAX_TOKENS unsafe=$UNSAFE_THRESHOLD min_confidence=$MIN_CONFIDENCE weak=$WEAK_DELTA strong=$STRONG_DELTA"
echo "Context windows: target=$TARGET_MAX_MODEL_LEN judge=$JUDGE_MAX_MODEL_LEN; judge full-prefix budget=$JUDGE_MAX_CONTEXT_CHARS chars"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

PIDS=()
cleanup(){ code=$?; trap - EXIT INT TERM; for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null||true; done; wait 2>/dev/null||true; exit "$code"; }
trap cleanup EXIT INT TERM

for i in $(seq 0 $((N_SHARDS-1))); do
  tgpu=$((2*i)); jgpu=$((2*i+1)); tport=$((8200+2*i)); jport=$((8201+2*i))
  CUDA_VISIBLE_DEVICES=$tgpu python3 -m vllm.entrypoints.openai.api_server \
    --model "$TARGET_MODEL" --port "$tport" --dtype bfloat16 \
    --max-model-len "$TARGET_MAX_MODEL_LEN" --gpu-memory-utilization 0.92 \
    --download-dir "$HF_HOME/hub" --no-enable-log-requests --disable-log-stats \
    2>"$LOG_DIR/front_ev_target${i}_${SLURM_JOB_ID:-$$}.log" &
  PIDS+=($!)
  CUDA_VISIBLE_DEVICES=$jgpu python3 -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" --port "$jport" --dtype bfloat16 \
    --tokenizer-mode mistral --config-format mistral --load-format mistral \
    --max-model-len "$JUDGE_MAX_MODEL_LEN" --gpu-memory-utilization 0.90 \
    --download-dir "$HF_HOME/hub" --no-enable-log-requests --disable-log-stats \
    2>"$LOG_DIR/front_ev_judge${i}_${SLURM_JOB_ID:-$$}.log" &
  PIDS+=($!)
done

for i in $(seq 0 $((N_SHARDS-1))); do
  for port in $((8200+2*i)) $((8201+2*i)); do
    for attempt in $(seq 1 180); do
      curl -sf "http://localhost:$port/health" >/dev/null 2>&1 && break
      [[ "$attempt" -eq 180 ]] && { echo "ERROR: server $port not ready"; exit 1; }
      sleep 10
    done
  done
done

WORKERS=()
for i in $(seq 0 $((N_SHARDS-1))); do
  python3 naacl/frontier_evidence_analysis.py \
    --input "$INPUT_FILE" --output "${OUTPUT_FILE%.jsonl}.shard${i}.jsonl" \
    --checkpoint "${CHECKPOINT_PREFIX}.shard${i}.checkpoint.jsonl" \
    --target-model "$TARGET_MODEL" --target-base-url "http://localhost:$((8200+2*i))" \
    --judge-model "$JUDGE_MODEL" --judge-base-url "http://localhost:$((8201+2*i))" \
    --seed "$SEED" \
    --unsafe-threshold "$UNSAFE_THRESHOLD" \
    --min-confidence "$MIN_CONFIDENCE" \
    --weak-delta "$WEAK_DELTA" \
    --strong-delta "$STRONG_DELTA" \
    --negative-control-tolerance "$NEGATIVE_CONTROL_TOLERANCE" \
    --max-turn-interventions "$MAX_TURN_INTERVENTIONS" \
    --max-positive-spans "$MAX_POSITIVE_SPANS" \
    --max-negative-spans "$MAX_NEGATIVE_SPANS" \
    --max-tokens "$MAX_TOKENS" \
    --judge-max-context-chars "$JUDGE_MAX_CONTEXT_CHARS" \
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
if missing: raise SystemExit(f'missing {len(missing)} evidence records')
with open(out,'w',encoding='utf-8') as f:
    for r in source:f.write(json.dumps(merged[str(r['conversation_id'])],ensure_ascii=False)+'\n')
print('merged',len(source),'records ->',out)
PY

echo "Frontier evidence complete: $OUTPUT_FILE"
