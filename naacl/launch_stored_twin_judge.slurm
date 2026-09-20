#!/bin/bash
#SBATCH --job-name=a_twin_judge
#SBATCH --account=V_cs_hat_capstone_mkhan74
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/a_twin_judge_%j.out
#SBATCH --error=logs/a_twin_judge_%j.err

set -euo pipefail

JUDGE_MODEL="${JUDGE_MODEL:-}"
PORT="${VLLM_PORT:-8000}"
MODEL_CACHE="${MODEL_CACHE:-$HOME/work/hf_models}"
CONDA_ENV="${CONDA_ENV:-$HOME/work/conda_envs/dataset_gen}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

INPUT_FILE="${INPUT_FILE:-$PROJECT_DIR/results-new/naacl_evidence_clean.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-$PROJECT_DIR/results-new/naacl_evidence_twins_judged.jsonl}"
STATS_FILE="${STATS_FILE:-$PROJECT_DIR/results-new/naacl_evidence_twins_judged_stats.json}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-$PROJECT_DIR/results-new/naacl_evidence_twins_judged.checkpoint.jsonl}"

CANDIDATE_FILE="${CANDIDATE_FILE:-$PROJECT_DIR/results-new/naacl_legacy_twins_restored_candidate.jsonl}"
CANDIDATE_STATS="${CANDIDATE_STATS:-$PROJECT_DIR/results-new/naacl_legacy_twins_restored_candidate_stats.json}"
EXCLUDED_FILE="${EXCLUDED_FILE:-$PROJECT_DIR/results-new/naacl_legacy_twins_restored_excluded.jsonl}"

mkdir -p "$PROJECT_DIR/logs" "$HOME/work/.tmp" "$HOME/work/.pip_cache"

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/.conda")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export TMPDIR="$HOME/work/.tmp"
export PIP_CACHE_DIR="$HOME/work/.pip_cache"
export HF_HOME="$MODEL_CACHE"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export VLLM_USE_FLASHINFER_SAMPLER=0

cd "$PROJECT_DIR"

echo "========================================================"
echo " Dataset A original-twin stored-response judge"
echo "========================================================"
echo "Input:                 $INPUT_FILE"
echo "Judge model request:   ${JUDGE_MODEL:-AUTO_FROM_PROVENANCE}"
echo "Target replay:         NO"
echo "Independent replay:    NO"
echo "Conversation rewrite:  NO"
echo "Output:                $OUTPUT_FILE"
echo "Candidate:             $CANDIDATE_FILE"
echo "========================================================"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: input not found: $INPUT_FILE"
    exit 1
fi

echo "--- Python syntax/import preflight ---"
python -m py_compile \
    naacl/audit_legacy_twin_restoration_input.py \
    naacl/judge_stored_twin_responses.py \
    naacl/restore_legacy_twins.py \
    naacl/prepare_dataset.py

echo "--- CPU twin-restoration input audit ---"
python naacl/audit_legacy_twin_restoration_input.py --input "$INPUT_FILE"

echo "--- Existing independent-validation provenance ---"
INDEPENDENT_MODELS=$(python - "$INPUT_FILE" <<'PY'
import json, sys
from collections import Counter
path = sys.argv[1]
c = Counter()
unknown = 0
with open(path) as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("label") != 0 or r.get("family") != "interactive_benign_twin":
            continue
        model = (
            r.get("independent_validation_model")
            or (r.get("independent_validation", {}) or {}).get("model_used")
            or (r.get("causal_validation", {}) or {}).get("model_used")
        )
        if model:
            c[str(model)] += 1
        else:
            unknown += 1
for model, n in sorted(c.items()):
    print(f"MODEL\t{model}\t{n}")
print(f"UNKNOWN\t{unknown}")
PY
)
echo "$INDEPENDENT_MODELS"

KNOWN_MODEL_COUNT=$(printf '%s\n' "$INDEPENDENT_MODELS" | awk '$1=="MODEL"{n++} END{print n+0}')
if [[ "$KNOWN_MODEL_COUNT" -ne 1 ]]; then
    echo "ERROR: expected exactly one known independent-validation model"
    exit 1
fi

RECORDED_MODEL=$(printf '%s\n' "$INDEPENDENT_MODELS" | awk '$1=="MODEL"{print $2}')
if [[ -z "$JUDGE_MODEL" ]]; then
    JUDGE_MODEL="$RECORDED_MODEL"
    echo "Judge model auto-selected from recorded provenance: $JUDGE_MODEL"
elif [[ "$RECORDED_MODEL" != "$JUDGE_MODEL" ]]; then
    echo "ERROR: requested judge model differs from recorded independent model"
    echo "  recorded: $RECORDED_MODEL"
    echo "  requested: $JUDGE_MODEL"
    echo "Use the recorded model so the restoration does not introduce a new judge regime."
    exit 1
fi

MODEL_DIR="$MODEL_CACHE/hub/models--$(echo "$JUDGE_MODEL" | sed 's#/#--#g')"
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: judge model not found in offline cache: $MODEL_DIR"
    exit 1
fi

echo "--- Starting vLLM judge server ---"
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server     --model "$JUDGE_MODEL"     --port "$PORT"     --dtype bfloat16     --max-model-len 8192     --gpu-memory-utilization 0.90     --download-dir "$HF_HOME/hub"     --no-enable-log-requests     --disable-log-stats     2>"logs/vllm_a_twin_judge_${SLURM_JOB_ID:-$$}.log" &
SERVER_PID=$!

cleanup() {
    local code=$?
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
    exit "$code"
}
trap cleanup EXIT

for attempt in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "vLLM ready"
        break
    fi
    if [[ "$attempt" -eq 180 ]]; then
        echo "ERROR: vLLM did not become ready"
        exit 1
    fi
    sleep 10
done

python naacl/judge_stored_twin_responses.py     --input "$INPUT_FILE"     --output "$OUTPUT_FILE"     --stats-output "$STATS_FILE"     --checkpoint "$CHECKPOINT_FILE"     --backend vllm     --model "$JUDGE_MODEL"     --base-url "http://localhost:$PORT"     --checkpoint-interval 10

echo "--- Building candidate restored Dataset A ---"
python naacl/restore_legacy_twins.py     --input "$OUTPUT_FILE"     --output "$CANDIDATE_FILE"     --stats-output "$CANDIDATE_STATS"     --excluded-output "$EXCLUDED_FILE"     --seed 42

echo "========================================================"
echo " Twin restoration candidate complete"
echo " Frozen dataset modified: NO"
echo " New target replay:       NO"
echo " New independent replay:  NO"
echo "========================================================"
cat "$CANDIDATE_STATS"
