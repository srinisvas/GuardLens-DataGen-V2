#!/bin/bash
# Phase 0: Freeze v11 + contamination baseline
#
# Runs on LOGIN NODE (CPU only). Total time: ~30-40 min.
#
# Usage:
#   cd ~/projects/GuardLens-DataGen-V2
#   bash slurm/00_freeze_and_contamination.sh
#
# Overrides:
#   V11_SOURCE=/path/to/somewhere bash slurm/00_freeze_and_contamination.sh
#   SKIP_FREEZE=1  bash ...    # if v11 is already frozen
#   SKIP_SEMANTIC=1 bash ...   # skip bge-m3 semantic checks

set -eu

REPO_ROOT="$HOME/projects/GuardLens-DataGen-V2"
RESULTS_ROOT="$HOME/work/results"
V11_SOURCE="${V11_SOURCE:-$REPO_ROOT/results-gen}"
V11_FINAL_JSONL="${V11_FINAL_JSONL:-final_dataset.jsonl}"

REF_MHJ="$REPO_ROOT/mhj_conversations.jsonl"
REF_HARMBENCH="${REF_HARMBENCH:-$HOME/data/harmbench/behaviors.jsonl}"
REF_JBB="${REF_JBB:-$HOME/data/jbb/prompts.jsonl}"
REF_WILDJB="${REF_WILDJB:-$HOME/data/wildjailbreak/eval.jsonl}"

cd "$REPO_ROOT"

# -----------------------------------------------------------
# Disk-space sanity check for HF cache (used by prestage_v12.py later)
# -----------------------------------------------------------
HF_CACHE_PARENT="${HF_HUB_CACHE:-$HOME/work/hf_models/hub}"
# Walk up to a directory that actually exists
probe="$HF_CACHE_PARENT"
while [ -n "$probe" ] && [ ! -d "$probe" ]; do
    probe=$(dirname "$probe")
done
if [ -n "$probe" ]; then
    free_gb=$(df -BG "$probe" 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
    if [ -n "$free_gb" ]; then
        echo "HF cache filesystem: $probe (${free_gb} GB free)"
        if [ "$free_gb" -lt 400 ]; then
            echo "  WARN: less than 400 GB free. Full v12 model stack needs ~313 GB."
            echo "  If prestage fails on disk-full, free space or set HF_HUB_CACHE elsewhere."
        fi
    fi
fi

# -----------------------------------------------------------
# Phase 0a: Freeze v11
# -----------------------------------------------------------
if [ "${SKIP_FREEZE:-0}" != "1" ]; then
    echo ""
    echo "======================================================"
    echo "  Phase 0a: Freeze v11 (dry-run first)"
    echo "======================================================"
    if [ ! -d "$V11_SOURCE" ]; then
        echo "ERROR: v11 source does not exist: $V11_SOURCE"
        echo "Set V11_SOURCE to the correct path, e.g.:"
        echo "  V11_SOURCE=$REPO_ROOT/results-gen bash slurm/00_freeze_and_contamination.sh"
        exit 1
    fi

    python freeze_v11_legacy.py --source "$V11_SOURCE" --dest "$RESULTS_ROOT"

    echo ""
    read -p "Dry-run looks right? Commit the freeze? [y/N] " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        python freeze_v11_legacy.py \
            --source "$V11_SOURCE" \
            --dest "$RESULTS_ROOT" \
            --commit \
            --overwrite
        (cd "$RESULTS_ROOT/dataset_v11_legacy" && python verify_manifest.py)
    else
        echo "Skipped freeze commit. Re-run this script when ready."
        exit 0
    fi
else
    echo "SKIP_FREEZE=1 — skipping Phase 0a"
fi

# -----------------------------------------------------------
# Phase 0b: Build reference list
# -----------------------------------------------------------
echo ""
echo "======================================================"
echo "  Phase 0b: Build reference list"
echo "======================================================"
REFS=()
[ -f "$REF_MHJ" ]        && REFS+=("mhj:$REF_MHJ")         || echo "  SKIP mhj: $REF_MHJ not found"
[ -f "$REF_HARMBENCH" ]  && REFS+=("harmbench:$REF_HARMBENCH") || echo "  SKIP harmbench: $REF_HARMBENCH not found"
[ -f "$REF_JBB" ]        && REFS+=("jailbreakbench:$REF_JBB") || echo "  SKIP jailbreakbench: $REF_JBB not found"
[ -f "$REF_WILDJB" ]     && REFS+=("wildjailbreak:$REF_WILDJB") || echo "  SKIP wildjailbreak: $REF_WILDJB not found"

V11_MERGED="$RESULTS_ROOT/dataset_v11_legacy/original_layout/$V11_FINAL_JSONL"
if [ -f "$V11_MERGED" ]; then
    REFS+=("v11_legacy:$V11_MERGED")
else
    echo "  SKIP v11_legacy: $V11_MERGED not found"
fi

if [ ${#REFS[@]} -eq 0 ]; then
    echo "ERROR: no reference corpora available."
    exit 1
fi

echo ""
echo "  Active references: ${#REFS[@]}"
for r in "${REFS[@]}"; do echo "    $r"; done

# -----------------------------------------------------------
# run_contamination: explicit exit-code handling.
#   0 = clean, 3 = quarantine (allowed), 1 = hard contamination (abort),
#   other = unexpected (abort)
# -----------------------------------------------------------
run_contamination() {
    local desc="$1"
    shift
    echo ""
    echo "  Running: $desc"
    set +e
    "$@"
    local rc=$?
    set -e
    case $rc in
        0) echo "  [$desc] CLEAN (exit 0)" ;;
        3) echo "  [$desc] QUARANTINE (exit 3) — review hits file" ;;
        1) echo "  [$desc] HARD CONTAMINATION (exit 1) — MUST fix before proceeding"
           echo "  See contamination_hits.jsonl in output dir"
           exit 1 ;;
        *) echo "  [$desc] UNEXPECTED EXIT CODE $rc — treating as fatal"
           exit $rc ;;
    esac
}

# -----------------------------------------------------------
# Phase 0c: Objective bank lexical check
# -----------------------------------------------------------
echo ""
echo "======================================================"
echo "  Phase 0c: Lexical check on objective bank"
echo "======================================================"
mkdir -p "$RESULTS_ROOT/v12_gen"
run_contamination "objective_bank_lexical" \
    python contamination_check.py \
        --candidate bench_objectives.jsonl \
        --references "${REFS[@]}" \
        --output-dir "$RESULTS_ROOT/v12_gen/objective_bank_contamination_lexical" \
        --jaccard-threshold 0.6

# -----------------------------------------------------------
# Phase 0d: Objective bank semantic check
# -----------------------------------------------------------
if [ "${SKIP_SEMANTIC:-0}" != "1" ]; then
    echo ""
    echo "======================================================"
    echo "  Phase 0d: Semantic check on objective bank"
    echo "  (first run downloads bge-m3, ~2.3GB)"
    echo "======================================================"
    run_contamination "objective_bank_semantic" \
        python contamination_check.py \
            --candidate bench_objectives.jsonl \
            --references "${REFS[@]}" \
            --output-dir "$RESULTS_ROOT/v12_gen/objective_bank_contamination_semantic" \
            --jaccard-threshold 0.6 \
            --semantic-check \
            --semantic-threshold 0.82
else
    echo "SKIP_SEMANTIC=1 — skipping Phase 0d"
fi

# -----------------------------------------------------------
# Phase 0e: v11 baseline overlap with external evals
# -----------------------------------------------------------
if [ "${SKIP_SEMANTIC:-0}" != "1" ] && [ -f "$V11_MERGED" ]; then
    echo ""
    echo "======================================================"
    echo "  Phase 0e: v11 vs external evals (paper baseline)"
    echo "  This runs on CPU. With v5 batching, ~10-15 min."
    echo "======================================================"
    EXT_REFS=()
    for r in "${REFS[@]}"; do
        [ "${r%%:*}" != "v11_legacy" ] && EXT_REFS+=("$r")
    done
    if [ ${#EXT_REFS[@]} -gt 0 ]; then
        run_contamination "v11_vs_external" \
            python contamination_check.py \
                --candidate "$V11_MERGED" \
                --references "${EXT_REFS[@]}" \
                --output-dir "$RESULTS_ROOT/v12_gen/v11_external_contamination" \
                --jaccard-threshold 0.6 \
                --semantic-check
        echo ""
        echo "Paper baseline number:"
        python -c "
import json
with open('$RESULTS_ROOT/v12_gen/v11_external_contamination/contamination_summary.json') as f:
    s = json.load(f)
print(f\"  n_records: {s['n_records']}\")
print(f\"  lexical hits: {s['n_records_lexical_hit']}\")
print(f\"  semantic-only hits: {s['n_records_semantic_only_hit']}\")
"
    fi
fi

echo ""
echo "======================================================"
echo "  Phase 0 complete."
echo "  Next: stage models, then submit Phase 1:"
echo "    python prestage_v12.py --config-file model_bench_configs.yaml"
echo "    sbatch slurm/01_bench_phase1.slurm"
echo "======================================================"