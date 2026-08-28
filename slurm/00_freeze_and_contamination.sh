#!/bin/bash
# Phase 0: Freeze v11 + contamination baseline
#
# Runs on LOGIN NODE (CPU only, no GPU allocation needed).
# Total time: ~40-60 min (mostly bge-m3 download on first run).
#
# Usage:
#   cd ~/projects/GuardLens-DataGen-V2
#   bash slurm/00_freeze_and_contamination.sh
#
# Prerequisites:
#   conda activate dataset_gen  (before invoking)

set -eu

REPO_ROOT="$HOME/projects/GuardLens-DataGen-V2"
RESULTS_ROOT="$HOME/work/results"
V11_SOURCE="${V11_SOURCE:-$HOME/work/results/guardlens_v11}"   # override with env var if v11 lives elsewhere

# Reference corpora. Only MHJ is confirmed present at repo root.
# Add others as you download them and re-run.
REF_MHJ="$REPO_ROOT/mhj_conversations.jsonl"
REF_HARMBENCH="${REF_HARMBENCH:-$HOME/data/harmbench/behaviors.jsonl}"
REF_JBB="${REF_JBB:-$HOME/data/jbb/prompts.jsonl}"
REF_WILDJB="${REF_WILDJB:-$HOME/data/wildjailbreak/eval.jsonl}"

cd "$REPO_ROOT"

echo "======================================================"
echo "  Phase 0a: Freeze v11 (dry-run)"
echo "======================================================"
if [ ! -d "$V11_SOURCE" ]; then
    echo "ERROR: v11 source directory does not exist: $V11_SOURCE"
    echo "Set V11_SOURCE env var to the correct path, e.g.:"
    echo "  V11_SOURCE=$REPO_ROOT/results-gen bash slurm/00_freeze_and_contamination.sh"
    exit 1
fi

python freeze_v11_legacy.py --source "$V11_SOURCE" --dest "$RESULTS_ROOT"

echo ""
read -p "Does the dry-run listing look right? Commit the freeze? [y/N] " confirm
if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    python freeze_v11_legacy.py --source "$V11_SOURCE" --dest "$RESULTS_ROOT" --commit
    cd "$RESULTS_ROOT/dataset_v11_legacy"
    python verify_manifest.py
    cd "$REPO_ROOT"
else
    echo "Skipped freeze commit. Re-run this script when ready."
    exit 0
fi

echo ""
echo "======================================================"
echo "  Phase 0b: Build reference list (skip missing corpora)"
echo "======================================================"
REFS=()
[ -f "$REF_MHJ" ]        && REFS+=("mhj:$REF_MHJ")         || echo "  SKIP mhj: $REF_MHJ not found"
[ -f "$REF_HARMBENCH" ]  && REFS+=("harmbench:$REF_HARMBENCH") || echo "  SKIP harmbench: $REF_HARMBENCH not found"
[ -f "$REF_JBB" ]        && REFS+=("jailbreakbench:$REF_JBB") || echo "  SKIP jailbreakbench: $REF_JBB not found"
[ -f "$REF_WILDJB" ]     && REFS+=("wildjailbreak:$REF_WILDJB") || echo "  SKIP wildjailbreak: $REF_WILDJB not found"

V11_MERGED="$RESULTS_ROOT/dataset_v11_legacy/original_layout/final_dataset.jsonl"
[ -f "$V11_MERGED" ]     && REFS+=("v11_legacy:$V11_MERGED") || echo "  SKIP v11_legacy: $V11_MERGED not found"

if [ ${#REFS[@]} -eq 0 ]; then
    echo "ERROR: no reference corpora available. Cannot run contamination check."
    exit 1
fi

echo ""
echo "  Active references: ${#REFS[@]}"
for r in "${REFS[@]}"; do echo "    $r"; done

echo ""
echo "======================================================"
echo "  Phase 0c: Contamination-check the objective bank"
echo "  (lexical only, fast)"
echo "======================================================"
mkdir -p "$RESULTS_ROOT/v12_gen"
python contamination_check.py \
    --candidate bench_objectives.jsonl \
    --references "${REFS[@]}" \
    --output-dir "$RESULTS_ROOT/v12_gen/objective_bank_contamination_lexical" \
    --jaccard-threshold 0.6 || { echo "LEXICAL contamination in objective bank — review before proceeding"; exit 1; }

echo ""
echo "======================================================"
echo "  Phase 0d: Semantic contamination check on bank"
echo "  (first run downloads bge-m3, ~2.3GB, one time)"
echo "======================================================"
python contamination_check.py \
    --candidate bench_objectives.jsonl \
    --references "${REFS[@]}" \
    --output-dir "$RESULTS_ROOT/v12_gen/objective_bank_contamination_semantic" \
    --jaccard-threshold 0.6 \
    --semantic-check \
    --semantic-threshold 0.82 || echo "Note: exit 3 (quarantine) is expected here — inspect hits and confirm"

echo ""
echo "======================================================"
echo "  Phase 0e: Baseline v11 overlap with external evals"
echo "  (~30 min with semantic; produces the paper's baseline number)"
echo "======================================================"
# Only use external eval sets (drop v11_legacy from refs since v11 is the candidate)
EXT_REFS=()
for r in "${REFS[@]}"; do
    [ "${r%%:*}" != "v11_legacy" ] && EXT_REFS+=("$r")
done

if [ ${#EXT_REFS[@]} -gt 0 ] && [ -f "$V11_MERGED" ]; then
    python contamination_check.py \
        --candidate "$V11_MERGED" \
        --references "${EXT_REFS[@]}" \
        --output-dir "$RESULTS_ROOT/v12_gen/v11_external_contamination" \
        --jaccard-threshold 0.6 \
        --semantic-check || echo "Note: contamination result is informational — record for paper"
    echo ""
    echo "Paper baseline number:"
    cat "$RESULTS_ROOT/v12_gen/v11_external_contamination/contamination_summary.json" | python -m json.tool | grep -E "n_records|contaminated"
else
    echo "SKIP: need external refs and v11 merged file"
fi

echo ""
echo "======================================================"
echo "  Phase 0 complete."
echo "  Next: submit Phase 1 with"
echo "    sbatch slurm/01_bench_phase1.slurm"
echo "======================================================"
