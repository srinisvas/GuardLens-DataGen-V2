#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OLD_FREEZE="${OLD_FREEZE:-$REPO_ROOT/results-naacl/final-data-freeze}"
CANDIDATE="${CANDIDATE:-$REPO_ROOT/results-naacl/final-data-freeze-semantic-turn-repair-candidate}"
REVIEW="${REVIEW:-$REPO_ROOT/results-naacl/review_b4_committed_2999.jsonl}"
REVIEW_MANIFEST="${REVIEW_MANIFEST:-$REPO_ROOT/results-naacl/review_b4_committed_2999.jsonl.review_manifest.json}"
LEGACY="${LEGACY:-$REPO_ROOT/results-new/naacl_legacy_prepared.jsonl}"

if [[ ! -d "$OLD_FREEZE" ]]; then
  echo "Missing old freeze: $OLD_FREEZE" >&2
  exit 2
fi
for path in "$REVIEW" "$REVIEW_MANIFEST" "$LEGACY"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 2
  fi
done

if [[ "$CANDIDATE" == "$OLD_FREEZE" ]]; then
  echo "Refusing to overwrite the current freeze" >&2
  exit 2
fi

RESULTS_ROOT="$(readlink -m "$REPO_ROOT/results-naacl")"
CANDIDATE_REAL="$(readlink -m "$CANDIDATE")"
case "$CANDIDATE_REAL" in
  "$RESULTS_ROOT"/final-data-freeze-semantic-turn-repair-candidate*) ;;
  *)
    echo "Refusing unsafe candidate path: $CANDIDATE_REAL" >&2
    echo "Candidate basename must start with final-data-freeze-semantic-turn-repair-candidate" >&2
    exit 2
    ;;
esac
CANDIDATE="$CANDIDATE_REAL"

if [[ "$(git rev-parse --abbrev-ref HEAD)" != "naacl-validity-repair-optimized" ]]; then
  echo "Run this rebuild only from branch naacl-validity-repair-optimized" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Tracked working-tree changes detected; commit or revert them before rebuilding" >&2
  exit 2
fi
CODE_SHA="$(git rev-parse HEAD)"

rm -rf "$CANDIDATE"
mkdir -p "$CANDIDATE"

echo "=== 1/9 Prepare Dataset B candidate ==="
python naacl/prepare_frontier_dataset.py \
  --input "$REVIEW" \
  --output "$CANDIDATE/dataset_b_primary.jsonl" \
  --benign-stress-output "$CANDIDATE/dataset_b_benign_stress.jsonl" \
  --excluded-output "$CANDIDATE/dataset_b_excluded.jsonl" \
  --stats-output "$CANDIDATE/dataset_b_primary.stats.json" \
  --expect-full-review-export

echo "=== 2/9 Audit prepared Dataset B ==="
python naacl/audit_frontier_dataset.py \
  --input "$CANDIDATE/dataset_b_primary.jsonl"

python naacl/audit_semantic_span_masking.py \
  --raw-input "$REVIEW" \
  --prepared-input "$CANDIDATE/dataset_b_primary.jsonl" \
  --enforce-reviewed-counts

echo "=== 3/9 Merge Dataset A + repaired Dataset B ==="
python naacl/merge_training_corpora.py \
  --legacy-input "$LEGACY" \
  --frontier-input "$CANDIDATE/dataset_b_primary.jsonl" \
  --output "$CANDIDATE/dataset_ab_primary.jsonl" \
  --stats-output "$CANDIDATE/dataset_ab_primary.stats.json" \
  --expect-final-naacl-counts

echo "=== 4/9 Recreate deterministic primary split ==="
python naacl/split_consolidated.py \
  --input "$CANDIDATE/dataset_ab_primary.jsonl" \
  --output-dir "$CANDIDATE/splits_primary" \
  --seed 42

echo "=== 5/9 Copy unchanged auxiliary corpus and attach to train only ==="
cp "$OLD_FREEZE/dataset_b_auxiliary_512.jsonl" "$CANDIDATE/dataset_b_auxiliary_512.jsonl"
cp "$OLD_FREEZE/dataset_b_auxiliary_512.stats.json" "$CANDIDATE/dataset_b_auxiliary_512.stats.json"

python naacl/attach_training_auxiliary.py \
  --primary-train "$CANDIDATE/splits_primary/train.jsonl" \
  --primary-dev "$CANDIDATE/splits_primary/dev.jsonl" \
  --primary-test "$CANDIDATE/splits_primary/test.jsonl" \
  --auxiliary-input "$CANDIDATE/dataset_b_auxiliary_512.jsonl" \
  --output-dir "$CANDIDATE/splits_primary_plus_train_auxiliary"

echo "=== 6/9 Confirm repaired semantic-turn contract on train/dev ==="
python naacl/audit_semantic_turn_consistency.py \
  --train "$CANDIDATE/splits_primary/train.jsonl" \
  --dev "$CANDIDATE/splits_primary/dev.jsonl" \
  --enforce-reviewed-counts \
  --output "$CANDIDATE/semantic_turn_consistency.json"

echo "=== 7/9 Prove exact old-vs-candidate delta ==="
python naacl/audit_semantic_turn_repair_delta.py \
  --before-frontier "$OLD_FREEZE/dataset_b_primary.jsonl" \
  --after-frontier "$CANDIDATE/dataset_b_primary.jsonl" \
  --before-stress "$OLD_FREEZE/dataset_b_benign_stress.jsonl" \
  --after-stress "$CANDIDATE/dataset_b_benign_stress.jsonl" \
  --before-excluded "$OLD_FREEZE/dataset_b_excluded.jsonl" \
  --after-excluded "$CANDIDATE/dataset_b_excluded.jsonl" \
  --before-auxiliary "$OLD_FREEZE/dataset_b_auxiliary_512.jsonl" \
  --after-auxiliary "$CANDIDATE/dataset_b_auxiliary_512.jsonl" \
  --before-merged "$OLD_FREEZE/dataset_ab_primary.jsonl" \
  --after-merged "$CANDIDATE/dataset_ab_primary.jsonl" \
  --before-primary-split-dir "$OLD_FREEZE/splits_primary" \
  --after-primary-split-dir "$CANDIDATE/splits_primary" \
  --before-aux-split-dir "$OLD_FREEZE/splits_primary_plus_train_auxiliary" \
  --after-aux-split-dir "$CANDIDATE/splits_primary_plus_train_auxiliary" \
  --output "$CANDIDATE/semantic_turn_repair_delta.json"

echo "=== 8/9 Run full final-data audit ==="
python naacl/audit_final_data_prep.py \
  --review-input "$REVIEW" \
  --review-manifest "$REVIEW_MANIFEST" \
  --legacy-input "$LEGACY" \
  --frontier-primary "$CANDIDATE/dataset_b_primary.jsonl" \
  --frontier-stress "$CANDIDATE/dataset_b_benign_stress.jsonl" \
  --frontier-excluded "$CANDIDATE/dataset_b_excluded.jsonl" \
  --merged-input "$CANDIDATE/dataset_ab_primary.jsonl" \
  --primary-split-dir "$CANDIDATE/splits_primary" \
  --auxiliary-input "$CANDIDATE/dataset_b_auxiliary_512.jsonl" \
  --auxiliary-candidate-split-dir "$CANDIDATE/splits_primary_plus_train_auxiliary" \
  --expect-final-naacl-counts \
  --report-output "$CANDIDATE/data_prep_freeze_report.json"

echo "=== 9/9 Record preparation code commit ==="
if [[ "$(git rev-parse HEAD)" != "$CODE_SHA" ]]; then
  echo "Repository HEAD changed during rebuild; candidate is invalid" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Tracked working tree changed during rebuild; candidate is invalid" >&2
  exit 2
fi
printf '%s\n' "$CODE_SHA" > "$CANDIDATE/data_prep_code_commit.txt"

echo
echo "CANDIDATE READY FOR REVIEW: $CANDIDATE"
echo "Current freeze was NOT modified."
echo "Inspect:"
echo "  $CANDIDATE/semantic_turn_consistency.json"
echo "  $CANDIDATE/semantic_turn_repair_delta.json"
echo "  $CANDIDATE/data_prep_freeze_report.json"
