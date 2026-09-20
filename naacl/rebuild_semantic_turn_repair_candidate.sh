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

assert_naacl_code_clean() {
  local dirty
  dirty="$(git status --porcelain=v1 --untracked-files=all -- naacl | grep -Ev '^\?\? naacl/(.*/)?__pycache__/|^\?\? naacl/.*\.pyc$' || true)"
  if [[ -n "$dirty" ]]; then
    echo "DatasetGen code tree under naacl/ is not clean:" >&2
    printf '%s\n' "$dirty" >&2
    echo "Only unrelated generated/log artifacts outside naacl/ may be dirty." >&2
    exit 2
  fi
}

verify_sha256() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "Pinned input/baseline hash mismatch: $path" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    exit 2
  fi
}

assert_naacl_code_clean

# Bind the rebuild to the exact three scientific inputs from the accepted freeze contract.
verify_sha256 "a71def97f7d2cf73388e0f31643ba48a444b14213f9d20c8763c9ff74ac0b0aa" "$LEGACY"
verify_sha256 "74fe911313e23fa75cf48ed195957ce71346d4ab517d99081ec3864d069af37d" "$REVIEW"
verify_sha256 "19bdd9884de0c231b3e76a1325f49e2dd85edf02345bedeb4062478e8849df58" "$REVIEW_MANIFEST"

# Bind every old-freeze artifact used as the comparison baseline.
verify_sha256 "eddae12529f5c520772973089c70c014fc1508e00dcc835faef78fb2b8d8281f" "$OLD_FREEZE/dataset_b_primary.jsonl"
verify_sha256 "ce21c83e349a3a3e0a41a523dc1bfc92669ed909f91494ee681a0133bd5b59a5" "$OLD_FREEZE/dataset_b_benign_stress.jsonl"
verify_sha256 "6599ac563096a67e499f8939e0767dda2b3da853ebe8f3aff475c9f51c41d968" "$OLD_FREEZE/dataset_b_excluded.jsonl"
verify_sha256 "76769282c545ce575677b26048dca69576c2398474168ae2bc21f467370abc5b" "$OLD_FREEZE/dataset_ab_primary.jsonl"
verify_sha256 "164a6327f54adedd8268d30c71d10ba56484f4984321505af33cbf3bb205ad1e" "$OLD_FREEZE/splits_primary/train.jsonl"
verify_sha256 "659394391b035fa5fa06607f304bd118872e44e61388704288cf27684034dbd0" "$OLD_FREEZE/splits_primary/dev.jsonl"
verify_sha256 "82771ea6ddef43a73f02de05e66d774f2c2f694bfb552929d28379c5a331cf45" "$OLD_FREEZE/splits_primary/test.jsonl"
verify_sha256 "f8e19e89dbbfcc2e41a3ff0bf560d6aa9d0d8e07a794f7306b2fc56daa00f594" "$OLD_FREEZE/dataset_b_auxiliary_512.jsonl"
verify_sha256 "3533198efdb55fe33087170c50d62d301969b841422e283c5ebdc71f10c7490f" "$OLD_FREEZE/splits_primary_plus_train_auxiliary/train.jsonl"
verify_sha256 "659394391b035fa5fa06607f304bd118872e44e61388704288cf27684034dbd0" "$OLD_FREEZE/splits_primary_plus_train_auxiliary/dev.jsonl"
verify_sha256 "82771ea6ddef43a73f02de05e66d774f2c2f694bfb552929d28379c5a331cf45" "$OLD_FREEZE/splits_primary_plus_train_auxiliary/test.jsonl"

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
assert_naacl_code_clean

# Recheck the scientific inputs at the end so a concurrent local process
# cannot alter them during the rebuild.
verify_sha256 "a71def97f7d2cf73388e0f31643ba48a444b14213f9d20c8763c9ff74ac0b0aa" "$LEGACY"
verify_sha256 "74fe911313e23fa75cf48ed195957ce71346d4ab517d99081ec3864d069af37d" "$REVIEW"
verify_sha256 "19bdd9884de0c231b3e76a1325f49e2dd85edf02345bedeb4062478e8849df58" "$REVIEW_MANIFEST"

printf '%s\n' "$CODE_SHA" > "$CANDIDATE/data_prep_code_commit.txt"

echo
echo "CANDIDATE READY FOR REVIEW: $CANDIDATE"
echo "Current freeze was NOT modified."
echo "Inspect:"
echo "  $CANDIDATE/semantic_turn_consistency.json"
echo "  $CANDIDATE/semantic_turn_repair_delta.json"
echo "  $CANDIDATE/data_prep_freeze_report.json"
 \
      || true
  )"
  if [[ -n "$dirty" ]]; then
    echo "DatasetGen code tree under naacl/ is not clean:" >&2
    printf '%s\n' "$dirty" >&2
    echo "Only unrelated generated/log artifacts outside naacl/ may be dirty." >&2
    exit 2
  fi
}

verify_sha256() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "Pinned input/baseline hash mismatch: $path" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    exit 2
  fi
}

assert_naacl_code_clean

# Bind the rebuild to the exact three scientific inputs from the accepted
# freeze contract. Dirty historical/generated files elsewhere in the repo do
# not matter if these bytes are exact.
verify_sha256 "a71def97f7d2cf73388e0f31643ba48a444b14213f9d20c8763c9ff74ac0b0aa" "$LEGACY"
verify_sha256 "74fe911313e23fa75cf48ed195957ce71346d4ab517d99081ec3864d069af37d" "$REVIEW"
verify_sha256 "19bdd9884de0c231b3e76a1325f49e2dd85edf02345bedeb4062478e8849df58" "$REVIEW_MANIFEST"

# Also bind every old-freeze artifact used as the comparison baseline. This
# prevents a locally edited freeze from making the exact-delta audit vacuous.
verify_sha256 "eddae12529f5c520772973089c70c014fc1508e00dcc835faef78fb2b8d8281f" "$OLD_FREEZE/dataset_b_primary.jsonl"
verify_sha256 "ce21c83e349a3a3e0a41a523dc1bfc92669ed909f91494ee681a0133bd5b59a5" "$OLD_FREEZE/dataset_b_benign_stress.jsonl"
verify_sha256 "6599ac563096a67e499f8939e0767dda2b3da853ebe8f3aff475c9f51c41d968" "$OLD_FREEZE/dataset_b_excluded.jsonl"
verify_sha256 "76769282c545ce575677b26048dca69576c2398474168ae2bc21f467370abc5b" "$OLD_FREEZE/dataset_ab_primary.jsonl"
verify_sha256 "164a6327f54adedd8268d30c71d10ba56484f4984321505af33cbf3bb205ad1e" "$OLD_FREEZE/splits_primary/train.jsonl"
verify_sha256 "659394391b035fa5fa06607f304bd118872e44e61388704288cf27684034dbd0" "$OLD_FREEZE/splits_primary/dev.jsonl"
verify_sha256 "82771ea6ddef43a73f02de05e66d774f2c2f694bfb552929d28379c5a331cf45" "$OLD_FREEZE/splits_primary/test.jsonl"
verify_sha256 "f8e19e89dbbfcc2e41a3ff0bf560d6aa9d0d8e07a794f7306b2fc56daa00f594" "$OLD_FREEZE/dataset_b_auxiliary_512.jsonl"
verify_sha256 "3533198efdb55fe33087170c50d62d301969b841422e283c5ebdc71f10c7490f" "$OLD_FREEZE/splits_primary_plus_train_auxiliary/train.jsonl"
verify_sha256 "659394391b035fa5fa06607f304bd118872e44e61388704288cf27684034dbd0" "$OLD_FREEZE/splits_primary_plus_train_auxiliary/dev.jsonl"
verify_sha256 "82771ea6ddef43a73f02de05e66d774f2c2f694bfb552929d28379c5a331cf45" "$OLD_FREEZE/splits_primary_plus_train_auxiliary/test.jsonl"

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
