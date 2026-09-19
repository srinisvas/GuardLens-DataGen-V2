# Dataset B auxiliary behavioral outcomes

This path is optional and separate from the primary GuardLens training corpus.
It uses the 512 Dataset B trajectories rejected by the frozen B2 behavioral
validation as additional trajectory-level detection examples. It does **not**
change Dataset A, the pair-complete Dataset B primary corpus, raw B4 evidence,
or the frozen v4/v5/v8 generation and counterfactual protocols.

## Why these 512 records are useful

B2 compares the realized Qwen trajectory with the source trajectory role.
The rejected set contains two useful behavioral outcomes:

- 190 benign-authored trajectories where Qwen nevertheless became unsafe. These
  receive `detection_label=1`.
- 322 malicious-authored trajectories where Qwen safely resisted the attack.
  These receive `detection_label=0`.

The original `label` is retained as `authoring_intent_label`. The realized
outcome is stored explicitly as `observed_behavior_label`, with
`detection_label` as the task-facing alias. For these auxiliary records the two
fields are identical and are independently audited against the frozen B2
trajectory decision.

These examples are **detection-only**. They must not provide token, span, pivot,
or counterfactual localization supervision. The preparation code therefore sets:

- `loss_weight = detection_loss_weight = 0.25`
- `pivot_loss_weight = 0.0`
- `span_loss_weight = 0.0`
- `pivot_supervision_ignore = true`
- `localization_supervision_ignore = true`
- `pivot_turn_id = null`
- `evidence_turn_ids = []`

The generic `loss_weight` is retained for compatibility with simple detection
loaders. A multi-task trainer must use the task-specific weights and must never
let the auxiliary examples contribute localization loss.

## Preparation and audit

Use the raw 2,999-record B4 review export. Do not rewrite it in place. Verify its
frozen byte hash and companion manifest before deriving either primary or
auxiliary data.

```bash
export REVIEW=results-naacl/review_b4_committed_2999.jsonl
export REVIEW_MANIFEST=results-naacl/review_b4_committed_2999.jsonl.review_manifest.json
export WORK=$HOME/staging/dataset_naacl/training_candidate
mkdir -p "$WORK"

python naacl/audit_review_export.py \
  --input "$REVIEW" \
  --manifest "$REVIEW_MANIFEST"

python naacl/prepare_frontier_auxiliary.py \
  --input "$REVIEW" \
  --output "$WORK/dataset_b_auxiliary_512.jsonl" \
  --stats-output "$WORK/dataset_b_auxiliary_512.stats.json" \
  --split-output-dir "$WORK/dataset_b_auxiliary_splits" \
  --expect-full-review-export

python naacl/audit_frontier_auxiliary.py \
  --input "$WORK/dataset_b_auxiliary_512.jsonl" \
  --expect-full-review-export
```

The full-export gate requires exactly 512 records with detection labels
`1:190` and `0:322`. If the review export changes, do not bypass that mismatch by
editing the expected counts. Re-audit the new artifact first.

## Primary Dataset B preparation

Primary Dataset B remains pair-complete and independent from the auxiliary set.
The known failed malicious record and its benign twin are excluded naturally
because `prepare_frontier_dataset.py` admits a pair only when both sides pass.
Standalone benign examples remain evaluation-only stress data.

```bash
python naacl/prepare_frontier_dataset.py \
  --input "$REVIEW" \
  --output "$WORK/dataset_b_primary.jsonl" \
  --benign-stress-output "$WORK/dataset_b_benign_stress.jsonl" \
  --excluded-output "$WORK/dataset_b_excluded.jsonl" \
  --stats-output "$WORK/dataset_b_primary.stats.json" \
  --expect-full-review-export

python naacl/audit_frontier_dataset.py --input "$WORK/dataset_b_primary.jsonl"
python naacl/audit_frontier_stress.py --input "$WORK/dataset_b_benign_stress.jsonl"

python naacl/audit_semantic_span_masking.py \
  --raw-input "$REVIEW" \
  --prepared-input "$WORK/dataset_b_primary.jsonl" \
  --enforce-reviewed-counts
```

For the audited review export, the expected primary result is 701 retained pairs,
or 1,402 records. Treat a different count as a review gate, not as permission to
relax pair admission.

## Candidate A+B merge

Dataset A is the repaired 1,052-record legacy primary corpus already tracked under
`results-new/naacl_legacy_prepared.jsonl`. Merge only the clean primary Dataset B
output above. Do not append Dataset A's separate benign stress pool.

```bash
python naacl/merge_training_corpora.py \
  --legacy-input results-new/naacl_legacy_prepared.jsonl \
  --frontier-input "$WORK/dataset_b_primary.jsonl" \
  --output "$WORK/dataset_ab_primary.jsonl" \
  --stats-output "$WORK/dataset_ab_primary.stats.json" \
  --expect-final-naacl-counts
```

The merge remains fail-closed for duplicate conversation IDs and exact normalized
user trajectories across independent split groups.

## Primary-only grouped split

```bash
python naacl/split_consolidated.py \
  --input "$WORK/dataset_ab_primary.jsonl" \
  --output-dir "$WORK/splits_primary" \
  --seed 42
```

The split keeps legacy pairs together and keeps all members of a Dataset B
`scenario_family` together.

## Candidate auxiliary training attachment

For a clean baseline-versus-auxiliary ablation, freeze the primary A+B split
first. Do **not** let auxiliary records change the dev/test composition or the
primary group assignment. Attach auxiliary outcomes to training only:

```bash
python naacl/attach_training_auxiliary.py \
  --primary-train "$WORK/splits_primary/train.jsonl" \
  --primary-dev "$WORK/splits_primary/dev.jsonl" \
  --primary-test "$WORK/splits_primary/test.jsonl" \
  --auxiliary-input "$WORK/dataset_b_auxiliary_512.jsonl" \
  --output-dir "$WORK/splits_primary_plus_train_auxiliary"
```

The attachment uses the already-frozen `consolidated_split_group` ownership.
Auxiliary records whose family is owned by primary dev/test are withheld from
training. Auxiliary records whose family is owned by primary train, or whose
family is absent from the primary corpus, may enter training. Dev and test are
copied unchanged and remain primary-only. This keeps held-out metrics directly
comparable with the primary baseline while preventing scenario-family leakage.

`split_consolidated.py --auxiliary-input` remains available for diagnostics, but
it is not the recommended ablation path because a joint re-split can alter
primary partition assignments and can put auxiliary examples into dev/test.


## Final data-preparation freeze audit

After the primary split and optional train-only auxiliary attachment are written,
run the model-independent final audit before treating any artifact as frozen:

```bash
python naacl/audit_final_data_prep.py \
  --review-input "$REVIEW" \
  --review-manifest "$REVIEW_MANIFEST" \
  --legacy-input results-new/naacl_legacy_prepared.jsonl \
  --frontier-primary "$WORK/dataset_b_primary.jsonl" \
  --merged-input "$WORK/dataset_ab_primary.jsonl" \
  --primary-split-dir "$WORK/splits_primary" \
  --auxiliary-input "$WORK/dataset_b_auxiliary_512.jsonl" \
  --auxiliary-candidate-split-dir "$WORK/splits_primary_plus_train_auxiliary" \
  --expect-final-naacl-counts \
  --report-output "$WORK/data_prep_freeze_report.json"
```

A pass requires all of the following:

- Dataset A has 1,052 records.
- Primary Dataset B has 1,402 records, or 701 complete pairs.
- The merged primary corpus has 2,454 records and 1,227 examples per class.
- Each source is label-balanced and has identical class-conditional user-turn
  and total-turn histograms.
- Stored normalized user-trajectory hashes recompute exactly.
- The primary train/dev/test files form an exact, disjoint partition with no
  group, pair, scenario-family, or exact-user-trajectory leakage.
- No internal conversation/pair identifier appears in model-visible text.
- The auxiliary artifact is exactly 512 records with 190 unsafe outcomes and
  322 safe outcomes, and its B2/B4 protocol provenance re-audits successfully.
- The auxiliary candidate contains auxiliary records only in train.
- Auxiliary train records cannot duplicate a primary dev/test user trajectory,
  even under a different scenario-family identifier.
- Candidate dev and test are byte-identical copies of the frozen primary
  dev/test files.

The final report also records where the reviewed construction-language phrases
remain visible in conversation text by source and class. This is diagnostic,
because those phrases are intentionally preserved in the raw conversation while
their adjudicated spans are removed from positive token supervision.

## Trainer integration gate

This repository checkout does not contain the final model-training package, so no
trainer is modified here. Before using the optional joint split, inspect the
actual trainer and require all of the following:

1. Auxiliary records use `detection_label` for trajectory detection.
2. `detection_loss_weight` is applied to detection loss.
3. `pivot_loss_weight=0` and `span_loss_weight=0` are respected exactly.
4. The trainer does not infer localization negatives from auxiliary records.
5. Unknown primary pivots remain masked rather than converted to true no-pivot
   negatives.
6. Token/span targets are mapped after the trainer's real tokenizer and truncation
   policy are known.
7. Family-preserving split assignments are consumed as supplied, not re-split at
   the record level.

The 21 reviewed construction-language spans are now masked upstream from positive
token supervision by `semantic_span_policy.py` while their raw B4 evidence status
and counterfactual deltas remain intact. The trainer must consume the prepared
`supervision_tier=ignore` / `semantic_token_supervision_ignore` state. The
remaining training concerns around v8 adapter semantics, legacy `max_turns=16`
behavior, and character clipping still block a training smoke. None of these
requires mutating the raw B4 artifact or running another generation job.

## Scope

No new GPU generation is required for this auxiliary path. It is a preparation,
audit, split, and training-integration option over already generated outcomes.
The primary A+B corpus should remain the baseline. Train with auxiliary outcomes
only as an explicit candidate or ablation so any gain or regression is measurable.
