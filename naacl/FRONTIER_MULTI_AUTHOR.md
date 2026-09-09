# GuardLens multi-author frontier source pipeline

This runbook is the canonical execution path for the merged GPT-5.6 + Astra
Stage-0 source corpus.

## Locked source design

Two independent authoring artifacts are processed together only after each passes
its own source audit:

- `guardlens_source_full_v3.jsonl` — GPT-5.6-authored source, 1,500 records
- `guardlens_source_astra_v3.jsonl` — Astra revision-2 source, 1,500 records

Merged Stage-0 expectations:

- 3,000 records
- 1,200 adversarial authoring-intent records
- 1,800 benign authoring-intent records
- 1,200 complete malicious/benign pairs = 2,400 primary candidates
- 600 standalone hard-benign records = stress candidates only
- 600 scenario families
- `pivot_turn_id`, `supervision_tier`, `loss_weight`, `judge_confidence` remain null
- `training_eligible=false`
- authoring intent remains generation metadata, never scientific ground truth

No source record is rewritten during merge. Existing `metadata.corpus_version` and
`metadata.generator` preserve author provenance.

## 1. CPU audits and merge

```bash
export OUT=$HOME/staging/dataset_gen_output
export GPT_SOURCE=$OUT/guardlens_source_full_v3.jsonl
export ASTRA_SOURCE=$OUT/guardlens_source_astra_v3.jsonl
export MERGED_SOURCE=$OUT/guardlens_source_merged_v3_3000.jsonl

python naacl/audit_frontier_source.py --input "$GPT_SOURCE"
python naacl/audit_frontier_source.py --input "$ASTRA_SOURCE"

python naacl/merge_frontier_sources.py \
  --inputs "$GPT_SOURCE" "$ASTRA_SOURCE" \
  --output "$MERGED_SOURCE" \
  --stats-output $OUT/guardlens_source_merged_v3_3000_stats.json \
  --expected-records-per-input 1500 \
  --expected-total 3000

python naacl/audit_frontier_source.py \
  --input "$MERGED_SOURCE" \
  --expected-records 3000 \
  --expected-pairs 1200 \
  --expected-standalone 600 \
  --expected-scenarios 600

python -m unittest naacl/test_frontier_cpu.py naacl/test_multi_author_source.py -v
```

The merge fails closed on conversation-ID, pair-ID, scenario-family, and normalized
complete-trajectory collisions across authors. The strict merged audit rechecks
all pair/scenario construction invariants and the paired primary length shortcut
gate.

## 2. Deterministic 20-record multi-author smoke

Select ten complete pairs. With two author corpora the selector enforces exactly
five pairs from each author while still seeking construction diversity.

```bash
python naacl/select_frontier_smoke_subset.py \
  --input "$MERGED_SOURCE" \
  --output $OUT/frontier_multi_author_smoke20.jsonl \
  --stats-output $OUT/frontier_multi_author_smoke20_stats.json \
  --pairs 10 \
  --seed 44
```

Inspect the stats and require `selection.source_authors` to contain 5 pairs from
each corpus.

## 3. Qwen2.5-32B smoke rollout

```bash
N_SHARDS=1 \
INPUT_FILE=$OUT/frontier_multi_author_smoke20.jsonl \
OUTPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_rollout.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_smoke_qwen32_rollout \
SOURCE_PREFLIGHT_MODE=schema \
sbatch --gres=gpu:1 naacl/launch_frontier_rollout.slurm
```

Do not proceed if model loading, record completion, or output integrity fails.

## 4. Mistral-24B smoke validation

```bash
N_SHARDS=1 \
INPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_rollout.jsonl \
OUTPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_validated.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_smoke_qwen32_validated \
sbatch --gres=gpu:1 naacl/launch_frontier_validation.slurm
```

Review validation-status counts, confidence, unsafe anchors, and representative
realized conversations from both authors before evidence replay.

## 5. Candidate materialization and paired-evidence smoke

```bash
python naacl/materialize_frontier_candidates.py \
  --input $OUT/frontier_multi_author_smoke_qwen32_validated.jsonl \
  --output $OUT/frontier_multi_author_smoke_qwen32_candidates.jsonl \
  --max-turn-candidates 4 \
  --spans-per-turn 2 \
  --controls 2

N_SHARDS=1 \
INPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_candidates.jsonl \
OUTPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_evidence.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_smoke_qwen32_evidence \
sbatch --gres=gpu:2 naacl/launch_frontier_evidence.slurm
```

The evidence smoke must demonstrate B1 rollout replay reproducibility and B2 judge
reproducibility. Any replay mismatch fails closed rather than creating evidence.

## 6. Full 3,000-record Qwen rollout

Only after the smoke passes:

```bash
N_SHARDS=2 \
INPUT_FILE="$MERGED_SOURCE" \
OUTPUT_FILE=$OUT/frontier_multi_author_qwen32_rollout.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_qwen32_rollout \
SOURCE_PREFLIGHT_MODE=strict \
sbatch naacl/launch_frontier_rollout.slurm
```

The launcher recognizes 3,000 records and automatically runs strict expectations
for 1,200 pairs, 600 standalone records, and 600 scenarios.

## 7. Full independent validation

```bash
N_SHARDS=2 \
INPUT_FILE=$OUT/frontier_multi_author_qwen32_rollout.jsonl \
OUTPUT_FILE=$OUT/frontier_multi_author_qwen32_validated.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_qwen32_validated \
sbatch naacl/launch_frontier_validation.slurm
```

Authoring intent does not decide validation. Malicious-authored trajectories that
Qwen safely refuses do not become positive training records.

## 8. Full candidate and evidence stages

```bash
python naacl/materialize_frontier_candidates.py \
  --input $OUT/frontier_multi_author_qwen32_validated.jsonl \
  --output $OUT/frontier_multi_author_qwen32_candidates.jsonl \
  --max-turn-candidates 4 \
  --spans-per-turn 2 \
  --controls 2

N_SHARDS=2 \
INPUT_FILE=$OUT/frontier_multi_author_qwen32_candidates.jsonl \
OUTPUT_FILE=$OUT/frontier_multi_author_qwen32_evidence.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_qwen32_evidence \
sbatch naacl/launch_frontier_evidence.slurm
```

## 9. Canonical Dataset B preparation

```bash
python naacl/prepare_frontier_dataset.py \
  --input $OUT/frontier_multi_author_qwen32_evidence.jsonl \
  --output $OUT/naacl_frontier_prepared.jsonl \
  --benign-stress-output $OUT/naacl_frontier_hard_benign_stress.jsonl \
  --excluded-output $OUT/naacl_frontier_excluded.jsonl \
  --stats-output $OUT/naacl_frontier_prepared_stats.json

python naacl/audit_frontier_dataset.py \
  --input $OUT/naacl_frontier_prepared.jsonl

python naacl/audit_frontier_stress.py \
  --input $OUT/naacl_frontier_hard_benign_stress.jsonl
```

Primary Dataset B remains complete-pair-only. A pair is retained only when the
malicious twin passes the Qwen/Mistral/evidence gates and the benign twin is
independently validated safe. No orphan twin enters primary training.

All validated standalone hard benigns from both authors remain evaluation-only.

## 10. Merge with frozen Dataset A and split once

```bash
python naacl/merge_training_corpora.py \
  --legacy-input $OUT/naacl_legacy_prepared.jsonl \
  --frontier-input $OUT/naacl_frontier_prepared.jsonl \
  --output $OUT/naacl_consolidated.jsonl \
  --stats-output $OUT/naacl_consolidated_stats.json \
  --seed 42

python naacl/split_consolidated.py \
  --input $OUT/naacl_consolidated.jsonl \
  --output-dir $OUT/naacl_splits \
  --train-frac 0.70 \
  --dev-frac 0.15 \
  --test-frac 0.15 \
  --seed 42
```

The final splitter keeps complete frontier scenario families indivisible and now
softly balances frontier author corpus and author×label in addition to the locked
source, label, length, difficulty, domain, pair-hardness, trajectory, mechanism,
and style axes.

## 11. Transformer handoff

Training remains on the reviewed 48-realized-turn GuardLens window. Before full
training, run the Transformer model-window audit and `smoke_naacl_window.slurm`.
Do not revert to the old 32-turn launcher.

## Cross-target robustness

Gemma-3-27B remains a separate robustness leg. Select the Gemma subset from the
merged Stage-0 source before inspecting Qwen outcomes. Gemma outputs never replace
Qwen-derived canonical labels or enter primary Dataset B preparation.
