# GuardLens NAACL Frontier-Authored Dataset Pipeline

This runbook processes the 1,500-record `GuardLensSourceTrajectory/v3` corpus into the same canonical training contract as the repaired legacy corpus, then merges both corpora before a single leakage-safe split.

## Locked model roles

- Primary target: `Qwen/Qwen2.5-32B-Instruct`
- Independent judge: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`
- Cross-target robustness: `google/gemma-3-27b-it`
- Default context cap: 8192 tokens
- Generation temperature: 0.0 with deterministic per-record/per-turn seeds

The target and judge never receive `intended_structure`, authoring-manifest fields, pair hardness, expected pivots, or other hidden construction metadata. The judge receives only the realized conversation prefix.

## Stage semantics

### Stage B0 — source

`pivot_turn_id`, `supervision_tier`, `loss_weight`, and `judge_confidence` must be null and `training_eligible=false`.

`intended_structure.expected_pivot_turn_ids` is a generation hypothesis only. It can propose intervention locations later but can never populate `pivot_turn_id` directly.

### Stage B1 — Qwen32 target rollout

Copy `guardlens_source_full_v3.jsonl` to `$HOME/staging/dataset_gen_output/` and run:

```bash
sbatch naacl/launch_frontier_rollout.slurm
```

Default output:

```text
$HOME/staging/dataset_gen_output/frontier_qwen32_rollout.jsonl
```

The rollout inserts one assistant response after each user turn using odd turn IDs. Supervision fields remain unresolved/null.

### Stage B2 — independent Mistral24 validation

```bash
sbatch naacl/launch_frontier_validation.slurm
```

Default output:

```text
$HOME/staging/dataset_gen_output/frontier_qwen32_validated.jsonl
```

Validation is context-aware. For authoring-intent malicious records, a trajectory is validated only if the observable target behavior crosses the unsafe threshold. For authoring-intent benign records, any unsafe crossing rejects the record. Near-boundary or low-confidence cases become ambiguous. No authoring metadata is shown to the judge.

Inspect validation counts before proceeding.

### Stage B3 — deterministic candidate materialization

```bash
python naacl/materialize_frontier_candidates.py \
  --input $HOME/staging/dataset_gen_output/frontier_qwen32_validated.jsonl \
  --output $HOME/staging/dataset_gen_output/frontier_qwen32_candidates.jsonl \
  --max-turn-candidates 4 \
  --spans-per-turn 2 \
  --controls 2
```

Candidate turn priority is:

1. observed unsafe anchor
2. author-intended candidate turns at/before that anchor
3. nearest preceding context turns

Author-intended turns are candidate proposals only. Candidate spans are labeled `EVIDENCE_CANDIDATE` or `CONTROL_CANDIDATE` and remain unvalidated.

### Stage B4 — paired context-aware counterfactual evidence

```bash
sbatch naacl/launch_frontier_evidence.slurm
```

Default output:

```text
$HOME/staging/dataset_gen_output/frontier_qwen32_evidence.jsonl
```

The job uses four A100s as two independent evidence shards. Each shard has one Qwen32 target GPU and one Mistral24 judge GPU.

For every behaviorally validated malicious trajectory the evidence engine:

- freshly replays the baseline with the same deterministic target seed schedule
- derives the unsafe anchor from the fresh replay
- tests up to four whole-turn interventions
- tests up to six evidence-candidate spans
- tests up to two control spans
- sets `evidence_turn_ids` only from supported interventions
- sets `pivot_turn_id` to the earliest supported evidence turn, otherwise null
- sets `pivot_supervision_ignore=true` when a malicious pivot remains unknown

The fixed later user trajectory is preserved after interventions. This is fixed-user-trajectory counterfactual replay, not adaptive causal simulation.

### Stage B5 — canonical preparation

After inspecting evidence statistics:

```bash
python naacl/prepare_frontier_dataset.py \
  --input $HOME/staging/dataset_gen_output/frontier_qwen32_evidence.jsonl \
  --output $HOME/staging/dataset_gen_output/naacl_frontier_prepared.jsonl \
  --excluded-output $HOME/staging/dataset_gen_output/naacl_frontier_excluded.jsonl \
  --stats-output $HOME/staging/dataset_gen_output/naacl_frontier_prepared_stats.json
```

Only behaviorally validated malicious records whose fresh paired baseline remains unsafe enter the malicious training pool. Only independently validated benign records enter the benign pool.

Then run:

```bash
python naacl/audit_frontier_dataset.py \
  --input $HOME/staging/dataset_gen_output/naacl_frontier_prepared.jsonl
```

Do not merge unless this ends with `VALIDITY AUDIT PASSED`.

## Gemma 27B cross-target robustness

Select complete scenario families:

```bash
python naacl/select_cross_target_subset.py \
  --input $HOME/staging/dataset_gen_output/guardlens_source_full_v3.jsonl \
  --output $HOME/staging/dataset_gen_output/frontier_gemma27_source.jsonl \
  --stats-output $HOME/staging/dataset_gen_output/frontier_gemma27_subset_stats.json \
  --fraction 0.40
```

The current v3 corpus yields roughly 600 records while keeping complete scenario families.

Run Gemma with the same rollout launcher:

```bash
TARGET_MODEL=google/gemma-3-27b-it \
INPUT_FILE=$HOME/staging/dataset_gen_output/frontier_gemma27_source.jsonl \
OUTPUT_FILE=$HOME/staging/dataset_gen_output/frontier_gemma27_rollout.jsonl \
CHECKPOINT_PREFIX=$HOME/staging/dataset_gen_output/frontier_gemma27_rollout \
sbatch naacl/launch_frontier_rollout.slurm
```

Judge with the same Mistral24 validator:

```bash
INPUT_FILE=$HOME/staging/dataset_gen_output/frontier_gemma27_rollout.jsonl \
OUTPUT_FILE=$HOME/staging/dataset_gen_output/frontier_gemma27_validated.jsonl \
CHECKPOINT_PREFIX=$HOME/staging/dataset_gen_output/frontier_gemma27_validated \
sbatch naacl/launch_frontier_validation.slurm
```

Compare behavior transfer:

```bash
python naacl/compare_cross_target.py \
  --primary $HOME/staging/dataset_gen_output/frontier_qwen32_validated.jsonl \
  --cross-target $HOME/staging/dataset_gen_output/frontier_gemma27_validated.jsonl \
  --output $HOME/staging/dataset_gen_output/frontier_qwen_gemma_transfer.json
```

Gemma outputs are robustness measurements and never overwrite the canonical Qwen-derived training labels.

If turn/span evidence transfer is desired, run candidate materialization on the Gemma-validated file and reuse `launch_frontier_evidence.slurm` with `TARGET_MODEL=google/gemma-3-27b-it` and Gemma-specific input/output/checkpoint paths.

## Merge Dataset A and Dataset B

Dataset A is frozen as the repaired legacy corpus. Dataset B must pass its own audit first.

```bash
python naacl/merge_training_corpora.py \
  --legacy-input $HOME/staging/dataset_gen_output/naacl_legacy_prepared.jsonl \
  --frontier-input $HOME/staging/dataset_gen_output/naacl_frontier_prepared.jsonl \
  --output $HOME/staging/dataset_gen_output/naacl_consolidated.jsonl \
  --stats-output $HOME/staging/dataset_gen_output/naacl_consolidated_stats.json \
  --seed 42
```

The merge fails closed if a training record has unresolved supervision or loss weight, or if conversation IDs collide.

## Final split — only after merge

Do not use the old pair-only splitter for the final corpus.

```bash
python naacl/split_consolidated.py \
  --input $HOME/staging/dataset_gen_output/naacl_consolidated.jsonl \
  --output-dir $HOME/staging/dataset_gen_output/naacl_splits \
  --train-frac 0.70 \
  --dev-frac 0.15 \
  --test-frac 0.15 \
  --seed 42
```

Frontier records are grouped by complete `metadata.scenario_family`. Legacy records preserve `pair_id` linkage where available. The splitter performs a final group-leakage assertion before writing the splits.

## Model cache preflight

The cluster runs offline, so confirm the three model snapshots exist before submission:

```bash
ls $HOME/work/hf_models/hub/models--Qwen--Qwen2.5-32B-Instruct
ls $HOME/work/hf_models/hub/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503
ls $HOME/work/hf_models/hub/models--google--gemma-3-27b-it
```

Primary training/evaluation should be based on the Qwen-derived canonical corpus. Gemma remains a cross-target robustness protocol.
