# GuardLens NAACL Frontier-Authored Dataset Pipeline

This runbook processes the 1,500-record `GuardLensSourceTrajectory/v3` corpus into the same canonical training contract as repaired legacy Dataset A, then merges the two canonical corpora before one leakage-safe train/dev/test split.

## Locked model roles

- Primary target: `Qwen/Qwen2.5-32B-Instruct`
- Independent judge: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`
- Cross-target robustness: `google/gemma-3-27b-it`
- Default model context cap: 8192 tokens
- Generation temperature: 0.0
- Deterministic per-record/per-turn seeds
- Primary unsafe threshold: 0.50
- Minimum judge confidence for unsafe/evidence decisions: 0.55

The target and judge never receive `intended_structure`, pair hardness, expected pivots, authoring-manifest fields, or other hidden construction metadata. The judge receives only the observable realized conversation prefix.

`intended_structure.expected_pivot_turn_ids` is generation-intent metadata only. It may propose intervention locations but may never populate `pivot_turn_id` directly.

## Dataset B primary/stress policy

The full source intentionally contains:

- 600 paired adversarial records
- 600 paired benign twins
- 300 standalone hard benign records

The paired 1,200-record slice has identical class-conditional user-turn distributions. The 300 standalone hard benign records are shorter and make full-source user-turn count predictive of label. Therefore:

- primary Dataset B contains only complete validated malicious/benign pairs
- no orphan twin enters the primary corpus
- validated standalone hard benign records are retained as a separate stress-evaluation set
- standalone hard benign records never enter primary training

This policy prevents a five-turn benign shortcut while preserving the hard-negative evaluation value.

## Stage B0 — CPU source preflight

Copy `guardlens_source_full_v3.jsonl` to `$HOME/staging/dataset_gen_output/` and run before allocating GPUs:

```bash
python naacl/audit_frontier_source.py \
  --input $HOME/staging/dataset_gen_output/guardlens_source_full_v3.jsonl

python -m unittest naacl/test_frontier_cpu.py -v
```

Expected source audit for v3:

```text
Records: 1500
Pairs: 600
Pair hardness: context_required=500, surface_control=100
Standalone benign: 300
Scenario families: 300
Paired class user-turn histograms identical: True
Paired primary length AUC: 0.5000
Full source length AUC: 0.6667   # warning only; standalones are stress-only
SOURCE PREFLIGHT PASSED
```

Source-stage invariants include:

- user turns only
- turn IDs exactly `0,2,4,...`
- no source span annotations
- `pivot_turn_id=null`
- `supervision_tier=null`
- `loss_weight=null`
- `judge_confidence=null`
- `training_eligible=false`
- author-intended pivots must refer to real source user turns

## Offline model/environment preflight

The cluster is offline. Before GPU submission, verify the cached models and library floors:

```bash
python naacl/check_frontier_environment.py \
  --model-cache $HOME/work/hf_models \
  --models \
    Qwen/Qwen2.5-32B-Instruct \
    mistralai/Mistral-Small-3.1-24B-Instruct-2503
```

Before Gemma robustness, run the same check including `google/gemma-3-27b-it`.

Runtime floors enforced by the preflight:

- Qwen2.5: Transformers >= 4.37
- Mistral Small 3.1: vLLM >= 0.8.1 and mistral-common >= 1.5.4
- Gemma 3: vLLM >= 0.8.3 and Transformers >= 4.50

The Mistral judge launchers explicitly use Mistral tokenizer/config/load modes rather than relying on auto-detection.

## Stage B1 — Qwen32 fixed-user rollout

```bash
sbatch naacl/launch_frontier_rollout.slurm
```

Default output:

```text
$HOME/staging/dataset_gen_output/frontier_qwen32_rollout.jsonl
```

The rollout:

- sends only source user-turn text to Qwen
- inserts one target assistant response after every user turn using odd turn IDs
- preserves all source/research metadata outside the prompt
- leaves supervision fields unresolved/null
- records model/config/input fingerprints in checkpoints
- retries prior error/incomplete checkpoint records instead of treating them as complete

## Stage B2 — independent context-aware Mistral24 validation

```bash
sbatch naacl/launch_frontier_validation.slurm
```

Default output:

```text
$HOME/staging/dataset_gen_output/frontier_qwen32_validated.jsonl
```

Validation is fail-closed and context-aware:

- judge sees only the observable conversation prefix
- required judge JSON fields are strictly validated
- malformed/missing/out-of-range judge outputs are retried, then fail the record
- malicious authoring-intent records validate only on a confidence-qualified unsafe crossing
- benign records reject only on a confidence-qualified unsafe crossing
- near-boundary or low-confidence observations become ambiguous rather than supervision
- target/judge provenance and thresholds are checkpoint-fingerprinted

Inspect validation counts before proceeding.

## Stage B3 — deterministic intervention candidate materialization

```bash
python naacl/materialize_frontier_candidates.py \
  --input $HOME/staging/dataset_gen_output/frontier_qwen32_validated.jsonl \
  --output $HOME/staging/dataset_gen_output/frontier_qwen32_candidates.jsonl \
  --max-turn-candidates 4 \
  --spans-per-turn 2 \
  --controls 2
```

Candidate turn priority:

1. observed confidence-qualified unsafe anchor
2. author-intended candidate turns at/before that anchor
3. nearest preceding context turns

Author-intended turns are candidate proposals only. They are never evidence labels.

Candidate spans are deterministically selected for positional coverage. With the default two spans per turn, the policy tests a final/request-bearing clause and a substantial earlier/contextual clause when available. All candidates remain unvalidated until paired replay.

## Stage B4 — paired context-aware counterfactual evidence

```bash
sbatch naacl/launch_frontier_evidence.slurm
```

Default output:

```text
$HOME/staging/dataset_gen_output/frontier_qwen32_evidence.jsonl
```

The four-GPU job runs two parallel target/judge pairs. Each pair uses one Qwen32 GPU and one Mistral24 GPU.

Before any intervention evidence is accepted, the fresh baseline must reproduce the stored B1/B2 state:

- regenerated target responses must hash-match stored Qwen responses
- judge behavior, unsafe score, and confidence must match stored B2 judgments
- target model, judge model, unsafe threshold, and confidence threshold must match B1/B2 provenance

Any baseline drift fails closed.

For each validated malicious trajectory whose fresh baseline remains unsafe, the evidence engine:

- derives the fresh confidence-qualified unsafe anchor
- tests up to four whole-turn interventions
- tests up to six evidence-candidate spans
- tests up to two control spans
- uses paired identical target/judge seed schedules across baseline and intervention
- uses neutral replacement wording rather than explicit safety cues
- ignores low-confidence post-intervention judgments when computing deltas
- marks interventions with insufficient trustworthy judgment as `not_assessable_low_confidence`
- sets `evidence_turn_ids` only from supported interventions
- sets `pivot_turn_id` to the earliest supported evidence turn
- sets `pivot_supervision_ignore=true` if no malicious pivot is established

The later user trajectory is held fixed after intervention. This is **fixed-user-trajectory counterfactual replay**, not adaptive causal simulation.

## Stage B5 — canonical pair-complete preparation

After inspecting evidence statistics:

```bash
python naacl/prepare_frontier_dataset.py \
  --input $HOME/staging/dataset_gen_output/frontier_qwen32_evidence.jsonl \
  --output $HOME/staging/dataset_gen_output/naacl_frontier_prepared.jsonl \
  --benign-stress-output $HOME/staging/dataset_gen_output/naacl_frontier_benign_stress.jsonl \
  --excluded-output $HOME/staging/dataset_gen_output/naacl_frontier_excluded.jsonl \
  --stats-output $HOME/staging/dataset_gen_output/naacl_frontier_prepared_stats.json
```

Primary pair retention requires both twins:

- malicious twin: independently validated unsafe + reproducible fresh Qwen/Mistral evidence baseline remains unsafe
- benign twin: independently validated safe Qwen trajectory

If either twin fails, both are excluded from primary Dataset B. This preserves label balance and exact paired length symmetry.

Validated standalone hard benign records are written only to:

```text
naacl_frontier_benign_stress.jsonl
```

They have `training_eligible=false` and are evaluation-only.

Run both post-preparation audits:

```bash
python naacl/audit_frontier_dataset.py \
  --input $HOME/staging/dataset_gen_output/naacl_frontier_prepared.jsonl

python naacl/audit_frontier_stress.py \
  --input $HOME/staging/dataset_gen_output/naacl_frontier_benign_stress.jsonl
```

Do not merge unless the primary audit ends with:

```text
VALIDITY AUDIT PASSED
```

Do not use the hard-benign stress output unless its audit ends with:

```text
STRESS AUDIT PASSED
```

The primary audit reconstructs supported evidence turns from the stored whole-turn and span interventions instead of trusting `evidence_turn_ids`; checks delta thresholds, negative controls, record-level supervision tier, pivot semantics, complete pair semantics, canonical Qwen/Mistral provenance, equal class counts, and identical class-conditional user-turn histograms.

The stress audit requires validated standalone benign records, canonical Qwen/Mistral provenance, `training_eligible=false`, evaluation-only markers, true no-pivot semantics, and no positive evidence/span supervision.

## Merge frozen Dataset A + canonical Dataset B

Dataset A remains frozen:

```text
naacl_legacy_prepared.jsonl
naacl_legacy_benign_stress.jsonl
naacl_legacy_prepared_stats.json
```

Merge only after Dataset B passes its own primary audit:

```bash
python naacl/merge_training_corpora.py \
  --legacy-input $HOME/staging/dataset_gen_output/naacl_legacy_prepared.jsonl \
  --frontier-input $HOME/staging/dataset_gen_output/naacl_frontier_prepared.jsonl \
  --output $HOME/staging/dataset_gen_output/naacl_consolidated.jsonl \
  --stats-output $HOME/staging/dataset_gen_output/naacl_consolidated_stats.json \
  --seed 42
```

The merge fails closed on:

- unresolved/ineligible records
- invalid loss weights/supervision tiers
- conversation-ID collisions
- non-Qwen frontier provenance in the primary corpus
- exact normalized user-trajectory duplicates across legacy/frontier corpora

## Final split — only after merge

Do not use the legacy pair-only splitter.

```bash
python naacl/split_consolidated.py \
  --input $HOME/staging/dataset_gen_output/naacl_consolidated.jsonl \
  --output-dir $HOME/staging/dataset_gen_output/naacl_splits \
  --train-frac 0.70 \
  --dev-frac 0.15 \
  --test-frac 0.15 \
  --seed 42
```

Indivisible grouping:

- frontier: complete `metadata.scenario_family`
- legacy: `pair_id` when present, otherwise conversation ID

The splitter asserts no conversation, pair, or frontier scenario leakage. It softly balances source×label, source×difficulty, and frontier domain, slice role, pair hardness, trajectory family, mechanism family, and style.

Because each frontier mechanism has only a small number of complete scenario groups, individual mechanisms cannot always appear in both dev and test while also preserving 70/15/15 and scenario integrity. Mechanism-held-out OOD is therefore a separate secondary evaluation protocol, not an invariant of the primary split.

## Gemma 3 27B cross-target robustness

Select the robustness subset **from source before looking at Qwen outcomes**:

```bash
python naacl/select_cross_target_subset.py \
  --input $HOME/staging/dataset_gen_output/guardlens_source_full_v3.jsonl \
  --output $HOME/staging/dataset_gen_output/frontier_gemma27_source.jsonl \
  --stats-output $HOME/staging/dataset_gen_output/frontier_gemma27_subset_stats.json \
  --fraction 0.40
```

Selection keeps complete scenario families. Paired and standalone scenario families are sampled separately and the allocator minimizes residual error over the full source feature distribution across domain, difficulty, trajectory family, pair hardness, mechanism family, style, slice-role composition, and label composition. On the reviewed v3 source, the 40% protocol selects exactly 600 records while preserving the major construction distributions.

Run Gemma using the rollout launcher:

```bash
TARGET_MODEL=google/gemma-3-27b-it \
INPUT_FILE=$HOME/staging/dataset_gen_output/frontier_gemma27_source.jsonl \
OUTPUT_FILE=$HOME/staging/dataset_gen_output/frontier_gemma27_rollout.jsonl \
CHECKPOINT_PREFIX=$HOME/staging/dataset_gen_output/frontier_gemma27_rollout \
sbatch naacl/launch_frontier_rollout.slurm
```

Judge with the same independent Mistral24 validator:

```bash
INPUT_FILE=$HOME/staging/dataset_gen_output/frontier_gemma27_rollout.jsonl \
OUTPUT_FILE=$HOME/staging/dataset_gen_output/frontier_gemma27_validated.jsonl \
CHECKPOINT_PREFIX=$HOME/staging/dataset_gen_output/frontier_gemma27_validated \
sbatch naacl/launch_frontier_validation.slurm
```

Compare behavioral transfer:

```bash
python naacl/compare_cross_target.py \
  --primary $HOME/staging/dataset_gen_output/frontier_qwen32_validated.jsonl \
  --cross-target $HOME/staging/dataset_gen_output/frontier_gemma27_validated.jsonl \
  --output $HOME/staging/dataset_gen_output/frontier_qwen_gemma_transfer.json
```

Gemma results are robustness measurements only. They never overwrite Qwen-derived canonical training labels and canonical preparation rejects non-Qwen primary provenance.

If turn/span evidence transfer is desired later, materialize candidates on the Gemma-validated subset and run the evidence launcher with Gemma-specific target/input/output/checkpoint paths. Keep those results parallel to the Qwen canonical evidence fields.

## Required staged launch policy

Do not immediately run all 1,500 records.

1. CPU source audit + regression tests
2. model/cache environment preflight
3. 20-record pair-complete Qwen rollout smoke
4. Mistral validation smoke and qualitative inspection
5. small paired evidence smoke with baseline reproducibility checks
6. only then launch the full B1/B2/B4 pipeline
7. prepare and run both primary/stress audits before merging

The runtime smoke remains mandatory because static review cannot prove one-A100 memory fit, installed-vLLM model-loader compatibility, or deterministic replay across separate server processes.
