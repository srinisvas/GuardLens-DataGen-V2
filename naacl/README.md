# GuardLens NAACL validity repair

This branch implements the bounded NAACL revision path. It does not regenerate
attacks and it does not rebuild the research project.

Base commit: `b71238099e9baaf75f8e551fb500bb1fe60c768b`

## Scientific scope

The repaired paper should claim **counterfactual evidence localization** and
**evidence-bearing turns/spans**, not causal identification.

The repair addresses the validity issues found in v11:

1. explicit independent-model behavioral validation;
2. fresh paired baseline/intervention replays;
3. no stored-assistant contamination;
4. no position-dependent 12-message replay window;
5. unmeasured effects represented as `null`, never zero;
6. span tests are not gated on success of a historical pivot intervention;
7. only supported evidence is exposed to attribution supervision;
8. unknown malicious pivots are ignored by the pivot loss rather than trained as
   a true no-pivot class;
9. benign training trajectories are length controlled;
10. the original untrimmed benign pool is retained as a stress set.

## Important protocol boundary

The original interactive Llama outcomes were judged in-loop by the Qwen
generator. They remain useful provenance, but a legacy Qwen-only target judgment
is **not sufficient** for repaired training eligibility.

The repaired behavioral gate requires at least one of:

- a fresh Llama replay that crosses the unsafe threshold under the Mistral judge;
- successful independent Mistral transfer validation.

The paired evidence experiment uses Llama to generate assistant responses and
Mistral only to judge those responses. The separate Mistral validation pass is a
behavioral transfer check, not attribution ground truth.

## 0. Inputs

Prefer the pre-causal adversarial artifact:

```bash
$HOME/staging/dataset_gen_output/combined_dedup.jsonl
```

Reuse the separately generated clean benign pool:

```bash
$HOME/staging/dataset_gen_output/benign_clean.jsonl
```

Do not rerun adversarial generation.

## 1. Independent Mistral behavioral validation

```bash
INPUT_FILE=$HOME/staging/dataset_gen_output/combined_dedup.jsonl \
OUTPUT_NAME=naacl_mistral_validation \
VAL_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
USE_COUNTERFACTUAL=false \
N_VAL_SHARDS=2 \
sbatch launch_val.slurm
```

Normalize provenance and transfer tiers:

```bash
python naacl/merge_independent_validation.py \
  --input $HOME/staging/dataset_gen_output/naacl_mistral_validation_validated.jsonl \
  --output $HOME/staging/dataset_gen_output/naacl_independent_merged.jsonl
```

Before continuing, inspect the printed model/provenance counts. The independent
validator should be Mistral and validation failures must not be interpreted as
negative outcomes.

## 2. Paired evidence replay

The hardened runner is `naacl/evidence_analysis_v3.py`, launched by:

```bash
INPUT_FILE=$HOME/staging/dataset_gen_output/naacl_independent_merged.jsonl \
OUTPUT_FILE=$HOME/staging/dataset_gen_output/naacl_evidence.jsonl \
sbatch naacl/launch_evidence.slurm
```

Default roles:

- GPU 0: `meta-llama/Meta-Llama-3-8B-Instruct` response-generating target
- GPU 1: `mistralai/Mistral-7B-Instruct-v0.3` outcome judge

The runner derives its intervention anchor from the **fresh replay**, verifies
span offsets before editing, prioritizes spans around the fresh unsafe
transition, treats large effects on negative controls as violations using
absolute delta, and fails closed on judge/runtime errors.

The default cap is six positive candidate spans and two negative controls per
malicious record.

### Resume warning

The default checkpoint is:

```bash
$HOME/staging/dataset_gen_output/naacl_evidence_v3.checkpoint.jsonl
```

Resume that checkpoint only with the same model, seed, thresholds and span caps.
If you intentionally change the protocol, delete it or set a new
`CHECKPOINT_FILE`. Do not mix results from different protocols.

The Slurm job runs the first audit automatically. You can also run:

```bash
python naacl/audit_repaired_dataset.py \
  --input $HOME/staging/dataset_gen_output/naacl_evidence.jsonl
```

Any evidence-analysis error must be fixed/rerun before dataset preparation.

## 3. Prepare repaired supervision and benign length control

```bash
python naacl/prepare_dataset.py \
  --evidence-input $HOME/staging/dataset_gen_output/naacl_evidence.jsonl \
  --benign-input $HOME/staging/dataset_gen_output/benign_clean.jsonl \
  --output $HOME/staging/dataset_gen_output/naacl_dataset.jsonl \
  --benign-stress-output $HOME/staging/dataset_gen_output/naacl_benign_untrimmed_stress.jsonl \
  --stats-output $HOME/staging/dataset_gen_output/naacl_dataset_stats.json \
  --seed 42
```

Then require the prepared-data invariants:

```bash
python naacl/audit_repaired_dataset.py \
  --input $HOME/staging/dataset_gen_output/naacl_dataset.jsonl \
  --require-prepared
```

Preparation performs three important compatibility repairs:

- unsupported malicious spans become `EVIDENCE_CANDIDATE` with ignored
  attribution supervision;
- annotated benign spans are explicit attribution negatives;
- a malicious record with no established evidence-bearing turn sets
  `pivot_supervision_ignore=true`. Benign `pivot_turn_id=None` remains a true
  supervised no-pivot example.

## 3A. Candidate restoration of original interactive benign twins

The current repaired primary artifact replaced the original interactive benign
twins with a separately generated benign pool in order to match user-turn
counts. Before changing the frozen dataset, audit whether the original twins can
be restored without replaying or rewriting any conversation.

The interactive generator stored Llama responses for both sides of each pair,
but only the malicious side received a generation-time llama_validation object.
The NAACL fail-closed normalization therefore made every original benign twin
ineligible because target-side validation provenance was missing, not because
all twins were observed unsafe.

The restoration audit repairs only that missing provenance:

- do not regenerate Llama responses;
- do not rerun the existing independent-model replay;
- do not rerun malicious counterfactual evidence;
- judge the already-stored benign Llama responses with
  `mistralai/Mistral-Small-3.1-24B-Instruct-2503`, the Dataset B judge;
- use Dataset B's frozen B2-v5 dual-boundary judge protocol
  (`frontier_context_judge_v5`, rubric `dual_boundary_union_v1`,
  conservative-union aggregation), thresholds (unsafe 0.50, ambiguous 0.35,
  minimum confidence 0.55), pair-ID seed policy, and 32K full-prefix judge runtime;
- fail closed unless the local v5 pass-A/pass-B prompt fingerprints match the
  frozen B4 export and the cached 24B revision is
  `68faf511d618ef198fef186659617cfd2eb8e33a`;
- retain the historical `mistralai/Mistral-7B-Instruct-v0.3` independent replay
  as a distinct second validation channel rather than overwriting or rerunning it;
- require both the 24B bridge adjudication and the existing 7B independent replay
  to validate the benign twin;
- preserve original pair_id, topic/setup, text, and natural trajectory length;
- build a separate candidate artifact before any frozen-data replacement.

Run:

```bash
sbatch naacl/launch_stored_twin_judge.slurm
```

The launcher is protocol-locked to the Dataset B 24B bridge judge and separately
fails closed unless the historical Dataset A independent-validation provenance is
uniformly `mistralai/Mistral-7B-Instruct-v0.3`. Its default outputs are:

```text
results-new/naacl_evidence_twins_bridge24b.jsonl
results-new/naacl_evidence_twins_bridge24b_stats.json
results-new/naacl_legacy_twins_restored_candidate.jsonl
results-new/naacl_legacy_twins_restored_candidate_stats.json
results-new/naacl_legacy_twins_restored_excluded.jsonl
```

This stage does not modify the current frozen train/dev/test artifacts. Inspect
the recovered-pair count and structural distributions before deciding whether
the candidate should replace the current legacy primary corpus.

Before allocating the judge GPU, the launcher now fails closed unless all of the
following hold:

- the restoration input contains exactly 545 original benign twins, 545
  validated interactive malicious siblings, and 526 final repaired malicious
  candidates;
- every benign twin matches the immutable pre-independent-validation source on
  observable turn IDs, roles, and text;
- each pair preserves the original shared setup prefix, target domain, style,
  and exact pair ID;
- the reconstructed 526 malicious records are exactly identical to the current
  frozen prepared Dataset A malicious records;
- the historical independent validator is uniformly
  `mistralai/Mistral-7B-Instruct-v0.3`;
- the 24B judge model revision, small model artifacts, package versions, A100
  hardware profile, CUDA runtime, driver, v5 prompt fingerprints, seed policy,
  and deterministic vLLM runtime match the frozen Dataset B B2 receipt;
- the offline 24B cache is complete.

For restored benign supervision, validated user-turn annotations become explicit
negative span targets, while any legacy assistant-turn span annotations remain
present only as ignored provenance because span localization is user-turn-only.
The active validation provenance is rewritten to describe the 24B bridge plus
the reused historical 7B replay; the old incomplete provenance is retained
separately for auditability.

The job holds an exclusive run lock, preserves an append-only resumable
checkpoint, removes stale derived outputs at start, and writes
`results-new/naacl_twin_bridge24b_completion.json` only after the bridge
artifact, restored candidate, statistics, and exclusions all succeed. The
completion receipt records SHA-256 digests for all inputs and outputs plus the
Git revision and Slurm job ID.

### 3B. Judge-capacity agreement audit

After the 24B bridge artifact exists, run a separate same-response agreement
audit with the historical 7B model. This audit does not affect corpus admission.

```bash
sbatch naacl/launch_twin_judge_agreement_7b.slurm
```

By default it deterministically samples 120 original A benign twins, judges the
same stored Llama responses with `mistralai/Mistral-7B-Instruct-v0.3` using the
same frozen B2-v5 dual-rubric prompts, thresholds, full-prefix context policy,
and seed schedule, then compares those decisions against the already-produced
24B bridge judgments.

The comparison reports exact status agreement, binary safe/non-safe agreement,
Cohen's kappa for both, max-unsafe-score correlation and mean absolute
difference, the status confusion matrix, and the disagreement record IDs.

Default output:

```text
results-new/naacl_twin_judge_agreement_7b_vs_24b.json
```

The 7B agreement run is a sensitivity analysis only. Do not use it to relabel or
change restored-pair membership selected by the precommitted 24B bridge gate.

## 4. Recreate splits

```bash
rm -rf $HOME/staging/dataset_gen_output/naacl_splits
python split_dataset.py \
  --input $HOME/staging/dataset_gen_output/naacl_dataset.jsonl \
  --output-dir $HOME/staging/dataset_gen_output/naacl_splits \
  --seed 42 \
  --human-benchmark 100 \
  --double-annotated 50
```

The Transformer training job independently checks conversation-ID and pair-ID
leakage across splits before training.

## 5. Retrain unchanged architecture/baselines

On `GuardLens-Transformer`, branch `naacl-validity-repair`:

```bash
SPLIT_DIR=$HOME/staging/dataset_gen_output/naacl_splits \
BASE_OUTPUT=$HOME/work/results/guardlens_naacl/checkpoints \
sbatch train_naacl.slurm
```

Before GPU training, `train_naacl.slurm` runs a **train/dev-only** length probe.
The held-out test set is not used to decide whether preprocessing is acceptable.
By default, dev length-only AUC above 0.65 stops training for investigation.

The same five existing variants are retrained:

- GuardLens
- GuardLens-NoFusion
- GuardLens-NoCF
- turn-level classifier
- ConversationDeBERTa

## 6. Targeted evaluation

```bash
SPLIT_DIR=$HOME/staging/dataset_gen_output/naacl_splits \
CKPT_DIR=$HOME/work/results/guardlens_naacl/checkpoints \
OUT_DIR=$HOME/work/results/guardlens_naacl/results \
sbatch eval_naacl.slurm
```

This runs, after the pipeline is frozen:

- held-out length-only shortcut probe;
- top-k evidence-turn localization;
- leave-one-turn-out baseline;
- attribution intervention metrics and evidence-tier analysis;
- utility grid;
- NoCF attribution/utility ablation;
- original untrimmed benign stress evaluation when the stress file exists.

Checkpoint loading falls back from `best_attribution.pt` to `best.pt` to
`best_detection.pt`, so evaluation does not fail merely because attribution F1
never exceeded the checkpoint-saving threshold.

## 7. External MHJ

If the NAACL manuscript retains the original MHJ generalization result, rerun
MHJ with the **new repaired checkpoint** before reporting it. Do not copy the old
v11 number into the revised paper. MHJ should be treated as external behavioral /
intervention validation, not as repaired counterfactual evidence ground truth.

## Go / no-go

Proceed to the manuscript rewrite only when all of the following hold:

1. both dataset audits pass;
2. no validated-malicious evidence jobs remain in error state;
3. the dev shortcut preflight passes, and the final held-out length-only probe
   is not a strong classifier;
4. a meaningful subset of malicious trajectories has supported evidence;
5. GuardLens retains the qualitative classification/localization/intervention
   story relative to direct baselines;
6. NoCF and LOTO comparisons do not erase the claimed contribution;
7. false positives remain acceptable on the untrimmed benign stress set;
8. every external result retained in the manuscript is rerun with repaired
   checkpoints.

If those conditions hold, stop and write the NAACL paper. Do not expand this
branch into the full Phase-2 rebuild.
