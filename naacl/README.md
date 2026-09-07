# GuardLens NAACL validity repair

This branch implements the **moderate** revision path. It does not rebuild the
research project and it does not regenerate attacks.

Base commit: `b71238099e9baaf75f8e551fb500bb1fe60c768b`

## Scope

The repair addresses the v11 validity problems that can directly undermine the
paper's claims:

1. Revalidate the existing trajectories with an independent model family.
2. Recompute counterfactual evidence with fresh paired baseline/intervention
   replays.
3. Remove stored-assistant contamination and the 12-message sliding window.
4. Represent untested effects as `null`, never as measured zero.
5. Test span interventions independently instead of gating them on pivot-CF
   success.
6. Hide construction-only spans from attribution supervision.
7. Length-match benign examples for the primary train/dev/test distribution.
8. Preserve the original untrimmed benign pool as an out-of-distribution stress
   test.

The model architecture is intentionally unchanged. The paper should be reframed
around **counterfactual evidence localization**, **evidence-bearing turns**, and
**evidence-bearing spans**, not causal identification.

## What is explicitly out of scope

Do not add a new attack generator, new architecture, new large model suite,
Phase-2 construction gate, new objective taxonomy, or a full dataset rebuild
unless the repaired experiment falsifies the original empirical story.

## 0. Inputs to reuse

Prefer the existing pre-causal merged adversarial artifact, typically:

```bash
$HOME/staging/dataset_gen_output/combined_dedup.jsonl
```

It must contain the generated user trajectories and the original
`llama_validation`. Using the old final malicious artifact is acceptable if the
pre-causal file is unavailable because `evidence_analysis.py` archives and
clears the old v11 evidence fields before recomputation.

Also reuse the separately generated clean benign pool, typically:

```bash
$HOME/staging/dataset_gen_output/benign_clean.jsonl
```

Do **not** rerun adversarial generation.

## 1. Independent behavioral validation

The old `merge_validations.py` treated the post-generation validation field as
Qwen validation, despite the v11 README describing Mistral. For this repair,
make the provenance explicit by rerunning the existing trajectories through
Mistral.

From the datagen repository:

```bash
INPUT_FILE=$HOME/staging/dataset_gen_output/combined_dedup.jsonl \
OUTPUT_NAME=naacl_mistral_validation \
VAL_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
USE_COUNTERFACTUAL=false \
N_VAL_SHARDS=2 \
sbatch launch_val.slurm
```

Normalize the result:

```bash
python naacl/merge_independent_validation.py \
  --input $HOME/staging/dataset_gen_output/naacl_mistral_validation_validated.jsonl \
  --output $HOME/staging/dataset_gen_output/naacl_independent_merged.jsonl
```

Inspect the printed transfer-tier counts and confirm that the independent model
is Mistral, not Qwen.

## 2. Paired evidence replay

This is the only substantial GPU repair. It uses two different model families:

- GPU 0: `meta-llama/Meta-Llama-3-8B-Instruct` as the response-generating target
- GPU 1: `mistralai/Mistral-7B-Instruct-v0.3` as the outcome judge

Baseline and intervention replays use identical per-turn seed schedules and
full history. Stored assistant turns are ignored.

```bash
INPUT_FILE=$HOME/staging/dataset_gen_output/naacl_independent_merged.jsonl \
OUTPUT_FILE=$HOME/staging/dataset_gen_output/naacl_evidence.jsonl \
sbatch naacl/launch_evidence.slurm
```

The default compute cap tests at most six positive candidate spans and two
negative-control spans per malicious record. This is intentionally bounded for
the moderate revision. Raise it only if evidence coverage is clearly too sparse.

The Slurm job automatically runs the first validity audit when replay finishes.
You can rerun it manually:

```bash
python naacl/audit_repaired_dataset.py \
  --input $HOME/staging/dataset_gen_output/naacl_evidence.jsonl
```

## 3. Prepare attribution supervision and length-match benign data

```bash
python naacl/prepare_dataset.py \
  --evidence-input $HOME/staging/dataset_gen_output/naacl_evidence.jsonl \
  --benign-input $HOME/staging/dataset_gen_output/benign_clean.jsonl \
  --output $HOME/staging/dataset_gen_output/naacl_dataset.jsonl \
  --benign-stress-output $HOME/staging/dataset_gen_output/naacl_benign_untrimmed_stress.jsonl \
  --stats-output $HOME/staging/dataset_gen_output/naacl_dataset_stats.json \
  --seed 42
```

Then enforce the post-prepare invariants:

```bash
python naacl/audit_repaired_dataset.py \
  --input $HOME/staging/dataset_gen_output/naacl_dataset.jsonl \
  --require-prepared
```

This step deliberately changes unestablished malicious span labels to
`EVIDENCE_CANDIDATE` with `supervision_tier=ignore`. That prevents the existing
Transformer loader's legacy label-name fallback from silently turning an
untested `MALICIOUS_TRIGGER` into positive attribution ground truth.

## 4. Recreate splits

```bash
python split_dataset.py \
  --input $HOME/staging/dataset_gen_output/naacl_dataset.jsonl \
  --output-dir $HOME/staging/dataset_gen_output/naacl_splits \
  --seed 42 \
  --human-benchmark 100 \
  --double-annotated 50
```

Pair linkage and the existing stratification logic are retained.

## 5. Retrain the unchanged model suite

Checkout the `naacl-validity-repair` branch in `GuardLens-Transformer` and run:

```bash
SPLIT_DIR=$HOME/staging/dataset_gen_output/naacl_splits \
BASE_OUTPUT=$HOME/work/results/guardlens_naacl/checkpoints \
sbatch train_naacl.slurm
```

This retrains the same five existing variants:

- GuardLens
- GuardLens-NoFusion
- GuardLens-NoCF
- turn-level classifier
- ConversationDeBERTa

Old v11 checkpoints are not overwritten.

## 6. Run the targeted reviewer/validity evaluations

```bash
SPLIT_DIR=$HOME/staging/dataset_gen_output/naacl_splits \
CKPT_DIR=$HOME/work/results/guardlens_naacl/checkpoints \
OUT_DIR=$HOME/work/results/guardlens_naacl/results \
sbatch eval_naacl.slurm
```

This produces:

- length-only shortcut probe
- top-k evidence-bearing turn hit rate
- leave-one-turn-out occlusion baseline
- attribution intervention metrics by evidence tier
- GuardLens vs surface-risk utility grid

## 7. Evaluate the original untrimmed benign distribution

Do not report only the length-matched benign test set. Preserve the harder
question of whether the repaired model still behaves well on the original long
benign distribution:

```bash
python -m guardlens.evaluation.eval_boundary_stress \
  --boundary-files $HOME/staging/dataset_gen_output/naacl_benign_untrimmed_stress.jsonl \
  --checkpoint $HOME/work/results/guardlens_naacl/checkpoints/guardlens/best.pt \
  --output $HOME/work/results/guardlens_naacl/results/untrimmed_benign_stress.json \
  --device cuda
```

Interpret the reported `false_positive_rate` as the key stress metric.

## Go / no-go decision

Proceed to the NAACL manuscript rewrite if all of the following are true:

1. Both dataset audits pass.
2. The length-only probe is no longer a strong classifier. A test AUC close to
   0.5 is ideal. If it remains above roughly 0.65, investigate the shortcut
   before treating classification results as clean.
3. A meaningful subset of malicious trajectories receives supported
   counterfactual evidence. Do not invent a zero delta for unsupported or
   unassessable cases.
4. GuardLens retains the qualitative headline story on classification,
   early/intervention utility, and localization relative to direct baselines.
5. False positives on the untrimmed benign stress set remain acceptably low.

If the repaired results preserve the central story, stop here and write the
NAACL paper. Do not turn this branch into the full Phase-2 project.
