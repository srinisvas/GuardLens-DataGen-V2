# GuardLens multi-author frontier pipeline — production protocol

This is the canonical path for the merged GPT-5.6 + Astra Stage-0 corpus. The
four A100-80GB GPUs are dedicated to Dataset B until completion.

## Frozen source contract

```text
records                         3000
adversarial authoring intent    1200
benign authoring intent         1800
paired scenarios                1200
standalone hard benign           600
scenario families                600
```

```bash
export OUT=$HOME/staging/dataset_gen_output
export MERGED_SOURCE=$OUT/guardlens_source_merged_v3_3000.jsonl
```

Authoring intent is construction metadata, never scientific ground truth.
Standalone hard benigns are evaluation-only. Primary Dataset B is complete-pair
only.

## Locked B1–B4 protocol

```text
target                         Qwen/Qwen2.5-32B-Instruct
judge                          mistralai/Mistral-Small-3.1-24B-Instruct-2503
B1                             frontier_fixed_user_rollout_v2
B2                             frontier_context_judge_v3
B4                             frontier_context_paired_counterfactual_v4
Qwen max output                2048 tokens
Qwen runtime context           16384 tokens
Qwen temperature               0.0
completion contract            finish_reason=stop + positive completion_tokens
Mistral runtime context        32768 tokens
judge observable prefix        100000 characters
judge context policy           full observable prefix or fail closed
judge max output               180 tokens
seed policy                    pair_id_if_present_else_conversation_id_v1
```

The 32K/100K judge envelope was chosen after the final B1 smoke. Its longest
completed observable trajectory was about 52.2K characters and contained about
11.1K Qwen completion tokens before adding user text and the judge prompt. The
previous 40K-character guard would therefore have rejected a valid rollout. The
100K character value is only a fail-closed serialization guard; Mistral's 32K
runtime token window remains the actual model-context limit.

Paired malicious/benign twins share one record-level seed derived from `pair_id`.
Standalone records derive their seed from `conversation_id`. B2 and B4 reuse the
B1-recorded seed rather than independently deriving a new seed.

### Natural target variance is intentional

Matched twins control the authored scenario, style, length, and construction
variables. They do **not** require byte-identical Qwen assistant histories. Even
with temperature 0 and a shared seed, independent GPU/batched inference can yield
small generation differences. That natural variance is retained rather than
copying one twin's assistant response into the other.

This is scientifically acceptable because the direct malicious-twin minus
benign-twin response difference is not the causal estimator. Counterfactual
evidence is assessed within each realized malicious trajectory in B4. Natural
between-twin assistant variation therefore contributes variance/noise and a more
realistic assistant-history distribution rather than defining the intervention.
We still audit label/source/shard balance so execution topology cannot become a
systematic label-correlated channel.

The B1 configuration fingerprint includes the 2048/16K envelope and seed policy.
B2 fingerprints the judge model, thresholds, seed policy, 32K runtime context,
100K observable-prefix guard, and context policy. Production B4 fingerprints the
target/judge runtime windows, paired-seed source, and execution optimization.

## Model-visible information boundary

Research metadata is retained in JSONL for provenance, grouping, supervision, and
auditing but is never serialized into target/judge prompts. Model requests may
contain only observable role/text conversation content plus the explicit B4
counterfactual replacement.

Researcher-side-only fields include labels, conversation/pair IDs, generator and
corpus version, scenario/mechanism family, `intended_structure`, expected pivots,
pair hardness, candidate annotations/origins, and split-group metadata.

`test_frontier_prompt_leakage.py` places secret canaries into those fields and
captures B1, B2, and B4 requests. Any canary appearing in a model request fails the
test. The GuardLens Transformer branch has a separate `test_metadata_leakage.py`
canary showing that construction/provenance metadata cannot change model-visible
turn text/role features.

## Full-throttle topology

### B1

Four Qwen servers, one per GPU. Two rollout workers per Qwen server. Eight logical
record shards total. vLLM continuous-batches independent worker requests.

### B2

Four Mistral servers, one per GPU. Four validation workers per judge server.
Sixteen logical shards total. Each Mistral server uses a 32K runtime context.

### B4

GPUs 0–2 run three Qwen servers. GPU 3 runs one shared Mistral server. Two
evidence workers per Qwen server produce six logical evidence shards. The shared
judge uses the same 32K/100K context contract as B2.

Current B4 code runs a complete fresh baseline first and requires exact B1 target
response and B2 judge reproduction before using its prefix-reuse optimization.
This is a separate within-record reproducibility gate; it is not a requirement
that malicious/benign twins have identical B1 histories. The B4 smoke must test
whether that stronger gate is realistic under the production batching topology.
If it fails, revisit the B4 baseline-pairing rule rather than forcing twin answers
to be identical.

## Parallel-output integrity

B1, B2, and B4 all use `merge_stage_shards.py`. The source JSONL is authoritative
for membership, shard ownership, and final order. Duplicate IDs, cross-shard
duplicates, unexpected IDs, wrong-shard records, missing records, and incorrect
shard counts are fatal.

## Mandatory CPU/static gate

```bash
python -m compileall -q naacl

python -m unittest \
  naacl/test_frontier_cpu.py \
  naacl/test_multi_author_source.py \
  naacl/test_generation_completion.py \
  naacl/test_frontier_evidence_fast.py \
  naacl/test_frontier_prompt_leakage.py \
  naacl/test_stage_shard_merge.py \
  naacl/test_frontier_validation_protocol.py \
  -v

bash -n naacl/launch_frontier_rollout.slurm
bash -n naacl/launch_frontier_validation.slurm
bash -n naacl/launch_frontier_evidence.slurm
```

Do not submit GPU jobs unless all commands pass.

## B1 production-topology smoke

The completed 20-record 2048/16K B1 smoke is the production B1 gate. Required
properties are 20/20 complete, 10/10 complete pairs, all assistant
`finish_reason=stop`, correct 2048/16K provenance, paired seed groups sharing one
record seed, strict shard merge, and `ROLLOUT AUDIT PASSED`. Byte-identical Qwen
answers across matched twins are explicitly **not** required.

## B2 smoke — 32K judge

Use the already completed B1 smoke; do not regenerate it for this judge-only
change.

```bash
GPU_COUNT=4 \
WORKERS_PER_GPU=4 \
JUDGE_MAX_MODEL_LEN=32768 \
JUDGE_MAX_CONTEXT_CHARS=100000 \
INPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_rollout_v2_2048_16k.jsonl \
OUTPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_validated_v3_2048_16k_j32k.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_smoke_qwen32_validated_v3_2048_16k_j32k \
sbatch naacl/launch_frontier_validation.slurm
```

Require all 20 records to end in a terminal B2 status (`validated`, `rejected`, or
`ambiguous`) with zero `incomplete` records and `B2 PROTOCOL AUDIT PASSED`.
Specifically verify that the longest ~52.2K-character smoke trajectory is judged
through all of its assistant turns without context overflow.

Inspect representative behavioral judgments, including clear unsafe-help cases,
benign twins, and target refusals, before scaling B2.

## B3 smoke

```bash
python naacl/materialize_frontier_candidates.py \
  --input $OUT/frontier_multi_author_smoke_qwen32_validated_v3_2048_16k_j32k.jsonl \
  --output $OUT/frontier_multi_author_smoke_qwen32_candidates_v3_2048_16k_j32k.jsonl \
  --max-turn-candidates 4 \
  --spans-per-turn 2 \
  --controls 2
```

Candidate fields are proposals only and never enter target/judge prompts.

## B4 smoke

```bash
TARGET_GPU_COUNT=3 \
WORKERS_PER_TARGET=2 \
INPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_candidates_v3_2048_16k_j32k.jsonl \
OUTPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_evidence_v4_2048_16k_j32k.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_smoke_qwen32_evidence_v4_2048_16k_j32k \
sbatch naacl/launch_frontier_evidence.slurm
```

The launcher runs the B1 and B2 protocol audits before allocating evidence work and
the B4 protocol audit after strict shard merge. Do not launch full B4 until this
smoke establishes the actual baseline-replay behavior under the production
continuous-batching topology.

## Full B1 — 3000 records

```bash
GPU_COUNT=4 \
WORKERS_PER_GPU=2 \
INPUT_FILE="$MERGED_SOURCE" \
OUTPUT_FILE=$OUT/frontier_multi_author_qwen32_rollout_v2_2048_16k.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_qwen32_rollout_v2_2048_16k \
SOURCE_PREFLIGHT_MODE=strict \
sbatch naacl/launch_frontier_rollout.slurm
```

If the 24-hour allocation expires, resubmit the identical command. Append-only
checkpoints reuse only records matching source/configuration fingerprints.

## Full B2

```bash
GPU_COUNT=4 \
WORKERS_PER_GPU=4 \
INPUT_FILE=$OUT/frontier_multi_author_qwen32_rollout_v2_2048_16k.jsonl \
OUTPUT_FILE=$OUT/frontier_multi_author_qwen32_validated_v3_2048_16k_j32k.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_qwen32_validated_v3_2048_16k_j32k \
sbatch naacl/launch_frontier_validation.slurm
```

## Full B3

```bash
python naacl/materialize_frontier_candidates.py \
  --input $OUT/frontier_multi_author_qwen32_validated_v3_2048_16k_j32k.jsonl \
  --output $OUT/frontier_multi_author_qwen32_candidates_v3_2048_16k_j32k.jsonl \
  --max-turn-candidates 4 \
  --spans-per-turn 2 \
  --controls 2
```

## Full B4

```bash
TARGET_GPU_COUNT=3 \
WORKERS_PER_TARGET=2 \
INPUT_FILE=$OUT/frontier_multi_author_qwen32_candidates_v3_2048_16k_j32k.jsonl \
OUTPUT_FILE=$OUT/frontier_multi_author_qwen32_evidence_v4_2048_16k_j32k.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_qwen32_evidence_v4_2048_16k_j32k \
sbatch naacl/launch_frontier_evidence.slurm
```

Resubmit the identical command after a 24-hour timeout. The six checkpoint files
resume independently.

## B5 Dataset B

```bash
python naacl/prepare_frontier_dataset.py \
  --input $OUT/frontier_multi_author_qwen32_evidence_v4_2048_16k_j32k.jsonl \
  --output $OUT/naacl_frontier_prepared.jsonl \
  --benign-stress-output $OUT/naacl_frontier_hard_benign_stress.jsonl \
  --excluded-output $OUT/naacl_frontier_excluded.jsonl \
  --stats-output $OUT/naacl_frontier_prepared_stats.json

python naacl/audit_frontier_dataset.py --input $OUT/naacl_frontier_prepared.jsonl
python naacl/audit_frontier_stress.py --input $OUT/naacl_frontier_hard_benign_stress.jsonl
```

Primary Dataset B stays complete-pair-only. Standalone hard benigns remain
training-ineligible evaluation stress.

## Merge frozen A + validated B and split exactly once

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

The primary split keeps complete frontier scenario families and legacy pairs
indivisible and asserts no conversation, pair, scenario-family, or exact normalized
user-trajectory leakage across train/dev/test. Dataset A remains frozen.
