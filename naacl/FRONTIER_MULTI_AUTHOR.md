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
Mistral runtime context        16384 tokens
judge observable prefix        40000 characters
judge context policy           full observable prefix or fail closed
seed policy                    pair_id_if_present_else_conversation_id_v1
```

Paired malicious/benign twins share one record-level seed derived from `pair_id`.
Standalone records derive their seed from `conversation_id`. B2 and B4 reuse the
B1-recorded seed rather than independently deriving a new seed. This prevents a
hidden random-seed difference from becoming a matched-pair confound.

The B1 configuration fingerprint includes the 2048/16K envelope and seed policy.
B2 fingerprints the judge model, thresholds, seed policy, 16K runtime context,
40K observable-prefix budget, and context policy. Production B4 fingerprints the
target/judge runtime windows, paired-seed source, and execution optimization.

## Model-visible information boundary

Research metadata is retained in JSONL for provenance, grouping, supervision, and
auditing but is never serialized into target/judge prompts. Model requests may
contain only observable role/text conversation content plus the explicit B4
counterfactual replacement.

The following are researcher-side only:

- label
- conversation/pair IDs
- author/generator and corpus version
- scenario/mechanism family
- `intended_structure`
- expected pivots
- pair hardness
- candidate annotations/origins
- split-group metadata

`test_frontier_prompt_leakage.py` places secret canaries into those fields and
captures B1, B2, and B4 requests. Any canary appearing in a model request fails the
test.

The GuardLens Transformer branch has a separate `test_metadata_leakage.py` canary
showing that construction/provenance metadata cannot change model-visible turn
text/role features.

## Full-throttle topology

### B1

Four Qwen servers, one per GPU. Two rollout workers per Qwen server. Eight logical
record shards total. vLLM continuous-batches independent worker requests.

### B2

Four Mistral servers, one per GPU. Four validation workers per judge server.
Sixteen logical shards total.

### B4

GPUs 0–2 run three Qwen servers. GPU 3 runs one shared Mistral server. Two
evidence workers per Qwen server produce six logical evidence shards. The shared
judge batches requests from all six workers.

B4 runs a complete fresh baseline first and requires exact B1 target-response and
B2 judge reproduction. Only after that does the optimized path reuse an unchanged
prefix before an intervention. The optimizer independently rechecks fresh baseline
response fingerprints against stored B1 assistant text. If prefix identity is not
proven it falls back to a full replay.

## Parallel-output integrity

B1, B2, and B4 all use `merge_stage_shards.py`. The source JSONL is authoritative
for membership, shard ownership, and final order. The merger fails on:

- duplicate conversation IDs
- duplicate IDs across shards
- unexpected IDs
- records in the wrong shard
- missing records
- incorrect shard record counts

There is no silent dictionary-overwrite merge path.

## Mandatory CPU/static gate

Run this after pulling the production branch and before allocating GPUs:

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

On the Transformer branch run:

```bash
python -m unittest test_metadata_leakage.py -v
```

before eventual GuardLens training.

## Final B1 production-topology smoke

Use the exact same previously selected 20 records:

```bash
GPU_COUNT=4 \
WORKERS_PER_GPU=2 \
INPUT_FILE=$OUT/frontier_multi_author_smoke20.jsonl \
OUTPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_rollout_v2_2048_16k.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_smoke_qwen32_rollout_v2_2048_16k \
SOURCE_PREFLIGHT_MODE=schema \
sbatch naacl/launch_frontier_rollout.slurm
```

Required pass:

```text
20/20 complete
10/10 complete pairs
all assistant finish_reason=stop
no instrumentation/context errors
paired seed groups share exactly one record seed
strict shard merge passes
ROLLOUT AUDIT PASSED
```

Do not select around a length failure. If any response still hits 2048 with
`finish_reason=length`, inspect before full B1.

## B2 smoke

After the locked Mistral-3.1 cache is ready:

```bash
GPU_COUNT=4 \
WORKERS_PER_GPU=4 \
INPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_rollout_v2_2048_16k.jsonl \
OUTPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_validated_v3_2048_16k.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_smoke_qwen32_validated_v3_2048_16k \
sbatch naacl/launch_frontier_validation.slurm
```

The launcher runs the B1 audit before loading Mistral, strict-merges B2 shards,
and then runs `audit_frontier_validation_protocol.py`. Require:

```text
B2 PROTOCOL AUDIT PASSED
```

Inspect validation counts and representative GPT/Astra malicious and benign
trajectories before B4.

## B3 smoke

```bash
python naacl/materialize_frontier_candidates.py \
  --input $OUT/frontier_multi_author_smoke_qwen32_validated_v3_2048_16k.jsonl \
  --output $OUT/frontier_multi_author_smoke_qwen32_candidates_v3_2048_16k.jsonl \
  --max-turn-candidates 4 \
  --spans-per-turn 2 \
  --controls 2
```

Candidate fields are proposals only and never enter target/judge prompts.

## B4 smoke — mandatory real determinism test

```bash
TARGET_GPU_COUNT=3 \
WORKERS_PER_TARGET=2 \
INPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_candidates_v3_2048_16k.jsonl \
OUTPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_evidence_v4_2048_16k.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_smoke_qwen32_evidence_v4_2048_16k \
sbatch naacl/launch_frontier_evidence.slurm
```

This smoke is mandatory because B1/B2 and B4 use different continuous-batching
topologies. Every malicious record reaching evidence must reproduce stored target
responses exactly and B2 judgments across behavior, harmfulness, refusal strength,
unsafe score, and confidence. The launcher also runs the post-merge B4 protocol
audit. Require:

```text
B4 PROTOCOL AUDIT PASSED
```

If target hashes or judge results drift, do not launch full B4.

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

The allocation has a 24-hour wall limit. If it expires, resubmit the identical
command. Append-only checkpoints reuse only records matching source/configuration
fingerprints.

## Full B2

```bash
GPU_COUNT=4 \
WORKERS_PER_GPU=4 \
INPUT_FILE=$OUT/frontier_multi_author_qwen32_rollout_v2_2048_16k.jsonl \
OUTPUT_FILE=$OUT/frontier_multi_author_qwen32_validated_v3_2048_16k.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_qwen32_validated_v3_2048_16k \
sbatch naacl/launch_frontier_validation.slurm
```

## Full B3

```bash
python naacl/materialize_frontier_candidates.py \
  --input $OUT/frontier_multi_author_qwen32_validated_v3_2048_16k.jsonl \
  --output $OUT/frontier_multi_author_qwen32_candidates_v3_2048_16k.jsonl \
  --max-turn-candidates 4 \
  --spans-per-turn 2 \
  --controls 2
```

## Full B4

```bash
TARGET_GPU_COUNT=3 \
WORKERS_PER_TARGET=2 \
INPUT_FILE=$OUT/frontier_multi_author_qwen32_candidates_v3_2048_16k.jsonl \
OUTPUT_FILE=$OUT/frontier_multi_author_qwen32_evidence_v4_2048_16k.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_qwen32_evidence_v4_2048_16k \
sbatch naacl/launch_frontier_evidence.slurm
```

Resubmit the identical command after a 24-hour timeout. The six checkpoint files
resume independently.

## B5 Dataset B

Only after the B4 protocol audit passes:

```bash
python naacl/prepare_frontier_dataset.py \
  --input $OUT/frontier_multi_author_qwen32_evidence_v4_2048_16k.jsonl \
  --output $OUT/naacl_frontier_prepared.jsonl \
  --benign-stress-output $OUT/naacl_frontier_hard_benign_stress.jsonl \
  --excluded-output $OUT/naacl_frontier_excluded.jsonl \
  --stats-output $OUT/naacl_frontier_prepared_stats.json

python naacl/audit_frontier_dataset.py \
  --input $OUT/naacl_frontier_prepared.jsonl

python naacl/audit_frontier_stress.py \
  --input $OUT/naacl_frontier_hard_benign_stress.jsonl
```

Primary Dataset B stays complete-pair-only. Standalone hard benigns stay
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
user-trajectory leakage across train/dev/test. Mechanism families are balanced in
the IID primary split; mechanism-held-out OOD remains a separate evaluation.

Dataset A remains frozen throughout.
