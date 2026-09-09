# GuardLens multi-author frontier source pipeline

This is the canonical execution path for the merged GPT-5.6 + Astra Stage-0 corpus.
The four A100-80GB GPUs are dedicated to Dataset B until completion.

## Locked source

- 3,000 records
- 1,200 adversarial authoring-intent records
- 1,800 benign authoring-intent records
- 1,200 paired scenarios = 2,400 primary candidates
- 600 standalone hard-benign stress candidates
- 600 scenario families
- author intent is generation metadata only, never scientific ground truth

Canonical source:

```bash
export OUT=$HOME/staging/dataset_gen_output
export MERGED_SOURCE=$OUT/guardlens_source_merged_v3_3000.jsonl
```

## Locked runtime/evidence protocol

- target: `Qwen/Qwen2.5-32B-Instruct`
- judge: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`
- B1: `frontier_fixed_user_rollout_v2`
- B2: `frontier_context_judge_v3`
- B4: `frontier_context_paired_counterfactual_v4`
- Qwen `max_tokens=2048`
- Qwen `max_model_len=16384`
- Qwen temperature `0.0`
- every target response must have `finish_reason=stop`
- Mistral runtime context `16384`
- judge observable-prefix budget `40000` characters
- judge policy: full observable prefix or fail closed

Old 320/640/1280-token checkpoints are not reusable because the B1 configuration
fingerprint contains the generation envelope. B2 additionally fingerprints the
actual Mistral runtime context window, so a validation artifact generated with a
different judge server context cannot reuse a reviewed checkpoint.

## Model-visible information boundary

Research metadata remains in JSONL for provenance, auditing, grouping, and
supervision, but is never serialized into target/judge prompts. B1, B2, and B4
model requests may contain only observable role/text conversation content plus the
explicit counterfactual replacement at B4. Pair IDs, labels, author/source fields,
scenario/mechanism families, `intended_structure`, expected pivots, and candidate
annotations are researcher-side only.

Executable canary tests inject secret values into these hidden fields and fail if
they appear in captured Qwen or Mistral requests. The GuardLens Transformer branch
has a separate training-side canary proving that changing construction/provenance
metadata cannot change model-visible turn text/role features.

## Full-throttle GPU topology

### B1 rollout

Four Qwen servers, one per A100. Default `WORKERS_PER_GPU=2`, giving eight logical
shards. Multiple workers share each vLLM server so independent trajectories can be
continuous-batched.

### B2 validation

Four Mistral servers, one per A100. Default `WORKERS_PER_GPU=4`, giving sixteen
logical shards. Judge responses are short JSON objects, so batching independent
conversation prefixes materially improves throughput.

### B4 evidence

Three Qwen servers use GPUs 0-2. One shared Mistral judge uses GPU 3. Default
`WORKERS_PER_TARGET=2`, giving six evidence shards. The judge batches requests from
all six workers.

B4 also uses execution-only prefix reuse. A fresh full baseline is still generated
and judged once and must exactly reproduce B1/B2. After that gate passes, an
intervention replay reuses the already-proven identical conversation prefix before
the intervention and regenerates/judges only the intervention turn and downstream
suffix. The optimizer independently verifies fresh baseline response fingerprints
against stored B1 assistant text. If that identity check fails, it falls back to a
full replay instead of reusing the prefix.

All vLLM servers enable prefix caching.

## Parallel-output integrity

B1, B2, and B4 use `merge_stage_shards.py` after workers finish. Source order and
`source_index % num_shards` are authoritative. Duplicate IDs, IDs in the wrong
shard, unexpected IDs, missing IDs, incorrect shard counts, and cross-shard
duplicates are fatal. Parallel output is never merged by silent dictionary
overwrite.

## Static gate — mandatory before any GPU submission

Run after every pull and before GPU submission:

```bash
python -m compileall -q naacl

python -m unittest \
  naacl/test_frontier_cpu.py \
  naacl/test_multi_author_source.py \
  naacl/test_generation_completion.py \
  naacl/test_frontier_evidence_fast.py \
  naacl/test_frontier_prompt_leakage.py \
  naacl/test_stage_shard_merge.py \
  -v

bash -n naacl/launch_frontier_rollout.slurm
bash -n naacl/launch_frontier_validation.slurm
bash -n naacl/launch_frontier_evidence.slurm
```

Do not submit the production GPU jobs unless this complete gate passes.

## Final 20-record B1 smoke at 2048 / 16K

Use the same previously selected 20 records. Run the production topology rather
than another one-GPU hour-long smoke:

```bash
GPU_COUNT=4 \
WORKERS_PER_GPU=2 \
INPUT_FILE=$OUT/frontier_multi_author_smoke20.jsonl \
OUTPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_rollout_v2_2048_16k.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_smoke_qwen32_rollout_v2_2048_16k \
SOURCE_PREFLIGHT_MODE=schema \
sbatch naacl/launch_frontier_rollout.slurm
```

Required pass condition:

- 20/20 `rollout_status=complete`
- 10/10 complete pairs
- every assistant generation `finish_reason=stop`
- no instrumentation errors
- no target-context errors
- strict shard merge passes

Do not select around length failures. If a record still reaches exactly 2048 and
ends with `finish_reason=length`, stop and inspect before the full launch.

## B2 smoke

First cache the locked Mistral 3.1 judge. Then:

```bash
GPU_COUNT=4 \
WORKERS_PER_GPU=4 \
INPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_rollout_v2_2048_16k.jsonl \
OUTPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_validated_v3_2048_16k.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_smoke_qwen32_validated_v3_2048_16k \
sbatch naacl/launch_frontier_validation.slurm
```

Inspect all 20 judgments, author balance, confidence, malicious unsafe rate, benign
safe rate, and earliest unsafe anchors before evidence. B2 provenance must report
`judge_max_model_len=16384`, `judge_max_context_chars=40000`, and the full-prefix
context policy.

## B3 candidate materialization

```bash
python naacl/materialize_frontier_candidates.py \
  --input $OUT/frontier_multi_author_smoke_qwen32_validated_v3_2048_16k.jsonl \
  --output $OUT/frontier_multi_author_smoke_qwen32_candidates_v3_2048_16k.jsonl \
  --max-turn-candidates 4 \
  --spans-per-turn 2 \
  --controls 2
```

## B4 evidence smoke — mandatory determinism gate

Run the production 3-target + 1-shared-judge topology on the smoke:

```bash
TARGET_GPU_COUNT=3 \
WORKERS_PER_TARGET=2 \
INPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_candidates_v3_2048_16k.jsonl \
OUTPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_evidence_v4_2048_16k.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_smoke_qwen32_evidence_v4_2048_16k \
sbatch naacl/launch_frontier_evidence.slurm
```

The decisive gate is baseline reproducibility. Every malicious record reaching B4
must reproduce the stored B1 target responses exactly and B2 judge outputs across
behavior, harmfulness, refusal strength, unsafe score, and confidence.

This smoke is mandatory even though the CPU differential test proves the prefix
optimization algebraically under deterministic stubs. It empirically verifies
that the real vLLM servers remain reproducible when B1/B2 and B4 use different
continuous-batching topologies. If baseline hashes or judge outputs drift, do not
run full B4.

## Full B1 — 3,000 records

```bash
GPU_COUNT=4 \
WORKERS_PER_GPU=2 \
INPUT_FILE="$MERGED_SOURCE" \
OUTPUT_FILE=$OUT/frontier_multi_author_qwen32_rollout_v2_2048_16k.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_qwen32_rollout_v2_2048_16k \
SOURCE_PREFLIGHT_MODE=strict \
sbatch naacl/launch_frontier_rollout.slurm
```

The job has a 24-hour wall limit. If it times out, submit the exact same command
again. Append-only checkpoints resume only records whose source and 2048/16K
configuration fingerprints match.

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

As with B1, resubmit the identical command if the 24-hour allocation expires.
Checkpoint fingerprints prevent stale protocol reuse.

## B5 canonical Dataset B

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

Primary Dataset B remains complete-pair-only. A malicious/benign twin pair enters
only when both sides pass their respective Qwen/Mistral/evidence gates. Standalone
hard benigns remain evaluation-only.

## Merge with frozen Dataset A and split once

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

The consolidated split asserts no conversation, pair, frontier scenario-family,
or exact normalized user-trajectory leakage across train/dev/test. Primary splits
are IID across mechanism families by design; mechanism-held-out evaluation is a
separate secondary protocol and must not be claimed from the primary split.

Only after the consolidated split passes do we hand off to the GuardLens
Transformer branch. Dataset A stays frozen throughout.
