
## Semantic-mask turn-consistency inspection

Before changing any frozen evidence-turn membership, run the read-only
train/dev inspection:

```bash
FREEZE="$HOME/projects/GuardLens-DataGen-V2/results-naacl/final-data-freeze"

python naacl/audit_semantic_turn_consistency.py \
  --train "$FREEZE/splits_primary/train.jsonl" \
  --dev "$FREEZE/splits_primary/dev.jsonl" \
  --enforce-reviewed-counts \
  --output /tmp/semantic_turn_consistency.json
```

The inspection never reads the held-out test split and never mutates data. It
classifies every semantically masked construction-language span/turn into:

- A: evidence-turn membership conflicts with an explicit whole-turn
  `not_supported` result and no eligible non-masked positive span remains.
- B: evidence-turn membership has no independent local support after masking
  and would rely on a downstream fallback.
- C: evidence-turn membership remains independently supported.
- D: the masked span has no evidence-turn membership dependency.

For the frozen primary corpus, `--enforce-reviewed-counts` requires the
previously audited 14 masked spans across 13 records. Any A/B finding is
reported for review; it does not mutate or silently repair the artifact.


### Approved semantic turn-supervision repair

After inspection establishes exactly four case-A turns and zero case-B turns,
the canonical preparation step reconciles only prepared record-level
`evidence_turn_ids`. Raw B4 evidence inside
`frontier_evidence_analysis` remains unchanged.

The repair rule is intentionally narrow:

- retain a masked evidence turn if the whole-turn intervention is supported;
- retain it if another eligible non-masked positive span independently supports
  that turn;
- remove it only when the whole-turn intervention is explicitly
  `not_supported` and the semantically masked span was the only positive
  support;
- fail closed on any other orphaned masked evidence-turn membership.

Changed records carry `semantic_turn_supervision_repair` provenance. The
expected frozen repair count is four turns across four conversations.

Use `audit_semantic_turn_repair_delta.py` against the old freeze and a
candidate regeneration before replacing any frozen artifact. The delta audit
requires every non-repair record to be identical at the JSON-object level,
requires split membership/order to remain unchanged, and requires the held-out
test JSONL to remain byte-identical.

Build and audit the candidate in one fail-closed CPU workflow:

```bash
bash naacl/rebuild_semantic_turn_repair_candidate.sh
```

The script writes a separate
`results-naacl/final-data-freeze-semantic-turn-repair-candidate` directory,
never overwrites the current freeze, reruns preparation/merge/split/auxiliary
attachment, proves the exact delta, runs the full final-data audit, and records
the preparation code commit.

# Optimized Dataset B execution

**Production length recovery:** the approved opt-in adaptive budget policy, checked state migration and exact commands are in [RECOVER_LENGTH_LIMIT.md](RECOVER_LENGTH_LIMIT.md). The fixed 2,048-token contract below remains the default and reference path. Adaptive execution uses B1 v4 and B4 v8, while the B2 v5 rubric stays unchanged. B1 remains at 16K; adaptive B4 uses Qwen's native 32K context envelope so a valid 4K/8K replay budget is not rejected merely because its counterfactual prefix is longer than the factual one.

Branch `naacl-validity-repair-optimized` starts at `7e43efc6bd2cd836a1bac71dc4f7aac9b616a8fb`. The original `naacl-validity-repair` branch remains the reference. This branch consolidates the current pipeline and changes scheduling and recovery. The review fixes add explicit production counts, trial-bound completion receipts, immutable checkpoint identities and output ownership. GPU equivalence and throughput must still be measured on the A100s before the 3,000-record production run.

## Current entry points

| Task | Current files | Frozen scientific protocol |
|---|---|---|
| B1 rollout | `launch_b1.slurm`, `frontier_rollout.py` | v3 fixed; v4 adaptive. Frozen review source uses v4 |
| B2 validation | `launch_b2.slurm`, `frontier_validation.py`, `frontier_judge.py` | `frontier_context_judge_v5`, `dual_boundary_union_v1` |
| B3 candidates, CPU | `materialize_frontier_candidates.py` | `frontier_candidate_materialization_v2` |
| B4 evidence | `launch_b4.slurm`, `frontier_evidence.py` | v6 fixed; v7/v8 adaptive by context envelope. Frozen review source uses v8 |
| Stage audits | `audit_frontier_rollout.py`, `audit_frontier_validation.py`, `audit_frontier_evidence.py` | Original completion, deterministic runtime, dual-pass and evidence checks |
| Dataset assembly, CPU | `prepare_frontier_dataset.py`, `audit_frontier_dataset.py`, `audit_frontier_stress.py` | Record-bound provenance audit. Frozen review source is adaptive v4/v5/v8 |
| Merge and split, CPU | `merge_training_corpora.py`, `split_consolidated.py`, `attach_training_auxiliary.py` | Grouped primary split, exact-hash leakage prevention, frozen eval partitions |
| Raw review freeze, CPU | `audit_review_export.py` | Exact B4 review SHA-256, manifest, counts, missing-record contract |
| Final data freeze, CPU | `audit_final_data_prep.py` | Raw-derived primary/stress/excluded membership, artifact hashes, A+B membership, shortcut controls, split isolation, exact auxiliary rejected-set contract |
| Execution | `run_stage.py`, `execution.py`, `launch_job.py`, `launch_stage.sh` | Bounded queues, request recovery, supervised replicas |
| Measurements | `probe_runtime.py`, `compare_outputs.py`, `report_performance.py` | Exact equality and measured performance |
| Regression tests | `tests/` | Archived reference tests plus active executor tests |
| Historical implementations | `legacy/` | Original files preserved unchanged |

The active pipeline has no runtime imports from `legacy/`. Run the supported entry points above. The old version wrappers and historical Slurm scripts remain reference material under `legacy/`.

## Quality contract

The production source had 3,000 intended records. The frozen review export used
for data preparation contains 2,999 records because one malicious B4 replay
exhausted all approved completion budgets. Its exact missing-record declaration,
JSONL SHA-256, contract fingerprint, and snapshot digest are pinned by
`audit_review_export.py`. B4 otherwise retains `not_applicable` rows rather
than silently dropping them.

| Invariant | Frozen value / behavior |
|---|---|
| Models | Qwen/Qwen2.5-32B-Instruct and mistralai/Mistral-Small-3.1-24B-Instruct-2503 |
| Inference | vLLM 0.28.0, A100 80GB, BF16, tensor parallel size 1, batch invariant, eager |
| Context / generation | B1 target 16,384; fixed B4 target 16,384; adaptive B4 target 32,768. Target output starts at 2,048 and adaptive mode retries only genuine length finishes at 4,096/8,192. Judge 32,768 / 180 tokens, 100,000 transcript characters |
| Seeds | Base 42, pair-aware record seeds, original per-turn, pass-B and retry offsets |
| Judge | Both unchanged prompts, three parsing attempts per pass, original conservative union and confidence rules |
| Evidence | Full fresh factual target replay and exact stored B1/B2 comparison before interventions |
| Interventions | Original caps of 4 turns, 6 positive spans and 2 controls, original replacement selection and ordering |
| Results | Full suffixes, raw passes, exact deltas, evidence turns, supervision tiers, loss weights and original record ordering |
| Failure behavior | No incomplete, truncated, drifted or unaudited output is published. A counterfactual that alone exceeds adaptive B4's 32K envelope is retained as explicitly unassessable with audited failure provenance; it never becomes causal supervision |

There is no span pruning, reduced dataset scope, changed precision, shorter context, sampled factual audit, or early termination of a scientifically assessable counterfactual. Existing proven-prefix reuse remains. Completed deterministic requests may be recovered by exact identity, but a B4 factual request cannot reuse a B1/B2 cache entry.

Model revisions are resolved from the existing cache's `refs/main`, checked for completeness, and pinned for model and tokenizer loading. Runtime versions, model/config/tokenizer fingerprints and server flags are recorded outside scientific outputs. If the cache or runtime changes, the executor rejects the old state directory.

## What runs concurrently

B1 uses a shared record queue across all target replicas. Turns within each conversation remain sequential. B2 submits immutable observable prefixes and both judge passes to a bounded pool, then assembles judgments in chronological order. B4 overlaps target generation with independent judgments and runs independent interventions concurrently after baseline verification. Each record owns its baseline, and intervention workers cannot overwrite another record's state.

Per-server request limits bound GPU pressure. Soft target affinity favors an equally loaded previous server but permits work to move to free capacity. Each HTTP worker reuses its own connection. All server statistics remain enabled.

Do not infer a safe concurrency of 6 or 8 from the A100 memory size alone. Inspect actual KV capacity in startup logs, prompt-token usage, preemptions, queueing and generated tokens/second. The approximately 43K target KV-token estimate in the review is a sizing hypothesis, not a measured capacity on this installation. Automatic chunked-prefill or scheduler changes are deliberately excluded.

## CPU verification

From the repository root, in the existing dataset environment:

```bash
bash naacl/tests/run.sh
```

The original 86 CPU tests exercise the archived reference. Additional tests exercise the active code and compare entire B1/B2/B4 results against the untouched original executors through a deterministic local HTTP fixture. They check unchanged request payloads, concurrency, hidden raw-pass drift, target drift, interruption recovery, invalid/truncated results, checkpoint corruption, and final dataset preparation. Simulated model equality does not establish numerical invariance on GPUs.

## GPU Gate 0 before throughput tuning

Create the Slurm log directory before submission. Use the already completed deterministic B1 smoke artifact as the probe input. This job starts the frozen servers, compares solo and mixed-load outputs, captures metrics, then releases the allocation without running a stage.

```bash
mkdir -p logs
export OUT=$HOME/staging/dataset_gen_output
export REF_B1=$OUT/frontier_multi_author_smoke_qwen32_rollout_v3_bi_eager_2048_16k.jsonl

INPUT_FILE=$REF_B1 \
PROBE_INPUT=$REF_B1 PROBE_ONLY=1 PROBE_LEVELS_JSON='[1,2,4]' \
STATE_DIR=$OUT/optimized_gate0_3plus1 \
sbatch naacl/launch_b4.slurm
```

The B4 input preflight requires B2 provenance, so for this probe-only command the launcher validates `PROBE_INPUT` as B1 and does not run the B4 input audit. It still checks the frozen servers and runtime. All target and judge replicas are probed. Repeat at higher concurrency only after reviewing the preceding level's measurements. A passing probe covers only its tested prefixes, servers and load levels. It does not replace the full equivalence smoke.

The probe selects observable prefixes across character lengths, including the longest. Real token counts come from the server response usage and metrics, not a character-to-token approximation. Judge prompts are constructed by the actual frozen dual-judge code.

## Full equivalence smoke

Use separate output/state paths so the original reference artifacts remain available. Run stages sequentially after each predecessor passes its audit.

```bash
B1_JOB_ID=$(RUN_MODE=smoke EXPECTED_RECORDS=20 \
INPUT_FILE=$OUT/frontier_multi_author_smoke20.jsonl \
OUTPUT_FILE=$OUT/optimized_smoke_b1.jsonl \
TARGET_INFLIGHT=2 RECORD_WORKERS=16 \
sbatch --parsable naacl/launch_b1.slurm)
B1_JOB_ID=${B1_JOB_ID%%;*}
```

```bash
B2_JOB_ID=$(RUN_MODE=smoke EXPECTED_RECORDS=20 \
INPUT_FILE=$OUT/optimized_smoke_b1.jsonl \
OUTPUT_FILE=$OUT/optimized_smoke_b2.jsonl \
JUDGE_INFLIGHT=4 RECORD_WORKERS=16 \
sbatch --parsable naacl/launch_b2.slurm)
B2_JOB_ID=${B2_JOB_ID%%;*}
```

```bash
python naacl/materialize_frontier_candidates.py \
  --input "$OUT/optimized_smoke_b2.jsonl" \
  --output "$OUT/optimized_smoke_b3.jsonl" \
  --max-turn-candidates 4 --spans-per-turn 2 --controls 2

B4_JOB_ID=$(RUN_MODE=smoke EXPECTED_RECORDS=20 \
INPUT_FILE=$OUT/optimized_smoke_b3.jsonl \
OUTPUT_FILE=$OUT/optimized_smoke_b4.jsonl \
TARGET_INFLIGHT=2 JUDGE_INFLIGHT=4 RECORD_WORKERS=12 INTERVENTION_WORKERS=12 \
sbatch --parsable naacl/launch_b4.slurm)
B4_JOB_ID=${B4_JOB_ID%%;*}
```

Compare all three GPU outputs with the completed deterministic reference artifacts. `compare_outputs.py` compares every scientific field, numerical type and record position. It does not mask differences in provenance, deltas, raw passes, weights or trajectories.

```bash
python naacl/compare_outputs.py \
  --reference "$REF_B1" --optimized "$OUT/optimized_smoke_b1.jsonl" --trial-id "$B1_JOB_ID"
python naacl/compare_outputs.py \
  --reference "$OUT/frontier_multi_author_smoke_qwen32_validated_v5_dual_bi_eager_2048_16k_j32k.jsonl" \
  --optimized "$OUT/optimized_smoke_b2.jsonl" --trial-id "$B2_JOB_ID"
python naacl/compare_outputs.py \
  --reference "$OUT/frontier_multi_author_smoke_qwen32_evidence_v6_bi_eager_2048_16k_j32k.jsonl" \
  --optimized "$OUT/optimized_smoke_b4.jsonl" --trial-id "$B4_JOB_ID"
```

Save the returned job IDs. Each comparison requires the ID of the trial being reviewed and a matching successful `<OUTPUT_FILE>.completion.json` receipt. The receipt binds the output digest and record count to the input fingerprint, code, runtime, scientific configuration and execution settings. The launcher preserves the previous receipt until preflight and runtime validation pass. Rejected submissions leave the previous trial verifiable, but cannot claim its success under their own job ID. Once admitted, a failed rerun invalidates success for that attempt while preserving the previous data file.

Require all stage audits, exact comparisons and zero replay errors. Then benchmark fresh state directories at target concurrency 2, 4, 6, and 8 as memory measurements permit. Keep all scientific settings fixed and compare complete outputs at each promoted level. Do not use a resumed/cached run as a throughput benchmark. Compare 3 target + 1 judge with 2 + 2 only if judge queue/latency measurements justify it. A topology change requires a new state directory and its own equality check.

No 48-hour or 72-hour completion promise is supported yet. Project duration from measured completed eligible records, realized tokens and the actual B2 eligibility census. Include startup, tail latency and all three GPU stages.

## Production and operational controls

After the GPU checks pass, use the same launchers with the full input paths and measured concurrency. `RUN_MODE=production` is the default and requires exactly 3,000 records in B1, B2 and B4. It never infers smoke mode from a smaller input. `RUN_MODE=smoke` must be explicit and defaults to `EXPECTED_RECORDS=20`. The production B1 preflight additionally enforces 1,200 pairs, 600 standalone records and 600 scenarios. B2/B4 preserve exact source membership and order. Final publication occurs only after the full audit succeeds.

| Environment variable | Default / use |
|---|---|
| `INPUT_FILE` | Required |
| `RUN_MODE`, `EXPECTED_RECORDS` | Production / 3,000 by default. Explicit smoke mode defaults to 20 |
| `OUTPUT_FILE` | Input stem + stage + `_optimized.jsonl` under `OUTPUT_DIR` |
| `STATE_DIR` | `<OUTPUT_FILE>.state` |
| `GPU_COUNT` | All allocated GPUs for B1/B2 |
| `TARGET_GPU_COUNT`, `JUDGE_GPU_COUNT` | B4 defaults to 3 targets and remaining GPUs for judging |
| `TARGET_BUDGET_POLICY` | `fixed_2048_v1` by default. Approved opt-in `length_retry_2048_4096_8192_v1` for recovery and downstream stages |
| `TARGET_INFLIGHT` | 2 outstanding requests per target replica |
| `JUDGE_INFLIGHT` | 4 outstanding requests per judge replica |
| `RECORD_WORKERS` | `max(8, 2 × targets × TARGET_INFLIGHT, judges × JUDGE_INFLIGHT)` |
| `INTERVENTION_WORKERS` | `max(4, 2 × targets × TARGET_INFLIGHT)` |
| `MODEL_CACHE`, `CONDA_ENV` | Existing `$HOME/work/hf_models` and `$HOME/work/conda_envs/dataset_gen` |
| `PORT_BASE` | 8300, allocation-local loopback ports |
| `PROBE_INPUT`, `PROBE_ONLY`, `PROBE_LEVELS_JSON` | Optional Gate 0 before stage execution or probe only |

Worker defaults scale with request limits. Explicit smaller worker overrides are honored but produce a startup warning when they restrict target concurrency. The launcher prints the resulting request capacities and chain ceilings.

`WORKERS_PER_GPU`, `WORKERS_PER_TARGET` and `N_SHARDS` are obsolete. Use the controls above. Changing `GPU_COUNT` does not change a Slurm allocation. Match it with `sbatch --gres=gpu:N`. The launchers use Slurm's assigned CUDA devices, require every allocated GPU to have a role, and stop if a server exits.

The default allocation is 12 hours with a ten-minute warning. Override with `sbatch --time=06:00:00` if desired. To pause cleanly, signal the supervisor:

```bash
scancel --signal=USR1 --batch JOB_ID
```

Resubmit with identical input, output, state, runtime and code to resume. Request concurrency and worker counts can change without invalidating completed work, but any promoted concurrency still needs GPU equivalence validation. Completed target responses, validated judge passes, interventions and records are journaled. Signals stop new admission while successful in-flight responses are saved. A hard kill may lose a request that had not committed yet. Transport failures stop admission because an HTTP timeout may leave work running on the server. Judge parsing retries and seed offsets remain unchanged. No failed/partial response is cached as success. In adaptive mode, length-limited attempts are retained in a separate diagnostic namespace so an interrupted escalation can resume. Record-local completion failures are persisted separately and do not stop independent records, but block final publication.

Only one allocation/executor can own a state directory or output destination. The supervisor retains the output lock throughout the allocation and passes ownership to its executor through an inherited file descriptor. Output/receipt writes use unique temporary files and synchronized atomic renames. State is SQLite with rollback journaling and full synchronization, requiring functioning POSIX locks and fsync on the shared filesystem. Do not copy a live state directory. Changing source content, model/runtime identity or active Python code requires a fresh state directory. Old deterministic whole-record checkpoints can be explicitly imported using `IMPORT_CHECKPOINTS_JSON='["/path/shard*.checkpoint.jsonl"]'`. Import requires matching input/config fingerprints, valid terminal outputs and no conflicting records. Nonterminal or incompatible imports fail rather than silently pass.

Inspect progress and performance with:

```bash
python naacl/report_performance.py --state-dir "$OUT/optimized_smoke_b4.jsonl.state"
```

For B4 supervision accrual, run the standard-library reporter **after the allocation
has exited, before submitting the next allocation**. It reads the saved executor
command to locate the original input, checks the input/record identities and every
result/failure digest, and opens SQLite read-only. It refuses to run while either
the allocation or executor owns the state. It writes only a report under
`STATE_DIR/reports/`, without changing journals or publication receipts.

```bash
# After the first production allocation (including any imported smoke records).
python naacl/tools/report_b4_progress.py \
  --state-dir "$B4_STATE" --job-id "$B4_JOB_ID"

# Keep this path before replacing B4_JOB_ID with the next submitted job ID.
export B4_PREVIOUS_REPORT="$B4_STATE/reports/b4-progress-${B4_JOB_ID}.json"

# After the next allocation exits.
python naacl/tools/report_b4_progress.py \
  --state-dir "$B4_STATE" --job-id "$B4_JOB_ID" \
  --previous "$B4_PREVIOUS_REPORT"
```

`cumulative` reports all committed records. `added_since_previous` reports the
snapshot difference, split into `newly_imported` and `newly_computed`. These include
separate `complete` and `not_applicable` counts, record tiers (`cf_strong_records`,
`cf_weak_records`, `llm_confirmed_records`), supported evidence turns, span tiers,
control violations and context-unassessable interventions. Record tiers are
exclusive: a record with both strong and weak spans is one `cf_strong` record,
while both span types are counted separately. Imports do not count as new work.

Report snapshots retain per-record IDs/digests and cannot be overwritten with
different contents. The previous snapshot must belong to the same state and
contract, and all previously counted records must remain identical. Keep import
source files available for provenance accounting. If an end-of-job snapshot is
missed, the next report lists every allocation in that reporting interval. The
database has no per-record completion timestamp, so exact historical daily totals
cannot be reconstructed retroactively. Without `--previous`, the interval begins
at the start of the state, not at the start of the latest allocation.

`records_without_complete_result` includes unresolved failures, which are also
listed separately. `pending_without_persisted_failure` excludes those failures.
`cached_nonrecord_results` counts saved subtasks/attempts across **both completed
and unfinished records**; it is not a count of partially completed records. Partial
work has no final record tier. This is a progress report, not a replacement for the
final scientific audit and publication receipt.

The reporter is deliberately under `naacl/tools/`. Adding it does not change the
executor's `naacl/*.py` code fingerprint or invalidate an existing resumable state.

Each state directory contains request timing/usage in `requests.jsonl` and an `allocation-JOB_ID/` directory with combined server logs, raw Prometheus/GPU samples and the executor command. A matching complete trial receipt with a verified output digest is the completeness signal. File existence alone is insufficient. Standalone `run_stage.py` runs generate and print a trial ID, or accept `--trial-id`; comparison requires that ID. Request latency sums are not GPU wall time.

The precommitted held-out judge sampling/evaluation utilities remain active. Sampling still uses score-free B1 output and excludes the frozen design manifest in `tests/fixtures/`. Dataset preparation preserves pair retention, standalone benign stress separation and loss weights. It directly validates the deterministic v3/v5/v6 fixed chain or the v4/v5/v8 adaptive chain.

vLLM argument reference used for the pinned launcher is [v0.28.0 serve](https://docs.vllm.ai/en/v0.28.0/cli/serve/). No server throughput measurements were available in the development workspace.

## Review regression gate

`tests/test_review_regressions.py` and the expanded HTTP integration tests cover the six review findings. They exercise 2,999-record rejection, stale-output rejection after a failed trial, completed-intervention recovery despite concurrent annotation updates, worker-capacity sizing, exclusive/inherited output ownership, unique atomic temporary files and authenticated solo/concurrent probes. The original scientific differential tests remain mandatory.
