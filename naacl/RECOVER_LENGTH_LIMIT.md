# Recover production B1 after a length-limited response

The approved opt-in policy is `length_retry_2048_4096_8192_v1`. It uses B1 rollout v4 and B4 evidence v7. B2 keeps the frozen v5 rubric, both passes, prompts, thresholds and aggregation. The fixed-policy v3/v5/v6 path remains available and continues to pass the original differential tests.

Each target response starts at 2,048 output tokens. Only `finish_reason=length` with exactly the requested output budget permits retry at 4,096, then 8,192. Each attempt restarts from the identical prompt and seed. The final response must end with `stop`, with valid usage. Every attempt's budget, finish reason, token count and response fingerprint is recorded. Short responses are not regenerated at larger budgets. The HTTP read timeout scales from 10 minutes at 2,048 to 20/40 minutes at 4,096/8,192, so the original transport deadline does not prematurely kill a longer eager decode. Slurm still applies its bounded shutdown/drain window. Models, BF16 precision, eager/batch-invariant runtime, target 16K context and judge 32K context remain unchanged.

B4 uses the same policy for factual and counterfactual turns. Its factual replay compares final response identity and the complete budget/usage trace against B1 before interventions. Adaptive span/control outputs also retain target attempt provenance. Increasing the output budget can still exhaust the fixed context window or judge transcript budget. Such failures are not repaired by truncating the prompt or dropping records.

Record-local completion failures are checkpointed separately, with detailed diagnostics in `failed-records.jsonl`. Independent records continue. Completed records and successful requests remain recoverable. No final dataset is published unless every input record passes. A known unresolved completion failure is reported on resume without repeatedly spending GPU time on the same deterministic failure. Transport, server and provenance failures remain fail-fast. Only explicit target context-envelope HTTP errors are treated as record-local.

## 1. Update and run the CPU gate

From the repository root, after the old allocation has ended:

```bash
git checkout naacl-validity-repair-optimized
git pull --ff-only
conda activate "$HOME/work/conda_envs/dataset_gen"
bash naacl/tests/run.sh

export OUT="$HOME/staging/dataset_naacl"
export OUTPUT_DIR="$OUT"
export B1_INPUT="$HOME/staging/dataset_gen_output/guardlens_source_merged_v3_3000.jsonl"
export TARGET_BUDGET_POLICY=length_retry_2048_4096_8192_v1
mkdir -p logs "$OUT"
unset PROBE_ONLY PROBE_INPUT PROBE_LEVELS_JSON IMPORT_CHECKPOINTS_JSON
```

## 2. Test the actual failing record on one GPU

This is an empirical completion check for the known blocked record. The CPU tests do not establish GPU completion at the increased budgets. This smoke is additional input for recovery and does not replace the full-production record-count gate.

```bash
export RECOVERY_INPUT="$OUT/recovery_b1_budget_v1_source.jsonl"
export RECOVERY_OUTPUT="$OUT/recovery_b1_budget_v1.jsonl"

python - <<'PY'
import json, os
from pathlib import Path
cid = 'f9bdfb49-366b-5b51-aff3-bb1068164eb3'
with open(os.environ['B1_INPUT']) as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
selected = [row for row in rows if row['conversation_id'] == cid]
assert len(selected) == 1, 'Expected exactly one matching source record'
Path(os.environ['RECOVERY_INPUT']).write_text(json.dumps(selected[0], ensure_ascii=False)+'\n')
PY

RECOVERY_JOB_ID=$( \
  RUN_MODE=smoke EXPECTED_RECORDS=1 \
  GPU_COUNT=1 TARGET_INFLIGHT=4 RECORD_WORKERS=8 \
  INPUT_FILE="$RECOVERY_INPUT" OUTPUT_FILE="$RECOVERY_OUTPUT" \
  STATE_DIR="$OUT/recovery_b1_budget_v1.state" \
  sbatch --parsable --gres=gpu:1 --time=12:00:00 naacl/launch_b1.slurm \
)
RECOVERY_JOB_ID=${RECOVERY_JOB_ID%%;*}
printf '%s\n' "$RECOVERY_JOB_ID" > "$OUT/recovery_b1_budget_v1.jobid"
tail -f "logs/opt_b1_${RECOVERY_JOB_ID}.out"
```

After the job ends, inspect its output and receipt. Require `B1 COMPLETE AND AUDITED` and Slurm `COMPLETED` with exit code `0:0`.

```bash
sacct -j "$RECOVERY_JOB_ID" --format=JobID,State,Elapsed,ExitCode
cat "logs/opt_b1_${RECOVERY_JOB_ID}.err"
python -m json.tool "$RECOVERY_OUTPUT.completion.json"

python - <<'PY'
import json, os
with open(os.environ['RECOVERY_OUTPUT']) as handle:
    record = json.loads(handle.readline())
assert record['rollout_status'] == 'complete'
for turn in record['turns']:
    if turn['role'] == 'assistant':
        p = turn['generation_provenance']
        print(turn['turn_id'], p['finish_reason'], p['completion_tokens'], p['max_tokens'], p['budget_attempts'])
PY
```

If this fails, retain its state and inspect `failed-records.jsonl`. Do not repeatedly resubmit a known terminal failure or raise the ceiling manually.

## 3. Migrate the old production state on CPU

The migration is restricted to the exact code contract at `c9ce9318cf37bbd7c6895d2aa2fa4d9eeb3ee240`, which generated the current production checkpoints. It takes exclusive locks on the old allocation/state, verifies every SQLite result digest, and reconstructs complete records and partial prefixes using exact request identities. It writes a new state directory and migration report. It neither edits the old database nor copies old failures as successes. A completed old 2,048-token request is a completed first attempt under the new policy, so its text, seed and measured usage can be reused.

Run once. If the destination already exists, inspect its `migration.json` instead of rerunning or deleting it.

```bash
export B1_STATE="$OUT/production_b1_3000_adaptive_v1.state"
export B1_OUTPUT="$OUT/production_b1_3000_adaptive_v1.jsonl"

python naacl/migrate_b1_state.py \
  --source-state "$OUT/production_b1_3000_c4.state" \
  --destination-state "$B1_STATE" \
  --input "$B1_INPUT" \
  --mode production --expected-records 3000
```

Require `B1 STATE MIGRATION PASSED`. The report gives the recovered complete-record and target-request counts. Do not bypass a contract mismatch by editing SQLite, fingerprints or the pinned migration manifest.

## 4. Resume all 3,000 records

Use the new state/output paths and the explicit new policy on every resubmission. Import the completed one-record recovery artifact to avoid regenerating it. The normal import checks verify its input/config fingerprints and completeness.

```bash
B1_JOB_ID=$( \
  TARGET_BUDGET_POLICY=length_retry_2048_4096_8192_v1 \
  RUN_MODE=production EXPECTED_RECORDS=3000 \
  GPU_COUNT=4 TARGET_INFLIGHT=4 RECORD_WORKERS=32 \
  INPUT_FILE="$B1_INPUT" OUTPUT_FILE="$B1_OUTPUT" STATE_DIR="$B1_STATE" \
  IMPORT_CHECKPOINTS_JSON="[\"$RECOVERY_OUTPUT\"]" \
  sbatch --parsable --time=12:00:00 naacl/launch_b1.slurm \
)
B1_JOB_ID=${B1_JOB_ID%%;*}
printf '%s\n' "$B1_JOB_ID" > "$OUT/production_b1_3000_adaptive_v1.jobid"
tail -f "logs/opt_b1_${B1_JOB_ID}.out"
```

Repeated allocations must retain the same input, state, code and runtime. Old fixed-protocol smoke outputs should not be compared directly with adaptive outputs using `compare_outputs.py`, because the approved protocol and attempt-provenance fields intentionally differ. Existing fixed-policy tests continue to compare every original scientific field. Before production B4, validate the adaptive GPU replay chain on the new artifacts. B4 scope remains undecided.

After full B1 passes, launch full B2 with `TARGET_BUDGET_POLICY=length_retry_2048_4096_8192_v1` and the completed adaptive B1 artifact. This option selects the accepted input contract. It does not change the Mistral judge generation budget or rubric.
