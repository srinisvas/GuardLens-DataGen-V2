# GuardLens Frontier Dataset B Smoke Run

Run this before any full 1,500-record frontier rollout. The smoke is source-only/outcome-blind, pair-complete, and deterministic.

## 0. Paths

```bash
export OUTPUT_DIR=$HOME/staging/dataset_gen_output
export SOURCE=$OUTPUT_DIR/guardlens_source_full_v3.jsonl
export SMOKE_SOURCE=$OUTPUT_DIR/frontier_smoke_source.jsonl
mkdir -p $OUTPUT_DIR
```

The 1,500-record source file must already exist at `$SOURCE`.

## 1. CPU source audit and regression tests

```bash
python naacl/audit_frontier_source.py --input $SOURCE
python -m unittest naacl/test_frontier_cpu.py -v
```

Do not continue unless the source audit ends with `SOURCE PREFLIGHT PASSED` and the regression suite passes.

## 2. Select 10 complete diverse pairs / 20 records

```bash
python naacl/select_frontier_smoke_subset.py \
  --input $SOURCE \
  --output $SMOKE_SOURCE \
  --stats-output $OUTPUT_DIR/frontier_smoke_source_stats.json \
  --pairs 10 \
  --seed 44

python naacl/audit_frontier_source.py \
  --input $SMOKE_SOURCE \
  --schema-only
```

The selector uses source metadata only. It keeps complete twins and deliberately covers both `context_required` and `surface_control` pair hardness when available.

## 3. Offline model/cache preflight

```bash
python naacl/check_frontier_environment.py \
  --model-cache $HOME/work/hf_models \
  --models \
    Qwen/Qwen2.5-32B-Instruct \
    mistralai/Mistral-Small-3.1-24B-Instruct-2503
```

Do not submit GPU jobs if the environment preflight fails.

## 4. B1 Qwen32 rollout smoke

Use one GPU and one shard so exactly the selected 20 records are processed.

```bash
N_SHARDS=1 \
INPUT_FILE=$SMOKE_SOURCE \
OUTPUT_FILE=$OUTPUT_DIR/frontier_smoke_qwen32_rollout.jsonl \
CHECKPOINT_PREFIX=$OUTPUT_DIR/frontier_smoke_qwen32_rollout \
SOURCE_PREFLIGHT_MODE=schema \
sbatch --gres=gpu:1 naacl/launch_frontier_rollout.slurm
```

After completion, require 20 `rollout_status=complete` records and no errors.

Quick summary:

```bash
python - <<'PY'
import json, collections, os
p=os.path.expanduser('~/staging/dataset_gen_output/frontier_smoke_qwen32_rollout.jsonl')
r=[json.loads(x) for x in open(p,encoding='utf-8') if x.strip()]
print('records',len(r))
print('rollout',collections.Counter(x.get('rollout_status') for x in r))
print('labels',collections.Counter(x.get('label') for x in r))
PY
```

## 5. B2 independent Mistral24 validation smoke

```bash
N_SHARDS=1 \
INPUT_FILE=$OUTPUT_DIR/frontier_smoke_qwen32_rollout.jsonl \
OUTPUT_FILE=$OUTPUT_DIR/frontier_smoke_qwen32_validated.jsonl \
CHECKPOINT_PREFIX=$OUTPUT_DIR/frontier_smoke_qwen32_validated \
sbatch --gres=gpu:1 naacl/launch_frontier_validation.slurm
```

Summarize before proceeding:

```bash
python - <<'PY'
import json, collections, os
p=os.path.expanduser('~/staging/dataset_gen_output/frontier_smoke_qwen32_validated.jsonl')
r=[json.loads(x) for x in open(p,encoding='utf-8') if x.strip()]
print('records',len(r))
print('validation',collections.Counter(x.get('validation_status') for x in r))
print('label×validation',collections.Counter((x.get('label'),x.get('validation_status')) for x in r))
for x in r:
    v=x.get('frontier_behavioral_validation',{}) or {}
    print(x.get('conversation_id'), 'label=',x.get('label'), 'status=',x.get('validation_status'),
          'max_unsafe=',v.get('max_unsafe_score'), 'anchor=',v.get('earliest_unsafe_user_turn_id'))
PY
```

Qualitatively inspect the realized assistant trajectories and judge decisions for all 20 records. The smoke should not be treated as a statistical result; its purpose is to catch prompt-formatting, judge-parsing, provenance, or obvious target/judge behavior failures before the full run.

## 6. B3 intervention candidates

```bash
python naacl/materialize_frontier_candidates.py \
  --input $OUTPUT_DIR/frontier_smoke_qwen32_validated.jsonl \
  --output $OUTPUT_DIR/frontier_smoke_qwen32_candidates.jsonl \
  --max-turn-candidates 4 \
  --spans-per-turn 2 \
  --controls 2
```

Only behaviorally validated malicious records should receive intervention candidates.

## 7. B4 paired evidence smoke

Use one target/judge pair = two GPUs.

```bash
N_SHARDS=1 \
INPUT_FILE=$OUTPUT_DIR/frontier_smoke_qwen32_candidates.jsonl \
OUTPUT_FILE=$OUTPUT_DIR/frontier_smoke_qwen32_evidence.jsonl \
CHECKPOINT_PREFIX=$OUTPUT_DIR/frontier_smoke_qwen32_evidence \
sbatch --gres=gpu:2 naacl/launch_frontier_evidence.slurm
```

The decisive runtime gate here is baseline reproducibility. Any target-response or judge-trajectory drift between B1/B2 and the fresh B4 baseline fails closed and must be understood before the full evidence run.

Summarize:

```bash
python - <<'PY'
import json, collections, os
p=os.path.expanduser('~/staging/dataset_gen_output/frontier_smoke_qwen32_evidence.jsonl')
r=[json.loads(x) for x in open(p,encoding='utf-8') if x.strip()]
print('evidence status',collections.Counter((x.get('frontier_evidence_analysis',{}) or {}).get('status') for x in r))
print('tiers',collections.Counter(x.get('supervision_tier') for x in r))
print('supported records',sum(bool(x.get('evidence_turn_ids')) for x in r))
for x in r:
    if x.get('label')==1 and x.get('validation_status')=='validated':
        a=x.get('frontier_evidence_analysis',{}) or {}
        print(x.get('conversation_id'), a.get('status'), 'baseline_reproduced=',a.get('baseline_reproduced_stored_rollout'),
              'evidence_turn_ids=',x.get('evidence_turn_ids'))
PY
```

## 8. Smoke decision

Proceed to the full 1,500-record B1/B2 pipeline only when all of the following hold:

- no rollout errors
- no judge parser/protocol errors
- no authoring metadata exposure
- behavior judgments look qualitatively sensible on all 20 records
- B4 fresh baselines reproduce B1/B2 exactly for every malicious record that reaches evidence analysis
- candidate offsets/replacements are aligned
- unsupported interventions remain unvalidated/ignored rather than becoming positive supervision

A low malicious success count in 10 pairs is not itself a software failure, but it is an empirical warning. If almost none of the authored malicious twins become unsafe on Qwen32, inspect the trajectories before spending the full GPU budget.
