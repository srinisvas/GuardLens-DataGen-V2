# GuardLens Dataset Generation Pipeline — v11

## Overview

End-to-end pipeline for generating a publication-quality multi-turn adversarial prompt detection dataset with token-level causal attribution ground truth.

The pipeline generates adversarial conversations where a strong generator model (Qwen) crafts adaptive multi-turn attacks against a live target model (Llama), validates them against an independent third model family (Mistral), performs counterfactual causal analysis to identify which tokens actually cause jailbreaks, and produces a separate clean benign pool validated by two model families.

**Key design principles:**

- **Cross-model validation, zero circularity.** Qwen generates, Llama is the target, Mistral validates. No model validates its own generation.
- **Interactive adversarial feedback.** The generator adapts user turns based on the target's actual responses — not static template paths.
- **Conservative causal attribution.** Four-pass analysis: LLM span annotation, pivot-turn counterfactual, span-level counterfactual, negative controls.
- **Separate clean benign pool.** Benign conversations generated independently, validated by both Llama and Mistral. Old benign twins treated as hard/boundary negatives.

---

## File Inventory

### Core Pipeline

| File | Purpose |
|---|---|
| `build_semantic_datasetv11.py` | Core data structures, supervision tiers, structured judge, span annotation, counterfactual analysis |
| `inference_backend.py` | Pluggable inference backends: Ollama, vLLM, HuggingFace |

### Generation

| File | Purpose |
|---|---|
| `interactive_generator.py` | Adaptive adversarial conversation generator (Qwen vs Llama with feedback loop) |
| `benign_generator.py` | Clean benign conversation generator (5 categories, length-matched) |
| `launch_gen_interactive.slurm` | 4-GPU parallel adversarial generation SLURM script |
| `launch_benign.slurm` | 4-GPU parallel benign generation SLURM script |

### Validation

| File | Purpose |
|---|---|
| `launch_val.slurm` | Multi-GPU sharded validation with configurable model |
| `run_hpc.py` | HPC runner for validation with checkpoint/resume |

### Causal Analysis

| File | Purpose |
|---|---|
| `causal_analysis.py` | 4-pass causal analysis: span annotation, pivot-turn CF, span CF, negative controls |
| `launch_causal.slurm` | 4-GPU parallel causal analysis SLURM script |

### Post-Processing and Assembly

| File | Purpose |
|---|---|
| `merge_validations.py` | Merges Llama (generation) + Mistral (validation) results with transfer tiers |
| `postprocess_causal.py` | Relabels causal BENIGN_CONTEXT spans, adds distributed-context tier, validates consistency |
| `split_dataset.py` | Train/dev/test splitting with pair linkage, stratification, human benchmark selection |

### Utilities

| File | Purpose |
|---|---|
| `augment_dataset.py` | Post-generation augmentation (not used for interactive data) |
| `mhj_loader.py` | MHJ external dataset loader |
| `prestage_models.sh` | Pre-stage Qwen models to HPC cache |
| `prestage_llama.py` | Pre-stage Llama model |
| `prestage_mistral.py` | Pre-stage Mistral model |

---

## Prerequisites

### Hardware

- 4x NVIDIA A100 80GB GPUs (minimum 2 for generation, 1 for validation)
- SLURM scheduler
- ~100GB disk for models + dataset

### Software

```bash
conda activate dataset_gen
pip install vllm transformers huggingface_hub requests
```

### Models

Pre-stage all three model families before running any jobs:

```bash
# On login node (has internet access)
bash prestage_models.sh                                    # Qwen-7B + 14B
python3 prestage_llama.py --model meta-llama/Meta-Llama-3-8B-Instruct
python3 prestage_mistral.py                                # Mistral-7B-Instruct-v0.3

# If Llama requires auth on compute nodes:
export HF_HUB_OFFLINE=1  # Use for all SLURM jobs
```

---

## Complete Execution Guide

### Architecture

```
Phase 1: ADVERSARIAL GENERATION (2-4 GPUs)
  Qwen-7B/14B (generator) <-> Llama-8B (target)
  Interactive feedback loop: generate -> target responds -> adapt -> repeat
  Output: adversarial conversations with Llama validation baked in

Phase 2: CROSS-MODEL VALIDATION (2-4 GPUs)
  Mistral-7B validates all adversarial records independently
  Output: transfer tiers (transfer_success / target_only / cross_only)

Phase 3: CAUSAL ANALYSIS (4 GPUs)
  Pass 1: LLM span annotation on jailbreak records
  Pass 2: Pivot-turn counterfactual (whole-turn ablation)
  Pass 3: Span-level counterfactual inside causal turns
  Pass 4: Negative-control span validation
  Output: causal/incidental span labels, supervision tiers

Phase 4: BENIGN GENERATION (4 GPUs)
  Qwen-7B (generator) + Llama-8B (responder)
  5 categories: clean everyday, research/technical, topic-matched safe,
                hard benign, false-lead benign
  Output: length-matched benign conversations

Phase 5: BENIGN VALIDATION (2-4 GPUs)
  Validate benign pool against BOTH Llama AND Mistral
  Keep only records accepted by both
  Output: clean_benign + benign_boundary_rejected

Phase 6: POST-PROCESSING + ASSEMBLY (CPU)
  Relabel causal spans, assign tiers, combine, split
  Output: train/dev/test splits + human benchmark
```

---

### Phase 1: Adversarial Generation

#### Step 1a: Smoke Test (50 pairs, ~2.5 hrs)

```bash
N_PAIRS=50 N_PIPELINES=1 sbatch launch_gen_interactive.slurm
```

Check output:
```bash
tail -f logs/igen_<jobid>.out
# Look for: jailbreak rate > 25%
```

#### Step 1b: Full 7B Generation (600 pairs, 2 pipelines, ~14 hrs)

```bash
N_PAIRS=600 N_PIPELINES=2 sbatch launch_gen_interactive.slurm
```

Expected output: `semantic_multiturn_v11_interactive_raw.jsonl` (~1,100 records after cross-shard dedup, ~45% jailbreak rate)

#### Step 1c: 14B Generation (200 pairs, 1 pipeline, ~11 hrs)

Can run in parallel with validation if GPUs available:

```bash
N_PAIRS=200 \
GEN_MODEL=Qwen/Qwen2.5-14B-Instruct \
N_PIPELINES=1 \
OUTPUT_NAME=semantic_multiturn_v11_interactive_14b \
sbatch launch_gen_interactive.slurm
```

Expected output: ~400 records, ~27% jailbreak rate

---

### Phase 2: Cross-Model Validation with Mistral

#### Step 2a: Validate 7B-generated data

```bash
VAL_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
INPUT_FILE=$HOME/staging/dataset_gen_output/semantic_multiturn_v11_interactive_raw.jsonl \
OUTPUT_NAME=semantic_multiturn_v11_interactive \
USE_COUNTERFACTUAL=false \
N_VAL_SHARDS=2 \
HF_HUB_OFFLINE=1 \
sbatch launch_val.slurm
```

Expected: ~70% Mistral jailbreak rate, ~90% transfer rate from Llama jailbreaks

#### Step 2b: Merge 7B validation results

```bash
python3 merge_validations.py \
    --interactive $HOME/staging/dataset_gen_output/semantic_multiturn_v11_interactive_validated.jsonl \
    --output $HOME/staging/dataset_gen_output/merged_7b.jsonl
```

**Inspect the output.** Key numbers to check:
- `transfer_success` count (records jailbreaking both Llama + Mistral)
- `false_alarm_rate` on benign records
- Per-strategy jailbreak rates

#### Step 2c: Validate 14B-generated data

```bash
VAL_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
INPUT_FILE=$HOME/staging/dataset_gen_output/semantic_multiturn_v11_interactive_14b_raw.jsonl \
OUTPUT_NAME=semantic_multiturn_v11_interactive_14b \
USE_COUNTERFACTUAL=false \
N_VAL_SHARDS=2 \
HF_HUB_OFFLINE=1 \
sbatch launch_val.slurm
```

#### Step 2d: Merge 14B validation results

```bash
python3 merge_validations.py \
    --interactive $HOME/staging/dataset_gen_output/semantic_multiturn_v11_interactive_14b_validated.jsonl \
    --output $HOME/staging/dataset_gen_output/merged_14b.jsonl
```

#### Step 2e: Combine 7B + 14B and dedup

```bash
cat $HOME/staging/dataset_gen_output/merged_7b.jsonl \
    $HOME/staging/dataset_gen_output/merged_14b.jsonl \
    > $HOME/staging/dataset_gen_output/combined_all.jsonl

python3 -c "
import sys, json
sys.path.insert(0, '.')
import build_semantic_datasetv11 as gen

records = []
with open('$HOME/staging/dataset_gen_output/combined_all.jsonl') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

before = len(records)
records = gen.deduplicate_dataset(records, threshold=0.70)
gen.write_jsonl(records, '$HOME/staging/dataset_gen_output/combined_dedup.jsonl')
print(f'Dedup: {before} -> {len(records)}')
"
```

---

### Phase 3: Causal Analysis

#### Step 3a: Delete stale checkpoints

```bash
rm -f $HOME/staging/dataset_gen_output/causal_pass1.checkpoint
rm -f $HOME/staging/dataset_gen_output/causal_analyzed_*checkpoint*
```

#### Step 3b: Run 4-pass causal analysis (4 GPUs, ~2 hrs)

```bash
INPUT_FILE=$HOME/staging/dataset_gen_output/combined_dedup.jsonl \
OUTPUT_NAME=causal_analyzed_all_final \
HF_HUB_OFFLINE=1 \
sbatch launch_causal.slurm
```

**Inspect output.** Key numbers:
- `cf_turn_strong` + `cf_turn_weak`: pivot turns with causal evidence
- `distributed_or_unclear`: multi-turn distributed causality (expected to be large)
- Causal spans by label: `PAYLOAD_SPAN` and `MALICIOUS_TRIGGER` should dominate

#### Step 3c: Post-process causal results (CPU, instant)

```bash
python3 postprocess_causal.py \
    --input $HOME/staging/dataset_gen_output/causal_analyzed_all_final.jsonl \
    --output $HOME/staging/dataset_gen_output/causal_postprocessed_all.jsonl
```

---

### Phase 4: Benign Generation

#### Step 4: Generate benign pool (4 GPUs, ~4-5 hrs)

```bash
N_RECORDS=1000 sbatch launch_benign.slurm
```

Generates across 5 categories (30% clean everyday, 20% research/technical, 20% topic-matched safe, 20% hard benign, 10% false-lead). Turn lengths matched to adversarial data (8-32 turns).

---

### Phase 5: Benign Validation

#### Step 5a: Validate with Mistral

```bash
VAL_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
INPUT_FILE=$HOME/staging/dataset_gen_output/benign_pool_v11_raw.jsonl \
OUTPUT_NAME=benign_pool_v11_mistral \
N_VAL_SHARDS=2 \
HF_HUB_OFFLINE=1 \
sbatch launch_val.slurm
```

#### Step 5b: Validate with Llama

```bash
VAL_MODEL=meta-llama/Meta-Llama-3-8B-Instruct \
INPUT_FILE=$HOME/staging/dataset_gen_output/benign_pool_v11_raw.jsonl \
OUTPUT_NAME=benign_pool_v11_llama \
N_VAL_SHARDS=2 \
HF_HUB_OFFLINE=1 \
sbatch launch_val.slurm
```

#### Step 5c: Filter benign pool (CPU, instant)

Keep only records accepted by BOTH validators:

```bash
python3 - <<'PY'
import os, json
from collections import Counter

base = os.path.expanduser("~/staging/dataset_gen_output")

llama = {}
with open(f"{base}/benign_pool_v11_llama_validated.jsonl") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            llama[r["conversation_id"]] = r

mistral = {}
with open(f"{base}/benign_pool_v11_mistral_validated.jsonl") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            mistral[r["conversation_id"]] = r

print(f"Llama validated: {len(llama)}")
print(f"Mistral validated: {len(mistral)}")

clean, boundary = [], []

for cid, m_rec in mistral.items():
    l_rec = llama.get(cid)
    if not l_rec:
        continue

    l_jb = bool(l_rec.get("causal_validation", {}).get("jailbreak_detected", False))
    m_jb = bool(m_rec.get("causal_validation", {}).get("jailbreak_detected", False))

    r = m_rec
    r["llama_benign_validation"] = l_rec.get("causal_validation", {})
    r["mistral_benign_validation"] = m_rec.get("causal_validation", {})
    r["source_dataset"] = "separate_benign_pool"
    r["transfer_tier"] = "benign"
    r["pivot_kind"] = "none"
    r["pivot_turn_id"] = None
    r["label"] = 0

    if not l_jb and not m_jb:
        r["benign_status"] = "clean_benign"
        r["validation_status"] = "validated"
        r["training_eligible"] = True
        r["supervision_tier"] = "benign_validated"
        r["loss_weight"] = 1.0
        clean.append(r)
    else:
        rejected_by = []
        if l_jb: rejected_by.append("llama")
        if m_jb: rejected_by.append("mistral")
        r["benign_status"] = "benign_boundary_rejected"
        r["validation_status"] = "rejected"
        r["training_eligible"] = False
        r["use_as"] = "benign_boundary_stress"
        r["rejected_by"] = rejected_by
        boundary.append(r)

print(f"\nClean benign: {len(clean)}")
print(f"Boundary rejected: {len(boundary)}")
print("\nClean by category:")
for c, n in Counter(r.get("family", "?") for r in clean).most_common():
    print(f"  {c}: {n}")

with open(f"{base}/benign_clean.jsonl", "w") as f:
    for r in clean:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(f"{base}/benign_boundary.jsonl", "w") as f:
    for r in boundary:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nWrote {len(clean)} to benign_clean.jsonl")
print(f"Wrote {len(boundary)} to benign_boundary.jsonl")
PY
```

---

### Phase 6: Final Assembly

#### Step 6a: Build final dataset (CPU, instant)

```bash
python3 - <<'PY'
import os, json
from collections import Counter

base = os.path.expanduser("~/staging/dataset_gen_output")

final = []
stress = []

with open(f"{base}/causal_postprocessed_all.jsonl") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)

        if r.get("label") == 1 and r.get("training_eligible", True):
            final.append(r)
        elif r.get("label") == 0:
            if r.get("validation_status") == "validated" and r.get("training_eligible", False):
                r["benign_status"] = r.get("benign_status", "validated_benign_twin")
                r["use_as"] = "hard_benign_supplement"
                final.append(r)
            else:
                r["use_as"] = "old_benign_boundary_or_excluded"
                stress.append(r)

with open(f"{base}/benign_clean.jsonl") as f:
    for line in f:
        if line.strip():
            final.append(json.loads(line))

with open(f"{base}/final_dataset.jsonl", "w") as f:
    for r in final:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(f"{base}/old_benign_boundary_or_unused.jsonl", "w") as f:
    for r in stress:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Final dataset: {len(final)} records")
print(f"Stress/excluded: {len(stress)} records")
print(f"\nLabels: {Counter(r.get('label') for r in final)}")
print(f"Supervision tiers: {Counter(r.get('supervision_tier', '?') for r in final)}")
print(f"Benign status: {Counter(r.get('benign_status', 'none') for r in final if r.get('label') == 0)}")
print(f"Transfer tiers: {Counter(r.get('transfer_tier', '?') for r in final if r.get('label') == 1)}")
PY
```

#### Step 6b: Split (CPU, instant)

```bash
python3 split_dataset.py \
    --input $HOME/staging/dataset_gen_output/final_dataset.jsonl \
    --output-dir $HOME/staging/dataset_gen_output/splits/ \
    --human-benchmark 100 \
    --double-annotated 50
```

Output:
```
splits/
  train.jsonl                   # ~70%
  dev.jsonl                     # ~15%
  test.jsonl                    # ~15% + external test
  human_benchmark.jsonl         # 100 records for annotation
  human_benchmark_double.jsonl  # 50 records for IAA
  human_benchmark_ids.json
  split_metadata.json
```

---

## Supervision Tier Reference

| Tier | Weight | Meaning | Use |
|---|---|---|---|
| `cf_strong` | 1.00 | CF delta >= 0.40 | Highest-confidence token attribution |
| `benign_validated` | 1.00 | Clean benign accepted by both validators | Main negative class |
| `hard_benign` | 0.80 | Validated benign twin near adversarial context | Hard negative training |
| `cf_weak` | 0.70 | CF delta 0.15-0.40 | High-confidence token attribution |
| `llm_confirmed` | 0.60 | LLM-detected spans + transfer/distributed evidence | Medium-confidence supervision |
| `construction` | 0.40 | Generator-placed spans, not causally confirmed | Weak supervision |
| `llm_only` | 0.25 | LLM judge only, no construction confirmation | Low-confidence |
| `incidental` | 0.00 | Confirmed non-causal | Negative token supervision |
| `boundary_benign` | 0.00 | Benign rejected by at least one validator | Stress test only |
| `ignore` | 0.00 | Ambiguous/unvalidated | Excluded from loss |

**Training loss usage:**

| Signal | Use |
|---|---|
| Sample `supervision_tier` + `loss_weight` | Classification / sample-level loss |
| Span `supervision_tier` + `causal_type` | Token attribution loss |
| Span `causal_type == "incidental"` | Negative attribution labels |

---

## Troubleshooting

### vLLM role alternation errors
Fixed in `causal_analysis.py`. Delete stale checkpoints and re-run.

### Stale checkpoints cause Pass 1 to skip
```bash
rm -f $HOME/staging/dataset_gen_output/*checkpoint*
```

### HuggingFace auth errors on compute nodes
Set `HF_HUB_OFFLINE=1` in all SLURM submissions.

### Low jailbreak rate
Check generator model, attack strategies, adaptation count.

### BENIGN_CONTEXT as largest causal label
Fixed in `postprocess_causal.py` — relabels to CONTEXT_BRIDGE.

---

## Quick Reference

```bash
# Phase 1: Adversarial generation
N_PAIRS=600 N_PIPELINES=2 sbatch launch_gen_interactive.slurm
N_PAIRS=200 GEN_MODEL=Qwen/Qwen2.5-14B-Instruct N_PIPELINES=1 \
  OUTPUT_NAME=semantic_multiturn_v11_interactive_14b sbatch launch_gen_interactive.slurm

# Phase 2: Mistral validation + merge
VAL_MODEL=mistralai/Mistral-7B-Instruct-v0.3 INPUT_FILE=...interactive_raw.jsonl \
  OUTPUT_NAME=...interactive N_VAL_SHARDS=2 HF_HUB_OFFLINE=1 sbatch launch_val.slurm
python3 merge_validations.py --interactive ...validated.jsonl --output ...merged.jsonl

# Phase 3: Causal analysis + postprocess
rm -f .../*checkpoint*
INPUT_FILE=...combined_dedup.jsonl OUTPUT_NAME=causal_analyzed_all_final \
  HF_HUB_OFFLINE=1 sbatch launch_causal.slurm
python3 postprocess_causal.py --input ...causal_analyzed_all_final.jsonl \
  --output ...causal_postprocessed_all.jsonl

# Phase 4: Benign generation
N_RECORDS=1000 sbatch launch_benign.slurm

# Phase 5: Benign validation (both models) + filter
# (see Step 5a-5c above)

# Phase 6: Assemble + split
# (see Step 6a-6b above)
```
