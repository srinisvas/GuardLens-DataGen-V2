# GuardLens Frontier Smoke Run

This file previously documented the original single-author 1,500-record smoke.
That path is retired for the NAACL validity-repair experiment.

The canonical source is now the merged 3,000-record GPT-5.6 + Astra corpus, and
the authoritative execution instructions are in:

```text
naacl/FRONTIER_MULTI_AUTHOR.md
```

Do not use the old 320-token or 640-token smoke outputs for validation/evidence.
The reviewed target envelope is:

```text
Qwen/Qwen2.5-32B-Instruct
max_tokens      = 1280
max_model_len   = 16384
temperature     = 0.0
finish_reason   = stop required
```

The deterministic 20-record smoke source remains:

```text
$HOME/staging/dataset_gen_output/frontier_multi_author_smoke20.jsonl
```

The reviewed B1 smoke command is:

```bash
export OUT=$HOME/staging/dataset_gen_output

N_SHARDS=1 \
MAX_TOKENS=1280 \
MAX_MODEL_LEN=16384 \
INPUT_FILE=$OUT/frontier_multi_author_smoke20.jsonl \
OUTPUT_FILE=$OUT/frontier_multi_author_smoke_qwen32_rollout_v2_1280_16k.jsonl \
CHECKPOINT_PREFIX=$OUT/frontier_multi_author_smoke_qwen32_rollout_v2_1280_16k \
SOURCE_PREFLIGHT_MODE=schema \
sbatch --gres=gpu:1 naacl/launch_frontier_rollout.slurm
```

Do not proceed to Mistral validation unless the rollout audit passes all 20
records with natural-stop completions. Continue B2–B5 only using the commands in
`FRONTIER_MULTI_AUTHOR.md`.
