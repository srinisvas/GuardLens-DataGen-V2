# GuardLens NAACL Frontier-Authored Dataset Pipeline

This document previously described the original 1,500-record single-author
Dataset B pipeline. That execution path is retired for the current NAACL
validity-repair experiment.

Use the canonical multi-author runbook instead:

```text
naacl/FRONTIER_MULTI_AUTHOR.md
```

## Current locked design

Stage 0 is the merged 3,000-record GPT-5.6 + Astra authored source corpus.

Model roles:

- primary target: `Qwen/Qwen2.5-32B-Instruct`
- independent judge: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`
- cross-target robustness: `google/gemma-3-27b-it`

Target generation envelope:

- `max_tokens=1280`
- `max_model_len=16384`
- `temperature=0.0`
- deterministic per-record/per-turn seeds
- every assistant turn must end with `finish_reason=stop`
- positive `completion_tokens` instrumentation is mandatory

Judge envelope:

- runtime context `16384`
- full observable-prefix budget `40000` characters
- no suffix-only context truncation

Primary/stress policy:

- paired records are the only primary Dataset B candidates
- both twins must pass their respective behavioral/evidence gates
- no orphan twin enters primary training
- standalone hard benign records remain evaluation-only stress records
- authoring intent and intended pivots are candidate-generation metadata, never
  scientific ground truth

The reviewed protocol identifiers remain:

```text
B1  frontier_fixed_user_rollout_v2
B2  frontier_context_judge_v3
B4  frontier_context_paired_counterfactual_v4
```

The token/context envelope is part of the recorded B1 configuration fingerprint
and is rechecked by rollout audit, validation/evidence launchers, B5 preparation,
post-B5 audits, and the Dataset A+B merge boundary. Old 320/640-token or 8K-
context artifacts are not valid inputs to the current pipeline.

For source merge, deterministic 20-record smoke, full B1–B5 commands, Dataset A+B
merge, final split, and Transformer handoff, follow `FRONTIER_MULTI_AUTHOR.md`.
