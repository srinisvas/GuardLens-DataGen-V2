#!/usr/bin/env python3
"""Stage B2 v5 under the frozen deterministic vLLM runtime contract."""
from __future__ import annotations

import validate_frontier_rollout_v5 as base
from frontier_runtime_determinism import assert_batch_invariant_env, runtime_provenance

_BASE_VALIDATION_CONFIG = base.validation_config


def validation_config(
    *,
    judge_model: str,
    base_seed: int,
    unsafe_threshold: float,
    ambiguous_threshold: float,
    min_confidence: float,
    judge_max_context_chars: int,
    judge_max_model_len: int,
):
    cfg = _BASE_VALIDATION_CONFIG(
        judge_model=judge_model,
        base_seed=base_seed,
        unsafe_threshold=unsafe_threshold,
        ambiguous_threshold=ambiguous_threshold,
        min_confidence=min_confidence,
        judge_max_context_chars=judge_max_context_chars,
        judge_max_model_len=judge_max_model_len,
    )
    cfg.update(runtime_provenance("judge"))
    return cfg


def main() -> None:
    assert_batch_invariant_env()
    base.validation_config = validation_config
    base.main()


if __name__ == "__main__":
    main()
