#!/usr/bin/env python3
"""Stage B1 rollout v3 with frozen deterministic vLLM runtime provenance."""
from __future__ import annotations

import rollout_frontier_source as base
from frontier_runtime_determinism import assert_batch_invariant_env, runtime_provenance

PROTOCOL = "frontier_fixed_user_rollout_v3"
_BASE_ROLLOUT_CONFIG = base.rollout_config


def rollout_config(*, model: str, base_seed: int, max_tokens: int, max_model_len: int):
    cfg = _BASE_ROLLOUT_CONFIG(
        model=model,
        base_seed=base_seed,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
    )
    cfg["protocol"] = PROTOCOL
    cfg.update(runtime_provenance())
    return cfg


def install() -> None:
    base.PROTOCOL = PROTOCOL
    base.rollout_config = rollout_config


def main() -> None:
    assert_batch_invariant_env()
    install()
    base.main()


if __name__ == "__main__":
    main()
