#!/usr/bin/env python3
"""Frozen deterministic inference-runtime contract for frontier Dataset B.

GuardLens B4 compares factual and counterfactual target behavior. Accidental
batch/scheduler numerical variance must therefore not enter the intervention
delta. Production B1/B2/B4 servers use vLLM batch invariance plus eager execution
on A100 (SM80) hardware.
"""
from __future__ import annotations

import os

RUNTIME_DETERMINISM = "vllm_batch_invariant_eager_v1"
VLLM_BATCH_INVARIANT_ENV = "VLLM_BATCH_INVARIANT"
VLLM_BATCH_INVARIANT_VALUE = "1"
ENFORCE_EAGER = True


def runtime_provenance(prefix: str = "") -> dict:
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}runtime_determinism": RUNTIME_DETERMINISM,
        f"{p}vllm_batch_invariant": True,
        f"{p}vllm_enforce_eager": True,
    }


def assert_batch_invariant_env() -> None:
    value = os.environ.get(VLLM_BATCH_INVARIANT_ENV)
    if value != VLLM_BATCH_INVARIANT_VALUE:
        raise RuntimeError(
            f"{VLLM_BATCH_INVARIANT_ENV}={value!r}; production deterministic "
            f"runtime requires {VLLM_BATCH_INVARIANT_ENV}=1"
        )


def assert_runtime_fields(mapping: dict, prefix: str = "") -> None:
    expected = runtime_provenance(prefix)
    for key, value in expected.items():
        if mapping.get(key) != value:
            raise RuntimeError(
                f"runtime provenance {key}={mapping.get(key)!r} != expected {value!r}"
            )
