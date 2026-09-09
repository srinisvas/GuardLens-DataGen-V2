#!/usr/bin/env python3
"""Offline environment/cache preflight for frontier GuardLens experiments.

The cluster is offline. Fail before starting vLLM servers when required model
snapshots or minimum library versions are absent. Version floors mirror the
model-family requirements used by the locked Qwen/Mistral/Gemma configuration.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import re
import sys
from typing import Optional, Tuple


def version_tuple(value: str) -> Tuple[int, ...]:
    numbers = re.findall(r"\d+", value.split("+")[0])
    return tuple(int(x) for x in numbers[:4]) if numbers else (0,)


def installed_version(*names: str) -> Optional[str]:
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def require_version(errors, package_label: str, found: Optional[str], minimum: str) -> None:
    if found is None:
        errors.append(f"{package_label} is not installed; require >= {minimum}")
        return
    if version_tuple(found) < version_tuple(minimum):
        errors.append(f"{package_label}={found} is too old; require >= {minimum}")


def cache_dir(model_cache: str, model_id: str) -> str:
    return os.path.join(model_cache, "hub", "models--" + model_id.replace("/", "--"))


def cache_has_snapshot(path: str) -> bool:
    snapshots = os.path.join(path, "snapshots")
    if not os.path.isdir(snapshots):
        return False
    for name in os.listdir(snapshots):
        candidate = os.path.join(snapshots, name)
        if os.path.isdir(candidate) and os.listdir(candidate):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()

    errors = []
    vllm_version = installed_version("vllm")
    transformers_version = installed_version("transformers")
    mistral_common_version = installed_version("mistral-common", "mistral_common")

    print("=== Frontier environment preflight ===")
    print(f"vLLM: {vllm_version or 'NOT INSTALLED'}")
    print(f"transformers: {transformers_version or 'NOT INSTALLED'}")
    print(f"mistral-common: {mistral_common_version or 'NOT INSTALLED'}")

    if vllm_version is None:
        errors.append("vLLM is not installed")
    if transformers_version is None:
        errors.append("transformers is not installed")

    for model_id in args.models:
        path = cache_dir(args.model_cache, model_id)
        ready = cache_has_snapshot(path)
        print(f"Model cache {model_id}: {'READY' if ready else 'MISSING/INCOMPLETE'} -> {path}")
        if not ready:
            errors.append(f"model cache missing/incomplete: {model_id}")

        lowered = model_id.lower()
        if "qwen2.5" in lowered:
            require_version(errors, "transformers", transformers_version, "4.37.0")
        if "mistral-small-3.1" in lowered:
            require_version(errors, "vLLM", vllm_version, "0.8.1")
            require_version(errors, "mistral-common", mistral_common_version, "1.5.4")
        if "gemma-3" in lowered:
            require_version(errors, "transformers", transformers_version, "4.50.0")

    if errors:
        print("ENVIRONMENT PREFLIGHT FAILED", file=sys.stderr)
        for error in sorted(set(errors)):
            print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
    print("ENVIRONMENT PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
