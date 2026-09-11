#!/usr/bin/env python3
"""Offline environment/cache preflight for frontier GuardLens experiments.

The cluster is offline. Fail before starting vLLM servers when required model
snapshots, model artifacts, or minimum library versions are absent. Version floors
mirror the model-family requirements used by the locked Qwen/Mistral/Gemma
configuration.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import re
import sys
from typing import List, Optional, Tuple


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


def snapshot_dirs(path: str) -> List[str]:
    root = os.path.join(path, "snapshots")
    if not os.path.isdir(root):
        return []
    return [
        os.path.join(root, name)
        for name in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, name))
    ]


def usable_file(path: str) -> bool:
    """Require a resolved, non-empty file; broken HF-cache symlinks fail."""
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def hf_weights_complete(snapshot: str) -> Tuple[bool, str]:
    """Validate unsharded or indexed Hugging Face weight material."""
    index_path = os.path.join(snapshot, "model.safetensors.index.json")
    if usable_file(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            files = sorted(set((payload.get("weight_map") or {}).values()))
        except Exception as exc:
            return False, f"invalid model.safetensors.index.json: {exc}"
        if not files:
            return False, "model.safetensors.index.json contains no weight files"
        missing = [name for name in files if not usable_file(os.path.join(snapshot, name))]
        if missing:
            return False, f"missing {len(missing)} indexed weight shard(s), e.g. {missing[:3]}"
        return True, f"{len(files)} indexed safetensor shard(s)"

    single = os.path.join(snapshot, "model.safetensors")
    if usable_file(single):
        return True, "model.safetensors"
    return False, "no complete HF safetensor weights found"


def snapshot_ready(snapshot: str, model_id: str) -> Tuple[bool, str]:
    lowered = model_id.lower()

    if "mistral-small-3.1" in lowered:
        required = ["consolidated.safetensors", "params.json", "tekken.json"]
        missing = [name for name in required if not usable_file(os.path.join(snapshot, name))]
        if missing:
            return False, f"Mistral-format snapshot missing {missing}"
        return True, "Mistral-format consolidated weights + params + Tekken tokenizer"

    if not usable_file(os.path.join(snapshot, "config.json")):
        return False, "config.json missing/unresolved"
    tokenizer_ok = any(
        usable_file(os.path.join(snapshot, name))
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    )
    if not tokenizer_ok:
        return False, "tokenizer artifact missing/unresolved"
    weights_ok, detail = hf_weights_complete(snapshot)
    if not weights_ok:
        return False, detail
    return True, detail


def cache_snapshot_status(path: str, model_id: str) -> Tuple[bool, str, Optional[str]]:
    snapshots = snapshot_dirs(path)
    if not snapshots:
        return False, "no snapshot directories", None
    failures = []
    for snapshot in snapshots:
        ready, detail = snapshot_ready(snapshot, model_id)
        if ready:
            return True, detail, snapshot
        failures.append(f"{os.path.basename(snapshot)}: {detail}")
    return False, "; ".join(failures[:3]), None


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
        ready, detail, snapshot = cache_snapshot_status(path, model_id)
        print(
            f"Model cache {model_id}: {'READY' if ready else 'MISSING/INCOMPLETE'} "
            f"-> {path}"
        )
        print(f"  artifact check: {detail}")
        if snapshot:
            print(f"  snapshot: {snapshot}")
        if not ready:
            errors.append(f"model cache missing/incomplete: {model_id}: {detail}")

        lowered = model_id.lower()
        if "qwen2.5" in lowered:
            require_version(errors, "transformers", transformers_version, "4.37.0")
        if "mistral-small-3.1" in lowered:
            require_version(errors, "vLLM", vllm_version, "0.8.1")
            require_version(errors, "mistral-common", mistral_common_version, "1.5.4")
        if "gemma-3" in lowered:
            require_version(errors, "vLLM", vllm_version, "0.8.3")
            require_version(errors, "transformers", transformers_version, "4.50.0")

    if errors:
        print("ENVIRONMENT PREFLIGHT FAILED", file=sys.stderr)
        for error in sorted(set(errors)):
            print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
    print("ENVIRONMENT PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
