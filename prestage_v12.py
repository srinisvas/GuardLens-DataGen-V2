"""
prestage_v12.py

Pre-download all models used by the v12 benchmark to the HuggingFace cache
so the compute nodes (which may not have HF access) can find them offline.

Reads model IDs from model_bench_configs.yaml, deduplicates, adds bge-m3
for the contamination check, and calls snapshot_download for each.

Set HF_HUB_OFFLINE=1 in the SLURM scripts after this succeeds so that any
missing model fails fast rather than trying to reach hf.co from a compute
node with no external network access.

Usage:
    # Requires HF_TOKEN in $HOME/.hf_token (needed for gated Meta/Google models).
    export HF_TOKEN=$(cat $HOME/.hf_token)
    export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
    export HF_HOME=$HOME/work/hf_models
    export HF_HUB_CACHE=$HOME/work/hf_models/hub

    python prestage_v12.py --config-file model_bench_configs.yaml

    # After this succeeds, the SLURM jobs will have every model on-disk
    # and can run with HF_HUB_OFFLINE=1.

Prerequisites (do this ONCE from the HF website while logged in):
    - Accept license for meta-llama/Llama-3.3-70B-Instruct
    - Accept license for google/gemma-4-31B-it
    - Accept license for mistralai/Mistral-Small-3.2-24B-Instruct-2506
    (Qwen and BAAI models are not gated.)
"""

import argparse
import os
import sys
import time
from typing import List, Set

import yaml


# Extra models not in the YAML (e.g. contamination-check embedder)
EXTRA_MODELS = [
    "BAAI/bge-m3",
]


def extract_model_ids(config_file: str) -> Set[str]:
    with open(config_file) as f:
        data = yaml.safe_load(f)
    ids = set()
    for conf in data.get("configurations", []):
        for role in ("generator", "target", "judge"):
            entry = conf.get(role) or {}
            model = entry.get("model")
            if model:
                ids.add(model)
    return ids


def stage_one(model_id: str, cache_dir: str) -> bool:
    from huggingface_hub import snapshot_download
    print(f"\n{'=' * 60}")
    print(f"  Staging: {model_id}")
    print(f"{'=' * 60}")
    t0 = time.time()
    try:
        local_dir = snapshot_download(
            repo_id=model_id, cache_dir=cache_dir,
            resume_download=True,
        )
        dt = time.time() - t0
        print(f"  DONE in {dt:.0f}s → {local_dir}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}", file=sys.stderr)
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", required=True)
    p.add_argument("--extra", nargs="*", default=[],
                   help="Additional model IDs to stage beyond the config file")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cache_dir = os.environ.get(
        "HF_HUB_CACHE",
        os.path.expanduser("~/work/hf_models/hub"),
    )
    print(f"HF cache directory: {cache_dir}")
    if "HF_TOKEN" not in os.environ:
        print("WARN: HF_TOKEN not set. Gated models (Meta, Google) will fail.",
              file=sys.stderr)

    ids = extract_model_ids(os.path.expanduser(args.config_file))
    for extra in EXTRA_MODELS + args.extra:
        ids.add(extra)

    print(f"\nModels to stage ({len(ids)}):")
    for m in sorted(ids):
        print(f"  {m}")

    if args.dry_run:
        print("\nDry run — no downloads performed.")
        return

    print(f"\nStarting downloads. First-time downloads of 70B / 31B models "
          f"can take 20-40 minutes each depending on bandwidth.")

    results = {}
    for m in sorted(ids):
        results[m] = stage_one(m, cache_dir)

    print(f"\n{'=' * 60}")
    print(f"  Staging summary")
    print(f"{'=' * 60}")
    n_ok = sum(1 for v in results.values() if v)
    n_fail = len(results) - n_ok
    for m, ok in sorted(results.items()):
        marker = "OK" if ok else "FAIL"
        print(f"  [{marker}] {m}")
    print(f"\n{n_ok} ok, {n_fail} failed (of {len(results)})")

    if n_fail > 0:
        print(f"\nSome models failed. Common causes:", file=sys.stderr)
        print(f"  - Model gated behind license (accept on HF website)", file=sys.stderr)
        print(f"  - HF_TOKEN not set or expired", file=sys.stderr)
        print(f"  - Model ID typo (check model_bench_configs.yaml)", file=sys.stderr)
        sys.exit(1)

    print(f"\nAll models staged. You can now safely set HF_HUB_OFFLINE=1 in "
          f"SLURM scripts.")


if __name__ == "__main__":
    main()