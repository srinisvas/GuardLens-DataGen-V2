"""
prestage_v12.py

Pre-download all models used by the v12 benchmark to the HuggingFace cache
so the compute nodes (which may not have HF access) can find them offline.

v2 corrections (post-review):
  - Removed resume_download=True: huggingface_hub v1.0 removed the argument
    and vLLM 0.28 requires hf_hub>=1.27. Downloads now resume automatically.
  - Added pre-run disk-space check. Full BF16 v12 stack requires ~313 GB
    (Llama-3.3-70B ~140G, Qwen3.5-27B ~54G, Gemma-4-31B ~62G,
    Mistral-Small-3.2 ~55G, bge-m3 ~2G). Warns/aborts if the cache
    filesystem has less than 350 GB free.

Reads model IDs from model_bench_configs.yaml, deduplicates, adds bge-m3
for contamination checks, calls snapshot_download for each.

After this succeeds, set HF_HUB_OFFLINE=1 in the SLURM scripts so any
missing model fails fast rather than hanging on compute nodes.

Usage:
    export HF_TOKEN=$(cat $HOME/.hf_token)
    export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
    export HF_HOME=$HOME/work/hf_models
    export HF_HUB_CACHE=$HOME/work/hf_models/hub

    python prestage_v12.py --config-file model_bench_configs.yaml

Prerequisites (ONCE from the HF website while logged in):
    - Accept license for meta-llama/Llama-3.3-70B-Instruct
    - Accept license for google/gemma-4-31B-it
    - Accept license for mistralai/Mistral-Small-3.2-24B-Instruct-2506
    (Qwen and BAAI models are not gated.)
"""

import argparse
import os
import shutil
import sys
import time
from typing import Set

import yaml


# Extra models not in the YAML (e.g. contamination-check embedder)
EXTRA_MODELS = [
    "BAAI/bge-m3",
]

# Rough BF16 disk footprint per model (GB). Used only for pre-flight sanity
# check. Actual values may vary ±10%.
MODEL_DISK_GB = {
    "meta-llama/Llama-3.3-70B-Instruct":                140,
    "Qwen/Qwen3.5-27B":                                  54,
    "google/gemma-4-31B-it":                             62,
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506":     55,
    "BAAI/bge-m3":                                        2,
}

MIN_FREE_GB = 350   # 313 GB models + ~40 GB temp/metadata headroom


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


def check_disk_space(cache_dir: str, required_gb: int) -> bool:
    """Verify the filesystem hosting cache_dir has at least required_gb free."""
    # Walk up to a directory that actually exists (for statvfs)
    probe = cache_dir
    while probe and not os.path.exists(probe):
        probe = os.path.dirname(probe)
    if not probe:
        print(f"WARN: cannot resolve any parent of {cache_dir} for disk check")
        return True
    stats = shutil.disk_usage(probe)
    free_gb = stats.free / (1024 ** 3)
    total_gb = stats.total / (1024 ** 3)
    print(f"Disk check on {probe}:")
    print(f"  free:  {free_gb:.1f} GB")
    print(f"  total: {total_gb:.1f} GB")
    print(f"  need:  {required_gb} GB (approximate)")
    if free_gb < required_gb:
        print(f"\nERROR: insufficient free space. Need at least {required_gb} GB, "
              f"have {free_gb:.1f} GB.", file=sys.stderr)
        print(f"Options:\n"
              f"  - Free space in {probe}\n"
              f"  - Point HF_HUB_CACHE to a bigger filesystem\n"
              f"  - Prestage a subset via --only-models",
              file=sys.stderr)
        return False
    return True


def stage_one(model_id: str, cache_dir: str) -> bool:
    from huggingface_hub import snapshot_download
    print(f"\n{'=' * 60}")
    print(f"  Staging: {model_id}")
    print(f"{'=' * 60}")
    t0 = time.time()
    try:
        # NOTE: resume_download removed in hf_hub 1.0. Auto-resume is the
        # default now.
        local_dir = snapshot_download(repo_id=model_id, cache_dir=cache_dir)
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
    p.add_argument("--only-models", nargs="*", default=None,
                   help="Restrict to this subset of model IDs "
                        "(useful if partial staging is needed)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-disk-check", action="store_true")
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

    if args.only_models:
        ids = {m for m in ids if m in set(args.only_models)}

    print(f"\nModels to stage ({len(ids)}):")
    approx_total_gb = 0
    for m in sorted(ids):
        est = MODEL_DISK_GB.get(m, 0)
        approx_total_gb += est
        est_str = f"~{est} GB" if est else "size unknown"
        print(f"  {m}  ({est_str})")
    print(f"  approximate total: ~{approx_total_gb} GB")

    if not args.skip_disk_check:
        required = max(approx_total_gb + 40, MIN_FREE_GB)
        if not check_disk_space(cache_dir, required):
            sys.exit(1)

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
        print(f"  - Model gated behind license (accept on HF website)",
              file=sys.stderr)
        print(f"  - HF_TOKEN not set or expired", file=sys.stderr)
        print(f"  - Model ID typo (check model_bench_configs.yaml)",
              file=sys.stderr)
        print(f"  - Disk filled during download (rerun after freeing space)",
              file=sys.stderr)
        sys.exit(1)

    print(f"\nAll models staged. Safe to set HF_HUB_OFFLINE=1 in SLURM scripts.")


if __name__ == "__main__":
    main()