#!/usr/bin/env python3
"""GuardLens v12 runtime compatibility preflight.

This script makes the pinned vLLM 0.28.0 environment usable on the KSU HPC
Python 3.11 nodes without requiring a system CUDA toolkit/nvcc.

It performs four checks/fixes before any model weights are loaded:

1. Verifies sqlite, CUDA visibility, vLLM version, and expected GPU count.
2. Applies the exact upstream FlashInfer 0.6.17 Python <=3.11 compatibility
   fix to the pinned FlashInfer package when needed. FlashInfer 0.6.16 uses
   ``array.array[int]`` in ``comm/fd_exchange.py``; on Python 3.11 that
   annotation is evaluated eagerly and crashes tensor-parallel startup.
   Upstream fixed it by adding ``from __future__ import annotations``.
3. Verifies ``flashinfer.comm`` imports after the compatibility fix.
4. Verifies the benchmark explicitly disabled FlashInfer sampling. The native
   vLLM sampler does not JIT-compile FlashInfer CUDA kernels and therefore does
   not require an external nvcc/CUDA_HOME on the compute node.

The patch is idempotent and only modifies a file that contains the known
problematic ``array.array[int]`` annotation. It prints package versions and the
before/after file hash so the runtime modification is auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sqlite3
import sys


_FUTURE_IMPORT = "from __future__ import annotations"
_PROBLEMATIC_ANNOTATION = "array.array[int]"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def patch_flashinfer_python311() -> tuple[Path | None, bool]:
    """Apply FlashInfer's upstream postponed-annotations fix when required."""
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("flashinfer package is not installed")

    root = Path(next(iter(spec.submodule_search_locations)))
    path = root / "comm" / "fd_exchange.py"
    if not path.is_file():
        raise RuntimeError(f"FlashInfer compatibility file not found: {path}")

    text = path.read_text(encoding="utf-8")
    before = _sha256(path)

    # Python 3.12+ supports array.array[int], so no source patch is necessary.
    if sys.version_info >= (3, 12):
        print(f"FlashInfer fd_exchange: Python {sys.version.split()[0]} needs no patch")
        print(f"FlashInfer fd_exchange sha256: {before}")
        return path, False

    if _FUTURE_IMPORT in text:
        print("FlashInfer fd_exchange: postponed annotations already present")
        print(f"FlashInfer fd_exchange sha256: {before}")
        return path, False

    if _PROBLEMATIC_ANNOTATION not in text:
        # A future/backported build may have fixed the annotation another way.
        print("FlashInfer fd_exchange: known Python 3.11 annotation bug not present")
        print(f"FlashInfer fd_exchange sha256: {before}")
        return path, False

    needle = "\nimport array\n"
    if needle not in text:
        raise RuntimeError(
            "Known FlashInfer annotation bug is present but expected 'import array' "
            "location was not found; refusing to patch an unexpected source layout"
        )

    # This is the exact semantic fix used upstream in FlashInfer 0.6.17.
    text = text.replace(
        needle,
        "\nfrom __future__ import annotations\n\nimport array\n",
        1,
    )
    path.write_text(text, encoding="utf-8")
    after = _sha256(path)

    print("FlashInfer fd_exchange: applied Python <=3.11 postponed-annotations patch")
    print(f"FlashInfer fd_exchange sha256 before: {before}")
    print(f"FlashInfer fd_exchange sha256 after:  {after}")
    return path, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--required-gpus",
        type=int,
        default=1,
        help="Minimum number of CUDA devices expected inside this Slurm allocation",
    )
    args = parser.parse_args()

    print("=== GuardLens v12 runtime compatibility preflight ===")
    print("Python:", sys.version.split()[0])
    print("sqlite:", sqlite3.sqlite_version)
    print("vLLM package:", _package_version("vllm"))
    print("flashinfer-python:", _package_version("flashinfer-python"))
    print("flashinfer-cubin:", _package_version("flashinfer-cubin"))

    # Keep the scientific benchmark independent of FlashInfer's JIT sampler.
    sampler_setting = os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")
    if sampler_setting != "0":
        raise RuntimeError(
            "VLLM_USE_FLASHINFER_SAMPLER must be explicitly set to 0 before "
            "runtime_compat_v12.py is called"
        )
    print("VLLM_USE_FLASHINFER_SAMPLER=0 (native sampler locked)")

    # Explicitly keep experimental FlashInfer all-reduce disabled. vLLM 0.28
    # defaults it off, but pinning the value prevents a future shell/default
    # change from altering the benchmark backend.
    fi_ar = os.environ.get("VLLM_ALLREDUCE_USE_FLASHINFER")
    if fi_ar != "0":
        raise RuntimeError(
            "VLLM_ALLREDUCE_USE_FLASHINFER must be explicitly set to 0"
        )
    print("VLLM_ALLREDUCE_USE_FLASHINFER=0")

    patch_flashinfer_python311()

    # Import the exact module path that broke the Llama TP=2 worker. Do this
    # after patching and before allocating/loading model weights.
    importlib.import_module("flashinfer.comm")
    print("flashinfer.comm import: OK")

    import torch
    import vllm
    from vllm.platforms import current_platform

    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    print("vLLM runtime:", vllm.__version__)
    print("CUDA available:", cuda_available)
    print("GPU count:", gpu_count)
    print("vLLM device:", current_platform.device_type)
    if cuda_available and gpu_count:
        print("GPU 0:", torch.cuda.get_device_name(0))

    if not cuda_available:
        raise RuntimeError("CUDA is not visible inside this Slurm job")
    if gpu_count < args.required_gpus:
        raise RuntimeError(
            f"Expected at least {args.required_gpus} GPUs, but only {gpu_count} are visible"
        )
    if current_platform.device_type != "cuda":
        raise RuntimeError(
            f"vLLM detected device={current_platform.device_type!r}, expected 'cuda'"
        )

    print("Runtime compatibility preflight: PASS")


if __name__ == "__main__":
    main()
