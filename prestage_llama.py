#!/usr/bin/env python3
"""
prestage_llama.py

Download meta-llama/Llama-3.1-8B-Instruct to the HPC model cache.
Run on the login node (not in a SLURM job — login nodes have internet).

Prerequisites:
  - huggingface_hub installed (pip install huggingface_hub)
  - HuggingFace account with Llama access approved
  - hf login (or HF_TOKEN env var set)

Usage:
  python3 prestage_llama.py

  # Custom cache dir:
  python3 prestage_llama.py --cache-dir /path/to/hf_models

  # If you need to authenticate:
  HF_TOKEN=hf_xxxx python3 prestage_llama.py
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Pre-stage Llama-3.0-8B-Instruct")
    parser.add_argument("--cache-dir", type=str,
                        default=os.path.expanduser("~/work/hf_models"),
                        help="HuggingFace cache directory")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-3.0-8B-Instruct",
                        help="Model to download")
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(args.cache_dir, "hub")

    print(f"Model:     {args.model}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"HF_HOME:   {os.environ['HF_HOME']}")
    print()

    # Check authentication
    token = os.environ.get("HF_TOKEN", None)
    if token:
        print("Using HF_TOKEN from environment.")
    else:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_path):
            print(f"Using cached token from {token_path}")
        else:
            print("WARNING: No HF token found.")
            print("  Llama models require Meta's license approval + HF login.")
            print("  Run: hf login")
            print("  Or set: export HF_TOKEN=hf_xxxx")
            print()
            resp = input("Continue anyway? [y/N] ").strip().lower()
            if resp != "y":
                sys.exit(1)

    print()
    print(f"Downloading {args.model}...")
    print("  This will take ~15-20 min on a fast connection (~16GB).")
    print()

    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            args.model,
            cache_dir=os.path.join(args.cache_dir, "hub"),
            token=token,
        )

        print()
        print(f"Download complete: {path}")

        # Verify
        import subprocess
        result = subprocess.run(
            ["du", "-sh", path],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"Size: {result.stdout.strip().split()[0]}")

        # Check model files exist
        expected = ["config.json", "tokenizer.json"]
        missing = [f for f in expected if not os.path.exists(os.path.join(path, f))]
        if missing:
            print(f"WARNING: Missing expected files: {missing}")
        else:
            print("Verified: config.json and tokenizer.json present.")

        print()
        print("Ready. Use in validation with:")
        print(f"  VAL_MODEL={args.model} sbatch launch_val.slurm")

    except Exception as e:
        print(f"ERROR: {e}")
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            print()
            print("This is likely an authentication/access issue.")
            print("Steps:")
            print("  1. Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
            print("  2. Accept Meta's license agreement")
            print("  3. Run: hf login")
            print("  4. Re-run this script")
        sys.exit(1)


if __name__ == "__main__":
    main()
