#!/usr/bin/env python3
"""Prestage Mistral-7B-Instruct-v0.3 to HPC model cache."""
import argparse, os, sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/work/hf_models"))
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(args.cache_dir, "hub")

    print(f"Model:     {args.model}")
    print(f"Cache dir: {args.cache_dir}")

    token = os.environ.get("HF_TOKEN", None)
    if not token:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_path):
            print(f"Using cached token from {token_path}")

    print(f"\nDownloading {args.model}...")
    from huggingface_hub import snapshot_download
    path = snapshot_download(args.model, cache_dir=os.path.join(args.cache_dir, "hub"), token=token)
    print(f"Done: {path}")

    import subprocess
    r = subprocess.run(["du", "-sh", path], capture_output=True, text=True)
    if r.returncode == 0:
        print(f"Size: {r.stdout.strip().split()[0]}")

if __name__ == "__main__":
    main()