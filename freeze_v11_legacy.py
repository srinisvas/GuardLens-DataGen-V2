"""
freeze_v11_legacy.py

v2 corrections (post-review):
  1. README no longer hard-codes a single generator model. v11 used a POOL
     (Qwen2.5-7B-Instruct + Qwen2.5-14B-Instruct, evidence: merged_7b.jsonl,
     merged_14b.jsonl in the artifact directory). README describes pools
     and points to actual manifests instead of guessing.
  2. Refuses to commit a zero-file freeze. Prior version happily created
     an empty legacy directory when --source pointed to a subdirectory
     with only nested folders and no files at the top level.
  3. Auto-detects manifest.json / dataset_manifest.json in the source and
     echoes its contents into the README so provenance comes from the
     actual pipeline outputs rather than my assumptions.
  4. Categorizes model-specific merge files (merged_7b.jsonl, merged_14b.jsonl,
     etc.) as such so the README makes the pool structure visible.

Safety:
  - Dry-run by default. Pass --commit to actually copy.
  - Refuses --commit if source has zero files.
  - Refuses if dest exists (unless --overwrite).
  - Never deletes source files. Move semantics are opt-in via --move.

Usage:
    python freeze_v11_legacy.py --source ./results-gen --dest ~/work/results
    python freeze_v11_legacy.py --source ./results-gen --dest ~/work/results --commit
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Optional


LEGACY_DIR_NAME = "dataset_v11_legacy"

MANIFEST_CANDIDATES = ["manifest.json", "dataset_manifest.json",
                        "config.json", "generation_config.json"]

V11_KNOWN_ISSUES = [
    "Baseline-model mismatch in causal_analysis.py Pass 2: baseline scores "
    "were computed on Qwen-7B while the CF replay ran on Mistral. Reported "
    "Delta values are not interpretable as intervention effects.",

    "Assistant-turn contamination in replay_conversation(): stored assistant "
    "responses were concatenated into the CF trajectory via a merge bug, "
    "systematically biasing Delta toward zero.",

    "Position-dependent 12-message sliding window in replay: late-turn "
    "interventions had systematically higher measured Delta than early-turn "
    "interventions purely as a truncation artifact.",

    "Pivot-CF gate on span analysis: only records where the pivot-turn CF "
    "succeeded entered span-level analysis, self-fulfilling the single-pivot "
    "pattern in the dataset.",

    "Length mismatch between malicious (mean ~10 user turns) and benign "
    "(mean ~19 user turns): a linear probe on turn count would separate the "
    "two non-trivially, so length is a viable shortcut in v11.",

    "Generator/validator family coupling (both Qwen family): flagged by "
    "SAC review as circularity in dataset construction.",
]

README_TEMPLATE = """# dataset_v11_legacy

**Frozen:** {freeze_date}
**Original location:** `{source_path}`
**Files:** {n_files} files, {total_mb:.1f} MB total

## Purpose

This directory is a frozen snapshot of the v11 dataset generation pipeline.
It is retained solely for use as a **historical-distribution behavioral
evaluation set** in the NAACL 2027 paper. Its role is to test whether the
v12-trained model generalizes to attacks generated against an older model
regime. It is NOT a source of attribution ground truth.

## Original model configuration

**Generator pools (based on files present in the frozen artifact):**

{model_pool_evidence}

**Target family:** Meta-Llama-3-8B-Instruct

**Validation:** multiple downstream validation passes across pipeline stages.
See any `manifest.json` echoed below and the file listing under "Files by
category" for exact per-job model choices.

## Manifests detected in source

{manifest_dump}

## Known issues (disclosed in paper)

The v11 causal-analysis outputs in this directory are **invalid, not merely
diagnostic**. The following issues are documented for reproducibility and
are addressed in the v12 pipeline (`causal_analysis_v12_3.py`):

{known_issues_list}

## Intended use

**DO:**
- Report v11 as a historical-distribution eval set for behavioral
  detection.
- Use v11 user turn text as a contamination reference set for v12
  generation.
- Cite v11 methods in the paper's related-work discussion.

**DO NOT:**
- Use v11 records as training data for the v12 model.
- Report v11 causal_analysis Delta values, evidence sets, or span-level
  causal labels as scientific claims.
- Use v11 attribution outputs as ground truth for evaluating v12's
  attribution performance.
- Modify any file in this directory.

## Manifest

See `freeze_manifest.json` for SHA256 hashes. Verify with
`python verify_manifest.py`.

## Files by category

{files_by_category}
"""


def sha256_file(path: str, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def categorize_file(filename: str) -> str:
    lower = filename.lower()
    if "causal" in lower and ".jsonl" in lower:
        return "Causal analysis output (v11, KNOWN INVALID)"
    # Model-pool-specific merges get their own category so the README
    # makes the multi-model pool visible.
    if lower.startswith("merged_") and lower.endswith(".jsonl"):
        return "Merged per-model-pool output"
    if "merged" in lower and ".jsonl" in lower:
        return "Merged pipeline output"
    if "final_dataset" in lower and ".jsonl" in lower:
        return "Post-processed final dataset artifact"
    if "interactive" in lower and ".jsonl" in lower:
        return "Interactive attack generation output"
    if "benign" in lower and ".jsonl" in lower:
        return "Benign twin generation output"
    if "qwen" in lower or "validated" in lower or "validation" in lower:
        return "Cross-model validation output"
    if "mhj" in lower and ".jsonl" in lower:
        return "MHJ external test set in v11 schema"
    if "final" in lower and ".jsonl" in lower:
        return "Post-processed dataset artifact"
    if "manifest" in lower and ".json" in lower:
        return "Pipeline manifest / config"
    if lower.endswith(".jsonl"):
        return "JSONL data (unclassified)"
    if lower.endswith(".py"):
        return "Pipeline source code"
    if lower.endswith((".slurm", ".sh")):
        return "SLURM / shell script"
    if lower.endswith((".md", ".txt", ".log")):
        return "Documentation / log"
    if lower.endswith(".json"):
        return "Config / metadata"
    return "Other"


def enumerate_source(source: str) -> List[Dict]:
    out = []
    if not os.path.isdir(source):
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    for root, _, files in os.walk(source):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, source)
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            out.append({"relpath": rel, "abspath": full, "size_bytes": size,
                        "category": categorize_file(f)})
    return sorted(out, key=lambda x: x["relpath"])


def render_files_by_category(files: List[Dict]) -> str:
    grouped: Dict[str, List[Dict]] = {}
    for f in files:
        grouped.setdefault(f["category"], []).append(f)
    lines = []
    for cat in sorted(grouped.keys()):
        lines.append(f"### {cat}")
        lines.append("")
        for f in grouped[cat]:
            mb = f["size_bytes"] / (1024 * 1024)
            size_s = f"{mb:.2f} MB" if mb >= 0.1 else f"{f['size_bytes']} B"
            lines.append(f"- `{f['relpath']}` ({size_s})")
        lines.append("")
    return "\n".join(lines)


def render_known_issues(issues: List[str]) -> str:
    return "\n".join(f"{i + 1}. {issue}" for i, issue in enumerate(issues))


def render_model_pool_evidence(files: List[Dict]) -> str:
    """Look at file names to infer which generator variants contributed
    to the pool. If we see merged_7b.jsonl and merged_14b.jsonl, we say
    so instead of guessing a single model."""
    pool_hints = []
    for f in files:
        low = f["relpath"].lower()
        if "merged_7b" in low:
            pool_hints.append("Qwen2.5-7B-Instruct (evidence: merged_7b.jsonl)")
        if "merged_14b" in low:
            pool_hints.append("Qwen2.5-14B-Instruct (evidence: merged_14b.jsonl)")
        if "merged_70b" in low:
            pool_hints.append("Some 70B model (evidence: merged_70b.jsonl)")
    pool_hints = sorted(set(pool_hints))
    if not pool_hints:
        return ("- Unable to determine generator pool from filenames alone. "
                "Consult `manifest.json` (echoed below) or per-job configs.")
    return "\n".join(f"- {p}" for p in pool_hints)


def find_and_dump_manifests(source: str) -> str:
    """If a manifest file exists in source, dump its contents into README."""
    dumps = []
    for candidate in MANIFEST_CANDIDATES:
        candidate_path = os.path.join(source, candidate)
        if os.path.isfile(candidate_path):
            try:
                with open(candidate_path) as f:
                    content = f.read()
                dumps.append(f"### `{candidate}`\n\n```json\n{content}\n```\n")
            except Exception as e:
                dumps.append(f"### `{candidate}` (read error: {e})\n")
    if not dumps:
        return ("_(No `manifest.json` / `dataset_manifest.json` / `config.json` "
                "found in source. Model provenance must be reconstructed from "
                "per-file inspection or job logs.)_")
    return "\n".join(dumps)


def freeze(source: str, dest_dir: str, commit: bool, move: bool,
           overwrite: bool, allow_empty: bool):
    if not os.path.isdir(source):
        print(f"ERROR: source does not exist: {source}", file=sys.stderr)
        sys.exit(2)

    legacy_root = os.path.join(dest_dir, LEGACY_DIR_NAME)
    if os.path.exists(legacy_root):
        if not overwrite:
            print(f"ERROR: {legacy_root} already exists. Pass --overwrite.",
                  file=sys.stderr)
            sys.exit(3)
        elif commit:
            print(f"  Overwriting existing {legacy_root}")

    files = enumerate_source(source)
    total_bytes = sum(f["size_bytes"] for f in files)
    total_mb = total_bytes / (1024 * 1024)

    print(f"\nFreeze plan:")
    print(f"  source: {source}")
    print(f"  dest:   {legacy_root}")
    print(f"  files:  {len(files)}  ({total_mb:.1f} MB)")
    print(f"  mode:   {'MOVE' if move else 'COPY'}")
    print(f"  commit: {commit}")

    if len(files) == 0 and not allow_empty:
        print(f"\nERROR: source directory has ZERO files. This is almost "
              f"certainly the wrong --source path.", file=sys.stderr)
        print(f"  Contents of {source}:", file=sys.stderr)
        try:
            for entry in os.listdir(source)[:20]:
                full = os.path.join(source, entry)
                marker = "d" if os.path.isdir(full) else "f"
                print(f"    [{marker}] {entry}", file=sys.stderr)
        except OSError:
            pass
        print(f"\n  If you REALLY want to freeze an empty directory "
              f"(e.g., for testing), pass --allow-empty.", file=sys.stderr)
        sys.exit(4)

    if not commit:
        print(f"\nDRY RUN. Pass --commit to execute.")
        return

    original_layout_dir = os.path.join(legacy_root, "original_layout")
    if os.path.exists(legacy_root) and overwrite:
        shutil.rmtree(legacy_root)
    os.makedirs(original_layout_dir, exist_ok=True)

    manifest = {
        "freeze_date_utc": datetime.utcnow().isoformat() + "Z",
        "source_absolute_path": os.path.abspath(source),
        "n_files": len(files),
        "total_bytes": total_bytes,
        "mode": "move" if move else "copy",
        "files": [],
    }

    for f in files:
        src, dst = f["abspath"], os.path.join(original_layout_dir, f["relpath"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if move:
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)
        manifest["files"].append({
            "relpath": f["relpath"], "size_bytes": f["size_bytes"],
            "sha256": sha256_file(dst), "category": f["category"],
        })

    with open(os.path.join(legacy_root, "freeze_manifest.json"), "w") as mf:
        json.dump(manifest, mf, indent=2, sort_keys=True)

    readme = README_TEMPLATE.format(
        freeze_date=manifest["freeze_date_utc"],
        source_path=os.path.abspath(source),
        n_files=len(files), total_mb=total_mb,
        model_pool_evidence=render_model_pool_evidence(files),
        manifest_dump=find_and_dump_manifests(source),
        known_issues_list=render_known_issues(V11_KNOWN_ISSUES),
        files_by_category=render_files_by_category(files),
    )
    with open(os.path.join(legacy_root, "README.md"), "w") as rf:
        rf.write(readme)

    verify_script = """#!/usr/bin/env python3
\"\"\"Recompute SHA256 of every file in original_layout/ and compare to freeze_manifest.json.\"\"\"
import hashlib, json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "freeze_manifest.json")) as f:
    m = json.load(f)
n_ok = n_bad = n_missing = 0
for entry in m["files"]:
    p = os.path.join(here, "original_layout", entry["relpath"])
    if not os.path.exists(p):
        n_missing += 1; print(f"MISSING: {entry['relpath']}"); continue
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        while True:
            b = fp.read(65536)
            if not b: break
            h.update(b)
    if h.hexdigest() == entry["sha256"]:
        n_ok += 1
    else:
        n_bad += 1; print(f"CORRUPT: {entry['relpath']}")
print(f"\\n{n_ok} ok, {n_bad} corrupt, {n_missing} missing (of {len(m['files'])})")
sys.exit(0 if n_bad == 0 and n_missing == 0 else 1)
"""
    vp = os.path.join(legacy_root, "verify_manifest.py")
    with open(vp, "w") as vf:
        vf.write(verify_script)
    os.chmod(vp, 0o755)
    print(f"\n Wrote {legacy_root}/")
    print(f"  {len(files)} files ({total_mb:.1f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--commit", action="store_true")
    p.add_argument("--move", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--allow-empty", action="store_true",
                   help="Allow committing a zero-file freeze (dangerous)")
    args = p.parse_args()
    freeze(source=os.path.expanduser(args.source),
           dest_dir=os.path.expanduser(args.dest),
           commit=args.commit, move=args.move, overwrite=args.overwrite,
           allow_empty=args.allow_empty)


if __name__ == "__main__":
    main()