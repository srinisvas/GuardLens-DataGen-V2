"""
freeze_v11_legacy.py

Freeze the current v11 dataset generation artifacts into a legacy directory
for use ONLY as a historical-distribution behavioral evaluation set in the
NAACL 2027 paper. After freezing, all v12 work operates against fresh
regenerated data. v11's causal-analysis outputs are known-invalid and are
never used as ground truth.

Safety:
  - Dry-run by default. Pass --commit to actually copy.
  - Refuses to run if the destination already exists (unless --overwrite).
  - Never deletes source files. Move semantics are opt-in via --move.

Usage:
    python freeze_v11_legacy.py --source ~/work/results/dataset_gen \
                                --dest ~/work/results
    python freeze_v11_legacy.py --source ~/work/results/dataset_gen \
                                --dest ~/work/results --commit
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Dict, List


LEGACY_DIR_NAME = "dataset_v11_legacy"


V11_MODEL_CONFIG = {
    "generator": "Qwen2.5-14B-Instruct (Alibaba family)",
    "target": "Meta-Llama-3-8B-Instruct (Meta family)",
    "validator_primary": "Qwen2.5-7B-Instruct (Alibaba family)",
    "validator_secondary": "Qwen2.5-14B-Instruct (Alibaba family, optional)",
    "family_pattern": (
        "Generator and validators were Qwen-family; target was Meta Llama. "
        "The circularity flagged by the SAC was generator/validator family "
        "coupling (both Alibaba), even though the attacked target belonged "
        "to a different family. This is what v12's multi-family design "
        "explicitly fixes: v12 rotates generator, target, and judge across "
        "at least three distinct families per configuration."
    ),
}

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

    "Generator/validator family coupling (both Qwen): flagged by SAC "
    "review as circularity in dataset construction.",
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
regime (2024-era Qwen + Llama-3-8B). It is NOT a source of attribution
ground truth — see "Intended use" below.

The final paper reports evaluation across three distribution slices:

| slice | source | role |
|-------|--------|------|
| in-distribution | new v12 pipeline (contemporary models) | training + IID eval |
| current external | MHJ, HarmBench, JailbreakBench, WildJailbreak eval | zero-shot transfer |
| **historical-distribution** | **this directory (v11)** | **older-attack-era behavioral eval** |

## Original model configuration

- generator: `{generator}`
- target: `{target}`
- validator (primary): `{validator_primary}`
- validator (secondary): `{validator_secondary}`

**Family pattern (important):** {family_pattern}

## Known issues (disclosed in paper)

The v11 causal-analysis outputs in this directory should be considered
**invalid, not merely diagnostic**. The following issues are documented
here for reproducibility and are addressed in the v12 pipeline
(`causal_analysis_v12_3.py`):

{known_issues_list}

## Intended use

**DO:**
- Report v11 as a historical-distribution eval set for behavioral
  detection (i.e. does the v12 model correctly classify v11
  conversations as malicious/benign given the trained detection head).
- Use v11 user turn text as a **contamination reference set** so that
  newly generated v12 data does not accidentally overlap with v11's
  attack surface.
- Cite v11 methods in the paper's related-work / history discussion.

**DO NOT:**
- Use v11 records as training data for the v12 model.
- Report v11 causal_analysis Delta values, evidence sets, or span-level
  causal labels as scientific claims. Those fields are known-invalid.
- Use v11 attribution outputs as ground truth for evaluating v12's
  attribution performance. If we need attribution evaluation on v11,
  it must come from independent human annotation, not from the frozen
  v11 causal_analysis fields.
- Modify any file in this directory. It is frozen; use as read-only.

## Manifest

See `freeze_manifest.json` for SHA256 hashes of every file in
`original_layout/`. Verify with `verify_manifest.py` before treating
these files as authoritative.

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
        return "Causal analysis output (v11, KNOWN INVALID — do not use as ground truth)"
    if "merged" in lower and ".jsonl" in lower:
        return "Merged pipeline output (safe as historical behavioral eval set)"
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


def freeze(source: str, dest_dir: str, commit: bool, move: bool,
           overwrite: bool):
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
        generator=V11_MODEL_CONFIG["generator"],
        target=V11_MODEL_CONFIG["target"],
        validator_primary=V11_MODEL_CONFIG["validator_primary"],
        validator_secondary=V11_MODEL_CONFIG["validator_secondary"],
        family_pattern=V11_MODEL_CONFIG["family_pattern"],
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--commit", action="store_true")
    p.add_argument("--move", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    freeze(source=os.path.expanduser(args.source),
           dest_dir=os.path.expanduser(args.dest),
           commit=args.commit, move=args.move, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
