#!/usr/bin/env python3
"""Locate the best pre-independent-validation Dataset A lineage artifact.

Scans JSONL files in a directory and compares the original benign-twin
conversation IDs plus observable turn IDs/roles/text against the current
restoration input. Reports exact/partial matches and whether each candidate
already contains later validation/evidence fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return out


def observable_hash(record: Dict) -> str:
    payload = [
        {
            "turn_id": turn.get("turn_id"),
            "role": turn.get("role"),
            "text": turn.get("text"),
        }
        for turn in record.get("turns", [])
    ]
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def is_original_twin(record: Dict) -> bool:
    return (
        record.get("label") == 0
        and record.get("family") == "interactive_benign_twin"
        and bool(record.get("pair_id"))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--search-dir", required=True)
    parser.add_argument("--expected-twins", type=int, default=545)
    args = parser.parse_args()

    current = [r for r in load_jsonl(args.input) if is_original_twin(r)]
    if len(current) != args.expected_twins:
        raise RuntimeError(
            f"restoration input has {len(current)} original twins; "
            f"expected {args.expected_twins}"
        )
    expected = {
        str(r.get("conversation_id")): observable_hash(r)
        for r in current
    }
    if len(expected) != len(current):
        raise RuntimeError("duplicate original-twin conversation IDs in restoration input")

    rows = []
    root = Path(args.search_dir)
    if not root.is_dir():
        raise RuntimeError(f"search directory does not exist: {root}")

    for path in sorted(root.glob("*.jsonl")):
        if path.resolve() == Path(args.input).resolve():
            continue
        try:
            records = load_jsonl(str(path))
        except Exception as exc:
            rows.append({
                "path": str(path),
                "status": "unreadable",
                "error": repr(exc),
            })
            continue

        by_id = {
            str(r.get("conversation_id")): r
            for r in records
            if r.get("conversation_id")
        }
        present = sorted(set(expected) & set(by_id))
        missing = sorted(set(expected) - set(by_id))
        mismatched = [
            cid for cid in present
            if observable_hash(by_id[cid]) != expected[cid]
        ]
        matched = len(present) - len(mismatched)

        twin_records = [by_id[cid] for cid in present]
        provenance = {
            "with_llama_validation": sum(
                bool(r.get("llama_validation")) for r in twin_records
            ),
            "with_causal_validation": sum(
                bool(r.get("causal_validation")) for r in twin_records
            ),
            "with_independent_validation": sum(
                bool(r.get("independent_validation")) for r in twin_records
            ),
            "with_evidence_analysis": sum(
                bool(r.get("evidence_analysis")) for r in twin_records
            ),
            "with_stored_target_validation": sum(
                bool(r.get("stored_target_validation")) for r in twin_records
            ),
        }

        exact = (
            len(present) == len(expected)
            and not missing
            and not mismatched
        )
        pre_independent_like = exact and (
            provenance["with_causal_validation"] == 0
            and provenance["with_independent_validation"] == 0
            and provenance["with_stored_target_validation"] == 0
        )

        rows.append({
            "path": str(path),
            "records": len(records),
            "present_twin_ids": len(present),
            "matching_twin_trajectories": matched,
            "missing_twin_ids": len(missing),
            "mismatched_twin_trajectories": len(mismatched),
            "exact_observable_match": exact,
            "pre_independent_like": pre_independent_like,
            "provenance": provenance,
            "missing_examples": missing[:5],
            "mismatch_examples": mismatched[:5],
        })

    rows.sort(
        key=lambda r: (
            not bool(r.get("pre_independent_like")),
            not bool(r.get("exact_observable_match")),
            -int(r.get("matching_twin_trajectories", 0)),
            str(r.get("path", "")),
        )
    )

    print(json.dumps({
        "restoration_input": args.input,
        "expected_original_twins": len(expected),
        "search_dir": str(root),
        "candidates": rows,
    }, indent=2, sort_keys=True))

    exact_pre = [r for r in rows if r.get("pre_independent_like")]
    exact_any = [r for r in rows if r.get("exact_observable_match")]

    print("\n=== SOURCE DISCOVERY SUMMARY ===")
    if exact_pre:
        print("Exact pre-independent-like candidates:")
        for row in exact_pre:
            print("  " + row["path"])
    elif exact_any:
        print("No exact pre-independent-like candidate found.")
        print("Exact observable matches that already contain later provenance:")
        for row in exact_any:
            print("  " + row["path"])
    else:
        print("No exact observable twin-lineage match found.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
