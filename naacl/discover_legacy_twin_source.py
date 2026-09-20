#!/usr/bin/env python3
"""Locate the pre-independent-validation source for restorable Dataset A twins.

The evidence artifact can contain more original benign twins than can enter the
final repaired Dataset A. This audit therefore distinguishes:

- all original benign twins present in the artifact;
- malicious records that passed the historical validation stage;
- final malicious candidates that survive the repaired evidence gate;
- the exact benign twins paired to those final malicious candidates.

Only the final-candidate twin population is required for restoration lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def is_validated_malicious(record: Dict) -> bool:
    return (
        record.get("label") == 1
        and record.get("family") == "interactive_adversarial"
        and record.get("validation_status") == "validated"
        and bool(record.get("pair_id"))
    )


def is_final_malicious_candidate(record: Dict) -> bool:
    if not is_validated_malicious(record):
        return False
    status = str((record.get("evidence_analysis", {}) or {}).get("status", "missing"))
    if status in {"error", "missing"}:
        return False
    fresh_target_unsafe = status == "complete"
    independent_success = bool(
        (record.get("validation_provenance", {}) or {}).get(
            "independent_success", False
        )
    )
    return fresh_target_unsafe or independent_success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--search-dir", required=True)
    parser.add_argument("--expected-all-twins", type=int, default=750)
    parser.add_argument("--expected-validated-malicious", type=int, default=545)
    parser.add_argument("--expected-final-pairs", type=int, default=526)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    all_twins = [r for r in records if is_original_twin(r)]
    validated_malicious = [r for r in records if is_validated_malicious(r)]
    final_malicious = [r for r in records if is_final_malicious_candidate(r)]

    if len(all_twins) != args.expected_all_twins:
        raise RuntimeError(
            f"restoration input has {len(all_twins)} original benign twins; "
            f"expected {args.expected_all_twins}"
        )
    if len(validated_malicious) != args.expected_validated_malicious:
        raise RuntimeError(
            f"restoration input has {len(validated_malicious)} validated malicious; "
            f"expected {args.expected_validated_malicious}"
        )
    if len(final_malicious) != args.expected_final_pairs:
        raise RuntimeError(
            f"restoration input has {len(final_malicious)} final malicious candidates; "
            f"expected {args.expected_final_pairs}"
        )

    all_twins_by_pair: Dict[str, List[Dict]] = {}
    for twin in all_twins:
        all_twins_by_pair.setdefault(str(twin.get("pair_id")), []).append(twin)

    final_pair_ids = [str(r.get("pair_id")) for r in final_malicious]
    if len(final_pair_ids) != len(set(final_pair_ids)):
        raise RuntimeError("final malicious candidates contain duplicate pair_id values")

    relevant_twins: List[Dict] = []
    missing_pairs = []
    duplicate_pairs = []
    for pair_id in final_pair_ids:
        twins = all_twins_by_pair.get(pair_id, [])
        if len(twins) == 0:
            missing_pairs.append(pair_id)
        elif len(twins) > 1:
            duplicate_pairs.append(pair_id)
        else:
            relevant_twins.append(twins[0])

    if missing_pairs or duplicate_pairs:
        raise RuntimeError(
            "final malicious pair linkage is not one-to-one: "
            f"missing={missing_pairs[:10]} duplicate={duplicate_pairs[:10]}"
        )
    if len(relevant_twins) != len(final_malicious):
        raise RuntimeError("final malicious/twin population size mismatch")

    expected = {
        str(r.get("conversation_id")): observable_hash(r)
        for r in relevant_twins
    }
    if len(expected) != len(relevant_twins):
        raise RuntimeError("duplicate restorable-twin conversation IDs in input")

    rows = []
    root = Path(args.search_dir)
    if not root.is_dir():
        raise RuntimeError(f"search directory does not exist: {root}")

    for path in sorted(root.glob("*.jsonl")):
        if path.resolve() == Path(args.input).resolve():
            continue
        try:
            candidate_records = load_jsonl(str(path))
        except Exception as exc:
            rows.append({
                "path": str(path),
                "status": "unreadable",
                "error": repr(exc),
            })
            continue

        by_id = {
            str(r.get("conversation_id")): r
            for r in candidate_records
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
            "records": len(candidate_records),
            "present_restorable_twin_ids": len(present),
            "matching_restorable_twin_trajectories": matched,
            "missing_restorable_twin_ids": len(missing),
            "mismatched_restorable_twin_trajectories": len(mismatched),
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
            -int(r.get("matching_restorable_twin_trajectories", 0)),
            str(r.get("path", "")),
        )
    )

    print(json.dumps({
        "restoration_input": args.input,
        "population": {
            "all_original_benign_twins": len(all_twins),
            "validated_interactive_malicious": len(validated_malicious),
            "final_malicious_candidates": len(final_malicious),
            "restorable_twin_lineage_population": len(relevant_twins),
            "excluded_original_twins_not_in_final_pair_universe": (
                len(all_twins) - len(relevant_twins)
            ),
        },
        "search_dir": str(root),
        "candidates": rows,
    }, indent=2, sort_keys=True))

    exact_pre = [r for r in rows if r.get("pre_independent_like")]
    exact_any = [r for r in rows if r.get("exact_observable_match")]

    print("\n=== SOURCE DISCOVERY SUMMARY ===")
    if exact_pre:
        print("Exact pre-independent-like candidates for the 526 final twin pairs:")
        for row in exact_pre:
            print("  " + row["path"])
    elif exact_any:
        print("No exact pre-independent-like candidate found.")
        print("Exact observable matches that already contain later provenance:")
        for row in exact_any:
            print("  " + row["path"])
    else:
        print("No exact observable match for the 526 final twin pairs.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
