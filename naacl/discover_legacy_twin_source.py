#!/usr/bin/env python3
"""Discover or reconstruct clean lineage for restorable Dataset A twins.

The evidence artifact contains 750 original benign twins, but only the 526 twins
paired to final repaired malicious candidates can enter the restored primary
Dataset A. This tool derives that exact 526-pair universe, recursively scans all
JSONL files under a staging directory (including shard inputs/checkpoints), and
looks for clean pre-independent copies of those twins.

A source copy is accepted for clean lineage only when:
- its conversation_id is in the 526-pair restoration universe;
- observable turn IDs/roles/text exactly match the restoration input; and
- it has no causal_validation, independent_validation, or
  stored_target_validation payload.

If no single file contains all 526 clean copies but multiple shards collectively
cover them, --materialize-output writes a deterministic lineage-only JSONL plus
a manifest that records each contributing source file and SHA-256 digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    nonempty = [(i, line) for i, line in enumerate(lines, 1) if line.strip()]
    for pos, (line_no, line) in enumerate(nonempty):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            # Checkpoint files can be interrupted during their last append.
            # Tolerate only a torn final non-empty line for discovery purposes.
            if pos == len(nonempty) - 1 and "checkpoint" in path.lower():
                break
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


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
    return status == "complete" or bool(
        (record.get("validation_provenance", {}) or {}).get(
            "independent_success", False
        )
    )


def is_pre_independent(record: Dict) -> bool:
    return not (
        bool(record.get("causal_validation"))
        or bool(record.get("independent_validation"))
        or bool(record.get("stored_target_validation"))
    )


def provenance_counts(records: Iterable[Dict]) -> Dict[str, int]:
    rows = list(records)
    return {
        "with_llama_validation": sum(bool(r.get("llama_validation")) for r in rows),
        "with_causal_validation": sum(bool(r.get("causal_validation")) for r in rows),
        "with_independent_validation": sum(
            bool(r.get("independent_validation")) for r in rows
        ),
        "with_evidence_analysis": sum(bool(r.get("evidence_analysis")) for r in rows),
        "with_stored_target_validation": sum(
            bool(r.get("stored_target_validation")) for r in rows
        ),
    }


def source_rank(path: Path) -> Tuple[int, str]:
    """Prefer immutable/input/raw sources over checkpoints when duplicates exist."""
    name = str(path).lower()
    if "_input" in name or name.endswith("_raw.jsonl") or "/raw/" in name:
        bucket = 0
    elif "checkpoint" not in name:
        bucket = 1
    else:
        bucket = 2
    return bucket, str(path)


def write_jsonl(records: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--search-dir", required=True)
    parser.add_argument("--expected-all-twins", type=int, default=750)
    parser.add_argument("--expected-validated-malicious", type=int, default=545)
    parser.add_argument("--expected-final-pairs", type=int, default=526)
    parser.add_argument(
        "--materialize-output",
        default="",
        help=(
            "If clean recursive shard coverage reaches all final twins, write one "
            "deterministic 526-record lineage artifact here."
        ),
    )
    parser.add_argument(
        "--manifest-output",
        default="",
        help="Optional manifest path for the materialized lineage artifact.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    records = load_jsonl(str(input_path))
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

    twins_by_pair: Dict[str, List[Dict]] = defaultdict(list)
    for twin in all_twins:
        twins_by_pair[str(twin.get("pair_id"))].append(twin)

    final_pair_ids = [str(r.get("pair_id")) for r in final_malicious]
    if len(final_pair_ids) != len(set(final_pair_ids)):
        raise RuntimeError("final malicious candidates contain duplicate pair_id values")

    relevant_twins = []
    for pair_id in final_pair_ids:
        twins = twins_by_pair.get(pair_id, [])
        if len(twins) != 1:
            raise RuntimeError(
                f"final pair {pair_id}: expected exactly one benign twin, "
                f"found {len(twins)}"
            )
        relevant_twins.append(twins[0])

    expected_hash = {
        str(r.get("conversation_id")): observable_hash(r)
        for r in relevant_twins
    }
    expected_record = {
        str(r.get("conversation_id")): r
        for r in relevant_twins
    }
    if len(expected_hash) != len(relevant_twins):
        raise RuntimeError("duplicate restorable-twin conversation IDs in input")

    root = Path(args.search_dir).resolve()
    if not root.is_dir():
        raise RuntimeError(f"search directory does not exist: {root}")

    materialize_path = (
        Path(args.materialize_output).resolve()
        if args.materialize_output
        else None
    )
    manifest_path = (
        Path(args.manifest_output).resolve()
        if args.manifest_output
        else (
            Path(str(materialize_path) + ".manifest.json")
            if materialize_path is not None
            else None
        )
    )

    rows = []
    clean_sources_by_cid: Dict[str, List[Tuple[Path, Dict]]] = defaultdict(list)
    exact_sources_by_cid: Dict[str, List[Tuple[Path, Dict]]] = defaultdict(list)
    unreadable = []

    jsonl_paths = sorted(root.rglob("*.jsonl"))
    for path in jsonl_paths:
        resolved = path.resolve()
        if resolved == input_path:
            continue
        if materialize_path is not None and resolved == materialize_path:
            continue

        try:
            candidate_records = load_jsonl(str(path))
        except Exception as exc:
            unreadable.append({"path": str(path), "error": repr(exc)})
            continue

        by_expected_id: Dict[str, List[Dict]] = defaultdict(list)
        for record in candidate_records:
            cid = str(record.get("conversation_id", ""))
            if cid in expected_hash:
                by_expected_id[cid].append(record)

        present = sorted(by_expected_id)
        exact_ids = []
        clean_exact_ids = []
        mismatch_ids = []
        conflicting_duplicate_ids = []
        exact_records = []

        for cid in present:
            copies = by_expected_id[cid]
            copy_hashes = {observable_hash(r) for r in copies}
            if len(copy_hashes) > 1:
                conflicting_duplicate_ids.append(cid)

            exact_copies = [
                r for r in copies
                if observable_hash(r) == expected_hash[cid]
            ]
            if not exact_copies:
                mismatch_ids.append(cid)
                continue

            exact_ids.append(cid)
            exact_records.append(exact_copies[0])
            exact_sources_by_cid[cid].append((path, exact_copies[0]))

            clean_copies = [r for r in exact_copies if is_pre_independent(r)]
            if clean_copies:
                clean_exact_ids.append(cid)
                clean_sources_by_cid[cid].append((path, clean_copies[0]))

        exact_full = (
            len(exact_ids) == len(expected_hash)
            and not mismatch_ids
            and not conflicting_duplicate_ids
        )
        clean_full = (
            len(clean_exact_ids) == len(expected_hash)
            and not mismatch_ids
            and not conflicting_duplicate_ids
        )

        rows.append({
            "path": str(path),
            "records": len(candidate_records),
            "present_restorable_twin_ids": len(present),
            "exact_observable_matches": len(exact_ids),
            "clean_pre_independent_matches": len(clean_exact_ids),
            "mismatched_restorable_twin_trajectories": len(mismatch_ids),
            "conflicting_duplicate_ids": len(conflicting_duplicate_ids),
            "exact_observable_full_coverage": exact_full,
            "clean_pre_independent_full_coverage": clean_full,
            "provenance_of_exact_matches": provenance_counts(exact_records),
            "mismatch_examples": mismatch_ids[:5],
            "duplicate_examples": conflicting_duplicate_ids[:5],
        })

    rows.sort(
        key=lambda r: (
            not bool(r.get("clean_pre_independent_full_coverage")),
            not bool(r.get("exact_observable_full_coverage")),
            -int(r.get("clean_pre_independent_matches", 0)),
            -int(r.get("exact_observable_matches", 0)),
            str(r.get("path", "")),
        )
    )

    clean_covered = set(clean_sources_by_cid)
    exact_covered = set(exact_sources_by_cid)
    missing_clean = sorted(set(expected_hash) - clean_covered)
    missing_exact = sorted(set(expected_hash) - exact_covered)

    payload = {
        "restoration_input": str(input_path),
        "population": {
            "all_original_benign_twins": len(all_twins),
            "validated_interactive_malicious": len(validated_malicious),
            "final_malicious_candidates": len(final_malicious),
            "restorable_twin_lineage_population": len(relevant_twins),
            "excluded_original_twins_not_in_final_pair_universe": (
                len(all_twins) - len(relevant_twins)
            ),
        },
        "search": {
            "root": str(root),
            "jsonl_files_scanned": len(jsonl_paths),
            "unreadable_files": len(unreadable),
        },
        "recursive_union": {
            "clean_pre_independent_coverage": len(clean_covered),
            "exact_observable_coverage_any_stage": len(exact_covered),
            "missing_clean_pre_independent": len(missing_clean),
            "missing_exact_any_stage": len(missing_exact),
            "missing_clean_examples": missing_clean[:10],
            "missing_exact_examples": missing_exact[:10],
        },
        "candidates": rows,
        "unreadable_examples": unreadable[:20],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    clean_full_files = [
        row["path"] for row in rows
        if row.get("clean_pre_independent_full_coverage")
    ]
    exact_full_files = [
        row["path"] for row in rows
        if row.get("exact_observable_full_coverage")
    ]

    print("\n=== SOURCE DISCOVERY SUMMARY ===")
    if clean_full_files:
        print("Single-file clean pre-independent candidates:")
        for path in clean_full_files:
            print("  " + path)
    elif len(clean_covered) == len(expected_hash):
        print(
            "No single clean file contains all 526 twins, but recursive clean "
            "shard/checkpoint coverage is complete: 526/526."
        )
    elif exact_full_files:
        print("No complete clean pre-independent coverage found.")
        print("Single-file exact observable matches from later stages:")
        for path in exact_full_files:
            print("  " + path)
    else:
        print(
            f"Clean pre-independent recursive coverage: "
            f"{len(clean_covered)}/{len(expected_hash)}"
        )
        print(
            f"Exact observable recursive coverage at any stage: "
            f"{len(exact_covered)}/{len(expected_hash)}"
        )

    if materialize_path is None:
        return

    if len(clean_covered) != len(expected_hash):
        raise RuntimeError(
            "cannot materialize clean lineage artifact: recursive clean coverage "
            f"is {len(clean_covered)}/{len(expected_hash)}"
        )

    selected_records = []
    selected_sources = {}
    contributing_files = Counter()
    for cid in sorted(expected_hash):
        sources = sorted(clean_sources_by_cid[cid], key=lambda item: source_rank(item[0]))
        source_path, source_record = sources[0]
        if observable_hash(source_record) != expected_hash[cid]:
            raise RuntimeError(f"{cid}: selected source observable hash mismatch")
        if not is_pre_independent(source_record):
            raise RuntimeError(f"{cid}: selected source is not pre-independent")
        selected_records.append(source_record)
        selected_sources[cid] = str(source_path.resolve())
        contributing_files[str(source_path.resolve())] += 1

    write_jsonl(selected_records, materialize_path)

    reread = load_jsonl(str(materialize_path))
    if len(reread) != len(expected_hash):
        raise RuntimeError("materialized lineage artifact record count mismatch")
    for record in reread:
        cid = str(record.get("conversation_id", ""))
        if cid not in expected_hash:
            raise RuntimeError(f"materialized unexpected conversation_id {cid}")
        if observable_hash(record) != expected_hash[cid]:
            raise RuntimeError(f"{cid}: materialized observable hash mismatch")
        if not is_pre_independent(record):
            raise RuntimeError(f"{cid}: materialized record is not pre-independent")

    manifest = {
        "kind": "dataset_a_restorable_twin_pre_independent_lineage_union_v1",
        "record_count": len(reread),
        "restoration_input": {
            "path": str(input_path),
            "sha256": file_sha256(input_path),
        },
        "search_root": str(root),
        "materialized_output": {
            "path": str(materialize_path),
            "sha256": file_sha256(materialize_path),
        },
        "contributing_files": [
            {
                "path": path,
                "records_selected": count,
                "sha256": file_sha256(Path(path)),
            }
            for path, count in sorted(contributing_files.items())
        ],
        "record_sources": selected_sources,
        "selection_policy": (
            "exact observable match + no causal/independent/stored-target "
            "validation; prefer input/raw, then non-checkpoint, then checkpoint"
        ),
    }

    if manifest_path is None:
        raise RuntimeError("internal error: manifest path not resolved")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(manifest_path) + ".tmp")
    tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, manifest_path)

    print("\nMATERIALIZED CLEAN LINEAGE ARTIFACT")
    print(f"  JSONL:    {materialize_path}")
    print(f"  SHA256:   {manifest['materialized_output']['sha256']}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Source files used: {len(contributing_files)}")


if __name__ == "__main__":
    main()
