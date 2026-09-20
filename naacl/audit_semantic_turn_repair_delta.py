#!/usr/bin/env python3
"""Prove that semantic turn-supervision repair changes only intended fields.

This audit compares the previous frozen artifacts with a candidate regeneration.
It derives the expected repair from the previous prepared records, requires every
non-repair record to remain byte-equivalent at the JSON-object level, and checks
that split membership/order does not move.

The raw B4 evidence stored in frontier_evidence_analysis is never rewritten.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from typing import Dict, Iterable, List, Mapping, Tuple

from frontier_common import load_jsonl
from prepare_frontier_dataset import SEMANTIC_TURN_REPAIR_VERSION
from semantic_span_policy import ADJUDICATION_VERSION, SUPPORTED_STATUSES

EXPECTED_REPAIRED_TURNS = 4
SPLITS = ("train", "dev", "test")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_byte_identical(before_path: str, after_path: str, *, where: str) -> Dict:
    before_sha = file_sha256(before_path)
    after_sha = file_sha256(after_path)
    if before_sha != after_sha:
        raise RuntimeError(f"{where}: artifact is not byte-identical")
    return {
        "status": "byte_identical",
        "sha256": before_sha,
    }


def _index(records: Iterable[Dict], where: str) -> Tuple[List[str], Dict[str, Dict]]:
    order: List[str] = []
    by_id: Dict[str, Dict] = {}
    for record in records:
        cid = str(record.get("conversation_id", ""))
        if not cid:
            raise RuntimeError(f"{where}: missing conversation_id")
        if cid in by_id:
            raise RuntimeError(f"{where}: duplicate conversation_id {cid}")
        order.append(cid)
        by_id[cid] = record
    return order, by_id


def _turn_statuses(record: Dict) -> Dict[int, str]:
    cid = str(record.get("conversation_id", ""))
    result: Dict[int, str] = {}
    analysis = record.get("frontier_evidence_analysis", {}) or {}
    for item in analysis.get("turn_interventions", []) or []:
        tid = item.get("turn_id")
        if isinstance(tid, bool) or not isinstance(tid, int):
            raise RuntimeError(f"{cid}: invalid intervention turn id {tid!r}")
        status = str(item.get("status", ""))
        old = result.get(tid)
        if old is not None and old != status:
            raise RuntimeError(
                f"{cid}: conflicting whole-turn statuses for turn {tid}: "
                f"{old!r} vs {status!r}"
            )
        result[tid] = status
    return result


def _eligible_nonmasked_positive(span: Dict) -> bool:
    if span.get("semantic_token_supervision_ignore") is True:
        return False
    status = str(span.get("evidence_status", ""))
    tier = str(span.get("supervision_tier", ""))
    causal_type = str(span.get("causal_type", ""))
    return (
        causal_type == "causal"
        and (
            (status == "supported_strong" and tier == "cf_strong")
            or (status == "supported_weak" and tier == "cf_weak")
        )
    )


def derive_case_a_turns(record: Dict) -> List[int]:
    """Derive repairable masked turns from the previous prepared artifact."""
    raw_ids = set(int(x) for x in record.get("evidence_turn_ids", []) or [])
    statuses = _turn_statuses(record)
    case_a: List[int] = []

    for turn in record.get("turns", []) or []:
        tid = int(turn.get("turn_id", -1))
        masked = [
            span
            for span in turn.get("span_annotations", []) or []
            if span.get("semantic_adjudication") == ADJUDICATION_VERSION
            and span.get("semantic_token_supervision_ignore") is True
        ]
        if not masked or tid not in raw_ids:
            continue

        for span in masked:
            if str(span.get("evidence_status", "")) not in SUPPORTED_STATUSES:
                raise RuntimeError(
                    f"{record.get('conversation_id')}: masked turn {tid} has "
                    "non-supported semantic evidence"
                )

        if statuses.get(tid) in SUPPORTED_STATUSES:
            continue
        if any(
            _eligible_nonmasked_positive(span)
            for span in turn.get("span_annotations", []) or []
        ):
            continue
        if statuses.get(tid) == "not_supported":
            case_a.append(tid)
            continue
        raise RuntimeError(
            f"{record.get('conversation_id')}: masked evidence turn {tid} "
            "has no independent support but is not an explicit case-A negative"
        )

    return sorted(case_a)


def expected_repaired_record(before: Dict) -> Tuple[Dict, List[int]]:
    expected = copy.deepcopy(before)
    case_a = derive_case_a_turns(before)
    if not case_a:
        return expected, []

    raw_ids = sorted({int(x) for x in before.get("evidence_turn_ids", []) or []})
    repaired_ids = [tid for tid in raw_ids if tid not in set(case_a)]

    expected["evidence_turn_ids"] = repaired_ids
    expected["pivot_turn_id"] = repaired_ids[0] if repaired_ids else None
    expected["pivot_supervision_ignore"] = not bool(repaired_ids)
    if len(repaired_ids) > 1:
        expected["pivot_kind"] = "distributed"
    elif len(repaired_ids) == 1:
        expected["pivot_kind"] = "contextual_pivot"
    else:
        expected["pivot_kind"] = "none"

    expected["semantic_turn_supervision_repair"] = {
        "version": SEMANTIC_TURN_REPAIR_VERSION,
        "removed_evidence_turn_ids": case_a,
        "raw_b4_evidence_turn_ids": raw_ids,
        "prepared_evidence_turn_ids": repaired_ids,
        "reason": (
            "semantic token mask removed the only eligible positive span "
            "while the whole-turn intervention was not_supported"
        ),
    }
    return expected, case_a


def audit_artifact_pair(
    before_path: str,
    after_path: str,
    *,
    where: str,
    expected_total_repaired_turns: int | None = None,
) -> Dict:
    before = load_jsonl(before_path)
    after = load_jsonl(after_path)
    before_order, before_by_id = _index(before, f"{where} before")
    after_order, after_by_id = _index(after, f"{where} after")

    if before_order != after_order:
        raise RuntimeError(f"{where}: record order/membership changed")

    changed_records: List[str] = []
    repair_turns: List[Dict] = []
    for cid in before_order:
        expected, tids = expected_repaired_record(before_by_id[cid])
        observed = after_by_id[cid]
        if observed != expected:
            raise RuntimeError(
                f"{where}: {cid} differs beyond the derived semantic turn repair"
            )
        if tids:
            changed_records.append(cid)
            repair_turns.extend(
                {"conversation_id": cid, "turn_id": tid} for tid in tids
            )

    if (
        expected_total_repaired_turns is not None
        and len(repair_turns) != expected_total_repaired_turns
    ):
        raise RuntimeError(
            f"{where}: repaired turns={len(repair_turns)} "
            f"!= expected {expected_total_repaired_turns}"
        )

    return {
        "records": len(before_order),
        "changed_records": len(changed_records),
        "changed_conversation_ids": changed_records,
        "repaired_turns": repair_turns,
        "before_sha256": file_sha256(before_path),
        "after_sha256": file_sha256(after_path),
    }


def _split_paths(root: str) -> Dict[str, str]:
    return {name: os.path.join(root, f"{name}.jsonl") for name in SPLITS}


def audit_split_pair(before_dir: str, after_dir: str, *, where: str) -> Dict:
    report = {}
    total_repairs = 0
    for split in ("train", "dev"):
        before_path = os.path.join(before_dir, f"{split}.jsonl")
        after_path = os.path.join(after_dir, f"{split}.jsonl")
        item = audit_artifact_pair(
            before_path,
            after_path,
            where=f"{where} {split}",
        )
        total_repairs += len(item["repaired_turns"])
        report[split] = item

    # Do not semantically inspect held-out test content. Prove only that the
    # candidate test file is byte-for-byte identical to the previous freeze.
    report["test"] = audit_byte_identical(
        os.path.join(before_dir, "test.jsonl"),
        os.path.join(after_dir, "test.jsonl"),
        where=f"{where} test",
    )

    if total_repairs != EXPECTED_REPAIRED_TURNS:
        raise RuntimeError(
            f"{where}: total repaired turns={total_repairs} "
            f"!= expected {EXPECTED_REPAIRED_TURNS}"
        )

    report["total_repaired_turns"] = total_repairs
    report["held_out_test_byte_identical"] = True
    report["held_out_test_semantically_inspected"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-frontier", required=True)
    parser.add_argument("--after-frontier", required=True)
    parser.add_argument("--before-stress", required=True)
    parser.add_argument("--after-stress", required=True)
    parser.add_argument("--before-excluded", required=True)
    parser.add_argument("--after-excluded", required=True)
    parser.add_argument("--before-auxiliary", required=True)
    parser.add_argument("--after-auxiliary", required=True)
    parser.add_argument("--before-merged")
    parser.add_argument("--after-merged")
    parser.add_argument("--before-primary-split-dir")
    parser.add_argument("--after-primary-split-dir")
    parser.add_argument("--before-aux-split-dir")
    parser.add_argument("--after-aux-split-dir")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = {
        "status": "passed",
        "repair_version": SEMANTIC_TURN_REPAIR_VERSION,
        "expected_repaired_turns": EXPECTED_REPAIRED_TURNS,
        "frontier_primary": audit_artifact_pair(
            args.before_frontier,
            args.after_frontier,
            where="frontier primary",
            expected_total_repaired_turns=EXPECTED_REPAIRED_TURNS,
        ),
        "frontier_stress": audit_byte_identical(
            args.before_stress,
            args.after_stress,
            where="frontier stress",
        ),
        "frontier_excluded": audit_byte_identical(
            args.before_excluded,
            args.after_excluded,
            where="frontier excluded",
        ),
        "auxiliary_input": audit_byte_identical(
            args.before_auxiliary,
            args.after_auxiliary,
            where="auxiliary input",
        ),
    }

    optional_pairs = [
        ("merged_primary", args.before_merged, args.after_merged),
    ]
    for name, before, after in optional_pairs:
        if bool(before) != bool(after):
            raise RuntimeError(f"{name}: both before and after paths are required")
        if before:
            report[name] = audit_artifact_pair(
                before,
                after,
                where=name,
                expected_total_repaired_turns=EXPECTED_REPAIRED_TURNS,
            )

    if bool(args.before_primary_split_dir) != bool(args.after_primary_split_dir):
        raise RuntimeError("primary splits: both before/after directories are required")
    if args.before_primary_split_dir:
        report["primary_splits"] = audit_split_pair(
            args.before_primary_split_dir,
            args.after_primary_split_dir,
            where="primary splits",
        )

    if bool(args.before_aux_split_dir) != bool(args.after_aux_split_dir):
        raise RuntimeError("auxiliary splits: both before/after directories are required")
    if args.before_aux_split_dir:
        report["auxiliary_splits"] = audit_split_pair(
            args.before_aux_split_dir,
            args.after_aux_split_dir,
            where="auxiliary candidate splits",
        )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.write("\n")
    print(rendered)


if __name__ == "__main__":
    main()
