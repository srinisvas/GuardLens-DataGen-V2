#!/usr/bin/env python3
"""Fail-closed audit for GuardLens detection-only auxiliary outcomes."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Dict, Iterable, List

from frontier_common import load_jsonl
from prepare_frontier_auxiliary import (
    AUXILIARY_SOURCE,
    AUXILIARY_WEIGHT,
    assert_expected_full_export_counts,
    detection_label_for_rejected,
    user_trajectory_hash,
)


def audit_record(record: Dict) -> None:
    cid = str(record.get("conversation_id", ""))
    if not cid:
        raise RuntimeError("auxiliary record missing conversation_id")
    if record.get("corpus_source") != AUXILIARY_SOURCE:
        raise RuntimeError(f"{cid}: unexpected corpus_source {record.get('corpus_source')!r}")
    if record.get("validation_status") != "rejected":
        raise RuntimeError(f"{cid}: auxiliary record is not B2-rejected")
    expected = detection_label_for_rejected(record)
    if record.get("detection_label") != expected:
        raise RuntimeError(f"{cid}: detection_label does not match realized B2 outcome")
    if record.get("authoring_intent_label") != record.get("label"):
        raise RuntimeError(f"{cid}: authoring label provenance changed")
    if record.get("training_eligible") is not True or record.get("auxiliary_detection_only") is not True:
        raise RuntimeError(f"{cid}: invalid auxiliary training admission flags")
    if record.get("use_as") != "auxiliary_detection_only":
        raise RuntimeError(f"{cid}: invalid auxiliary use_as")
    if record.get("primary_pair_complete") is not False:
        raise RuntimeError(f"{cid}: auxiliary record may not claim primary-pair membership")
    if record.get("supervision_tier") != "auxiliary_detection":
        raise RuntimeError(f"{cid}: invalid auxiliary supervision tier")
    for field, expected_weight in [
        ("loss_weight", AUXILIARY_WEIGHT),
        ("detection_loss_weight", AUXILIARY_WEIGHT),
        ("pivot_loss_weight", 0.0),
        ("span_loss_weight", 0.0),
    ]:
        if record.get(field) != expected_weight:
            raise RuntimeError(f"{cid}: {field}={record.get(field)!r}, expected {expected_weight}")
    if record.get("pivot_supervision_ignore") is not True:
        raise RuntimeError(f"{cid}: pivot supervision is not masked")
    if record.get("localization_supervision_ignore") is not True:
        raise RuntimeError(f"{cid}: localization supervision is not masked")
    if record.get("pivot_turn_id") is not None or record.get("evidence_turn_ids") != []:
        raise RuntimeError(f"{cid}: auxiliary record carries localization targets")

    metadata = record.get("metadata", {}) or {}
    scenario = str(metadata.get("scenario_family", "")).strip()
    expected_group = f"frontier::{scenario}"
    if not scenario or metadata.get("consolidated_split_group") != expected_group:
        raise RuntimeError(f"{cid}: invalid family-preserving split group")
    if metadata.get("normalized_user_trajectory_hash") != user_trajectory_hash(record):
        raise RuntimeError(f"{cid}: normalized user-trajectory hash mismatch")


def audit(records: Iterable[Dict], *, expect_full_review_export: bool = False) -> Dict:
    rows: List[Dict] = list(records)
    ids = set()
    groups = defaultdict(set)
    for record in rows:
        audit_record(record)
        cid = str(record["conversation_id"])
        if cid in ids:
            raise RuntimeError(f"duplicate auxiliary conversation_id: {cid}")
        ids.add(cid)
        groups[record["metadata"]["consolidated_split_group"]].add(
            int(record["detection_label"])
        )
    if expect_full_review_export:
        assert_expected_full_export_counts(rows)
    return {
        "status": "passed",
        "records": len(rows),
        "detection_labels": dict(Counter(str(r["detection_label"]) for r in rows)),
        "authoring_labels": dict(Counter(str(r["authoring_intent_label"]) for r in rows)),
        "scenario_families": len(groups),
        "detection_only_masking": "passed",
        "split_grouping": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--expect-full-review-export", action="store_true")
    args = parser.parse_args()
    report = audit(
        load_jsonl(args.input),
        expect_full_review_export=args.expect_full_review_export,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
