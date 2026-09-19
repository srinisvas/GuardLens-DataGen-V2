#!/usr/bin/env python3
"""Audit semantic masking of reviewed construction-language evidence spans."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, Iterable, Tuple

from frontier_common import load_jsonl
from semantic_span_policy import (
    ADJUDICATION_VERSION,
    SUPPORTED_STATUSES,
    iter_flagged_spans,
)

EXPECTED_RAW_REVIEWED_SPANS = 21
EXPECTED_PRIMARY_MASKED_SPANS = 14
EXPECTED_PRIMARY_MASKED_RECORDS = 13


def span_key(record: Dict, turn: Dict, span: Dict) -> Tuple:
    return (
        str(record.get("conversation_id", "")),
        int(turn.get("turn_id")),
        int(span.get("char_start")),
        int(span.get("char_end")),
        str(span.get("text", "")),
    )


def raw_flagged_map(records: Iterable[Dict]) -> Dict[Tuple, str]:
    result = {}
    for record, turn, span in iter_flagged_spans(records):
        key = span_key(record, turn, span)
        if key in result:
            raise RuntimeError(f"duplicate raw reviewed span key: {key}")
        result[key] = str(span.get("evidence_status"))
    return result


def prepared_masked_map(records: Iterable[Dict]) -> Dict[Tuple, str]:
    result = {}
    for record in records:
        for turn in record.get("turns", []) or []:
            for span in turn.get("span_annotations", []) or []:
                if span.get("semantic_adjudication") != ADJUDICATION_VERSION:
                    continue
                key = span_key(record, turn, span)
                if key in result:
                    raise RuntimeError(f"duplicate prepared masked span key: {key}")
                if span.get("semantic_token_supervision_ignore") is not True:
                    raise RuntimeError(f"{key}: semantic token ignore flag missing")
                if span.get("supervision_tier") != "ignore":
                    raise RuntimeError(f"{key}: masked span still has positive supervision tier")
                status = str(span.get("evidence_status", ""))
                if status not in SUPPORTED_STATUSES:
                    raise RuntimeError(f"{key}: raw supported evidence status was rewritten")
                result[key] = status
    return result


def audit(raw_records, prepared_records, *, enforce_reviewed_counts=False) -> Dict:
    raw = raw_flagged_map(raw_records)
    prepared = prepared_masked_map(prepared_records)

    for key, status in prepared.items():
        if key not in raw:
            raise RuntimeError(f"prepared semantic mask has no matching raw reviewed span: {key}")
        if raw[key] != status:
            raise RuntimeError(f"prepared mask changed raw evidence status for {key}")

    masked_records = {key[0] for key in prepared}
    if enforce_reviewed_counts:
        if len(raw) != EXPECTED_RAW_REVIEWED_SPANS:
            raise RuntimeError(
                f"reviewed raw construction span count changed: {len(raw)} "
                f"!= {EXPECTED_RAW_REVIEWED_SPANS}"
            )
        if len(prepared) != EXPECTED_PRIMARY_MASKED_SPANS:
            raise RuntimeError(
                f"primary semantic mask count changed: {len(prepared)} "
                f"!= {EXPECTED_PRIMARY_MASKED_SPANS}"
            )
        if len(masked_records) != EXPECTED_PRIMARY_MASKED_RECORDS:
            raise RuntimeError(
                f"primary semantic masked-record count changed: {len(masked_records)} "
                f"!= {EXPECTED_PRIMARY_MASKED_RECORDS}"
            )

    return {
        "status": "passed",
        "adjudication_version": ADJUDICATION_VERSION,
        "raw_reviewed_supported_spans": len(raw),
        "prepared_masked_spans": len(prepared),
        "prepared_masked_records": len(masked_records),
        "prepared_statuses": dict(Counter(prepared.values())),
        "raw_evidence_preserved": True,
        "positive_token_supervision_removed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-input", required=True)
    parser.add_argument("--prepared-input", required=True)
    parser.add_argument("--enforce-reviewed-counts", action="store_true")
    args = parser.parse_args()
    report = audit(
        load_jsonl(args.raw_input),
        load_jsonl(args.prepared_input),
        enforce_reviewed_counts=args.enforce_reviewed_counts,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
