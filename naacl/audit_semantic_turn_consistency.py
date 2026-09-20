#!/usr/bin/env python3
"""Read-only inspection of semantic span masking versus turn supervision.

This module answers one narrow question before any frozen-data repair:
when a reviewed construction-language span is masked from positive token
supervision, does the containing turn still have independent intervention
support for membership in evidence_turn_ids?

The inspection never mutates records and never reads the held-out test split.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from frontier_common import load_jsonl
from semantic_span_policy import ADJUDICATION_VERSION, SUPPORTED_STATUSES

EXPECTED_PRIMARY_MASKED_SPANS = 14
EXPECTED_PRIMARY_MASKED_RECORDS = 13

POSITIVE_TIER_STATUS = {
    "cf_strong": "supported_strong",
    "cf_weak": "supported_weak",
}

CASE_A = "A_explicit_turn_negative_masked_only_support"
CASE_B = "B_orphan_evidence_id_after_mask"
CASE_C = "C_independently_supported_after_mask"
CASE_D = "D_mask_has_no_turn_membership_dependency"

CASE_DESCRIPTIONS = {
    CASE_A: (
        "Turn is in evidence_turn_ids, its whole-turn intervention is "
        "not_supported, and no eligible non-masked positive span remains."
    ),
    CASE_B: (
        "Turn is in evidence_turn_ids, but after semantic masking it has "
        "neither a supported whole-turn intervention nor an eligible "
        "non-masked positive span. Current downstream logic would need an "
        "evidence-id fallback rather than direct local evidence."
    ),
    CASE_C: (
        "Turn is in evidence_turn_ids and remains independently supported by "
        "a supported whole-turn intervention and/or an eligible non-masked "
        "positive span."
    ),
    CASE_D: (
        "The semantically masked span is on a turn that is not in "
        "evidence_turn_ids, so masking has no turn-membership dependency."
    ),
}


def _cid(record: Dict) -> str:
    return str(record.get("conversation_id", "")) or "<missing>"


def _turn_id(turn: Dict) -> int:
    raw = turn.get("turn_id")
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise RuntimeError(f"invalid turn_id={raw!r}")
    return raw


def is_masked_reviewed_span(span: Dict) -> bool:
    return (
        span.get("semantic_adjudication") == ADJUDICATION_VERSION
        and span.get("semantic_token_supervision_ignore") is True
    )


def is_eligible_nonmasked_positive_span(span: Dict) -> bool:
    """Mirror the downstream positive span contract after semantic masking."""
    if span.get("semantic_token_supervision_ignore") is True:
        return False
    tier = str(span.get("supervision_tier", ""))
    status = str(span.get("evidence_status", ""))
    return (
        tier in POSITIVE_TIER_STATUS
        and status == POSITIVE_TIER_STATUS[tier]
        and str(span.get("causal_type", "")) == "causal"
    )


def _turn_intervention_status(record: Dict, tid: int) -> Optional[str]:
    analysis = record.get("frontier_evidence_analysis", {}) or {}
    matches = [
        str(item.get("status", ""))
        for item in analysis.get("turn_interventions", []) or []
        if item.get("turn_id") == tid
    ]
    if not matches:
        return None
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise RuntimeError(
            f"{_cid(record)}: turn {tid} has conflicting whole-turn "
            f"intervention statuses {unique}"
        )
    return unique[0]


def _validate_masked_span(record: Dict, turn: Dict, span: Dict) -> None:
    cid = _cid(record)
    tid = _turn_id(turn)
    status = str(span.get("evidence_status", ""))
    if status not in SUPPORTED_STATUSES:
        raise RuntimeError(
            f"{cid}: turn {tid} semantic mask has unsupported raw status {status!r}"
        )
    if span.get("supervision_tier") != "ignore":
        raise RuntimeError(
            f"{cid}: turn {tid} semantic mask still has positive supervision tier"
        )
    if span.get("causal_type") != "unvalidated":
        raise RuntimeError(
            f"{cid}: turn {tid} semantic mask still has causal attribution type"
        )


def classify_masked_turn(record: Dict, turn: Dict) -> Dict:
    """Classify one turn containing at least one semantically masked span."""
    cid = _cid(record)
    tid = _turn_id(turn)

    masked = [
        span
        for span in turn.get("span_annotations", []) or []
        if is_masked_reviewed_span(span)
    ]
    if not masked:
        raise RuntimeError(f"{cid}: turn {tid} has no semantic masks to classify")
    for span in masked:
        _validate_masked_span(record, turn, span)

    evidence_ids_raw = record.get("evidence_turn_ids", []) or []
    evidence_ids = set()
    for raw in evidence_ids_raw:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise RuntimeError(
                f"{cid}: evidence_turn_ids contains non-integer value {raw!r}"
            )
        evidence_ids.add(raw)

    analysis_ids_raw = (
        (record.get("frontier_evidence_analysis", {}) or {})
        .get("evidence_turn_ids", evidence_ids_raw)
        or []
    )
    analysis_ids = set()
    for raw in analysis_ids_raw:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise RuntimeError(
                f"{cid}: frontier_evidence_analysis.evidence_turn_ids "
                f"contains non-integer value {raw!r}"
            )
        analysis_ids.add(raw)
    if evidence_ids != analysis_ids:
        raise RuntimeError(
            f"{cid}: record evidence_turn_ids={sorted(evidence_ids)} differs "
            f"from frontier_evidence_analysis.evidence_turn_ids="
            f"{sorted(analysis_ids)}"
        )

    turn_status = _turn_intervention_status(record, tid)
    whole_turn_supported = turn_status in SUPPORTED_STATUSES
    whole_turn_negative = turn_status == "not_supported"

    independent_spans = [
        span
        for span in turn.get("span_annotations", []) or []
        if is_eligible_nonmasked_positive_span(span)
    ]
    has_independent_span = bool(independent_spans)
    in_evidence_ids = tid in evidence_ids

    if not in_evidence_ids:
        case = CASE_D
        if whole_turn_supported or has_independent_span:
            raise RuntimeError(
                f"{cid}: turn {tid} has independent supported evidence but is "
                "missing from evidence_turn_ids"
            )
    elif whole_turn_supported or has_independent_span:
        case = CASE_C
    elif whole_turn_negative:
        case = CASE_A
    else:
        case = CASE_B

    return {
        "conversation_id": cid,
        "turn_id": tid,
        "in_evidence_turn_ids": in_evidence_ids,
        "whole_turn_status": turn_status,
        "whole_turn_supported": whole_turn_supported,
        "whole_turn_negative": whole_turn_negative,
        "masked_span_count": len(masked),
        "masked_span_statuses": sorted(
            str(span.get("evidence_status", "")) for span in masked
        ),
        "eligible_nonmasked_positive_span_count": len(independent_spans),
        "eligible_nonmasked_positive_span_statuses": sorted(
            str(span.get("evidence_status", "")) for span in independent_spans
        ),
        "case": case,
        "recommended_turn_effect": (
            "remove_positive_membership_and_keep_explicit_negative"
            if case == CASE_A
            else "requires_policy_review_before_repair"
            if case == CASE_B
            else "retain_positive_membership"
            if case == CASE_C
            else "no_turn_membership_change"
        ),
    }


def inspect_splits(
    splits: Mapping[str, Iterable[Dict]],
    *,
    enforce_reviewed_counts: bool = False,
) -> Dict:
    """Inspect all semantic masks in the supplied non-test splits."""
    details: List[Dict] = []
    masked_span_count = 0
    masked_record_ids = set()
    masked_turn_keys = set()

    for split_name, records in splits.items():
        if split_name == "test":
            raise RuntimeError("held-out test split must not be supplied")
        for record in records:
            cid = _cid(record)
            for turn in record.get("turns", []) or []:
                masked = [
                    span
                    for span in turn.get("span_annotations", []) or []
                    if is_masked_reviewed_span(span)
                ]
                if not masked:
                    continue
                tid = _turn_id(turn)
                masked_span_count += len(masked)
                masked_record_ids.add(cid)
                key = (cid, tid)
                if key in masked_turn_keys:
                    raise RuntimeError(
                        f"{cid}: turn {tid} encountered twice while inspecting splits"
                    )
                masked_turn_keys.add(key)
                item = classify_masked_turn(record, turn)
                item["split"] = split_name
                item["masked_spans"] = [
                    {
                        "char_start": span.get("char_start"),
                        "char_end": span.get("char_end"),
                        "text": str(span.get("text", "")),
                        "evidence_status": span.get("evidence_status"),
                        "counterfactual_delta": span.get("counterfactual_delta"),
                        "semantic_original_supervision_tier": span.get(
                            "semantic_original_supervision_tier"
                        ),
                    }
                    for span in masked
                ]
                details.append(item)

    if enforce_reviewed_counts:
        if masked_span_count != EXPECTED_PRIMARY_MASKED_SPANS:
            raise RuntimeError(
                f"semantic masked span count changed: {masked_span_count} "
                f"!= {EXPECTED_PRIMARY_MASKED_SPANS}"
            )
        if len(masked_record_ids) != EXPECTED_PRIMARY_MASKED_RECORDS:
            raise RuntimeError(
                f"semantic masked record count changed: {len(masked_record_ids)} "
                f"!= {EXPECTED_PRIMARY_MASKED_RECORDS}"
            )

    case_turns = Counter(item["case"] for item in details)
    case_spans = Counter()
    for item in details:
        case_spans[item["case"]] += int(item["masked_span_count"])

    # A/B are findings, not structural audit failures. The purpose of this
    # module is to quantify them before any data mutation.
    repair_candidates = [
        item for item in details if item["case"] in {CASE_A, CASE_B}
    ]

    return {
        "status": "inspection_complete",
        "held_out_test_accessed": False,
        "adjudication_version": ADJUDICATION_VERSION,
        "masked_spans": masked_span_count,
        "masked_turns": len(masked_turn_keys),
        "masked_records": len(masked_record_ids),
        "case_descriptions": CASE_DESCRIPTIONS,
        "case_counts_by_turn": {
            key: case_turns.get(key, 0) for key in CASE_DESCRIPTIONS
        },
        "case_counts_by_span": {
            key: case_spans.get(key, 0) for key in CASE_DESCRIPTIONS
        },
        "repair_candidate_turns": len(repair_candidates),
        "silent_fallback_case_B_turns": case_turns.get(CASE_B, 0),
        "details": sorted(
            details,
            key=lambda item: (
                item["split"],
                item["conversation_id"],
                item["turn_id"],
            ),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output")
    parser.add_argument("--enforce-reviewed-counts", action="store_true")
    args = parser.parse_args()

    report = inspect_splits(
        {
            "train": load_jsonl(args.train),
            "dev": load_jsonl(args.dev),
        },
        enforce_reviewed_counts=args.enforce_reviewed_counts,
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
