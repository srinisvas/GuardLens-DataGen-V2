#!/usr/bin/env python3
"""Fail-closed audit for canonical frontier-authored GuardLens records."""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter

from frontier_common import load_jsonl

SUPPORTED = {"supported_strong", "supported_weak"}


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    errors = []
    seen = set()
    tiers = Counter()
    pivots = Counter()

    for r in records:
        cid = str(r.get("conversation_id", ""))
        if not cid:
            errors.append("record missing conversation_id")
        elif cid in seen:
            errors.append(f"duplicate conversation_id: {cid}")
        seen.add(cid)

        pair_id = r.get("pair_id")
        if pair_id == "":
            errors.append(f"{cid}: empty-string pair_id")
        metadata = r.get("metadata", {}) or {}
        scenario = metadata.get("scenario_family")
        if not scenario:
            errors.append(f"{cid}: missing scenario_family")
        if metadata.get("split_group") != scenario:
            errors.append(f"{cid}: split_group differs from scenario_family")
        intended = r.get("intended_structure", {}) or {}
        if intended.get("annotation_status") != "generation_intent_only_not_ground_truth":
            errors.append(f"{cid}: authoring intent metadata lost its non-ground-truth marker")

        if not r.get("training_eligible", False):
            errors.append(f"{cid}: ineligible record present in prepared dataset")
        tier = r.get("supervision_tier")
        tiers[tier] += 1
        if tier is None or tier == "ignore":
            errors.append(f"{cid}: unresolved/ignored supervision tier in prepared dataset")
        if not is_number(r.get("loss_weight")) or float(r.get("loss_weight", 0)) <= 0:
            errors.append(f"{cid}: invalid prepared loss_weight={r.get('loss_weight')!r}")

        label = r.get("label")
        evidence_turns = sorted({int(x) for x in r.get("evidence_turn_ids", [])})
        pivot = r.get("pivot_turn_id")
        ignore = bool(r.get("pivot_supervision_ignore", False))

        if label == 1:
            analysis = r.get("frontier_evidence_analysis", {}) or {}
            if r.get("validation_status") != "validated":
                errors.append(f"{cid}: malicious prepared record is not independently validated")
            if analysis.get("status") != "complete":
                errors.append(f"{cid}: malicious prepared record lacks complete paired evidence baseline")
            if evidence_turns:
                pivots["malicious_supported"] += 1
                if pivot != evidence_turns[0]:
                    errors.append(f"{cid}: pivot is not earliest supported evidence turn")
                if ignore:
                    errors.append(f"{cid}: supported malicious pivot is ignored")
            else:
                pivots["malicious_unknown_ignored"] += 1
                if pivot is not None:
                    errors.append(f"{cid}: unsupported malicious pivot is non-null")
                if not ignore:
                    errors.append(f"{cid}: unknown malicious pivot must be ignored")
        elif label == 0:
            pivots["benign_true_no_pivot"] += 1
            if r.get("validation_status") != "validated":
                errors.append(f"{cid}: benign prepared record is not independently validated")
            if pivot is not None or evidence_turns:
                errors.append(f"{cid}: benign prepared record contains evidence pivot")
            if ignore:
                errors.append(f"{cid}: benign true no-pivot is incorrectly ignored")
            if tier != "benign_validated":
                errors.append(f"{cid}: benign tier is {tier!r}, expected benign_validated")
        else:
            errors.append(f"{cid}: unsupported label={label!r}")

        for turn in r.get("turns", []):
            text = str(turn.get("text", ""))
            for span in turn.get("span_annotations", []):
                status = str(span.get("evidence_status", "unassessed"))
                delta = span.get("counterfactual_delta")
                start, end = span.get("char_start"), span.get("char_end")
                span_text = str(span.get("text", ""))
                if isinstance(start, int) and isinstance(end, int) and span_text:
                    if not (0 <= start < end <= len(text)) or text[start:end] != span_text:
                        errors.append(f"{cid}: stale/misaligned span offsets")
                if label == 1 and status in SUPPORTED:
                    if span.get("causal_type") != "causal":
                        errors.append(f"{cid}: supported span not causal")
                    if span.get("supervision_tier") not in {"cf_strong", "cf_weak"}:
                        errors.append(f"{cid}: supported span lacks evidence supervision tier")
                    if not is_number(delta):
                        errors.append(f"{cid}: supported span has nonnumeric delta")
                elif label == 1 and status != "negative_control_supported":
                    if span.get("causal_type") == "causal":
                        errors.append(f"{cid}: unsupported span visible as causal")
                    if span.get("supervision_tier") != "ignore":
                        errors.append(f"{cid}: unsupported malicious span tier is not ignore")

    print("=== Frontier prepared dataset audit ===")
    print(f"Records: {len(records)}")
    print(f"Labels: {dict(Counter(r.get('label') for r in records))}")
    print(f"Supervision tiers: {dict(tiers)}")
    print(f"Pivot supervision modes: {dict(pivots)}")
    print(f"Scenario families: {len({(r.get('metadata',{}) or {}).get('scenario_family') for r in records})}")
    if errors:
        print("VALIDITY AUDIT FAILED", file=sys.stderr)
        for error in errors[:100]:
            print(f"ERROR: {error}", file=sys.stderr)
        if len(errors) > 100:
            print(f"... {len(errors)-100} additional errors", file=sys.stderr)
        sys.exit(2)
    print("VALIDITY AUDIT PASSED")


if __name__ == "__main__":
    main()
