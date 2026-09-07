#!/usr/bin/env python3
"""CPU-only invariant audit for the repaired NAACL dataset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from typing import Dict, Iterable, List

SUPPORTED = {"supported_strong", "supported_weak"}
NOT_MEASURED = {
    "unassessed",
    "not_assessable_preexisting_unsafe",
    "not_assessable_baseline_not_unsafe",
    "not_assessable_span_alignment",
    "not_assessable_missing_turn",
}
MEASURED = SUPPORTED | {"not_supported", "negative_control_supported", "negative_control_violated"}


def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return out


def n_user(record: Dict) -> int:
    return sum(1 for t in record.get("turns", []) if str(t.get("role", "")).lower() == "user")


def iter_spans(record: Dict) -> Iterable[Dict]:
    for turn in record.get("turns", []):
        for span in turn.get("span_annotations", []):
            yield span


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--require-prepared", action="store_true")
    parser.add_argument("--max-length-gap", type=float, default=1.5)
    parser.add_argument("--weak-delta", type=float, default=0.25)
    parser.add_argument("--strong-delta", type=float, default=0.40)
    parser.add_argument("--negative-control-tolerance", type=float, default=0.15)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    errors, warnings = [], []
    evidence_statuses, tiers, analysis_statuses = Counter(), Counter(), Counter()
    seen_ids = set()

    for record in records:
        cid = str(record.get("conversation_id", ""))
        is_malicious = record.get("label") == 1
        if not cid:
            errors.append("record missing conversation_id")
        elif cid in seen_ids:
            errors.append(f"duplicate conversation_id: {cid}")
        seen_ids.add(cid)

        tiers[record.get("supervision_tier", "unknown")] += 1
        analysis_status = str(record.get("evidence_analysis", {}).get("status", "missing"))
        analysis_statuses[analysis_status] += 1
        if is_malicious and record.get("validation_status") == "validated":
            if analysis_status == "error" and record.get("training_eligible", True):
                errors.append(f"{cid}: evidence error record is still training eligible")
            if analysis_status == "missing":
                errors.append(f"{cid}: validated malicious record has no evidence analysis status")

        evidence_turns = set(int(x) for x in record.get("evidence_turn_ids", []))
        pivot = record.get("pivot_turn_id")
        if pivot is not None and int(pivot) not in evidence_turns:
            errors.append(f"{cid}: pivot_turn_id {pivot} is not supported by evidence_turn_ids")

        for span in iter_spans(record):
            status = str(span.get("evidence_status", "unassessed"))
            delta = span.get("counterfactual_delta")
            if is_malicious:
                evidence_statuses[status] += 1
                if status in NOT_MEASURED and delta is not None:
                    errors.append(f"{cid}: status={status} must have null delta, got {delta!r}")
                if status in MEASURED and not is_number(delta):
                    errors.append(f"{cid}: measured status={status} has non-numeric delta={delta!r}")
                if status == "supported_strong":
                    if is_number(delta) and delta < args.strong_delta:
                        errors.append(f"{cid}: strong evidence below threshold: {delta}")
                    if span.get("causal_type") != "causal":
                        errors.append(f"{cid}: strong supported span not legacy causal-compatible")
                if status == "supported_weak":
                    if is_number(delta) and not (args.weak_delta <= delta < args.strong_delta):
                        errors.append(f"{cid}: weak evidence outside threshold band: {delta}")
                    if span.get("causal_type") != "causal":
                        errors.append(f"{cid}: weak supported span not legacy causal-compatible")
                if status == "negative_control_supported":
                    if is_number(delta) and abs(delta) >= args.negative_control_tolerance:
                        errors.append(f"{cid}: negative control effect too large: {delta}")
                    if span.get("causal_type") != "incidental":
                        errors.append(f"{cid}: supported negative control not incidental")

                if args.require_prepared:
                    label = span.get("label", "")
                    if status not in SUPPORTED and status != "negative_control_supported":
                        if label != "EVIDENCE_CANDIDATE":
                            errors.append(f"{cid}: unestablished span visible as {label!r} after prepare")
                        if span.get("supervision_tier") != "ignore":
                            errors.append(f"{cid}: unestablished span tier is not ignore after prepare")

    malicious = [r for r in records if r.get("label") == 1]
    benign = [r for r in records if r.get("label") == 0]
    mal_lengths = [n_user(r) for r in malicious]
    ben_lengths = [n_user(r) for r in benign]
    mal_mean = sum(mal_lengths) / len(mal_lengths) if mal_lengths else 0.0
    ben_mean = sum(ben_lengths) / len(ben_lengths) if ben_lengths else 0.0
    gap = abs(mal_mean - ben_mean)

    if args.require_prepared and malicious and benign and gap > args.max_length_gap:
        errors.append(f"mean user-turn gap {gap:.3f} exceeds {args.max_length_gap:.3f}")
    elif malicious and benign and gap > args.max_length_gap:
        warnings.append(f"mean user-turn gap is {gap:.3f}; prepare length matching before training")

    print("=== NAACL repaired dataset audit ===")
    print(f"Records: {len(records)}")
    print(f"Malicious: {len(malicious)}  Benign: {len(benign)}")
    print(f"Mean user turns: malicious={mal_mean:.3f} benign={ben_mean:.3f} gap={gap:.3f}")
    print(f"Evidence-analysis statuses: {dict(analysis_statuses)}")
    print(f"Malicious span statuses: {dict(evidence_statuses)}")
    print(f"Supervision tiers: {dict(tiers)}")

    for warning in warnings:
        print(f"WARN: {warning}")
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
