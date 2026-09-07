#!/usr/bin/env python3
"""CPU-only invariant audit for the repaired NAACL dataset.

Run this after ``evidence_analysis.py`` and again after ``prepare_dataset.py``.
The command exits non-zero on validity violations that would make a paper rerun
unsafe to interpret.
"""

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
    "not_assessable_span_alignment",
}


def load_jsonl(path: str) -> List[Dict]:
    out: List[Dict] = []
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
    return sum(
        1 for t in record.get("turns", [])
        if str(t.get("role", "")).lower() == "user"
    )


def iter_spans(record: Dict) -> Iterable[Dict]:
    for turn in record.get("turns", []):
        for span in turn.get("span_annotations", []):
            yield span


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--require-prepared",
        action="store_true",
        help="Also enforce the post-prepare EVIDENCE_CANDIDATE masking invariants",
    )
    parser.add_argument(
        "--max-length-gap",
        type=float,
        default=1.5,
        help="Maximum allowed absolute malicious/benign mean user-turn gap",
    )
    args = parser.parse_args()

    records = load_jsonl(args.input)
    errors: List[str] = []
    warnings: List[str] = []
    evidence_statuses: Counter = Counter()
    tiers: Counter = Counter()

    seen_ids = set()
    for record in records:
        cid = str(record.get("conversation_id", ""))
        if not cid:
            errors.append("record missing conversation_id")
        elif cid in seen_ids:
            errors.append(f"duplicate conversation_id: {cid}")
        seen_ids.add(cid)

        tiers[record.get("supervision_tier", "unknown")] += 1
        evidence_turns = set(int(x) for x in record.get("evidence_turn_ids", []))
        pivot = record.get("pivot_turn_id")
        if pivot is not None and evidence_turns and int(pivot) not in evidence_turns:
            errors.append(f"{cid}: pivot_turn_id {pivot} is not in evidence_turn_ids")
        if pivot is not None and not evidence_turns and record.get("label") == 1:
            errors.append(f"{cid}: malicious pivot exists without supported evidence turn")

        for span in iter_spans(record):
            status = str(span.get("evidence_status", "unassessed"))
            evidence_statuses[status] += 1
            delta = span.get("counterfactual_delta")

            if status in SUPPORTED:
                if not is_number(delta):
                    errors.append(f"{cid}: {status} span has non-numeric delta={delta!r}")
                if span.get("causal_type") != "causal":
                    errors.append(f"{cid}: supported span is not legacy causal-compatible")
            elif status in NOT_MEASURED:
                if delta is not None:
                    errors.append(
                        f"{cid}: unmeasured span status={status} must have null delta, got {delta!r}"
                    )

            if status == "negative_control_supported":
                if span.get("causal_type") != "incidental":
                    errors.append(f"{cid}: supported negative control is not incidental")
                if not is_number(delta):
                    errors.append(f"{cid}: supported negative control has no measured delta")

            if args.require_prepared:
                label = span.get("label", "")
                if status not in SUPPORTED and status != "negative_control_supported":
                    if label != "EVIDENCE_CANDIDATE":
                        errors.append(
                            f"{cid}: unestablished span remains visible as label={label!r} after prepare"
                        )
                    if span.get("supervision_tier") != "ignore":
                        errors.append(
                            f"{cid}: unestablished span tier must be ignore after prepare"
                        )

    malicious = [r for r in records if r.get("label") == 1]
    benign = [r for r in records if r.get("label") == 0]
    mal_lengths = [n_user(r) for r in malicious]
    ben_lengths = [n_user(r) for r in benign]

    mal_mean = sum(mal_lengths) / len(mal_lengths) if mal_lengths else 0.0
    ben_mean = sum(ben_lengths) / len(ben_lengths) if ben_lengths else 0.0
    gap = abs(mal_mean - ben_mean)

    if args.require_prepared and malicious and benign and gap > args.max_length_gap:
        errors.append(
            f"mean user-turn length gap remains {gap:.3f}, above {args.max_length_gap:.3f}"
        )
    elif malicious and benign and gap > args.max_length_gap:
        warnings.append(
            f"mean user-turn length gap is {gap:.3f}; run prepare_dataset.py before training"
        )

    print("=== NAACL repaired dataset audit ===")
    print(f"Records: {len(records)}")
    print(f"Malicious: {len(malicious)}  Benign: {len(benign)}")
    print(f"Mean user turns: malicious={mal_mean:.3f} benign={ben_mean:.3f} gap={gap:.3f}")
    print(f"Evidence statuses: {dict(evidence_statuses)}")
    print(f"Supervision tiers: {dict(tiers)}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  WARN: {warning}")

    if errors:
        print("\nVALIDITY AUDIT FAILED", file=sys.stderr)
        for error in errors[:100]:
            print(f"  ERROR: {error}", file=sys.stderr)
        if len(errors) > 100:
            print(f"  ... {len(errors) - 100} additional errors", file=sys.stderr)
        sys.exit(2)

    print("\nVALIDITY AUDIT PASSED")


if __name__ == "__main__":
    main()
