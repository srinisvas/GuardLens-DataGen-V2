#!/usr/bin/env python3
"""Stage B3: materialize deterministic intervention candidates.

Authoring metadata may propose where to test but never supplies the answer.
All created spans are unvalidated candidates. Final evidence labels are assigned
only by paired counterfactual replay in ``frontier_evidence_analysis.py``.
"""
from __future__ import annotations

import argparse
import copy
import re
from collections import Counter
from typing import Dict, List, Tuple

from frontier_common import load_jsonl, write_jsonl

MIN_SPAN_CHARS = 18
MIN_SPAN_WORDS = 4


def user_turns(record: Dict) -> List[Dict]:
    return [t for t in record.get("turns", []) if str(t.get("role", "")).lower() == "user"]


def _trimmed_span(text: str, start: int, end: int):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end - start < MIN_SPAN_CHARS:
        return None
    if len(text[start:end].split()) < MIN_SPAN_WORDS:
        return None
    return start, end


def extract_candidate_spans(text: str, max_spans: int = 2) -> List[Tuple[int, int]]:
    candidates: List[Tuple[int, int]] = []
    for match in re.finditer(r"[^,;:!?\.]+(?:[,;:!?\.]|$)", text):
        span = _trimmed_span(text, match.start(), match.end())
        if span:
            candidates.append(span)
    if not candidates:
        for match in re.finditer(r"[^.!?]+(?:[.!?]|$)", text):
            span = _trimmed_span(text, match.start(), match.end())
            if span:
                candidates.append(span)
    if not candidates and len(text.split()) >= MIN_SPAN_WORDS:
        words = list(re.finditer(r"\S+", text))
        start_word = max(0, len(words) - min(12, len(words)))
        span = _trimmed_span(text, words[start_word].start(), words[-1].end())
        if span:
            candidates.append(span)
    unique = sorted(set(candidates), key=lambda x: (-x[1], -(x[1] - x[0])))
    return unique[:max_spans]


def candidate_turn_ids(record: Dict, max_turns: int = 4):
    users = user_turns(record)
    valid_ids = [int(t.get("turn_id", -1)) for t in users]
    observed_anchor = (record.get("frontier_behavioral_validation", {}) or {}).get(
        "earliest_unsafe_user_turn_id"
    )
    if observed_anchor is None:
        return [], {}
    observed_anchor = int(observed_anchor)
    origins = {observed_anchor: "observed_unsafe_anchor"}
    ordered = [observed_anchor]

    intended = record.get("intended_structure", {}) or {}
    for raw in intended.get("expected_pivot_turn_ids", []) or []:
        try:
            tid = int(raw)
        except (TypeError, ValueError):
            continue
        if tid in valid_ids and tid <= observed_anchor and tid not in origins:
            origins[tid] = "author_intended_candidate_only"
            ordered.append(tid)

    for tid in sorted((x for x in valid_ids if x < observed_anchor), reverse=True):
        if tid not in origins:
            origins[tid] = "near_anchor_context_candidate"
            ordered.append(tid)
        if len(ordered) >= max_turns:
            break
    return ordered[:max_turns], origins


def materialize_record(record: Dict, *, max_turn_candidates: int, spans_per_turn: int, controls: int) -> Dict:
    r = copy.deepcopy(record)
    if r.get("label") != 1 or r.get("validation_status") != "validated":
        r["candidate_materialization"] = {"status": "not_applicable"}
        return r

    turn_ids, origins = candidate_turn_ids(r, max_turn_candidates)
    turn_map = {int(t.get("turn_id", -1)): t for t in user_turns(r)}
    span_count = 0
    for tid in turn_ids:
        turn = turn_map.get(tid)
        if not turn:
            continue
        text = str(turn.get("text", ""))
        existing = list(turn.get("span_annotations", []))
        for ordinal, (start, end) in enumerate(extract_candidate_spans(text, spans_per_turn)):
            existing.append({
                "label": "EVIDENCE_CANDIDATE",
                "text": text[start:end],
                "char_start": start,
                "char_end": end,
                "causal_type": "unvalidated",
                "supervision_tier": None,
                "counterfactual_delta": None,
                "evidence_status": "unassessed",
                "candidate_origin": origins.get(tid, "candidate"),
                "candidate_rank_within_turn": ordinal,
                "annotation_status": "candidate_only_not_ground_truth",
            })
            span_count += 1
        turn["span_annotations"] = existing

    control_turn_ids = []
    if turn_ids:
        anchor = turn_ids[0]
        pool = [
            int(t.get("turn_id", -1))
            for t in user_turns(r)
            if int(t.get("turn_id", -1)) < anchor and int(t.get("turn_id", -1)) not in turn_ids
        ]
        for tid in pool[:controls]:
            turn = turn_map[tid]
            text = str(turn.get("text", ""))
            spans = extract_candidate_spans(text, 1)
            if not spans:
                continue
            start, end = spans[0]
            annotations = list(turn.get("span_annotations", []))
            annotations.append({
                "label": "CONTROL_CANDIDATE",
                "text": text[start:end],
                "char_start": start,
                "char_end": end,
                "causal_type": "unvalidated",
                "supervision_tier": None,
                "counterfactual_delta": None,
                "evidence_status": "unassessed",
                "candidate_origin": "non_candidate_context_control",
                "annotation_status": "candidate_only_not_ground_truth",
            })
            turn["span_annotations"] = annotations
            control_turn_ids.append(tid)

    r["candidate_materialization"] = {
        "status": "complete",
        "protocol": "frontier_candidate_materialization_v1",
        "candidate_turn_ids": turn_ids,
        "candidate_turn_origins": {str(k): v for k, v in origins.items() if k in turn_ids},
        "control_turn_ids": control_turn_ids,
        "evidence_candidate_spans": span_count,
        "author_intended_pivots_used_as_ground_truth": False,
        "author_intended_pivots_used_for_candidate_proposal": True,
    }
    r["source_stage"] = "intervention_candidates_materialized"
    r["use_as"] = "input_for_paired_counterfactual_evidence"
    return r


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-turn-candidates", type=int, default=4)
    parser.add_argument("--spans-per-turn", type=int, default=2)
    parser.add_argument("--controls", type=int, default=2)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    output = [
        materialize_record(
            r,
            max_turn_candidates=args.max_turn_candidates,
            spans_per_turn=args.spans_per_turn,
            controls=args.controls,
        )
        for r in records
    ]
    write_jsonl(output, args.output)
    statuses = Counter((r.get("candidate_materialization", {}) or {}).get("status", "missing") for r in output)
    span_counts = Counter()
    for r in output:
        for t in r.get("turns", []):
            for s in t.get("span_annotations", []):
                span_counts[s.get("label", "unknown")] += 1
    print(f"Candidate materialization: {dict(statuses)}")
    print(f"Candidate spans: {dict(span_counts)}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
