#!/usr/bin/env python3
"""Fail-closed audit for canonical frontier-authored GuardLens primary records.

The audit reconstructs evidence supervision from the stored intervention results
rather than trusting derived fields such as ``evidence_turn_ids`` or the record-
level supervision tier. It also rechecks pair semantics after rollout/evidence
processing so no schema transformation can silently break the authored twin
construction.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict

from frontier_common import load_jsonl

SUPPORTED = {"supported_strong", "supported_weak"}
EXPECTED_WEIGHTS = {
    "cf_strong": 1.00,
    "cf_weak": 0.70,
    "llm_confirmed": 0.60,
    "benign_validated": 1.00,
}
DEFAULT_TARGET = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_JUDGE = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


def is_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def user_turns(record):
    return [
        t for t in record.get("turns", [])
        if str(t.get("role", "")).lower() == "user"
    ]


def n_user(record) -> int:
    return len(user_turns(record))


def all_spans_with_turn(record):
    for turn in record.get("turns", []):
        if str(turn.get("role", "")).lower() != "user":
            continue
        tid = int(turn.get("turn_id", -1))
        for span in turn.get("span_annotations", []):
            yield tid, turn, span


def expected_record_tier(record) -> str:
    statuses = [
        str(span.get("evidence_status", "unassessed"))
        for _, _, span in all_spans_with_turn(record)
    ]
    if "supported_strong" in statuses:
        return "cf_strong"
    if "supported_weak" in statuses:
        return "cf_weak"
    return "llm_confirmed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-target-model", default=DEFAULT_TARGET)
    parser.add_argument("--expected-judge-model", default=DEFAULT_JUDGE)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    errors = []
    seen = set()
    tiers = Counter()
    pivots = Counter()
    pair_groups = defaultdict(list)

    for r in records:
        cid = str(r.get("conversation_id", ""))
        if not cid:
            errors.append("record missing conversation_id")
        elif cid in seen:
            errors.append(f"duplicate conversation_id: {cid}")
        seen.add(cid)

        pair_id = r.get("pair_id")
        if pair_id in (None, ""):
            errors.append(f"{cid}: primary frontier record must belong to a retained twin pair")
        else:
            pair_groups[str(pair_id)].append(r)
        if r.get("primary_pair_complete") is not True:
            errors.append(f"{cid}: primary_pair_complete marker is not true")

        metadata = r.get("metadata", {}) or {}
        scenario = metadata.get("scenario_family")
        if not scenario:
            errors.append(f"{cid}: missing scenario_family")
        if metadata.get("split_group") != scenario:
            errors.append(f"{cid}: split_group differs from scenario_family")
        intended = r.get("intended_structure", {}) or {}
        if intended.get("annotation_status") != "generation_intent_only_not_ground_truth":
            errors.append(f"{cid}: authoring intent metadata lost its non-ground-truth marker")
        if r.get("authoring_intent_label") != r.get("label"):
            errors.append(f"{cid}: authoring intent provenance no longer matches retained class")

        rollout = r.get("rollout_provenance", {}) or {}
        validation = r.get("frontier_behavioral_validation", {}) or {}
        if rollout.get("target_model") != args.expected_target_model:
            errors.append(f"{cid}: unexpected primary rollout target")
        if validation.get("judge_model") != args.expected_judge_model:
            errors.append(f"{cid}: unexpected primary validation judge")
        if r.get("canonical_target_model") != args.expected_target_model:
            errors.append(f"{cid}: canonical target model marker mismatch")
        if r.get("canonical_judge_model") != args.expected_judge_model:
            errors.append(f"{cid}: canonical judge model marker mismatch")
        if rollout.get("authoring_metadata_exposed_to_target") is not False:
            errors.append(f"{cid}: target metadata-exposure provenance is not false")
        if validation.get("authoring_metadata_exposed_to_judge") is not False:
            errors.append(f"{cid}: judge metadata-exposure provenance is not false")

        if not r.get("training_eligible", False):
            errors.append(f"{cid}: ineligible record present in prepared dataset")
        tier = r.get("supervision_tier")
        tiers[tier] += 1
        if tier not in EXPECTED_WEIGHTS:
            errors.append(f"{cid}: unresolved/invalid supervision tier {tier!r}")
        weight = r.get("loss_weight")
        if not is_number(weight) or float(weight) <= 0:
            errors.append(f"{cid}: invalid prepared loss_weight={weight!r}")
        elif tier in EXPECTED_WEIGHTS and not math.isclose(
            float(weight), EXPECTED_WEIGHTS[tier], abs_tol=1e-9
        ):
            errors.append(f"{cid}: loss_weight={weight!r} does not match tier {tier!r}")

        label = r.get("label")
        evidence_turns = sorted({int(x) for x in r.get("evidence_turn_ids", [])})
        pivot = r.get("pivot_turn_id")
        ignore = bool(r.get("pivot_supervision_ignore", False))
        physical_user_ids = {int(t.get("turn_id", -1)) for t in user_turns(r)}

        if label == 1:
            analysis = r.get("frontier_evidence_analysis", {}) or {}
            materialization = r.get("candidate_materialization", {}) or {}
            if r.get("validation_status") != "validated":
                errors.append(f"{cid}: malicious prepared record is not independently validated")
            if analysis.get("status") != "complete":
                errors.append(f"{cid}: malicious prepared record lacks complete paired evidence baseline")
            if analysis.get("baseline_reproduced_stored_rollout") is not True:
                errors.append(f"{cid}: malicious paired baseline did not reproduce stored rollout")
            if analysis.get("target_model") != args.expected_target_model:
                errors.append(f"{cid}: malicious evidence target mismatch")
            if analysis.get("judge_model") != args.expected_judge_model:
                errors.append(f"{cid}: malicious evidence judge mismatch")
            if analysis.get("author_intended_pivots_used_as_ground_truth") is not False:
                errors.append(f"{cid}: author-intended pivots leaked into evidence ground truth")
            if materialization.get("author_intended_pivots_used_as_ground_truth") is not False:
                errors.append(f"{cid}: candidate materialization ground-truth marker is not false")

            anchor = analysis.get("fresh_anchor_turn_id")
            try:
                anchor_int = int(anchor)
            except (TypeError, ValueError):
                anchor_int = None
            if anchor_int not in physical_user_ids:
                errors.append(f"{cid}: fresh unsafe anchor is not a realized user turn")

            try:
                weak_thr = float(analysis.get("weak_delta_threshold"))
                strong_thr = float(analysis.get("strong_delta_threshold"))
                control_tol = float(analysis.get("negative_control_tolerance"))
            except (TypeError, ValueError):
                weak_thr = strong_thr = control_tol = math.nan
                errors.append(f"{cid}: evidence thresholds are missing/non-numeric")

            reconstructed_turns = set()
            for intervention in analysis.get("turn_interventions", []) or []:
                status = str(intervention.get("status", ""))
                tid = int(intervention.get("turn_id", -1))
                delta = intervention.get("delta")
                if tid not in physical_user_ids:
                    errors.append(f"{cid}: turn intervention references missing user turn {tid}")
                if status in SUPPORTED:
                    reconstructed_turns.add(tid)
                    if not is_number(delta):
                        errors.append(f"{cid}: supported turn intervention has nonnumeric delta")
                    elif status == "supported_strong" and float(delta) < strong_thr:
                        errors.append(f"{cid}: strong turn intervention below strong threshold")
                    elif status == "supported_weak" and not (
                        weak_thr <= float(delta) < strong_thr
                    ):
                        errors.append(f"{cid}: weak turn intervention inconsistent with thresholds")

            strong_span = False
            weak_span = False
            for tid, turn, span in all_spans_with_turn(r):
                text = str(turn.get("text", ""))
                status = str(span.get("evidence_status", "unassessed"))
                delta = span.get("counterfactual_delta")
                start, end = span.get("char_start"), span.get("char_end")
                span_text = str(span.get("text", ""))
                if isinstance(start, int) and isinstance(end, int) and span_text:
                    if not (0 <= start < end <= len(text)) or text[start:end] != span_text:
                        errors.append(f"{cid}: stale/misaligned span offsets")

                if status in SUPPORTED:
                    reconstructed_turns.add(tid)
                    strong_span = strong_span or status == "supported_strong"
                    weak_span = weak_span or status == "supported_weak"
                    if span.get("causal_type") != "causal":
                        errors.append(f"{cid}: supported span not causal")
                    expected_span_tier = "cf_strong" if status == "supported_strong" else "cf_weak"
                    if span.get("supervision_tier") != expected_span_tier:
                        errors.append(f"{cid}: supported span supervision tier disagrees with status")
                    if not is_number(delta):
                        errors.append(f"{cid}: supported span has nonnumeric delta")
                    elif status == "supported_strong" and float(delta) < strong_thr:
                        errors.append(f"{cid}: strong span below strong threshold")
                    elif status == "supported_weak" and not (
                        weak_thr <= float(delta) < strong_thr
                    ):
                        errors.append(f"{cid}: weak span inconsistent with thresholds")
                elif status == "negative_control_supported":
                    if span.get("causal_type") != "incidental" or span.get("supervision_tier") != "incidental":
                        errors.append(f"{cid}: supported negative control not marked incidental")
                    if not is_number(delta) or not abs(float(delta)) < control_tol:
                        errors.append(f"{cid}: supported negative control violates tolerance")
                else:
                    if span.get("causal_type") == "causal":
                        errors.append(f"{cid}: unsupported span visible as causal")
                    if span.get("supervision_tier") != "ignore":
                        errors.append(f"{cid}: unsupported malicious span tier is not ignore")
                    if status == "negative_control_violated" and is_number(delta):
                        if abs(float(delta)) < control_tol:
                            errors.append(f"{cid}: violated negative control is actually within tolerance")

            reconstructed = sorted(reconstructed_turns)
            if reconstructed != evidence_turns:
                errors.append(
                    f"{cid}: evidence_turn_ids {evidence_turns} != reconstructed supported turns {reconstructed}"
                )
            analysis_turns = sorted({int(x) for x in analysis.get("evidence_turn_ids", [])})
            if analysis_turns != evidence_turns:
                errors.append(f"{cid}: record and analysis evidence_turn_ids disagree")
            if anchor_int is not None and any(tid > anchor_int for tid in evidence_turns):
                errors.append(f"{cid}: evidence turn occurs after fresh unsafe anchor")

            expected_tier = "cf_strong" if strong_span else "cf_weak" if weak_span else "llm_confirmed"
            if tier != expected_tier:
                errors.append(
                    f"{cid}: record supervision tier {tier!r} != evidence-derived {expected_tier!r}"
                )

            if evidence_turns:
                pivots["malicious_supported"] += 1
                if pivot != evidence_turns[0]:
                    errors.append(f"{cid}: pivot is not earliest supported evidence turn")
                if ignore:
                    errors.append(f"{cid}: supported malicious pivot is ignored")
                expected_kind = "distributed" if len(evidence_turns) > 1 else "contextual_pivot"
                if r.get("pivot_kind") != expected_kind:
                    errors.append(f"{cid}: pivot_kind disagrees with supported evidence-turn count")
            else:
                pivots["malicious_unknown_ignored"] += 1
                if pivot is not None:
                    errors.append(f"{cid}: unsupported malicious pivot is non-null")
                if not ignore:
                    errors.append(f"{cid}: unknown malicious pivot must be ignored")
                if r.get("pivot_kind") != "none":
                    errors.append(f"{cid}: no-evidence malicious record must have pivot_kind=none")

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
            if r.get("pivot_kind") != "none":
                errors.append(f"{cid}: benign record must have pivot_kind=none")
            analysis = r.get("frontier_evidence_analysis", {}) or {}
            if analysis.get("status") != "not_applicable":
                errors.append(f"{cid}: benign evidence analysis must be not_applicable")
            for _, turn, span in all_spans_with_turn(r):
                text = str(turn.get("text", ""))
                start, end = span.get("char_start"), span.get("char_end")
                span_text = str(span.get("text", ""))
                if isinstance(start, int) and isinstance(end, int) and span_text:
                    if not (0 <= start < end <= len(text)) or text[start:end] != span_text:
                        errors.append(f"{cid}: stale/misaligned benign span offsets")
        else:
            errors.append(f"{cid}: unsupported label={label!r}")

    for pair_id, group in pair_groups.items():
        labels = Counter(r.get("label") for r in group)
        if len(group) != 2 or labels != Counter({0: 1, 1: 1}):
            errors.append(
                f"pair {pair_id}: primary pair is incomplete, n={len(group)} labels={dict(labels)}"
            )
            continue
        malicious = next(r for r in group if r.get("label") == 1)
        benign = next(r for r in group if r.get("label") == 0)
        mal_users = user_turns(malicious)
        ben_users = user_turns(benign)
        if len(mal_users) != len(ben_users):
            errors.append(f"pair {pair_id}: user-turn counts differ across twins")
            continue
        if not mal_users or mal_users[0].get("text") != ben_users[0].get("text"):
            errors.append(f"pair {pair_id}: first user turn is no longer byte-identical")
        mh = (malicious.get("intended_structure", {}) or {}).get("pair_hardness")
        bh = (benign.get("intended_structure", {}) or {}).get("pair_hardness")
        if mh != bh:
            errors.append(f"pair {pair_id}: pair_hardness differs across twins")
        elif mh == "context_required":
            if mal_users[-1].get("text") != ben_users[-1].get("text"):
                errors.append(f"pair {pair_id}: context-required final turn is no longer identical")
        elif mh == "surface_control":
            if mal_users[-1].get("text") == ben_users[-1].get("text"):
                errors.append(f"pair {pair_id}: surface-control final turns unexpectedly identical")
        else:
            errors.append(f"pair {pair_id}: unsupported pair_hardness={mh!r}")

    label_counts = Counter(r.get("label") for r in records)
    if label_counts.get(0, 0) != label_counts.get(1, 0):
        errors.append(f"primary frontier labels are not exactly balanced: {dict(label_counts)}")
    mal_hist = Counter(n_user(r) for r in records if r.get("label") == 1)
    ben_hist = Counter(n_user(r) for r in records if r.get("label") == 0)
    if mal_hist != ben_hist:
        errors.append(
            f"primary frontier user-turn histograms differ by class: mal={dict(mal_hist)} ben={dict(ben_hist)}"
        )

    print("=== Frontier prepared dataset audit ===")
    print(f"Records: {len(records)}")
    print(f"Labels: {dict(label_counts)}")
    print(f"Retained complete pairs: {len(pair_groups)}")
    print(f"User-turn histograms identical: {mal_hist == ben_hist}")
    print(f"Supervision tiers: {dict(tiers)}")
    print(f"Pivot supervision modes: {dict(pivots)}")
    print(
        "Scenario families: "
        f"{len({(r.get('metadata',{}) or {}).get('scenario_family') for r in records})}"
    )
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
