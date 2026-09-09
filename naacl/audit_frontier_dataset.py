#!/usr/bin/env python3
"""Fail-closed audit for canonical frontier-authored GuardLens primary records."""
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
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def n_user(record) -> int:
    return sum(
        str(t.get("role", "")).lower() == "user"
        for t in record.get("turns", [])
    )


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
            errors.append(
                f"{cid}: loss_weight={weight!r} does not match tier {tier!r}"
            )

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
            if analysis.get("baseline_reproduced_stored_rollout") is not True:
                errors.append(f"{cid}: malicious paired baseline did not reproduce stored rollout")
            if analysis.get("target_model") != args.expected_target_model:
                errors.append(f"{cid}: malicious evidence target mismatch")
            if analysis.get("judge_model") != args.expected_judge_model:
                errors.append(f"{cid}: malicious evidence judge mismatch")
            if analysis.get("author_intended_pivots_used_as_ground_truth") is not False:
                errors.append(f"{cid}: author-intended pivots leaked into evidence ground truth")
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

    for pair_id, group in pair_groups.items():
        labels = Counter(r.get("label") for r in group)
        if len(group) != 2 or labels != Counter({0: 1, 1: 1}):
            errors.append(
                f"pair {pair_id}: primary pair is incomplete, n={len(group)} labels={dict(labels)}"
            )
            continue
        malicious = next(r for r in group if r.get("label") == 1)
        benign = next(r for r in group if r.get("label") == 0)
        if n_user(malicious) != n_user(benign):
            errors.append(f"pair {pair_id}: user-turn counts differ across twins")

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
