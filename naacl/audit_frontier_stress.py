#!/usr/bin/env python3
"""Fail-closed audit for the frontier standalone hard-benign stress set.

These records are scientifically useful as hard benign/FPR stress cases but are
excluded from primary training because their five-turn construction creates a
source-level length shortcut. This audit prevents them from drifting back into
training or acquiring positive pivot/span supervision.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter

from frontier_common import load_jsonl

DEFAULT_TARGET = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_JUDGE = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


def is_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


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
    lengths = Counter()

    for r in records:
        cid = str(r.get("conversation_id", ""))
        if not cid:
            errors.append("record missing conversation_id")
        elif cid in seen:
            errors.append(f"duplicate conversation_id: {cid}")
        seen.add(cid)

        if r.get("label") != 0:
            errors.append(f"{cid}: stress set contains non-benign label={r.get('label')!r}")
        if r.get("pair_id") not in (None, ""):
            errors.append(f"{cid}: standalone stress record unexpectedly has pair_id")

        metadata = r.get("metadata", {}) or {}
        if metadata.get("slice_role") != "standalone_benign":
            errors.append(f"{cid}: stress record slice_role is not standalone_benign")
        if not metadata.get("scenario_family"):
            errors.append(f"{cid}: missing scenario_family")
        if metadata.get("split_group") != metadata.get("scenario_family"):
            errors.append(f"{cid}: split_group differs from scenario_family")

        intended = r.get("intended_structure", {}) or {}
        if intended.get("annotation_status") != "generation_intent_only_not_ground_truth":
            errors.append(f"{cid}: authoring intent lost non-ground-truth marker")
        if r.get("authoring_intent_label") != 0:
            errors.append(f"{cid}: authoring intent label is not benign")

        if r.get("primary_pair_complete") is not False:
            errors.append(f"{cid}: stress record primary_pair_complete must be false")
        if r.get("training_eligible") is not False:
            errors.append(f"{cid}: stress record must be training_eligible=false")
        if r.get("use_as") != "benign_stress_evaluation_only":
            errors.append(f"{cid}: stress record use_as is not evaluation-only")
        if r.get("source_stage") != "canonical_stress_record":
            errors.append(f"{cid}: unexpected stress source_stage={r.get('source_stage')!r}")
        if r.get("benign_status") != "frontier_standalone_hard_benign_stress":
            errors.append(f"{cid}: unexpected benign_status")

        if r.get("validation_status") != "validated":
            errors.append(f"{cid}: stress record is not independently validated")
        rollout = r.get("rollout_provenance", {}) or {}
        validation = r.get("frontier_behavioral_validation", {}) or {}
        if rollout.get("target_model") != args.expected_target_model:
            errors.append(f"{cid}: unexpected stress rollout target")
        if validation.get("judge_model") != args.expected_judge_model:
            errors.append(f"{cid}: unexpected stress judge")
        if rollout.get("authoring_metadata_exposed_to_target") is not False:
            errors.append(f"{cid}: target metadata-exposure provenance is not false")
        if validation.get("authoring_metadata_exposed_to_judge") is not False:
            errors.append(f"{cid}: judge metadata-exposure provenance is not false")
        if r.get("canonical_target_model") != args.expected_target_model:
            errors.append(f"{cid}: canonical target model marker mismatch")
        if r.get("canonical_judge_model") != args.expected_judge_model:
            errors.append(f"{cid}: canonical judge model marker mismatch")

        # sanitize_benign resolves the benign record semantics before marking the
        # stress record non-trainable. Keep those resolved semantics intact while
        # making the training exclusion explicit.
        if r.get("supervision_tier") != "benign_validated":
            errors.append(f"{cid}: expected benign_validated supervision semantics")
        weight = r.get("loss_weight")
        if not is_number(weight) or not math.isclose(float(weight), 1.0, abs_tol=1e-9):
            errors.append(f"{cid}: benign stress loss_weight must remain 1.0 metadata")

        if r.get("pivot_turn_id") is not None:
            errors.append(f"{cid}: benign stress pivot_turn_id must be null")
        if r.get("pivot_kind") != "none":
            errors.append(f"{cid}: benign stress pivot_kind must be none")
        if bool(r.get("pivot_supervision_ignore", False)):
            errors.append(f"{cid}: benign true no-pivot must not be marked unknown/ignored")
        if r.get("evidence_turn_ids", []) not in (None, []):
            errors.append(f"{cid}: benign stress record contains evidence turns")

        evidence = r.get("frontier_evidence_analysis", {}) or {}
        if evidence.get("status") != "not_applicable":
            errors.append(f"{cid}: benign stress evidence status must be not_applicable")

        for turn in r.get("turns", []):
            text = str(turn.get("text", ""))
            for span in turn.get("span_annotations", []):
                if span.get("causal_type") == "causal":
                    errors.append(f"{cid}: benign stress span visible as causal")
                if span.get("supervision_tier") not in {"incidental", None}:
                    errors.append(f"{cid}: unexpected benign stress span tier")
                start, end = span.get("char_start"), span.get("char_end")
                span_text = str(span.get("text", ""))
                if isinstance(start, int) and isinstance(end, int) and span_text:
                    if not (0 <= start < end <= len(text)) or text[start:end] != span_text:
                        errors.append(f"{cid}: stale/misaligned stress span offsets")

        lengths[n_user(r)] += 1

    print("=== Frontier hard-benign stress audit ===")
    print(f"Records: {len(records)}")
    print(f"User-turn histogram: {dict(lengths)}")
    print(
        "Scenario families: "
        f"{len({(r.get('metadata',{}) or {}).get('scenario_family') for r in records})}"
    )
    if errors:
        print("STRESS AUDIT FAILED", file=sys.stderr)
        for error in errors[:100]:
            print(f"ERROR: {error}", file=sys.stderr)
        if len(errors) > 100:
            print(f"... {len(errors)-100} additional errors", file=sys.stderr)
        sys.exit(2)
    print("STRESS AUDIT PASSED")


if __name__ == "__main__":
    main()
