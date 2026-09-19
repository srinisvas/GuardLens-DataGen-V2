#!/usr/bin/env python3
"""Prepare detection-only auxiliary outcomes from rejected Dataset B trajectories.

This dataset is intentionally separate from the pair-complete primary corpus.
B2 rejection means the realized target behavior disagreed with the source
trajectory intent: benign-authored trajectories that became unsafe are positive
detection examples, while malicious-authored trajectories that were safely
resisted are negative detection examples.

Auxiliary records carry no pivot or span-localization supervision. They use a
small detection-only loss weight and preserve the original authoring label for
provenance. ``metadata.consolidated_split_group`` is set from ``scenario_family``
so a later joint split can never separate auxiliary records from related primary
Dataset B records.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

from audit_frontier_evidence import audit_record as audit_b4_record
from frontier_common import (
    DEFAULT_JUDGE_MAX_CONTEXT_CHARS,
    DEFAULT_JUDGE_MAX_MODEL_LEN,
    DEFAULT_TARGET_MAX_TOKENS,
    load_jsonl,
    write_jsonl,
)

AUXILIARY_SOURCE = "frontier_authored_v3_auxiliary"
AUXILIARY_WEIGHT = 0.25
DEFAULT_TARGET = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_JUDGE = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
EXPECTED_REJECTED = 512
EXPECTED_UNSAFE_BENIGN_AUTHORED = 190
EXPECTED_SAFE_MALICIOUS_AUTHORED = 322
SPLITS = ("train", "dev", "test")


def user_trajectory_hash(record: Dict) -> str:
    texts = [
        str(turn.get("text", "")).strip()
        for turn in record.get("turns", [])
        if str(turn.get("role", "")).lower() == "user"
    ]
    normalized = "\n<USER_TURN>\n".join(" ".join(text.split()) for text in texts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validation(record: Dict) -> Dict:
    value = record.get("frontier_behavioral_validation", {}) or {}
    if value.get("status") != record.get("validation_status"):
        raise RuntimeError(
            f"{record.get('conversation_id')}: top-level and B2 validation status disagree"
        )
    return value


def evidence_target_max_model_len(record: Dict) -> int:
    cid = str(record.get("conversation_id", ""))
    analysis = record.get("frontier_evidence_analysis", {}) or {}
    try:
        value = int(analysis["target_max_model_len"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{cid}: missing/invalid B4 target context envelope") from exc
    if value not in {16384, 32768}:
        raise RuntimeError(f"{cid}: unsupported B4 target context envelope {value}")
    return value


def assert_auxiliary_source_provenance(
    record: Dict,
    *,
    expected_target: str = DEFAULT_TARGET,
    expected_judge: str = DEFAULT_JUDGE,
) -> None:
    audit_b4_record(
        record,
        target_model=expected_target,
        judge_model=expected_judge,
        max_tokens=DEFAULT_TARGET_MAX_TOKENS,
        target_max_model_len=evidence_target_max_model_len(record),
        judge_max_model_len=DEFAULT_JUDGE_MAX_MODEL_LEN,
        judge_max_context_chars=DEFAULT_JUDGE_MAX_CONTEXT_CHARS,
    )


def recompute_validation_status(record: Dict) -> str:
    cid = str(record.get("conversation_id", ""))
    label = record.get("label")
    if label not in (0, 1):
        raise RuntimeError(f"{cid}: unsupported source label {label!r}")
    info = validation(record)
    try:
        unsafe_threshold = float(info["unsafe_threshold"])
        ambiguous_threshold = float(info["ambiguous_threshold"])
        min_confidence = float(info["min_confidence_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{cid}: missing B2 decision thresholds") from exc

    trajectory = info.get("trajectory", []) or []
    if not trajectory:
        raise RuntimeError(f"{cid}: missing B2 validation trajectory")
    try:
        scores = [float(item["unsafe_score"]) for item in trajectory]
        confidences = [float(item["confidence"]) for item in trajectory]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{cid}: invalid B2 trajectory score/confidence") from exc

    raw_unsafe = [i for i, score in enumerate(scores) if score >= unsafe_threshold]
    confident_unsafe = [
        i for i in raw_unsafe if confidences[i] >= min_confidence
    ]
    near_boundary = any(
        ambiguous_threshold <= score < unsafe_threshold for score in scores
    )
    max_index = max(range(len(scores)), key=scores.__getitem__)
    uncertain_risk = (
        bool(raw_unsafe) and not bool(confident_unsafe)
    ) or confidences[max_index] < min_confidence

    if bool(info.get("near_boundary", False)) != near_boundary:
        raise RuntimeError(f"{cid}: stored B2 near_boundary disagrees with trajectory")
    if bool(info.get("uncertain_risk", False)) != uncertain_risk:
        raise RuntimeError(f"{cid}: stored B2 uncertain_risk disagrees with trajectory")

    if label == 1:
        expected = (
            "validated"
            if confident_unsafe
            else "ambiguous"
            if uncertain_risk or near_boundary
            else "rejected"
        )
    else:
        expected = (
            "rejected"
            if confident_unsafe
            else "ambiguous"
            if uncertain_risk or near_boundary
            else "validated"
        )
    if info.get("status") != expected or record.get("validation_status") != expected:
        raise RuntimeError(
            f"{cid}: B2 status does not recompute from trajectory: "
            f"stored={record.get('validation_status')!r} expected={expected!r}"
        )
    return expected


def detection_label_for_rejected(record: Dict) -> int:
    """Return the behavior label implied by a rejected source-intent example."""
    label = record.get("label")
    if label not in (0, 1):
        raise RuntimeError(f"{record.get('conversation_id')}: unsupported source label {label!r}")
    info = validation(record)
    if recompute_validation_status(record) != "rejected":
        raise RuntimeError(f"{record.get('conversation_id')}: auxiliary input is not B2-rejected")

    max_unsafe = info.get("max_unsafe_score")
    confidence = info.get("max_risk_confidence")
    threshold = float(info.get("unsafe_threshold", 0.5))
    min_confidence = float(info.get("min_confidence_threshold", 0.55))
    uncertain = bool(info.get("uncertain_risk", False))
    if not isinstance(max_unsafe, (int, float)) or isinstance(max_unsafe, bool):
        raise RuntimeError(f"{record.get('conversation_id')}: missing B2 max_unsafe_score")
    if confidence is not None and (
        not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
    ):
        raise RuntimeError(f"{record.get('conversation_id')}: invalid B2 confidence")
    if uncertain:
        raise RuntimeError(f"{record.get('conversation_id')}: rejected B2 record is uncertainty-gated")

    # For benign-authored records rejection must mean an unsafe realized target
    # trajectory. For malicious-authored records rejection must mean the attack
    # did not produce a confidence-qualified unsafe target trajectory.
    if label == 0:
        if float(max_unsafe) < threshold:
            raise RuntimeError(
                f"{record.get('conversation_id')}: benign-authored rejection is not unsafe"
            )
        if confidence is not None and float(confidence) < min_confidence:
            raise RuntimeError(
                f"{record.get('conversation_id')}: benign-authored unsafe outcome is low-confidence"
            )
        return 1
    if float(max_unsafe) >= threshold and (
        confidence is None or float(confidence) >= min_confidence
    ):
        raise RuntimeError(
            f"{record.get('conversation_id')}: malicious-authored rejected record still qualifies unsafe"
        )
    return 0


def build_auxiliary_record(record: Dict) -> Dict | None:
    if record.get("validation_status") != "rejected":
        return None
    cid = str(record.get("conversation_id", ""))
    if not cid:
        raise RuntimeError("record missing conversation_id")
    scenario = str((record.get("metadata", {}) or {}).get("scenario_family", "")).strip()
    if not scenario:
        raise RuntimeError(f"{cid}: missing scenario_family")

    out = copy.deepcopy(record)
    authoring_label = int(out["label"])
    detection_label = detection_label_for_rejected(out)
    metadata = out.setdefault("metadata", {})
    metadata["consolidated_split_group"] = f"frontier::{scenario}"
    metadata["normalized_user_trajectory_hash"] = user_trajectory_hash(out)

    out["authoring_intent_label"] = authoring_label
    out["observed_behavior_label"] = detection_label
    out["detection_label"] = detection_label
    out["corpus_source"] = AUXILIARY_SOURCE
    out["source_stage"] = "canonical_auxiliary_detection_record"
    out["use_as"] = "auxiliary_detection_only"
    out["training_eligible"] = True
    out["auxiliary_detection_only"] = True
    out["primary_pair_complete"] = False
    out["supervision_tier"] = "auxiliary_detection"
    out["loss_weight"] = AUXILIARY_WEIGHT
    out["detection_loss_weight"] = AUXILIARY_WEIGHT
    out["pivot_loss_weight"] = 0.0
    out["span_loss_weight"] = 0.0
    out["pivot_supervision_ignore"] = True
    out["localization_supervision_ignore"] = True
    out["pivot_turn_id"] = None
    out["pivot_kind"] = "none"
    out["evidence_turn_ids"] = []
    return out


def prepare_auxiliary(records: Iterable[Dict]) -> List[Dict]:
    output = []
    seen = set()
    for record in records:
        prepared = build_auxiliary_record(record)
        if prepared is None:
            continue
        cid = str(prepared["conversation_id"])
        if cid in seen:
            raise RuntimeError(f"duplicate auxiliary conversation_id: {cid}")
        seen.add(cid)
        output.append(prepared)
    return output


def group_auxiliary(records: Iterable[Dict]) -> Dict[str, List[Dict]]:
    groups = defaultdict(list)
    for record in records:
        group = str((record.get("metadata", {}) or {}).get("consolidated_split_group", ""))
        if not group:
            raise RuntimeError(f"{record.get('conversation_id')}: missing auxiliary split group")
        groups[group].append(record)
    return dict(groups)


def split_auxiliary(
    records: List[Dict],
    *,
    fractions: Dict[str, float],
    seed: int,
) -> Dict[str, List[Dict]]:
    """Deterministically split complete scenario families without leakage."""
    groups = group_auxiliary(records)
    rng = random.Random(seed)
    target = {name: len(records) * fractions[name] for name in SPLITS}
    counts = {name: 0 for name in SPLITS}
    class_counts = {name: Counter() for name in SPLITS}
    global_classes = Counter(int(r["detection_label"]) for r in records)
    class_targets = {
        name: {label: n * fractions[name] for label, n in global_classes.items()}
        for name in SPLITS
    }

    items: List[Tuple[str, List[Dict]]] = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    assigned = {name: [] for name in SPLITS}
    for group_id, group in items:
        signature = Counter(int(r["detection_label"]) for r in group)
        best_name = None
        best_score = None
        for name in SPLITS:
            fill = (counts[name] + len(group)) / max(target[name], 1.0)
            class_fill = []
            for label, amount in signature.items():
                wanted = class_targets[name].get(label, 0.0)
                if wanted > 0:
                    class_fill.append((class_counts[name][label] + amount) / wanted)
            score = 0.75 * fill + 0.25 * (sum(class_fill) / len(class_fill) if class_fill else fill)
            score += rng.random() * 1e-9
            if best_score is None or score < best_score:
                best_score, best_name = score, name
        assigned[best_name].append((group_id, group))
        counts[best_name] += len(group)
        class_counts[best_name].update(signature)

    result = {}
    owner = {}
    for name in SPLITS:
        subset = [record for _, group in assigned[name] for record in group]
        rng.shuffle(subset)
        result[name] = subset
        for record in subset:
            group = str(record["metadata"]["consolidated_split_group"])
            previous = owner.setdefault(group, name)
            if previous != name:
                raise RuntimeError(f"auxiliary family leakage: {group} in {previous} and {name}")
    return result


def summary(records: List[Dict]) -> Dict:
    return {
        "records": len(records),
        "authoring_labels": dict(Counter(str(r["authoring_intent_label"]) for r in records)),
        "detection_labels": dict(Counter(str(r["detection_label"]) for r in records)),
        "scenario_families": len({r["metadata"]["consolidated_split_group"] for r in records}),
        "policy": {
            "source": "B2-rejected realized trajectories only",
            "label": "realized unsafe behavior, distinct from authoring intent",
            "loss": "detection-only weight 0.25; pivot/span losses zero",
            "split": "scenario_family is indivisible and compatible with primary frontier grouping",
        },
    }


def assert_expected_full_export_counts(records: List[Dict]) -> None:
    if len(records) != EXPECTED_REJECTED:
        raise RuntimeError(
            f"expected {EXPECTED_REJECTED} B2-rejected auxiliary records, found {len(records)}"
        )
    detection = Counter(int(r["detection_label"]) for r in records)
    if detection != Counter({1: EXPECTED_UNSAFE_BENIGN_AUTHORED, 0: EXPECTED_SAFE_MALICIOUS_AUTHORED}):
        raise RuntimeError(
            "unexpected auxiliary detection-label counts: "
            f"{dict(detection)} expected {{1: {EXPECTED_UNSAFE_BENIGN_AUTHORED}, "
            f"0: {EXPECTED_SAFE_MALICIOUS_AUTHORED}}}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--split-output-dir")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--dev-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-target-model", default=DEFAULT_TARGET)
    parser.add_argument("--expected-judge-model", default=DEFAULT_JUDGE)
    parser.add_argument(
        "--expect-full-review-export",
        action="store_true",
        help="Require the audited 512 = 190 unsafe-benign + 322 safe-malicious counts.",
    )
    args = parser.parse_args()

    fractions = {"train": args.train_frac, "dev": args.dev_frac, "test": args.test_frac}
    if abs(sum(fractions.values()) - 1.0) > 1e-8 or min(fractions.values()) <= 0:
        raise ValueError("train/dev/test fractions must be positive and sum to 1")

    source_records = load_jsonl(args.input)
    for record in source_records:
        if record.get("validation_status") == "rejected":
            assert_auxiliary_source_provenance(
                record,
                expected_target=args.expected_target_model,
                expected_judge=args.expected_judge_model,
            )
    records = prepare_auxiliary(source_records)
    if args.expect_full_review_export:
        assert_expected_full_export_counts(records)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_jsonl(records, args.output)

    stats = summary(records)
    if args.split_output_dir:
        splits = split_auxiliary(records, fractions=fractions, seed=args.seed)
        os.makedirs(args.split_output_dir, exist_ok=True)
        for name, subset in splits.items():
            write_jsonl(subset, os.path.join(args.split_output_dir, f"{name}.jsonl"))
        stats["split"] = {
            "seed": args.seed,
            "fractions": fractions,
            "records": {name: len(subset) for name, subset in splits.items()},
            "detection_labels": {
                name: dict(Counter(str(r["detection_label"]) for r in subset))
                for name, subset in splits.items()
            },
        }

    os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
    with open(args.stats_output, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
