#!/usr/bin/env python3
"""Materialize legacy Dataset A detection-only auxiliary benign controls.

The restored semantic twins remain the immutable primary Dataset A. This script
reintroduces the historically validated broad benign pool only as train-time
trajectory-detection auxiliary data.

Scientific contract
-------------------
* Primary A is never modified.
* Auxiliary rows must originate from the pre-length-matching validated benign
  pool, not from the old trimmed/matched primary artifact.
* Conversation text is never modified.
* Auxiliary rows receive detection supervision only. Turn/pivot/span
  localization is explicitly masked.
* Exact conversation-ID and normalized user-trajectory overlap is removed
  against primary A and any optional external corpora (for example Dataset B).
* The default auxiliary detection weight is 0.25, matching the existing
  GuardLens-Transformer auxiliary-detection contract used for Dataset B.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_DETECTION_WEIGHT = 0.25


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(records: Iterable[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def n_user_turns(record: Dict) -> int:
    return sum(
        str(turn.get("role", "")).lower() == "user"
        for turn in record.get("turns", [])
    )


def observable_turn_hash(record: Dict) -> str:
    observable = [
        {
            "turn_id": turn.get("turn_id"),
            "role": turn.get("role"),
            "text": turn.get("text"),
        }
        for turn in record.get("turns", [])
    ]
    payload = json.dumps(
        observable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalized_user_trajectory_hash(record: Dict) -> str:
    texts = [
        str(turn.get("text", "")).strip()
        for turn in record.get("turns", [])
        if str(turn.get("role", "")).lower() == "user"
    ]
    normalized = "\n<USER_TURN>\n".join(" ".join(text.split()) for text in texts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_primary(records: Sequence[Dict], expected_pairs: int) -> None:
    if len(records) != 2 * expected_pairs:
        raise RuntimeError(
            f"primary A expected {2 * expected_pairs} records, got {len(records)}"
        )
    labels = Counter(record.get("label") for record in records)
    if labels != Counter({0: expected_pairs, 1: expected_pairs}):
        raise RuntimeError(
            f"primary A expected {expected_pairs} records per label, got {dict(labels)}"
        )

    ids = [str(record.get("conversation_id", "")) for record in records]
    if any(not cid for cid in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("primary A contains missing/duplicate conversation_id values")

    groups = defaultdict(list)
    for record in records:
        pair_id = str(record.get("pair_id", ""))
        if not pair_id:
            raise RuntimeError(
                f"{record.get('conversation_id')}: primary A record missing pair_id"
            )
        groups[pair_id].append(record)

    if len(groups) != expected_pairs:
        raise RuntimeError(
            f"primary A expected {expected_pairs} pair groups, got {len(groups)}"
        )
    for pair_id, group in groups.items():
        labels = Counter(record.get("label") for record in group)
        if len(group) != 2 or labels != Counter({0: 1, 1: 1}):
            raise RuntimeError(
                f"primary A pair {pair_id} is not one malicious + one benign"
            )


def validate_broad_benign_source(records: Sequence[Dict], expected_records: int) -> None:
    if len(records) != expected_records:
        raise RuntimeError(
            f"broad benign source expected {expected_records} records, got {len(records)}"
        )

    ids = [str(record.get("conversation_id", "")) for record in records]
    if any(not cid for cid in ids) or len(ids) != len(set(ids)):
        raise RuntimeError(
            "broad benign source contains missing/duplicate conversation_id values"
        )

    for record in records:
        cid = str(record.get("conversation_id", ""))
        if record.get("label") != 0:
            raise RuntimeError(f"{cid}: broad benign source contains non-benign label")
        if record.get("validation_status") != "validated":
            raise RuntimeError(
                f"{cid}: broad benign source is not historically validated"
            )
        if record.get("training_eligible") is not True:
            raise RuntimeError(
                f"{cid}: broad benign source is not historically training eligible"
            )
        if record.get("supervision_tier") != "benign_validated":
            raise RuntimeError(
                f"{cid}: expected benign_validated source tier, got "
                f"{record.get('supervision_tier')!r}"
            )
        if (record.get("metadata", {}) or {}).get("naacl_length_match"):
            raise RuntimeError(
                f"{cid}: source is a trimmed length-matched derivative; "
                "use the full pre-matching benign pool"
            )
        if observable_turn_hash(record) == observable_turn_hash({"turns": []}):
            raise RuntimeError(f"{cid}: empty observable trajectory")
        declared = record.get("conversation_length")
        if declared is not None and int(declared) != len(record.get("turns", [])):
            raise RuntimeError(
                f"{cid}: conversation_length differs from physical turn count"
            )


def auxiliary_copy(record: Dict, detection_weight: float) -> Dict:
    if (
        isinstance(detection_weight, bool)
        or not isinstance(detection_weight, (int, float))
        or not math.isfinite(float(detection_weight))
        or float(detection_weight) <= 0
    ):
        raise RuntimeError("detection weight must be finite and positive")

    before = observable_turn_hash(record)
    out = copy.deepcopy(record)

    metadata = out.setdefault("metadata", {})
    metadata["pre_auxiliary_supervision_tier"] = out.get("supervision_tier")
    metadata["pre_auxiliary_loss_weight"] = out.get("loss_weight")
    metadata["auxiliary_source"] = "legacy_validated_broad_benign_pre_length_match"
    metadata["conversation_text_modified"] = False
    metadata["localization_targets_masked"] = True

    out["auxiliary_detection_only"] = True
    out["use_as"] = "auxiliary_detection_only"
    out["corpus_role"] = "detection_auxiliary"
    out["detection_label"] = 0
    out["detection_loss_weight"] = float(detection_weight)
    out["loss_weight"] = float(detection_weight)
    out["supervision_tier"] = "auxiliary_detection"
    out["localization_supervision_ignore"] = True
    out["pivot_supervision_ignore"] = True
    out["pivot_loss_weight"] = 0.0
    out["span_loss_weight"] = 0.0

    # Preserve historical annotations as provenance, but they are not training
    # targets in this derived view. Explicit record-level masking is the
    # contract consumed by GuardLens-Transformer.
    out["auxiliary_original_annotations_preserved"] = True

    after = observable_turn_hash(out)
    if before != after:
        raise RuntimeError(
            f"{record.get('conversation_id')}: auxiliary conversion modified "
            "observable conversation text"
        )
    return out


def turn_hist(records: Sequence[Dict]) -> Dict[str, int]:
    return {
        str(k): v
        for k, v in sorted(Counter(n_user_turns(record) for record in records).items())
    }


def auc_from_values(negative: Sequence[int], positive: Sequence[int]) -> float:
    if not negative or not positive:
        return float("nan")
    neg_counts = Counter(negative)
    wins = 0.0
    for value in positive:
        wins += sum(count for x, count in neg_counts.items() if value > x)
        wins += 0.5 * neg_counts.get(value, 0)
    return wins / (len(negative) * len(positive))


def threshold_metrics(
    negative: Sequence[int],
    positive: Sequence[int],
    threshold: int,
) -> Dict:
    tp = sum(value > threshold for value in positive)
    fn = len(positive) - tp
    fp = sum(value > threshold for value in negative)
    tn = len(negative) - fp
    tpr = tp / len(positive)
    tnr = tn / len(negative)
    return {
        "rule": f"predict malicious iff n_user_turns > {threshold}",
        "threshold": threshold,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "tpr": tpr,
        "tnr": tnr,
        "balanced_accuracy": 0.5 * (tpr + tnr),
    }


def turn_count_diagnostic(records: Sequence[Dict]) -> Dict:
    negative = [
        n_user_turns(record)
        for record in records
        if int(record.get("detection_label", record.get("label"))) == 0
    ]
    positive = [
        n_user_turns(record)
        for record in records
        if int(record.get("detection_label", record.get("label"))) == 1
    ]
    thresholds = sorted(set(negative + positive))
    best = max(
        (threshold_metrics(negative, positive, threshold) for threshold in thresholds),
        key=lambda item: (item["balanced_accuracy"], -item["threshold"]),
    )
    return {
        "labels": {"0": len(negative), "1": len(positive)},
        "turn_count_auc": auc_from_values(negative, positive),
        "best_threshold": best,
        "fixed_threshold_gt_10": threshold_metrics(negative, positive, 10),
        "mean_user_turns": {
            "0": statistics.mean(negative),
            "1": statistics.mean(positive),
        },
        "histogram": {
            "0": {str(k): v for k, v in sorted(Counter(negative).items())},
            "1": {str(k): v for k, v in sorted(Counter(positive).items())},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-input", required=True)
    parser.add_argument("--benign-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--excluded-output", default=None)
    parser.add_argument(
        "--exclude-against",
        action="append",
        default=[],
        help=(
            "Optional JSONL corpus whose conversation IDs and normalized user "
            "trajectories must not occur in the auxiliary output. Repeatable."
        ),
    )
    parser.add_argument("--expected-primary-pairs", type=int, default=516)
    parser.add_argument("--expected-source-records", type=int, default=721)
    parser.add_argument(
        "--detection-loss-weight",
        type=float,
        default=DEFAULT_DETECTION_WEIGHT,
    )
    args = parser.parse_args()

    primary = load_jsonl(args.primary_input)
    source = load_jsonl(args.benign_source)
    validate_primary(primary, args.expected_primary_pairs)
    validate_broad_benign_source(source, args.expected_source_records)

    forbidden_ids = {str(record.get("conversation_id", "")) for record in primary}
    forbidden_hashes = {
        normalized_user_trajectory_hash(record) for record in primary
    }
    external_counts = {}
    for path in args.exclude_against:
        rows = load_jsonl(path)
        external_counts[path] = len(rows)
        forbidden_ids.update(str(record.get("conversation_id", "")) for record in rows)
        forbidden_hashes.update(
            normalized_user_trajectory_hash(record) for record in rows
        )

    seen_source_hashes = set()
    auxiliary = []
    excluded = []
    reasons = Counter()

    for record in source:
        cid = str(record.get("conversation_id", ""))
        trajectory_hash = normalized_user_trajectory_hash(record)

        reason = None
        if cid in forbidden_ids:
            reason = "conversation_id_overlap"
        elif trajectory_hash in forbidden_hashes:
            reason = "normalized_user_trajectory_overlap"
        elif trajectory_hash in seen_source_hashes:
            reason = "duplicate_normalized_user_trajectory_within_aux_source"

        if reason is not None:
            rejected = copy.deepcopy(record)
            rejected["auxiliary_exclusion_reason"] = reason
            excluded.append(rejected)
            reasons[reason] += 1
            continue

        seen_source_hashes.add(trajectory_hash)
        auxiliary.append(auxiliary_copy(record, args.detection_loss_weight))

    output_ids = [str(record.get("conversation_id", "")) for record in auxiliary]
    if len(output_ids) != len(set(output_ids)):
        raise RuntimeError("auxiliary output has duplicate conversation IDs")

    # A derived detection-training view is useful as a population diagnostic,
    # but is not written here. Primary splitting must happen independently and
    # auxiliary rows are intended for train only, matching Dataset B practice.
    detection_view = list(primary) + list(auxiliary)
    diagnostics = turn_count_diagnostic(detection_view)

    write_jsonl(auxiliary, args.output)
    if args.excluded_output:
        write_jsonl(excluded, args.excluded_output)

    stats = {
        "primary_a": {
            "records": len(primary),
            "pairs": args.expected_primary_pairs,
            "labels": dict(Counter(str(r.get("label")) for r in primary)),
            "turn_count_diagnostic": turn_count_diagnostic(primary),
        },
        "historical_broad_benign_source": {
            "records": len(source),
            "user_turn_histogram": turn_hist(source),
            "mean_user_turns": statistics.mean(n_user_turns(r) for r in source),
            "policy": (
                "validated full benign records before old one-to-one prefix "
                "length matching"
            ),
        },
        "auxiliary_detection": {
            "records": len(auxiliary),
            "detection_labels": dict(
                Counter(str(r.get("detection_label")) for r in auxiliary)
            ),
            "detection_loss_weight": float(args.detection_loss_weight),
            "localization_supervision": "fully_masked",
            "user_turn_histogram": turn_hist(auxiliary),
            "mean_user_turns": (
                statistics.mean(n_user_turns(r) for r in auxiliary)
                if auxiliary else None
            ),
        },
        "excluded": {
            "records": len(excluded),
            "reasons": dict(reasons),
        },
        "exclude_against": external_counts,
        "primary_plus_aux_detection_population": {
            "records": len(detection_view),
            "turn_count_diagnostic": diagnostics,
            "note": (
                "population diagnostic only; primary split should be frozen "
                "before auxiliary records are added to train"
            ),
        },
        "contract": {
            "primary_a_modified": False,
            "conversation_text_modified": False,
            "auxiliary_detection_only": True,
            "auxiliary_train_only": True,
            "localization_supervision_ignore": True,
            "pivot_supervision_ignore": True,
            "pivot_loss_weight": 0.0,
            "span_loss_weight": 0.0,
            "default_detection_loss_weight": DEFAULT_DETECTION_WEIGHT,
            "mirrors_guardlens_transformer_auxiliary_contract": True,
        },
    }

    os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
    with open(args.stats_output, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)

    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"Wrote Dataset A detection auxiliary: {args.output}")
    if args.excluded_output:
        print(f"Wrote excluded records: {args.excluded_output}")


if __name__ == "__main__":
    main()
