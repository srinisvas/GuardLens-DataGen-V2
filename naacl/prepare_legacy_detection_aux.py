#!/usr/bin/env python3
"""Materialize Dataset A detection-only auxiliary benign controls.

Primary Dataset A is the immutable restored semantic-twin corpus. This script
reintroduces the historically validated broad benign population only as
trajectory-detection auxiliary data.

The script is intentionally fail-closed:
* the exact restored-primary SHA-256 is pinned by default;
* the exact broad-benign Git blob is pinned by default;
* source trajectories must be complete alternating user/assistant records;
* conversation text is never modified;
* auxiliary localization/pivot/span supervision is always masked;
* exact ID and normalized-user-trajectory overlap is removed against primary A
  and optional external corpora;
* records exceeding the configured model turn ceiling stop the run by default;
* both unweighted and detection-loss-weighted turn-count diagnostics are
  reported so auxiliary weighting cannot hide a shortcut.
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
from typing import Dict, Iterable, List, Sequence


EXPECTED_PRIMARY_SHA256 = (
    "f9021672150696b2c3a367b1a1c67edafa4ce2f1a6a83fc1293870eaa7e6c34f"
)
EXPECTED_BENIGN_SOURCE_GIT_BLOB = "de1c979a2f634e8302a27d0d3d79014a5509e7a9"
DEFAULT_DETECTION_WEIGHT = 1.0
DEFAULT_MAX_PHYSICAL_TURNS = 64


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


def write_json(obj: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: str) -> str:
    size = os.path.getsize(path)
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("ascii"))
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_trajectory(record: Dict, *, source_name: str) -> None:
    cid = str(record.get("conversation_id", "")) or "<missing>"
    turns = record.get("turns", [])
    if not isinstance(turns, list) or not turns:
        raise RuntimeError(f"{source_name}:{cid}: empty trajectory")

    expected_role = "user"
    for idx, turn in enumerate(turns):
        if turn.get("turn_id") != idx:
            raise RuntimeError(
                f"{source_name}:{cid}: turn_id={turn.get('turn_id')!r} "
                f"at index {idx}; expected contiguous 0..N-1"
            )
        role = str(turn.get("role", "")).lower()
        if role != expected_role:
            raise RuntimeError(
                f"{source_name}:{cid}: expected role {expected_role}, got {role!r} "
                f"at turn {idx}"
            )
        text = turn.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(
                f"{source_name}:{cid}: empty/non-string text at turn {idx}"
            )
        expected_role = "assistant" if expected_role == "user" else "user"

    if expected_role != "user":
        raise RuntimeError(
            f"{source_name}:{cid}: trajectory ends with unmatched user turn"
        )

    declared = record.get("conversation_length")
    if declared is not None and int(declared) != len(turns):
        raise RuntimeError(
            f"{source_name}:{cid}: conversation_length={declared} differs "
            f"from physical turns={len(turns)}"
        )


def validate_primary(
    records: Sequence[Dict],
    expected_pairs: int,
    *,
    expected_sha256: str,
    actual_sha256: str,
) -> None:
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(
            "primary A SHA-256 mismatch; refusing to materialize auxiliary. "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
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
        validate_trajectory(record, source_name="primary")
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


def validate_broad_benign_source(
    records: Sequence[Dict],
    expected_records: int,
    *,
    expected_git_blob: str,
    actual_git_blob: str,
) -> None:
    if expected_git_blob and actual_git_blob != expected_git_blob:
        raise RuntimeError(
            "broad benign source Git-blob mismatch; refusing to materialize "
            f"auxiliary. expected={expected_git_blob} actual={actual_git_blob}"
        )
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
        validate_trajectory(record, source_name="broad_benign")
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
    # Preserve historical loss_weight as provenance semantics. The Transformer
    # contract reads detection_loss_weight for auxiliary detection.
    out["supervision_tier"] = "auxiliary_detection"
    out["localization_supervision_ignore"] = True
    out["pivot_supervision_ignore"] = True
    out["pivot_loss_weight"] = 0.0
    out["span_loss_weight"] = 0.0
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


def record_detection_label(record: Dict) -> int:
    value = record.get("detection_label", record.get("label"))
    if isinstance(value, bool) or value not in (0, 1):
        raise RuntimeError(
            f"{record.get('conversation_id')}: invalid detection label {value!r}"
        )
    return int(value)


def record_detection_weight(record: Dict) -> float:
    if record.get("auxiliary_detection_only") is True or (
        record.get("use_as") == "auxiliary_detection_only"
    ):
        value = record.get("detection_loss_weight")
    else:
        value = 1.0
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise RuntimeError(
            f"{record.get('conversation_id')}: invalid detection weight {value!r}"
        )
    return float(value)


def auc_from_values(negative: Sequence[int], positive: Sequence[int]) -> float:
    if not negative or not positive:
        return float("nan")
    neg_counts = Counter(negative)
    wins = 0.0
    for value in positive:
        wins += sum(count for x, count in neg_counts.items() if value > x)
        wins += 0.5 * neg_counts.get(value, 0)
    return wins / (len(negative) * len(positive))


def weighted_auc(
    negative: Sequence[tuple[int, float]],
    positive: Sequence[tuple[int, float]],
) -> float:
    neg_mass = sum(weight for _, weight in negative)
    pos_mass = sum(weight for _, weight in positive)
    if neg_mass <= 0 or pos_mass <= 0:
        return float("nan")
    wins = 0.0
    for p_value, p_weight in positive:
        for n_value, n_weight in negative:
            credit = 1.0 if p_value > n_value else 0.5 if p_value == n_value else 0.0
            wins += credit * p_weight * n_weight
    return wins / (pos_mass * neg_mass)


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


def weighted_threshold_metrics(
    negative: Sequence[tuple[int, float]],
    positive: Sequence[tuple[int, float]],
    threshold: int,
) -> Dict:
    pos_mass = sum(weight for _, weight in positive)
    neg_mass = sum(weight for _, weight in negative)
    tp = sum(weight for value, weight in positive if value > threshold)
    fp = sum(weight for value, weight in negative if value > threshold)
    fn = pos_mass - tp
    tn = neg_mass - fp
    tpr = tp / pos_mass
    tnr = tn / neg_mass
    return {
        "rule": f"predict malicious iff n_user_turns > {threshold}",
        "threshold": threshold,
        "weighted_tp": tp,
        "weighted_fn": fn,
        "weighted_tn": tn,
        "weighted_fp": fp,
        "tpr": tpr,
        "tnr": tnr,
        "balanced_accuracy": 0.5 * (tpr + tnr),
    }


def turn_count_diagnostic(records: Sequence[Dict]) -> Dict:
    negative = [
        n_user_turns(record)
        for record in records
        if record_detection_label(record) == 0
    ]
    positive = [
        n_user_turns(record)
        for record in records
        if record_detection_label(record) == 1
    ]
    if not negative or not positive:
        raise RuntimeError("turn-count diagnostic requires both detection classes")

    weighted_negative = [
        (n_user_turns(record), record_detection_weight(record))
        for record in records
        if record_detection_label(record) == 0
    ]
    weighted_positive = [
        (n_user_turns(record), record_detection_weight(record))
        for record in records
        if record_detection_label(record) == 1
    ]

    thresholds = sorted(set(negative + positive))
    best = max(
        (threshold_metrics(negative, positive, threshold) for threshold in thresholds),
        key=lambda item: (item["balanced_accuracy"], -item["threshold"]),
    )
    weighted_best = max(
        (
            weighted_threshold_metrics(
                weighted_negative, weighted_positive, threshold
            )
            for threshold in thresholds
        ),
        key=lambda item: (item["balanced_accuracy"], -item["threshold"]),
    )
    return {
        "labels": {"0": len(negative), "1": len(positive)},
        "unweighted": {
            "turn_count_auc": auc_from_values(negative, positive),
            "best_threshold": best,
            "fixed_threshold_gt_10": threshold_metrics(negative, positive, 10),
        },
        "effective_detection_weighted": {
            "negative_mass": sum(weight for _, weight in weighted_negative),
            "positive_mass": sum(weight for _, weight in weighted_positive),
            "turn_count_auc": weighted_auc(weighted_negative, weighted_positive),
            "best_threshold": weighted_best,
            "fixed_threshold_gt_10": weighted_threshold_metrics(
                weighted_negative, weighted_positive, 10
            ),
        },
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
        "--expected-primary-sha256",
        default=EXPECTED_PRIMARY_SHA256,
    )
    parser.add_argument(
        "--expected-benign-source-git-blob",
        default=EXPECTED_BENIGN_SOURCE_GIT_BLOB,
    )
    parser.add_argument(
        "--detection-loss-weight",
        type=float,
        default=DEFAULT_DETECTION_WEIGHT,
        help=(
            "Detection weight for validated A auxiliary benigns. Default 1.0; "
            "do not reuse B's 0.25 weight without an explicit scientific reason."
        ),
    )
    parser.add_argument(
        "--max-physical-turns",
        type=int,
        default=DEFAULT_MAX_PHYSICAL_TURNS,
        help="Current GuardLens model turn ceiling; 0 disables the ceiling audit.",
    )
    parser.add_argument(
        "--overlength-policy",
        choices=["fail", "exclude"],
        default="fail",
        help=(
            "Fail by default if a validated auxiliary record exceeds the model "
            "turn ceiling. 'exclude' must be chosen explicitly."
        ),
    )
    args = parser.parse_args()

    if args.max_physical_turns < 0:
        raise RuntimeError("--max-physical-turns must be >= 0")

    primary_sha = file_sha256(args.primary_input)
    source_blob = git_blob_sha1(args.benign_source)
    primary = load_jsonl(args.primary_input)
    source = load_jsonl(args.benign_source)

    validate_primary(
        primary,
        args.expected_primary_pairs,
        expected_sha256=args.expected_primary_sha256,
        actual_sha256=primary_sha,
    )
    validate_broad_benign_source(
        source,
        args.expected_source_records,
        expected_git_blob=args.expected_benign_source_git_blob,
        actual_git_blob=source_blob,
    )

    overlength = [
        record
        for record in source
        if args.max_physical_turns
        and len(record.get("turns", [])) > args.max_physical_turns
    ]
    if overlength and args.overlength_policy == "fail":
        max_seen = max(len(record.get("turns", [])) for record in overlength)
        raise RuntimeError(
            f"{len(overlength)}/{len(source)} validated broad benign records "
            f"exceed max_physical_turns={args.max_physical_turns}; max={max_seen}. "
            "No output was written. Either raise the model turn ceiling after "
            "review or rerun explicitly with --overlength-policy exclude."
        )
    overlength_ids = {
        str(record.get("conversation_id", "")) for record in overlength
    }

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
        if cid in overlength_ids:
            reason = "exceeds_model_turn_ceiling"
        elif cid in forbidden_ids:
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

    detection_view = list(primary) + list(auxiliary)
    diagnostics = turn_count_diagnostic(detection_view)

    # All checks and diagnostics complete before any artifact is written.
    write_jsonl(auxiliary, args.output)
    if args.excluded_output:
        write_jsonl(excluded, args.excluded_output)

    stats = {
        "input_integrity": {
            "primary_sha256": primary_sha,
            "expected_primary_sha256": args.expected_primary_sha256,
            "benign_source_git_blob": source_blob,
            "expected_benign_source_git_blob": args.expected_benign_source_git_blob,
        },
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
            "max_physical_turns": max(len(r.get("turns", [])) for r in source),
            "over_model_turn_ceiling": len(overlength),
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
            "max_physical_turns": (
                max(len(r.get("turns", [])) for r in auxiliary)
                if auxiliary else None
            ),
        },
        "excluded": {
            "records": len(excluded),
            "reasons": dict(reasons),
        },
        "model_turn_contract": {
            "max_physical_turns": args.max_physical_turns,
            "overlength_policy": args.overlength_policy,
        },
        "exclude_against": external_counts,
        "primary_plus_aux_detection_population": {
            "records": len(detection_view),
            "turn_count_diagnostic": diagnostics,
            "note": (
                "Population diagnostic only. Primary splitting must be frozen "
                "before auxiliary records are added to train."
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
            "transformer_auxiliary_schema_compatible": True,
        },
    }

    write_json(stats, args.stats_output)
    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"Wrote Dataset A detection auxiliary: {args.output}")
    if args.excluded_output:
        print(f"Wrote excluded records: {args.excluded_output}")


if __name__ == "__main__":
    main()
