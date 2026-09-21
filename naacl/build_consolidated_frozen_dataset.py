#!/usr/bin/env python3
"""Build the frozen GuardLens consolidated primary and train-only auxiliary view.

Order is deliberate and fail-closed:
1. Validate and merge paired primary Dataset A + paired primary Dataset B.
2. Split the primary corpus only, preserving A pair IDs and B scenario groups.
3. Freeze primary train/dev/test.
4. Validate A/B detection-only auxiliary records against all primary records.
5. Append auxiliaries to TRAIN ONLY.
6. Verify byte identity of primary dev/test and run leakage/shortcut diagnostics.

No auxiliary record participates in split allocation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from prepare_frontier_dataset import assert_expected_provenance


DEFAULT_B_TARGET = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_B_JUDGE = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

A_PRIMARY_RECORDS = 1032
A_PRIMARY_PAIRS = 516
A_AUX_RECORDS = 721
B_PRIMARY_RECORDS = 1402
B_PRIMARY_PAIRS = 701
B_AUX_RECORDS = 424
B_AUX_LABELS = Counter({0: 271, 1: 153})
B_AUX_DETECTION_WEIGHT = 0.25
EXPECTED_A_PRIMARY_SHA256 = (
    "f9021672150696b2c3a367b1a1c67edafa4ce2f1a6a83fc1293870eaa7e6c34f"
)
EXPECTED_A_AUX_SHA256 = (
    "9d17f3b094957ba2626184e98c33dde81a153a55e91fbb94799fbe7a0ae577ad"
)
SPLITS = ("train", "dev", "test")


def load_jsonl(path: str) -> List[Dict]:
    out: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return out


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


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_user_hash(record: Dict) -> str:
    parts = [
        " ".join(str(turn.get("text", "")).strip().split())
        for turn in record.get("turns", [])
        if str(turn.get("role", "")).lower() == "user"
    ]
    return hashlib.sha256(
        "\n<USER_TURN>\n".join(parts).encode("utf-8")
    ).hexdigest()


def physical_turn_hash(record: Dict) -> str:
    payload = [
        {
            "turn_id": t.get("turn_id"),
            "role": t.get("role"),
            "text": t.get("text"),
        }
        for t in record.get("turns", [])
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def user_turns(record: Dict) -> int:
    return sum(
        str(turn.get("role", "")).lower() == "user"
        for turn in record.get("turns", [])
    )


def validate_trajectory(record: Dict, source: str, max_turns: int) -> None:
    cid = str(record.get("conversation_id", "")) or "<missing>"
    turns = record.get("turns")
    if not isinstance(turns, list) or not turns:
        raise RuntimeError(f"{source}:{cid}: empty trajectory")
    if len(turns) > max_turns:
        raise RuntimeError(
            f"{source}:{cid}: {len(turns)} physical turns exceed max_turns={max_turns}"
        )
    expected_role = "user"
    for index, turn in enumerate(turns):
        if turn.get("turn_id") != index:
            raise RuntimeError(
                f"{source}:{cid}: non-contiguous turn_id at index {index}"
            )
        role = str(turn.get("role", "")).lower()
        if role != expected_role:
            raise RuntimeError(
                f"{source}:{cid}: expected {expected_role}, got {role!r} at {index}"
            )
        text = turn.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"{source}:{cid}: empty/non-string text at {index}")
        expected_role = "assistant" if expected_role == "user" else "user"
    if expected_role != "user":
        raise RuntimeError(f"{source}:{cid}: trajectory ends on unmatched user turn")
    declared = record.get("conversation_length")
    if declared is not None and int(declared) != len(turns):
        raise RuntimeError(
            f"{source}:{cid}: declared conversation_length mismatch"
        )


def validate_a_primary(records: Sequence[Dict], max_turns: int) -> None:
    if len(records) != A_PRIMARY_RECORDS:
        raise RuntimeError(
            f"A primary expected {A_PRIMARY_RECORDS}, got {len(records)}"
        )
    labels = Counter(r.get("label") for r in records)
    if labels != Counter({0: A_PRIMARY_PAIRS, 1: A_PRIMARY_PAIRS}):
        raise RuntimeError(f"A primary label mismatch: {dict(labels)}")

    groups = defaultdict(list)
    for record in records:
        validate_trajectory(record, "A-primary", max_turns)
        cid = str(record.get("conversation_id", ""))
        if not cid:
            raise RuntimeError("A primary missing conversation_id")
        pair_id = str(record.get("pair_id", ""))
        if not pair_id:
            raise RuntimeError(f"{cid}: A primary missing pair_id")
        groups[pair_id].append(record)

    if len(groups) != A_PRIMARY_PAIRS:
        raise RuntimeError(
            f"A primary expected {A_PRIMARY_PAIRS} pair IDs, got {len(groups)}"
        )
    for pair_id, group in groups.items():
        if len(group) != 2 or Counter(r.get("label") for r in group) != Counter({0: 1, 1: 1}):
            raise RuntimeError(
                f"A primary pair {pair_id} is not one benign + one malicious"
            )


def frontier_scenario(record: Dict) -> str:
    metadata = record.get("metadata", {}) or {}
    return str(metadata.get("scenario_family", ""))


def validate_b_primary(
    records: Sequence[Dict],
    max_turns: int,
    *,
    expected_target: str,
    expected_judge: str,
) -> None:
    if len(records) != B_PRIMARY_RECORDS:
        raise RuntimeError(
            f"B primary expected {B_PRIMARY_RECORDS}, got {len(records)}"
        )
    labels = Counter(r.get("label") for r in records)
    if labels != Counter({0: B_PRIMARY_PAIRS, 1: B_PRIMARY_PAIRS}):
        raise RuntimeError(f"B primary label mismatch: {dict(labels)}")

    pair_groups = defaultdict(list)
    scenarios = defaultdict(list)
    for record in records:
        validate_trajectory(record, "B-primary", max_turns)
        cid = str(record.get("conversation_id", ""))
        if not cid:
            raise RuntimeError("B primary missing conversation_id")
        if record.get("training_eligible") is not True:
            raise RuntimeError(f"{cid}: B primary is not training eligible")
        try:
            assert_expected_provenance(
                record,
                expected_target=expected_target,
                expected_judge=expected_judge,
                require_evidence=(record.get("label") == 1),
            )
        except Exception as exc:
            raise RuntimeError(
                f"{cid}: B canonical protocol-chain validation failed: {exc}"
            ) from exc
        if record.get("canonical_target_model") != expected_target:
            raise RuntimeError(
                f"{cid}: B target model mismatch: "
                f"{record.get('canonical_target_model')!r}"
            )
        if record.get("canonical_judge_model") != expected_judge:
            raise RuntimeError(
                f"{cid}: B judge model mismatch: "
                f"{record.get('canonical_judge_model')!r}"
            )
        if record.get("primary_pair_complete") is not True:
            raise RuntimeError(f"{cid}: B primary lacks primary_pair_complete=true")

        pair_id = str(record.get("pair_id", ""))
        if not pair_id:
            raise RuntimeError(f"{cid}: B primary missing pair_id")
        pair_groups[pair_id].append(record)

        scenario = frontier_scenario(record)
        if not scenario:
            raise RuntimeError(f"{cid}: B primary missing metadata.scenario_family")
        scenarios[scenario].append(record)

    if len(pair_groups) != B_PRIMARY_PAIRS:
        raise RuntimeError(
            f"B primary expected {B_PRIMARY_PAIRS} pair IDs, got {len(pair_groups)}"
        )
    for pair_id, group in pair_groups.items():
        if len(group) != 2 or Counter(r.get("label") for r in group) != Counter({0: 1, 1: 1}):
            raise RuntimeError(
                f"B primary pair {pair_id} is not one benign + one malicious"
            )


def validate_aux(
    records: Sequence[Dict],
    *,
    name: str,
    expected_records: int,
    max_turns: int,
    expected_labels: Counter | None = None,
    expected_weight: float | None = None,
) -> None:
    if len(records) != expected_records:
        raise RuntimeError(
            f"{name} expected {expected_records} records, got {len(records)}"
        )
    observed_labels = Counter()
    observed_weights = Counter()
    for record in records:
        validate_trajectory(record, name, max_turns)
        cid = str(record.get("conversation_id", ""))
        if not cid:
            raise RuntimeError(f"{name}: missing conversation_id")
        if record.get("auxiliary_detection_only") is not True and (
            record.get("use_as") != "auxiliary_detection_only"
        ):
            raise RuntimeError(f"{name}:{cid}: not marked detection-only auxiliary")
        detection_label = record.get("detection_label")
        if isinstance(detection_label, bool) or detection_label not in (0, 1):
            raise RuntimeError(f"{name}:{cid}: invalid detection_label")
        observed_labels[int(detection_label)] += 1
        weight = record.get("detection_loss_weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) <= 0
        ):
            raise RuntimeError(f"{name}:{cid}: invalid detection_loss_weight")
        observed_weights[float(weight)] += 1
        if expected_weight is not None and not math.isclose(
            float(weight), float(expected_weight), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(
                f"{name}:{cid}: detection weight {weight} differs from "
                f"expected {expected_weight}"
            )
        if record.get("localization_supervision_ignore") is not True:
            raise RuntimeError(f"{name}:{cid}: localization must be ignored")
        if record.get("pivot_supervision_ignore") is not True:
            raise RuntimeError(f"{name}:{cid}: pivot supervision must be ignored")
        if record.get("pivot_loss_weight") != 0.0:
            raise RuntimeError(f"{name}:{cid}: pivot_loss_weight must be 0")
        if record.get("span_loss_weight") != 0.0:
            raise RuntimeError(f"{name}:{cid}: span_loss_weight must be 0")

    if expected_labels is not None and observed_labels != expected_labels:
        raise RuntimeError(
            f"{name}: detection-label counts {dict(observed_labels)} differ "
            f"from expected {dict(expected_labels)}"
        )


def canonical_primary_record(record: Dict, corpus: str) -> Dict:
    out = copy.deepcopy(record)
    metadata = out.setdefault("metadata", {})
    if corpus == "A":
        pair_id = str(out.get("pair_id", ""))
        split_group = f"A::pair::{pair_id}"
        out["corpus_source"] = "legacy_restored_primary"
    elif corpus == "B":
        scenario = frontier_scenario(out)
        split_group = f"B::scenario::{scenario}"
        out["corpus_source"] = "frontier_authored_v3"
    else:
        raise RuntimeError(f"unsupported corpus {corpus!r}")

    metadata["consolidated_split_group"] = split_group
    metadata["normalized_user_trajectory_hash"] = normalized_user_hash(out)
    metadata["observable_turn_hash"] = physical_turn_hash(out)
    return out


def assert_global_primary_uniqueness(records: Sequence[Dict]) -> None:
    ids = Counter(str(r.get("conversation_id", "")) for r in records)
    duplicate_ids = [cid for cid, count in ids.items() if count > 1]
    if duplicate_ids:
        raise RuntimeError(
            f"duplicate primary conversation IDs: {duplicate_ids[:10]}"
        )

    hash_groups = defaultdict(set)
    for record in records:
        metadata = record.get("metadata", {}) or {}
        hash_groups[metadata["normalized_user_trajectory_hash"]].add(
            metadata["consolidated_split_group"]
        )
    bad = {h: groups for h, groups in hash_groups.items() if len(groups) > 1}
    if bad:
        preview = list(bad.items())[:10]
        raise RuntimeError(
            f"{len(bad)} normalized user trajectories span independent primary "
            f"groups. examples={preview}"
        )


def group_records(records: Sequence[Dict]) -> Dict[str, List[Dict]]:
    groups = defaultdict(list)
    for record in records:
        group = str(
            (record.get("metadata", {}) or {}).get("consolidated_split_group", "")
        )
        if not group:
            raise RuntimeError(
                f"{record.get('conversation_id')}: missing consolidated_split_group"
            )
        groups[group].append(record)
    return dict(groups)


def group_signature(group: Sequence[Dict]) -> Counter:
    sig = Counter()
    for record in group:
        source = str(record.get("corpus_source"))
        label = str(record.get("label"))
        turns = str(user_turns(record))
        sig[("source", source)] += 1
        sig[("label", label)] += 1
        sig[("source_label", source, label)] += 1
        sig[("source_label_turns", source, label, turns)] += 1

        if source == "frontier_authored_v3":
            metadata = record.get("metadata", {}) or {}
            intended = record.get("intended_structure", {}) or {}
            sig[("B_domain", str(record.get("target_domain", "unknown")))] += 1
            sig[("B_slice", str(metadata.get("slice_role", "unknown")))] += 1
            sig[("B_hardness", str(intended.get("pair_hardness", "unknown")))] += 1
            sig[("B_trajectory", str(intended.get("trajectory_family", "unknown")))] += 1
        else:
            sig[("A_family", str(record.get("family", "unknown")))] += 1
    return sig


def split_groups(
    groups: Dict[str, List[Dict]],
    *,
    fractions: Dict[str, float],
    seed: int,
) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    total = sum(len(group) for group in groups.values())
    target_total = {name: total * fractions[name] for name in SPLITS}

    global_sig = Counter()
    for group in groups.values():
        global_sig.update(group_signature(group))
    target_sig = {
        name: {key: value * fractions[name] for key, value in global_sig.items()}
        for name in SPLITS
    }

    assigned = {name: [] for name in SPLITS}
    count = {name: 0 for name in SPLITS}
    sig_count = {name: Counter() for name in SPLITS}

    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)

    for group_id, group in items:
        signature = group_signature(group)
        candidates = []
        for split_name in SPLITS:
            total_fill = (
                count[split_name] + len(group)
            ) / max(target_total[split_name], 1.0)
            feature_fills = []
            for key, amount in signature.items():
                target = target_sig[split_name].get(key, 0.0)
                if target > 0:
                    feature_fills.append(
                        (sig_count[split_name][key] + amount) / target
                    )
            feature_fill = (
                sum(feature_fills) / len(feature_fills)
                if feature_fills else total_fill
            )
            score = 0.72 * total_fill + 0.28 * feature_fill
            candidates.append((score, rng.random(), split_name))
        _, _, chosen = min(candidates)
        assigned[chosen].append((group_id, group))
        count[chosen] += len(group)
        sig_count[chosen].update(signature)

    result: Dict[str, List[Dict]] = {}
    for split_name in SPLITS:
        result[split_name] = [
            record
            for _, group in assigned[split_name]
            for record in group
        ]
        rng.shuffle(result[split_name])
    return result


def assert_primary_split_integrity(splits: Dict[str, List[Dict]]) -> None:
    id_owner: Dict[str, str] = {}
    pair_owner: Dict[Tuple[str, str], str] = {}
    group_owner: Dict[str, str] = {}
    hash_owner: Dict[str, str] = {}

    for split_name, records in splits.items():
        for record in records:
            cid = str(record.get("conversation_id", ""))
            if cid in id_owner:
                raise RuntimeError(
                    f"conversation {cid} appears in {id_owner[cid]} and {split_name}"
                )
            id_owner[cid] = split_name

            metadata = record.get("metadata", {}) or {}
            group = str(metadata.get("consolidated_split_group", ""))
            previous = group_owner.setdefault(group, split_name)
            if previous != split_name:
                raise RuntimeError(
                    f"group {group} appears in {previous} and {split_name}"
                )

            trajectory_hash = str(metadata.get("normalized_user_trajectory_hash", ""))
            previous = hash_owner.setdefault(trajectory_hash, split_name)
            if previous != split_name:
                raise RuntimeError(
                    "normalized user trajectory appears across primary splits"
                )

            pair_id = str(record.get("pair_id", ""))
            key = (str(record.get("corpus_source")), pair_id)
            previous = pair_owner.setdefault(key, split_name)
            if previous != split_name:
                raise RuntimeError(
                    f"pair {key} appears in {previous} and {split_name}"
                )


def check_split_ratios(
    splits: Dict[str, List[Dict]],
    fractions: Dict[str, float],
    max_group: int,
) -> None:
    total = sum(len(records) for records in splits.values())
    tolerance = max_group / total + 0.005
    for split_name in SPLITS:
        actual = len(splits[split_name]) / total
        if abs(actual - fractions[split_name]) > tolerance:
            raise RuntimeError(
                f"{split_name} ratio {actual:.4f} exceeds tolerance "
                f"around target {fractions[split_name]:.4f}"
            )


def detection_label(record: Dict) -> int:
    value = (
        record.get("detection_label")
        if (
            record.get("auxiliary_detection_only") is True
            or record.get("use_as") == "auxiliary_detection_only"
        )
        else record.get("label")
    )
    if isinstance(value, bool) or value not in (0, 1):
        raise RuntimeError(
            f"{record.get('conversation_id')}: invalid effective detection label"
        )
    return int(value)


def detection_weight(record: Dict) -> float:
    if (
        record.get("auxiliary_detection_only") is True
        or record.get("use_as") == "auxiliary_detection_only"
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
            f"{record.get('conversation_id')}: invalid effective detection weight"
        )
    return float(value)


def weighted_auc(records: Sequence[Dict]) -> float:
    positives = [
        (user_turns(r), detection_weight(r))
        for r in records if detection_label(r) == 1
    ]
    negatives = [
        (user_turns(r), detection_weight(r))
        for r in records if detection_label(r) == 0
    ]
    pos_mass = sum(weight for _, weight in positives)
    neg_mass = sum(weight for _, weight in negatives)
    if not positives or not negatives or pos_mass <= 0 or neg_mass <= 0:
        raise RuntimeError("turn-count AUC requires both classes")
    credit = 0.0
    for p_value, p_weight in positives:
        for n_value, n_weight in negatives:
            c = 1.0 if p_value > n_value else 0.5 if p_value == n_value else 0.0
            credit += c * p_weight * n_weight
    return credit / (pos_mass * neg_mass)


def unweighted_auc(records: Sequence[Dict]) -> float:
    copied = []
    for record in records:
        item = copy.deepcopy(record)
        if (
            item.get("auxiliary_detection_only") is True
            or item.get("use_as") == "auxiliary_detection_only"
        ):
            item["detection_loss_weight"] = 1.0
        copied.append(item)
    return weighted_auc(copied)


def describe(records: Sequence[Dict]) -> Dict:
    return {
        "records": len(records),
        "labels": dict(Counter(str(detection_label(r)) for r in records)),
        "primary_labels": dict(
            Counter(
                str(r.get("label"))
                for r in records
                if not (
                    r.get("auxiliary_detection_only") is True
                    or r.get("use_as") == "auxiliary_detection_only"
                )
            )
        ),
        "corpus_source": dict(
            Counter(str(r.get("corpus_source", "unknown")) for r in records)
        ),
        "auxiliary_records": sum(
            1 for r in records
            if (
                r.get("auxiliary_detection_only") is True
                or r.get("use_as") == "auxiliary_detection_only"
            )
        ),
        "max_physical_turns": max(len(r.get("turns", [])) for r in records),
        "turn_count_auc_unweighted": unweighted_auc(records),
        "turn_count_auc_detection_weighted": weighted_auc(records),
    }


def make_aux_copy(record: Dict, source: str) -> Dict:
    out = copy.deepcopy(record)
    out["corpus_source"] = source
    metadata = out.setdefault("metadata", {})
    metadata["normalized_user_trajectory_hash"] = normalized_user_hash(out)
    metadata["observable_turn_hash"] = physical_turn_hash(out)
    return out


def filter_aux_against_primary_and_each_other(
    a_aux: Sequence[Dict],
    b_aux: Sequence[Dict],
    primary: Sequence[Dict],
) -> Tuple[List[Dict], List[Dict], Dict]:
    forbidden_ids = {str(r.get("conversation_id", "")) for r in primary}
    forbidden_hashes = {normalized_user_hash(r) for r in primary}

    seen_ids = set(forbidden_ids)
    seen_hashes = set(forbidden_hashes)
    kept_a: List[Dict] = []
    kept_b: List[Dict] = []
    excluded = Counter()

    for source_name, records, target in (
        ("A_aux", a_aux, kept_a),
        ("B_aux", b_aux, kept_b),
    ):
        for record in records:
            cid = str(record.get("conversation_id", ""))
            h = normalized_user_hash(record)
            reason = None
            if cid in seen_ids:
                reason = "conversation_id_overlap"
            elif h in seen_hashes:
                reason = "normalized_user_trajectory_overlap"
            if reason:
                excluded[f"{source_name}:{reason}"] += 1
                continue
            seen_ids.add(cid)
            seen_hashes.add(h)
            target.append(
                make_aux_copy(
                    record,
                    "legacy_detection_aux" if source_name == "A_aux"
                    else "frontier_detection_aux",
                )
            )

    return kept_a, kept_b, dict(excluded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-primary", required=True)
    parser.add_argument("--b-primary", required=True)
    parser.add_argument("--a-aux", required=True)
    parser.add_argument("--b-aux", required=True)
    parser.add_argument("--b-primary-sha256", required=True)
    parser.add_argument("--b-aux-sha256", required=True)
    parser.add_argument("--expected-b-target-model", default=DEFAULT_B_TARGET)
    parser.add_argument("--expected-b-judge-model", default=DEFAULT_B_JUDGE)
    parser.add_argument(
        "--a-primary-sha256",
        default=EXPECTED_A_PRIMARY_SHA256,
    )
    parser.add_argument(
        "--a-aux-sha256",
        default=EXPECTED_A_AUX_SHA256,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--dev-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--max-turns", type=int, default=64)
    args = parser.parse_args()

    fractions = {
        "train": args.train_frac,
        "dev": args.dev_frac,
        "test": args.test_frac,
    }
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
        raise RuntimeError("split fractions must sum to 1")
    if min(fractions.values()) <= 0:
        raise RuntimeError("all split fractions must be positive")

    input_paths = {
        "a_primary": args.a_primary,
        "b_primary": args.b_primary,
        "a_aux": args.a_aux,
        "b_aux": args.b_aux,
    }
    actual_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    expected_hashes = {
        "a_primary": args.a_primary_sha256,
        "b_primary": args.b_primary_sha256,
        "a_aux": args.a_aux_sha256,
        "b_aux": args.b_aux_sha256,
    }
    mismatches = {
        name: {"expected": expected_hashes[name], "actual": actual_hashes[name]}
        for name in input_paths
        if actual_hashes[name] != expected_hashes[name]
    }
    if mismatches:
        raise RuntimeError(f"input SHA-256 mismatch: {mismatches}")

    a_primary = load_jsonl(args.a_primary)
    b_primary = load_jsonl(args.b_primary)
    a_aux = load_jsonl(args.a_aux)
    b_aux = load_jsonl(args.b_aux)

    validate_a_primary(a_primary, args.max_turns)
    validate_b_primary(
        b_primary,
        args.max_turns,
        expected_target=args.expected_b_target_model,
        expected_judge=args.expected_b_judge_model,
    )
    validate_aux(
        a_aux,
        name="A-aux",
        expected_records=A_AUX_RECORDS,
        max_turns=args.max_turns,
        expected_labels=Counter({0: A_AUX_RECORDS}),
        expected_weight=1.0,
    )
    validate_aux(
        b_aux,
        name="B-aux",
        expected_records=B_AUX_RECORDS,
        max_turns=args.max_turns,
        expected_labels=B_AUX_LABELS,
        expected_weight=B_AUX_DETECTION_WEIGHT,
    )

    primary = [
        canonical_primary_record(record, "A") for record in a_primary
    ] + [
        canonical_primary_record(record, "B") for record in b_primary
    ]
    if len(primary) != A_PRIMARY_RECORDS + B_PRIMARY_RECORDS:
        raise RuntimeError("unexpected consolidated primary record count")
    if Counter(r.get("label") for r in primary) != Counter({0: 1217, 1: 1217}):
        raise RuntimeError("consolidated primary is not exactly 1217/1217 balanced")

    assert_global_primary_uniqueness(primary)
    groups = group_records(primary)
    splits = split_groups(groups, fractions=fractions, seed=args.seed)
    assert_primary_split_integrity(splits)
    check_split_ratios(
        splits,
        fractions,
        max_group=max(len(group) for group in groups.values()),
    )

    kept_a_aux, kept_b_aux, aux_exclusions = filter_aux_against_primary_and_each_other(
        a_aux, b_aux, primary
    )
    if len(kept_a_aux) != A_AUX_RECORDS:
        raise RuntimeError(
            f"A auxiliary overlap removed {A_AUX_RECORDS - len(kept_a_aux)} records; "
            "review before freezing"
        )
    if len(kept_b_aux) != B_AUX_RECORDS:
        raise RuntimeError(
            f"B auxiliary overlap removed {B_AUX_RECORDS - len(kept_b_aux)} records; "
            "review before freezing"
        )

    rng = random.Random(args.seed)
    train_with_aux = (
        list(splits["train"]) + list(kept_a_aux) + list(kept_b_aux)
    )
    rng.shuffle(train_with_aux)

    # Aux must be train-only. Dev/test are exact primary subsets.
    primary_dev_ids = {str(r.get("conversation_id", "")) for r in splits["dev"]}
    primary_test_ids = {str(r.get("conversation_id", "")) for r in splits["test"]}
    aux_ids = {
        str(r.get("conversation_id", ""))
        for r in list(kept_a_aux) + list(kept_b_aux)
    }
    if aux_ids & primary_dev_ids or aux_ids & primary_test_ids:
        raise RuntimeError("auxiliary record leaked into dev/test")

    # No primary test access is needed downstream to create the train+aux artifact.
    os.makedirs(args.output_dir, exist_ok=True)
    primary_path = os.path.join(args.output_dir, "primary_all.jsonl")
    train_path = os.path.join(args.output_dir, "train_primary.jsonl")
    dev_path = os.path.join(args.output_dir, "dev_primary.jsonl")
    test_path = os.path.join(args.output_dir, "test_primary.jsonl")
    train_aux_path = os.path.join(args.output_dir, "train_with_aux.jsonl")
    a_aux_path = os.path.join(args.output_dir, "a_detection_aux.jsonl")
    b_aux_path = os.path.join(args.output_dir, "b_detection_aux.jsonl")

    write_jsonl(primary, primary_path)
    write_jsonl(splits["train"], train_path)
    write_jsonl(splits["dev"], dev_path)
    write_jsonl(splits["test"], test_path)
    write_jsonl(train_with_aux, train_aux_path)
    write_jsonl(kept_a_aux, a_aux_path)
    write_jsonl(kept_b_aux, b_aux_path)

    output_hashes = {
        name: sha256_file(path)
        for name, path in {
            "primary_all": primary_path,
            "train_primary": train_path,
            "dev_primary": dev_path,
            "test_primary": test_path,
            "train_with_aux": train_aux_path,
            "a_detection_aux": a_aux_path,
            "b_detection_aux": b_aux_path,
        }.items()
    }

    metadata = {
        "input_integrity": {
            "actual_sha256": actual_hashes,
            "expected_sha256": expected_hashes,
        },
        "counts": {
            "primary_all": len(primary),
            "a_primary": len(a_primary),
            "b_primary": len(b_primary),
            "a_detection_aux": len(kept_a_aux),
            "b_detection_aux": len(kept_b_aux),
            "train_primary": len(splits["train"]),
            "dev_primary": len(splits["dev"]),
            "test_primary": len(splits["test"]),
            "train_with_aux": len(train_with_aux),
        },
        "primary_splits": {
            name: describe(records) for name, records in splits.items()
        },
        "train_with_aux": describe(train_with_aux),
        "auxiliary_exclusions": aux_exclusions,
        "policy": {
            "primary_split_before_auxiliary_attachment": True,
            "a_grouping": "original generation-time pair_id",
            "b_grouping": "scenario_family",
            "auxiliary_train_only": True,
            "dev_is_primary_only": True,
            "test_is_primary_only": True,
            "normalized_user_trajectory_overlap_forbidden": True,
            "max_turns": args.max_turns,
            "seed": args.seed,
            "fractions": fractions,
            "b_expected_target_model": args.expected_b_target_model,
            "b_expected_judge_model": args.expected_b_judge_model,
            "b_aux_expected_labels": dict(B_AUX_LABELS),
            "b_aux_expected_detection_weight": B_AUX_DETECTION_WEIGHT,
        },
        "output_sha256": output_hashes,
    }
    write_json(metadata, os.path.join(args.output_dir, "freeze_manifest.json"))
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
