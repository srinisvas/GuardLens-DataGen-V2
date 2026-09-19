#!/usr/bin/env python3
"""Leakage-safe train/dev/test split for merged GuardLens corpora.

Primary grouping uses ``metadata.consolidated_split_group``. Frontier records are
therefore grouped by complete scenario_family; legacy records retain pair linkage
when available. An optional detection-only auxiliary corpus can be split jointly
with the primary A+B corpus. Auxiliary records use ``detection_label`` for soft
balancing, but share the exact same ``frontier::<scenario_family>`` grouping as
primary Dataset B so related records can never cross partitions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List

from frontier_common import load_jsonl, write_jsonl

SPLITS = ("train", "dev", "test")


def n_user_turns(record: Dict) -> int:
    return sum(
        str(t.get("role", "")).lower() == "user"
        for t in record.get("turns", [])
    )


def training_label(record: Dict):
    """Return the task-appropriate detection label for split balancing."""
    if "detection_label" in record:
        return record.get("detection_label")
    return record.get("label")


def is_frontier(record: Dict) -> bool:
    return str(record.get("corpus_source", "")).startswith("frontier_authored_v3")


def frontier_author(record: Dict) -> str:
    metadata = record.get("metadata", {}) or {}
    return str(
        metadata.get("corpus_version")
        or metadata.get("generator")
        or record.get("seed_source")
        or "unknown_source"
    )


def group_records(records: List[Dict]) -> Dict[str, List[Dict]]:
    groups = defaultdict(list)
    for r in records:
        group = (r.get("metadata", {}) or {}).get("consolidated_split_group")
        if not group:
            raise RuntimeError(f"{r.get('conversation_id')}: missing consolidated_split_group")
        groups[str(group)].append(r)
    return dict(groups)


def group_signature(group: List[Dict]) -> Counter:
    """Return soft-balancing features for one indivisible split group."""
    c = Counter()
    for r in group:
        source = str(r.get("corpus_source", "unknown"))
        label = str(training_label(r))
        difficulty = str(r.get("difficulty", "unknown"))
        user_len = str(n_user_turns(r))
        c[("label", label)] += 1
        c[("source", source)] += 1
        c[("source_label", source, label)] += 1
        c[("source_difficulty", source, difficulty)] += 1
        c[("source_label_user_turns", source, label, user_len)] += 1

        if is_frontier(r):
            metadata = r.get("metadata", {}) or {}
            intended = r.get("intended_structure", {}) or {}
            author = frontier_author(r)
            c[("frontier_author", author)] += 1
            c[("frontier_author_label", author, label)] += 1
            c[("frontier_domain", str(r.get("target_domain", "unknown")))] += 1
            c[("frontier_slice_role", str(metadata.get("slice_role", "unknown")))] += 1
            c[("frontier_pair_hardness", str(intended.get("pair_hardness", "none")))] += 1
            c[("frontier_trajectory_family", str(intended.get("trajectory_family", "unknown")))] += 1
            c[("frontier_mechanism_family", str(metadata.get("mechanism_family", "unknown")))] += 1
            c[("frontier_style", str(r.get("style", "unknown")))] += 1
        else:
            c[("legacy_family", str(r.get("family", "unknown")))] += 1
    return c


def split_groups(groups: Dict[str, List[Dict]], fractions: Dict[str, float], seed: int):
    rng = random.Random(seed)
    total_records = sum(len(v) for v in groups.values())
    target_total = {s: total_records * fractions[s] for s in SPLITS}

    global_sig = Counter()
    for g in groups.values():
        global_sig.update(group_signature(g))
    target_sig = {
        s: {k: v * fractions[s] for k, v in global_sig.items()}
        for s in SPLITS
    }

    assigned = {s: [] for s in SPLITS}
    counts = {s: 0 for s in SPLITS}
    sig_counts = {s: Counter() for s in SPLITS}

    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda x: len(x[1]), reverse=True)

    for group_id, group in items:
        gsig = group_signature(group)
        best_split = None
        best_score = None
        for split_name in SPLITS:
            total_fill = (counts[split_name] + len(group)) / max(target_total[split_name], 1.0)
            signature_fills = []
            for key, amount in gsig.items():
                target = target_sig[split_name].get(key, 0.0)
                if target > 0:
                    signature_fills.append((sig_counts[split_name][key] + amount) / target)
            signature_fill = (
                sum(signature_fills) / len(signature_fills)
                if signature_fills else total_fill
            )
            score = 0.72 * total_fill + 0.28 * signature_fill + rng.random() * 1e-9
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name
        assigned[best_split].append((group_id, group))
        counts[best_split] += len(group)
        sig_counts[best_split].update(gsig)

    output = {}
    for split_name in SPLITS:
        output[split_name] = [r for _, group in assigned[split_name] for r in group]
        rng.shuffle(output[split_name])
    return output


def assert_no_leakage(splits: Dict[str, List[Dict]]) -> None:
    owner = {}
    ids = set()
    pair_owner = {}
    scenario_owner = {}
    hash_owner = {}
    for split_name, records in splits.items():
        for r in records:
            cid = str(r.get("conversation_id", ""))
            if cid in ids:
                raise RuntimeError(f"duplicate conversation_id across split material: {cid}")
            ids.add(cid)
            metadata = r.get("metadata", {}) or {}
            group = str(metadata.get("consolidated_split_group", "")).strip()
            if not group:
                raise RuntimeError(f"{cid}: missing consolidated_split_group in split material")
            previous = owner.setdefault(group, split_name)
            if previous != split_name:
                raise RuntimeError(f"split leakage: group {group} appears in {previous} and {split_name}")

            trajectory_hash = metadata.get("normalized_user_trajectory_hash")
            if trajectory_hash:
                previous = hash_owner.setdefault(str(trajectory_hash), split_name)
                if previous != split_name:
                    raise RuntimeError(
                        f"exact user-trajectory leakage: hash appears in {previous} and {split_name}"
                    )

            pair_id = r.get("pair_id")
            if pair_id not in (None, ""):
                key = (str(r.get("corpus_source")), str(pair_id))
                previous = pair_owner.setdefault(key, split_name)
                if previous != split_name:
                    raise RuntimeError(f"pair leakage: {key} appears in {previous} and {split_name}")

            if is_frontier(r):
                scenario = str(metadata.get("scenario_family", ""))
                if not scenario:
                    raise RuntimeError(f"{cid}: frontier record missing scenario_family")
                previous = scenario_owner.setdefault(scenario, split_name)
                if previous != split_name:
                    raise RuntimeError(
                        f"frontier scenario leakage: {scenario} appears in {previous} and {split_name}"
                    )


def describe(records: List[Dict]) -> Dict:
    frontier = [r for r in records if is_frontier(r)]
    return {
        "n": len(records),
        "labels": dict(Counter(str(training_label(r)) for r in records)),
        "authoring_labels": dict(Counter(str(r.get("label")) for r in records)),
        "sources": dict(Counter(str(r.get("corpus_source")) for r in records)),
        "source_label": dict(Counter(
            f"{r.get('corpus_source')}|{training_label(r)}" for r in records
        )),
        "source_label_user_turns": dict(Counter(
            f"{r.get('corpus_source')}|{training_label(r)}|{n_user_turns(r)}"
            for r in records
        )),
        "difficulty": dict(Counter(str(r.get("difficulty", "unknown")) for r in records)),
        "supervision_tiers": dict(Counter(str(r.get("supervision_tier")) for r in records)),
        "groups": len({(r.get("metadata", {}) or {}).get("consolidated_split_group") for r in records}),
        "frontier_author_corpora": dict(Counter(frontier_author(r) for r in frontier)),
        "frontier_author_label": dict(Counter(
            f"{frontier_author(r)}|{training_label(r)}" for r in frontier
        )),
        "frontier_scenario_families": len({
            (r.get("metadata", {}) or {}).get("scenario_family") for r in frontier
        }),
        "frontier_domains": dict(Counter(str(r.get("target_domain", "unknown")) for r in frontier)),
        "frontier_slice_roles": dict(Counter(
            str((r.get("metadata", {}) or {}).get("slice_role", "unknown")) for r in frontier
        )),
        "frontier_pair_hardness": dict(Counter(
            str((r.get("intended_structure", {}) or {}).get("pair_hardness", "none")) for r in frontier
        )),
        "frontier_trajectory_family": dict(Counter(
            str((r.get("intended_structure", {}) or {}).get("trajectory_family", "unknown")) for r in frontier
        )),
        "frontier_mechanism_families": len({
            (r.get("metadata", {}) or {}).get("mechanism_family") for r in frontier
        }),
    }


def assert_size_tolerance(
    splits: Dict[str, List[Dict]],
    fractions: Dict[str, float],
    max_group_size: int,
) -> None:
    total = sum(len(v) for v in splits.values())
    if total == 0:
        raise RuntimeError("cannot split an empty dataset")
    tolerance = max_group_size / total + 0.005
    for name in SPLITS:
        if not splits[name]:
            raise RuntimeError(f"split {name} is empty")
        actual = len(splits[name]) / total
        if abs(actual - fractions[name]) > tolerance:
            raise RuntimeError(
                f"split {name} ratio {actual:.4f} differs from target "
                f"{fractions[name]:.4f} beyond tolerance {tolerance:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Canonical primary A+B corpus")
    parser.add_argument(
        "--auxiliary-input",
        help="Optional detection-only Dataset B auxiliary corpus. When present, split jointly.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--dev-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total_fraction = args.train_frac + args.dev_frac + args.test_frac
    if not math.isclose(total_fraction, 1.0, abs_tol=1e-8):
        raise ValueError("train/dev/test fractions must sum to 1")
    if min(args.train_frac, args.dev_frac, args.test_frac) <= 0:
        raise ValueError("all train/dev/test fractions must be positive")
    fractions = {"train": args.train_frac, "dev": args.dev_frac, "test": args.test_frac}

    primary_records = load_jsonl(args.input)
    auxiliary_records = load_jsonl(args.auxiliary_input) if args.auxiliary_input else []
    records = primary_records + auxiliary_records
    ids = [str(r.get("conversation_id", "")) for r in records]
    duplicates = [cid for cid, n in Counter(ids).items() if cid and n > 1]
    if duplicates:
        raise RuntimeError(f"duplicate conversation_ids across primary/auxiliary inputs: {duplicates[:10]}")

    groups = group_records(records)
    splits = split_groups(groups, fractions, args.seed)
    assert_no_leakage(splits)
    assert_size_tolerance(splits, fractions, max(len(g) for g in groups.values()))

    os.makedirs(args.output_dir, exist_ok=True)
    for name, subset in splits.items():
        write_jsonl(subset, os.path.join(args.output_dir, f"{name}.jsonl"))

    metadata = {
        "input_records": len(records),
        "primary_input_records": len(primary_records),
        "auxiliary_input_records": len(auxiliary_records),
        "joint_auxiliary_split": bool(args.auxiliary_input),
        "input_groups": len(groups),
        "seed": args.seed,
        "fractions": fractions,
        "group_policy": (
            "metadata.consolidated_split_group; frontier scenario_family is shared across primary and "
            "auxiliary records; legacy pairs never cross partitions"
        ),
        "label_policy": "detection_label when present, otherwise label",
        "balance_policy": (
            "soft balance on task label, source, source×label, source×difficulty, source×label×user_turn_count, "
            "and for frontier: author corpus, author×label, target_domain, slice_role, pair_hardness, "
            "trajectory_family, mechanism_family, style"
        ),
        "splits": {name: describe(subset) for name, subset in splits.items()},
        "leakage_check": "passed",
        "ratio_tolerance_check": "passed",
    }
    with open(os.path.join(args.output_dir, "split_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
