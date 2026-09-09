#!/usr/bin/env python3
"""Leakage-safe train/dev/test split for merged GuardLens corpora.

Primary grouping uses ``metadata.consolidated_split_group``. Frontier records are
therefore grouped by scenario_family, keeping all twins and scenario variants in
a single partition. Legacy records retain pair linkage when available.
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


def group_records(records: List[Dict]) -> Dict[str, List[Dict]]:
    groups = defaultdict(list)
    for r in records:
        group = (r.get("metadata", {}) or {}).get("consolidated_split_group")
        if not group:
            raise RuntimeError(f"{r.get('conversation_id')}: missing consolidated_split_group")
        groups[str(group)].append(r)
    return dict(groups)


def group_signature(group: List[Dict]) -> Counter:
    c = Counter()
    for r in group:
        c[("label", str(r.get("label")))] += 1
        c[("source", str(r.get("corpus_source")))] += 1
        c[("difficulty", str(r.get("difficulty", "unknown")))] += 1
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
        for s in SPLITS:
            new_total = counts[s] + len(group)
            total_err = ((new_total - target_total[s]) / max(target_total[s], 1.0)) ** 2
            sig_err = 0.0
            for key, amount in gsig.items():
                target = max(target_sig[s].get(key, 0.0), 1.0)
                new_value = sig_counts[s][key] + amount
                sig_err += ((new_value - target) / target) ** 2
            score = 4.0 * total_err + 0.25 * sig_err + rng.random() * 1e-9
            if best_score is None or score < best_score:
                best_score = score
                best_split = s
        assigned[best_split].append((group_id, group))
        counts[best_split] += len(group)
        sig_counts[best_split].update(gsig)

    output = {}
    for s in SPLITS:
        output[s] = [r for _, group in assigned[s] for r in group]
        rng.shuffle(output[s])
    return output


def assert_no_leakage(splits: Dict[str, List[Dict]]) -> None:
    owner = {}
    ids = set()
    for split_name, records in splits.items():
        for r in records:
            cid = str(r.get("conversation_id", ""))
            if cid in ids:
                raise RuntimeError(f"duplicate conversation_id across split material: {cid}")
            ids.add(cid)
            group = str((r.get("metadata", {}) or {}).get("consolidated_split_group", ""))
            previous = owner.setdefault(group, split_name)
            if previous != split_name:
                raise RuntimeError(f"split leakage: group {group} appears in {previous} and {split_name}")


def describe(records: List[Dict]) -> Dict:
    return {
        "n": len(records),
        "labels": dict(Counter(str(r.get("label")) for r in records)),
        "sources": dict(Counter(str(r.get("corpus_source")) for r in records)),
        "difficulty": dict(Counter(str(r.get("difficulty", "unknown")) for r in records)),
        "supervision_tiers": dict(Counter(str(r.get("supervision_tier")) for r in records)),
        "groups": len({(r.get("metadata", {}) or {}).get("consolidated_split_group") for r in records}),
        "frontier_scenario_families": len({
            (r.get("metadata", {}) or {}).get("scenario_family")
            for r in records if r.get("corpus_source") == "frontier_authored_v3"
        }),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--dev-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total_fraction = args.train_frac + args.dev_frac + args.test_frac
    if not math.isclose(total_fraction, 1.0, abs_tol=1e-8):
        raise ValueError("train/dev/test fractions must sum to 1")
    fractions = {"train": args.train_frac, "dev": args.dev_frac, "test": args.test_frac}

    records = load_jsonl(args.input)
    groups = group_records(records)
    splits = split_groups(groups, fractions, args.seed)
    assert_no_leakage(splits)

    os.makedirs(args.output_dir, exist_ok=True)
    for name, subset in splits.items():
        write_jsonl(subset, os.path.join(args.output_dir, f"{name}.jsonl"))

    metadata = {
        "input_records": len(records),
        "input_groups": len(groups),
        "seed": args.seed,
        "fractions": fractions,
        "group_policy": "metadata.consolidated_split_group; frontier scenario_family and legacy pairs never cross partitions",
        "splits": {name: describe(subset) for name, subset in splits.items()},
        "leakage_check": "passed",
    }
    with open(os.path.join(args.output_dir, "split_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
