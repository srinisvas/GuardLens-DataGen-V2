#!/usr/bin/env python3
"""Select a deterministic, diverse, pair-complete frontier smoke subset.

The default selects 10 complete malicious/benign twin pairs (20 records) from
GuardLensSourceTrajectory/v3. Selection uses source metadata only and is outcome
blind. It greedily favors underrepresented construction features so the smoke
covers context-required and surface-control pairs plus varied domains, styles,
difficulties, trajectory families, and mechanisms.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

from frontier_common import load_jsonl, write_jsonl


def pair_features(group):
    malicious = next(r for r in group if r.get("label") == 1)
    metadata = malicious.get("metadata", {}) or {}
    intended = malicious.get("intended_structure", {}) or {}
    return {
        "pair_hardness": str(intended.get("pair_hardness", "unknown")),
        "difficulty": str(malicious.get("difficulty", "unknown")),
        "domain": str(malicious.get("target_domain", "unknown")),
        "style": str(malicious.get("style", "unknown")),
        "trajectory_family": str(intended.get("trajectory_family", "unknown")),
        "mechanism_family": str(metadata.get("mechanism_family", "unknown")),
    }


def validate_pairs(records):
    groups = defaultdict(list)
    for record in records:
        pair_id = record.get("pair_id")
        if pair_id in (None, ""):
            continue
        groups[str(pair_id)].append(record)

    valid = {}
    for pair_id, group in groups.items():
        labels = Counter(r.get("label") for r in group)
        if len(group) != 2 or labels != Counter({0: 1, 1: 1}):
            raise RuntimeError(
                f"source pair {pair_id} is not exactly one malicious + one benign twin: "
                f"n={len(group)} labels={dict(labels)}"
            )
        valid[pair_id] = group
    if not valid:
        raise RuntimeError("no complete source pairs found")
    return valid


def select_pairs(groups, n_pairs, seed):
    rng = random.Random(seed)
    remaining = list(groups.items())
    rng.shuffle(remaining)
    selected = []
    counts = defaultdict(Counter)

    while remaining and len(selected) < n_pairs:
        best_idx = None
        best_score = None
        for idx, (pair_id, group) in enumerate(remaining):
            features = pair_features(group)
            # Lower score is better. Rare/unused feature values are preferred.
            score = 0.0
            for key, value in features.items():
                score += counts[key][value]
            # Give pair-hardness coverage extra priority because the 500/100
            # source imbalance can otherwise omit surface controls in a tiny smoke.
            score += 2.0 * counts["pair_hardness"][features["pair_hardness"]]
            score += rng.random() * 1e-9
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        pair_id, group = remaining.pop(best_idx)
        selected.append((pair_id, group))
        for key, value in pair_features(group).items():
            counts[key][value] += 1

    if len(selected) != n_pairs:
        raise RuntimeError(f"requested {n_pairs} pairs, selected {len(selected)}")

    # Require both construction hardness classes in the default-scale smoke
    # whenever the source offers them.
    available_hardness = {
        pair_features(group)["pair_hardness"] for group in groups.values()
    }
    selected_hardness = {
        pair_features(group)["pair_hardness"] for _, group in selected
    }
    if n_pairs >= 2 and len(available_hardness) >= 2 and len(selected_hardness) < 2:
        raise RuntimeError("smoke selection failed to cover both pair-hardness classes")
    return selected


def summarize(selected):
    pair_rows = [pair_features(group) for _, group in selected]
    return {
        "pairs": len(selected),
        "records": 2 * len(selected),
        "pair_hardness": dict(Counter(x["pair_hardness"] for x in pair_rows)),
        "difficulty": dict(Counter(x["difficulty"] for x in pair_rows)),
        "domains": dict(Counter(x["domain"] for x in pair_rows)),
        "styles": dict(Counter(x["style"] for x in pair_rows)),
        "trajectory_family": dict(Counter(x["trajectory_family"] for x in pair_rows)),
        "mechanism_families": len({x["mechanism_family"] for x in pair_rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()
    if args.pairs <= 0:
        raise ValueError("pairs must be positive")

    source = load_jsonl(args.input)
    groups = validate_pairs(source)
    selected = select_pairs(groups, args.pairs, args.seed)
    selected_ids = {pair_id for pair_id, _ in selected}

    # Preserve source ordering so downstream sharding/checkpoint behavior is
    # deterministic and easy to compare with the full run.
    output = [r for r in source if str(r.get("pair_id")) in selected_ids]
    if len(output) != 2 * args.pairs:
        raise RuntimeError(
            f"pair-complete smoke should contain {2 * args.pairs} records, got {len(output)}"
        )
    write_jsonl(output, args.output)

    stats = {
        "source_records": len(source),
        "source_complete_pairs": len(groups),
        "selection": summarize(selected),
        "selected_pair_ids": [pair_id for pair_id, _ in selected],
        "seed": args.seed,
        "policy": "source-only outcome-blind diverse complete-pair smoke selection",
    }
    with open(args.stats_output, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    print(json.dumps(stats, indent=2))
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
