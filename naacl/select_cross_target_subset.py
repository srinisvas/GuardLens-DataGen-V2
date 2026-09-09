#!/usr/bin/env python3
"""Select a scenario-family-complete subset for Gemma cross-target robustness."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from typing import Dict, List

from frontier_common import load_jsonl, write_jsonl


def stratum(group: List[Dict]):
    first = group[0]
    intended = first.get("intended_structure", {}) or {}
    return (
        first.get("target_domain", "unknown"),
        first.get("difficulty", "unknown"),
        intended.get("trajectory_family", "unknown"),
        intended.get("pair_hardness", "none"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--fraction", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if not (0 < args.fraction <= 1):
        raise ValueError("fraction must be in (0,1]")

    records = load_jsonl(args.input)
    groups = defaultdict(list)
    for r in records:
        scenario = (r.get("metadata", {}) or {}).get("scenario_family")
        if not scenario:
            raise RuntimeError(f"{r.get('conversation_id')}: missing scenario_family")
        groups[scenario].append(r)

    strata = defaultdict(list)
    for scenario, group in groups.items():
        strata[stratum(group)].append((scenario, group))

    rng = random.Random(args.seed)
    selected_groups = []
    for _, items in sorted(strata.items(), key=lambda x: str(x[0])):
        rng.shuffle(items)
        n = max(1, int(round(len(items) * args.fraction)))
        selected_groups.extend(items[:n])

    selected = [r for _, group in selected_groups for r in group]
    rng.shuffle(selected)
    write_jsonl(selected, args.output)
    stats = {
        "source_records": len(records),
        "source_scenario_families": len(groups),
        "selected_records": len(selected),
        "selected_scenario_families": len(selected_groups),
        "record_fraction": round(len(selected) / len(records), 4) if records else 0,
        "labels": dict(Counter(str(r.get("label")) for r in selected)),
        "difficulty": dict(Counter(str(r.get("difficulty")) for r in selected)),
        "domains": dict(Counter(str(r.get("target_domain")) for r in selected)),
        "policy": "complete scenario families; stratified by domain, difficulty, trajectory family, and pair hardness",
        "target_model": "google/gemma-3-27b-it",
    }
    with open(args.stats_output, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
