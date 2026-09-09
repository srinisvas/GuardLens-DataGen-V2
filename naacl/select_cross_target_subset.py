#!/usr/bin/env python3
"""Select a scenario-family-complete subset for Gemma cross-target robustness.

Selection is outcome-blind and occurs from the source corpus. Complete scenario
families are indivisible. Paired scenario families and standalone hard-benign
families are sampled separately so the robustness subset preserves the source
construction mix. Within each construction type, a deterministic greedy allocator
minimizes residual error over the full scenario-feature distribution, including
author corpus for multi-author sources.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from frontier_common import load_jsonl, write_jsonl


def author_corpus(record: Dict) -> str:
    metadata = record.get("metadata", {}) or {}
    return str(
        metadata.get("corpus_version")
        or metadata.get("generator")
        or record.get("seed_source")
        or "unknown_source"
    )


def group_kind(group: List[Dict]) -> str:
    paired = [r.get("pair_id") not in (None, "") for r in group]
    if all(paired):
        return "paired"
    if not any(paired):
        return "standalone"
    raise RuntimeError("scenario family mixes paired and standalone records")


def feature_counter(group: List[Dict]) -> Counter:
    c = Counter()
    for r in group:
        metadata = r.get("metadata", {}) or {}
        intended = r.get("intended_structure", {}) or {}
        c[("author_corpus", author_corpus(r))] += 1
        c[("label", str(r.get("label")))] += 1
        c[("domain", str(r.get("target_domain", "unknown")))] += 1
        c[("difficulty", str(r.get("difficulty", "unknown")))] += 1
        c[("trajectory_family", str(intended.get("trajectory_family", "unknown")))] += 1
        c[("pair_hardness", str(intended.get("pair_hardness", "none")))] += 1
        c[("slice_role", str(metadata.get("slice_role", "unknown")))] += 1
        c[("mechanism_family", str(metadata.get("mechanism_family", "unknown")))] += 1
        c[("style", str(r.get("style", "unknown")))] += 1
    return c


def select_balanced_groups(
    items: List[Tuple[str, List[Dict]]],
    *,
    fraction: float,
    rng: random.Random,
) -> List[Tuple[str, List[Dict]]]:
    """Greedily minimize full normalized residual to the target distribution."""
    if not items:
        return []
    n_select = max(1, min(len(items), int(round(len(items) * fraction))))

    feature_cache = {}
    global_features = Counter()
    for scenario, group in items:
        features = feature_counter(group)
        feature_cache[scenario] = features
        global_features.update(features)

    target = {key: value * fraction for key, value in global_features.items()}
    keys = list(target)

    remaining = list(items)
    rng.shuffle(remaining)
    selected: List[Tuple[str, List[Dict]]] = []
    selected_features = Counter()

    while len(selected) < n_select:
        best_idx = None
        best_score = None
        for idx, (scenario, _) in enumerate(remaining):
            candidate = feature_cache[scenario]
            score = 0.0
            for key in keys:
                after = selected_features[key] + candidate.get(key, 0)
                denom = max(target[key], 1.0)
                score += ((after - target[key]) / denom) ** 2
            score += rng.random() * 1e-12
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        scenario, group = remaining.pop(best_idx)
        selected.append((scenario, group))
        selected_features.update(feature_cache[scenario])

    return selected


def summarize(records: List[Dict]) -> Dict:
    return {
        "records": len(records),
        "author_corpora": dict(Counter(author_corpus(r) for r in records)),
        "labels": dict(Counter(str(r.get("label")) for r in records)),
        "difficulty": dict(Counter(str(r.get("difficulty")) for r in records)),
        "domains": dict(Counter(str(r.get("target_domain")) for r in records)),
        "trajectory_family": dict(Counter(
            str((r.get("intended_structure", {}) or {}).get("trajectory_family", "unknown"))
            for r in records
        )),
        "pair_hardness": dict(Counter(
            str((r.get("intended_structure", {}) or {}).get("pair_hardness", "none"))
            for r in records
        )),
        "slice_role": dict(Counter(
            str((r.get("metadata", {}) or {}).get("slice_role", "unknown"))
            for r in records
        )),
        "styles": dict(Counter(str(r.get("style", "unknown")) for r in records)),
        "mechanism_families": len({
            (r.get("metadata", {}) or {}).get("mechanism_family") for r in records
        }),
    }


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
    ids = [str(r.get("conversation_id", "")) for r in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("input contains duplicate conversation_id values")

    groups = defaultdict(list)
    for r in records:
        scenario = (r.get("metadata", {}) or {}).get("scenario_family")
        if not scenario:
            raise RuntimeError(f"{r.get('conversation_id')}: missing scenario_family")
        groups[str(scenario)].append(r)

    by_kind = defaultdict(list)
    for scenario, group in groups.items():
        by_kind[group_kind(group)].append((scenario, group))

    rng = random.Random(args.seed)
    selected_groups: List[Tuple[str, List[Dict]]] = []
    kind_stats = {}
    for kind in sorted(by_kind):
        chosen = select_balanced_groups(
            by_kind[kind],
            fraction=args.fraction,
            rng=rng,
        )
        selected_groups.extend(chosen)
        kind_stats[kind] = {
            "source_scenario_families": len(by_kind[kind]),
            "selected_scenario_families": len(chosen),
            "source_records": sum(len(g) for _, g in by_kind[kind]),
            "selected_records": sum(len(g) for _, g in chosen),
        }

    selected = [r for _, group in selected_groups for r in group]
    rng.shuffle(selected)
    write_jsonl(selected, args.output)
    stats = {
        "source_records": len(records),
        "source_scenario_families": len(groups),
        "selected_records": len(selected),
        "selected_scenario_families": len(selected_groups),
        "record_fraction": round(len(selected) / len(records), 4) if records else 0,
        "scenario_fraction": round(len(selected_groups) / len(groups), 4) if groups else 0,
        "selection_by_kind": kind_stats,
        "source_distribution": summarize(records),
        "selected_distribution": summarize(selected),
        "policy": (
            "source-only outcome-blind selection; complete scenario families; paired and standalone "
            "families sampled separately; full normalized residual minimized across author corpus, domain, "
            "difficulty, trajectory family, pair hardness, slice role, mechanism family, style, and label"
        ),
        "target_model": "google/gemma-3-27b-it",
        "seed": args.seed,
    }
    with open(args.stats_output, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
