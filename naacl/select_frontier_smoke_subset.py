#!/usr/bin/env python3
"""Select a deterministic, diverse, pair-complete frontier smoke subset.

For a multi-author merged source, selection is balanced exactly across author
corpora when the requested pair count permits it. Selection uses source metadata
only and is outcome blind.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

from frontier_common import load_jsonl, write_jsonl


def source_author(record):
    metadata = record.get("metadata", {}) or {}
    return str(
        metadata.get("corpus_version")
        or metadata.get("generator")
        or record.get("seed_source")
        or "unknown_source"
    )


def pair_features(group):
    malicious = next(r for r in group if r.get("label") == 1)
    metadata = malicious.get("metadata", {}) or {}
    intended = malicious.get("intended_structure", {}) or {}
    return {
        "source_author": source_author(malicious),
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
        authors = {source_author(r) for r in group}
        if len(authors) != 1:
            raise RuntimeError(f"pair {pair_id} spans author corpora: {sorted(authors)}")
        valid[pair_id] = group
    if not valid:
        raise RuntimeError("no complete source pairs found")
    return valid


def author_quotas(groups, n_pairs):
    by_author = defaultdict(list)
    for pair_id, group in groups.items():
        by_author[pair_features(group)["source_author"]].append(pair_id)
    authors = sorted(by_author)
    if n_pairs < len(authors):
        raise RuntimeError(
            f"requested {n_pairs} pairs but source has {len(authors)} author corpora; "
            "cannot cover every author"
        )
    base, rem = divmod(n_pairs, len(authors))
    quotas = {
        author: base + (1 if idx < rem else 0)
        for idx, author in enumerate(authors)
    }
    for author, quota in quotas.items():
        if quota > len(by_author[author]):
            raise RuntimeError(
                f"author {author} has {len(by_author[author])} complete pairs, quota={quota}"
            )
    return quotas


def select_pairs(groups, n_pairs, seed):
    rng = random.Random(seed)
    remaining = list(groups.items())
    rng.shuffle(remaining)
    selected = []
    counts = defaultdict(Counter)
    quotas = author_quotas(groups, n_pairs)
    selected_by_author = Counter()

    while remaining and len(selected) < n_pairs:
        best_idx = None
        best_score = None
        for idx, (pair_id, group) in enumerate(remaining):
            features = pair_features(group)
            author = features["source_author"]
            if selected_by_author[author] >= quotas[author]:
                continue

            # Lower score is better. Rare/unused values are preferred. Author
            # balance itself is a hard quota rather than merely a soft feature.
            score = 0.0
            for key, value in features.items():
                if key != "source_author":
                    score += counts[key][value]
            score += 2.0 * counts["pair_hardness"][features["pair_hardness"]]
            score += rng.random() * 1e-9
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            raise RuntimeError(
                f"could not satisfy author quotas {quotas}; selected={dict(selected_by_author)}"
            )
        pair_id, group = remaining.pop(best_idx)
        selected.append((pair_id, group))
        features = pair_features(group)
        selected_by_author[features["source_author"]] += 1
        for key, value in features.items():
            counts[key][value] += 1

    if len(selected) != n_pairs:
        raise RuntimeError(f"requested {n_pairs} pairs, selected {len(selected)}")
    if dict(selected_by_author) != quotas:
        raise RuntimeError(
            f"author-balanced smoke selection failed: selected={dict(selected_by_author)} quotas={quotas}"
        )

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
        "source_authors": dict(Counter(x["source_author"] for x in pair_rows)),
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

    output = [r for r in source if str(r.get("pair_id")) in selected_ids]
    if len(output) != 2 * args.pairs:
        raise RuntimeError(
            f"pair-complete smoke should contain {2 * args.pairs} records, got {len(output)}"
        )
    write_jsonl(output, args.output)

    stats = {
        "source_records": len(source),
        "source_complete_pairs": len(groups),
        "source_author_pairs": dict(Counter(
            pair_features(group)["source_author"] for group in groups.values()
        )),
        "selection": summarize(selected),
        "selected_pair_ids": [pair_id for pair_id, _ in selected],
        "seed": args.seed,
        "policy": "source-only outcome-blind diverse complete-pair smoke selection with exact author quotas",
    }
    with open(args.stats_output, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    print(json.dumps(stats, indent=2))
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
