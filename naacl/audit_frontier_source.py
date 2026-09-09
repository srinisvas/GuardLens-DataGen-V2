#!/usr/bin/env python3
"""CPU-only preflight audit for GuardLensSourceTrajectory/v3.

The strict mode verifies the complete 1,500-record construction contract. Use
``--schema-only`` for smoke-test or preselected robustness subsets; that mode
still validates every source record and uniqueness but intentionally skips full-
corpus pair/scenario/count expectations.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from typing import Dict, List

from frontier_common import assert_frontier_source_record, load_jsonl


def user_turns(record: Dict) -> List[Dict]:
    return [
        t for t in record.get("turns", [])
        if str(t.get("role", "")).lower() == "user"
    ]


def n_user(record: Dict) -> int:
    return len(user_turns(record))


def full_user_sequence(record: Dict):
    return tuple(str(t.get("text", "")) for t in user_turns(record))


def auc_from_scores(labels: List[int], scores: List[float]) -> float:
    """Tie-aware binary ROC AUC via pairwise ordering; small CPU diagnostic."""
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--expected-records", type=int, default=1500)
    parser.add_argument("--expected-pairs", type=int, default=600)
    parser.add_argument("--expected-standalone", type=int, default=300)
    parser.add_argument("--expected-scenarios", type=int, default=300)
    parser.add_argument("--max-primary-length-auc", type=float, default=0.65)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    errors = []
    warnings = []
    ids = set()
    pairs = defaultdict(list)
    scenarios = defaultdict(list)
    full_sequences = Counter()

    for record in records:
        cid = str(record.get("conversation_id", ""))
        try:
            assert_frontier_source_record(record)
        except Exception as exc:
            errors.append(f"{cid or '<missing>'}: {exc}")
        if cid in ids:
            errors.append(f"duplicate conversation_id: {cid}")
        ids.add(cid)
        pair_id = record.get("pair_id")
        if pair_id not in (None, ""):
            pairs[str(pair_id)].append(record)
        scenario = str((record.get("metadata", {}) or {}).get("scenario_family", ""))
        scenarios[scenario].append(record)
        full_sequences[full_user_sequence(record)] += 1

    duplicate_sequences = sum(n - 1 for n in full_sequences.values() if n > 1)
    if duplicate_sequences:
        errors.append(f"duplicate complete user trajectories: {duplicate_sequences}")

    labels = Counter(r.get("label") for r in records)
    if args.schema_only:
        print("=== Frontier source schema preflight ===")
        print(f"Records: {len(records)}  Labels: {dict(labels)}")
        print(f"Scenario families represented: {len(scenarios)}")
        if errors:
            print("SOURCE PREFLIGHT FAILED", file=sys.stderr)
            for error in errors[:100]:
                print(f"ERROR: {error}", file=sys.stderr)
            sys.exit(2)
        print("SOURCE PREFLIGHT PASSED (schema-only subset mode)")
        return

    if len(records) != args.expected_records:
        errors.append(f"expected {args.expected_records} source records, got {len(records)}")
    expected_labels = {1: args.expected_pairs, 0: args.expected_pairs + args.expected_standalone}
    if dict(labels) != expected_labels:
        errors.append(f"unexpected label counts {dict(labels)}, expected {expected_labels}")
    if len(pairs) != args.expected_pairs:
        errors.append(f"expected {args.expected_pairs} pair IDs, got {len(pairs)}")

    paired_records = []
    pair_hardness = Counter()
    for pair_id, group in pairs.items():
        group_labels = Counter(r.get("label") for r in group)
        if len(group) != 2 or group_labels != Counter({0: 1, 1: 1}):
            errors.append(f"pair {pair_id}: n={len(group)} labels={dict(group_labels)}")
            continue
        malicious = next(r for r in group if r.get("label") == 1)
        benign = next(r for r in group if r.get("label") == 0)
        paired_records.extend(group)
        if n_user(malicious) != n_user(benign):
            errors.append(f"pair {pair_id}: user-turn counts differ")
        mal_users = user_turns(malicious)
        ben_users = user_turns(benign)
        if not mal_users or not ben_users or mal_users[0].get("text") != ben_users[0].get("text"):
            errors.append(f"pair {pair_id}: first user turn is not byte-identical")
        mh = (malicious.get("intended_structure", {}) or {}).get("pair_hardness")
        bh = (benign.get("intended_structure", {}) or {}).get("pair_hardness")
        if mh != bh:
            errors.append(f"pair {pair_id}: pair_hardness differs across twins")
        pair_hardness[str(mh)] += 1
        if mh == "context_required":
            if mal_users[-1].get("text") != ben_users[-1].get("text"):
                errors.append(f"pair {pair_id}: context-required final user turn differs")
        elif mh == "surface_control":
            if mal_users[-1].get("text") == ben_users[-1].get("text"):
                errors.append(f"pair {pair_id}: surface-control final user turn is identical")
        else:
            errors.append(f"pair {pair_id}: unsupported pair_hardness={mh!r}")

    standalone = [r for r in records if r.get("pair_id") in (None, "")]
    if len(standalone) != args.expected_standalone:
        errors.append(f"expected {args.expected_standalone} standalone records, got {len(standalone)}")
    if any(r.get("label") != 0 for r in standalone):
        errors.append("standalone source slice contains non-benign records")
    if any((r.get("metadata", {}) or {}).get("slice_role") != "standalone_benign" for r in standalone):
        errors.append("standalone records do not all have metadata.slice_role=standalone_benign")

    if len(scenarios) != args.expected_scenarios:
        errors.append(f"expected {args.expected_scenarios} scenario families, got {len(scenarios)}")
    paired_scenarios = standalone_scenarios = 0
    for scenario, group in scenarios.items():
        is_paired = [r.get("pair_id") not in (None, "") for r in group]
        if all(is_paired):
            paired_scenarios += 1
            if len(group) != 6 or len({r.get("pair_id") for r in group}) != 3:
                errors.append(f"scenario {scenario}: paired family must contain 3 complete pairs / 6 records")
        elif not any(is_paired):
            standalone_scenarios += 1
            if len(group) != 3:
                errors.append(f"scenario {scenario}: standalone family must contain 3 records")
        else:
            errors.append(f"scenario {scenario}: mixes paired and standalone construction")

    paired_mal_hist = Counter(n_user(r) for r in paired_records if r.get("label") == 1)
    paired_ben_hist = Counter(n_user(r) for r in paired_records if r.get("label") == 0)
    if paired_mal_hist != paired_ben_hist:
        errors.append(
            f"paired source user-turn histograms differ: mal={dict(paired_mal_hist)} ben={dict(paired_ben_hist)}"
        )

    paired_labels = [int(r.get("label")) for r in paired_records]
    paired_lengths = [n_user(r) for r in paired_records]
    paired_auc = auc_from_scores(paired_labels, paired_lengths)
    if paired_auc > args.max_primary_length_auc:
        errors.append(
            f"paired primary-eligible length AUC={paired_auc:.4f} exceeds {args.max_primary_length_auc:.4f}"
        )

    all_auc = auc_from_scores(
        [int(r.get("label")) for r in records],
        [n_user(r) for r in records],
    )
    if all_auc > args.max_primary_length_auc:
        warnings.append(
            f"full-source user-turn length AUC={all_auc:.4f}; expected because standalone hard-benign stress records are shorter. "
            "They must remain evaluation-only and must not enter primary training."
        )

    print("=== Frontier source preflight audit ===")
    print(f"Records: {len(records)}  Labels: {dict(labels)}")
    print(f"Pairs: {len(pairs)}  Pair hardness: {dict(pair_hardness)}")
    print(f"Standalone benign: {len(standalone)}")
    print(f"Scenario families: {len(scenarios)} paired={paired_scenarios} standalone={standalone_scenarios}")
    print(f"Paired class user-turn histograms identical: {paired_mal_hist == paired_ben_hist}")
    print(f"Paired primary length AUC: {paired_auc:.4f}")
    print(f"Full source length AUC: {all_auc:.4f}")
    for warning in warnings:
        print(f"WARN: {warning}")
    if errors:
        print("SOURCE PREFLIGHT FAILED", file=sys.stderr)
        for error in errors[:100]:
            print(f"ERROR: {error}", file=sys.stderr)
        if len(errors) > 100:
            print(f"... {len(errors)-100} additional errors", file=sys.stderr)
        sys.exit(2)
    print("SOURCE PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
