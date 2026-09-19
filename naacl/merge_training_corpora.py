#!/usr/bin/env python3
"""Merge frozen legacy Dataset A and canonical frontier Dataset B.

The merge is deliberately fail-closed. Besides identifier collisions, exact
normalized user trajectories are forbidden from crossing corpus or split-group
boundaries because they would create content leakage even when IDs differ.
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
from typing import Dict

from frontier_common import load_jsonl, write_jsonl
from prepare_frontier_dataset import assert_expected_provenance

DEFAULT_TARGET = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_JUDGE = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
EXPECTED_LEGACY_RECORDS = 1052
EXPECTED_FRONTIER_RECORDS = 1402
EXPECTED_COMBINED_RECORDS = 2454
EXPECTED_COMBINED_PER_LABEL = 1227


def user_trajectory_hash(record: Dict) -> str:
    texts = [
        str(t.get("text", "")).strip()
        for t in record.get("turns", [])
        if str(t.get("role", "")).lower() == "user"
    ]
    normalized = "\n<USER_TURN>\n".join(" ".join(x.split()) for x in texts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def n_user_turns(record: Dict) -> int:
    return sum(
        str(t.get("role", "")).lower() == "user"
        for t in record.get("turns", [])
    )


def n_total_turns(record: Dict) -> int:
    return len(record.get("turns", []))


def assert_source_shortcut_invariants(records, source: str) -> None:
    rows = [r for r in records if r.get("corpus_source") == source]
    labels = Counter(r.get("label") for r in rows)
    if labels.get(0, 0) != labels.get(1, 0):
        raise RuntimeError(
            f"{source}: source itself is label-predictive because label counts differ: {dict(labels)}"
        )
    user_hist = {
        label: Counter(n_user_turns(r) for r in rows if r.get("label") == label)
        for label in (0, 1)
    }
    total_hist = {
        label: Counter(n_total_turns(r) for r in rows if r.get("label") == label)
        for label in (0, 1)
    }
    if user_hist[0] != user_hist[1]:
        raise RuntimeError(
            f"{source}: class-conditional user-turn histograms differ: "
            f"benign={dict(user_hist[0])} malicious={dict(user_hist[1])}"
        )
    if total_hist[0] != total_hist[1]:
        raise RuntimeError(
            f"{source}: class-conditional total-turn histograms differ: "
            f"benign={dict(total_hist[0])} malicious={dict(total_hist[1])}"
        )


def canonicalize(
    record: Dict,
    source: str,
    *,
    expected_frontier_target: str,
    expected_frontier_judge: str,
) -> Dict:
    r = copy.deepcopy(record)
    cid = str(r.get("conversation_id", ""))
    if not cid:
        raise RuntimeError("record missing conversation_id")
    if not r.get("training_eligible", False):
        raise RuntimeError(f"{cid}: merge input contains training-ineligible record")
    if r.get("supervision_tier") in {None, "ignore", "construction"}:
        raise RuntimeError(f"{cid}: unresolved supervision tier")
    loss = r.get("loss_weight")
    if (
        not isinstance(loss, (int, float))
        or isinstance(loss, bool)
        or not math.isfinite(float(loss))
        or float(loss) <= 0
    ):
        raise RuntimeError(f"{cid}: unresolved/invalid loss_weight")

    r["corpus_source"] = source
    metadata = r.setdefault("metadata", {})
    if source == "frontier_authored_v3":
        try:
            assert_expected_provenance(
                r,
                expected_target=expected_frontier_target,
                expected_judge=expected_frontier_judge,
                require_evidence=(r.get("label") == 1),
            )
        except Exception as exc:
            raise RuntimeError(f"{cid}: frontier protocol-chain validation failed: {exc}") from exc
        if r.get("primary_pair_complete") is not True:
            raise RuntimeError(f"{cid}: frontier merge input is not from a complete retained pair")
        if r.get("canonical_target_model") != expected_frontier_target:
            raise RuntimeError(
                f"{cid}: frontier target {r.get('canonical_target_model')!r} is not canonical primary target"
            )
        if r.get("canonical_judge_model") != expected_frontier_judge:
            raise RuntimeError(
                f"{cid}: frontier judge {r.get('canonical_judge_model')!r} is not canonical primary judge"
            )
        group = metadata.get("scenario_family")
        if not group:
            raise RuntimeError(f"{cid}: frontier record missing scenario_family")
        group = f"frontier::{group}"
    elif source == "legacy_repaired":
        pair_id = r.get("pair_id")
        if pair_id not in (None, ""):
            group = f"legacy::pair::{pair_id}"
        else:
            group = f"legacy::conversation::{cid}"
    else:
        raise RuntimeError(f"{cid}: unsupported corpus source {source!r}")

    metadata["consolidated_split_group"] = group
    metadata["normalized_user_trajectory_hash"] = user_trajectory_hash(r)
    return r


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-input", required=True)
    parser.add_argument("--frontier-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-frontier-target-model", default=DEFAULT_TARGET)
    parser.add_argument("--expected-frontier-judge-model", default=DEFAULT_JUDGE)
    parser.add_argument(
        "--expect-final-naacl-counts",
        action="store_true",
        help="Require A=1,052, B=1,402, merged=2,454 and 1,227 records per label.",
    )
    args = parser.parse_args()

    legacy = [
        canonicalize(
            r,
            "legacy_repaired",
            expected_frontier_target=args.expected_frontier_target_model,
            expected_frontier_judge=args.expected_frontier_judge_model,
        )
        for r in load_jsonl(args.legacy_input)
    ]
    frontier = [
        canonicalize(
            r,
            "frontier_authored_v3",
            expected_frontier_target=args.expected_frontier_target_model,
            expected_frontier_judge=args.expected_frontier_judge_model,
        )
        for r in load_jsonl(args.frontier_input)
    ]
    combined = legacy + frontier

    if args.expect_final_naacl_counts:
        if len(legacy) != EXPECTED_LEGACY_RECORDS:
            raise RuntimeError(
                f"expected {EXPECTED_LEGACY_RECORDS} legacy records, found {len(legacy)}"
            )
        if len(frontier) != EXPECTED_FRONTIER_RECORDS:
            raise RuntimeError(
                f"expected {EXPECTED_FRONTIER_RECORDS} frontier records, found {len(frontier)}"
            )
        if len(combined) != EXPECTED_COMBINED_RECORDS:
            raise RuntimeError(
                f"expected {EXPECTED_COMBINED_RECORDS} merged records, found {len(combined)}"
            )
        expected_labels = Counter(
            {0: EXPECTED_COMBINED_PER_LABEL, 1: EXPECTED_COMBINED_PER_LABEL}
        )
        observed_labels = Counter(r.get("label") for r in combined)
        if observed_labels != expected_labels:
            raise RuntimeError(
                f"unexpected merged label counts {dict(observed_labels)}; "
                f"expected {dict(expected_labels)}"
            )

    assert_source_shortcut_invariants(combined, "legacy_repaired")
    assert_source_shortcut_invariants(combined, "frontier_authored_v3")

    ids = [str(r.get("conversation_id", "")) for r in combined]
    duplicates = [cid for cid, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise RuntimeError(f"duplicate conversation_ids across corpora: {duplicates[:10]}")

    # Exact normalized content must not bridge independent split groups. The
    # same content inside one indivisible group is harmless because it can never
    # cross train/dev/test; across groups it is a leakage path.
    hash_groups = defaultdict(set)
    hash_sources = defaultdict(set)
    for r in combined:
        metadata = r.get("metadata", {}) or {}
        trajectory_hash = metadata.get("normalized_user_trajectory_hash")
        hash_groups[trajectory_hash].add(metadata.get("consolidated_split_group"))
        hash_sources[trajectory_hash].add(r.get("corpus_source"))
    cross_group_exact = [h for h, groups in hash_groups.items() if len(groups) > 1]
    if cross_group_exact:
        examples = [
            {
                "hash": h,
                "groups": sorted(str(x) for x in hash_groups[h]),
                "sources": sorted(str(x) for x in hash_sources[h]),
            }
            for h in cross_group_exact[:10]
        ]
        raise RuntimeError(
            "exact normalized user trajectories occur across independent split groups; "
            f"leakage risk for {len(cross_group_exact)} hashes. Examples: {examples}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(combined)
    write_jsonl(combined, args.output)
    stats = {
        "records": len(combined),
        "corpus_source": dict(Counter(r.get("corpus_source") for r in combined)),
        "labels": dict(Counter(r.get("label") for r in combined)),
        "supervision_tiers": dict(Counter(r.get("supervision_tier") for r in combined)),
        "split_groups": len({(r.get("metadata", {}) or {}).get("consolidated_split_group") for r in combined}),
        "frontier_scenario_families": len({(r.get("metadata", {}) or {}).get("scenario_family") for r in frontier}),
        "cross_group_exact_user_trajectory_duplicates": 0,
        "expected_frontier_target_model": args.expected_frontier_target_model,
        "expected_frontier_judge_model": args.expected_frontier_judge_model,
        "frontier_protocol_chain_rechecked": True,
        "source_shortcut_checks": {
            "per_source_label_balance": "passed",
            "per_source_user_turn_histogram_match": "passed",
            "per_source_total_turn_histogram_match": "passed",
        },
        "policy": "merge canonical records first; perform a single group-aware split afterward",
    }
    os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
    with open(args.stats_output, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    print(json.dumps(stats, indent=2))
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
