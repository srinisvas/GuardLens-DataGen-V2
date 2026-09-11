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


def user_trajectory_hash(record: Dict) -> str:
    texts = [
        str(t.get("text", "")).strip()
        for t in record.get("turns", [])
        if str(t.get("role", "")).lower() == "user"
    ]
    normalized = "\n<USER_TURN>\n".join(" ".join(x.split()) for x in texts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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
        "policy": "merge canonical records first; perform a single group-aware split afterward",
    }
    os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
    with open(args.stats_output, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    print(json.dumps(stats, indent=2))
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
