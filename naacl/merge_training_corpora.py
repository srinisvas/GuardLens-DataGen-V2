#!/usr/bin/env python3
"""Merge frozen legacy Dataset A and canonical frontier Dataset B."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import Counter
from typing import Dict

from frontier_common import load_jsonl, write_jsonl


def canonicalize(record: Dict, source: str) -> Dict:
    r = copy.deepcopy(record)
    cid = str(r.get("conversation_id", ""))
    if not cid:
        raise RuntimeError("record missing conversation_id")
    if not r.get("training_eligible", False):
        raise RuntimeError(f"{cid}: merge input contains training-ineligible record")
    if r.get("supervision_tier") in {None, "ignore", "construction"}:
        raise RuntimeError(f"{cid}: unresolved supervision tier")
    loss = r.get("loss_weight")
    if not isinstance(loss, (int, float)) or loss <= 0:
        raise RuntimeError(f"{cid}: unresolved loss_weight")

    r["corpus_source"] = source
    metadata = r.setdefault("metadata", {})
    if source == "frontier_authored_v3":
        group = metadata.get("scenario_family")
        if not group:
            raise RuntimeError(f"{cid}: frontier record missing scenario_family")
        group = f"frontier::{group}"
    else:
        pair_id = r.get("pair_id")
        if pair_id not in (None, ""):
            group = f"legacy::pair::{pair_id}"
        else:
            group = f"legacy::conversation::{cid}"
    metadata["consolidated_split_group"] = group
    return r


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-input", required=True)
    parser.add_argument("--frontier-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    legacy = [canonicalize(r, "legacy_repaired") for r in load_jsonl(args.legacy_input)]
    frontier = [canonicalize(r, "frontier_authored_v3") for r in load_jsonl(args.frontier_input)]
    combined = legacy + frontier

    ids = [str(r.get("conversation_id", "")) for r in combined]
    duplicates = [cid for cid, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise RuntimeError(f"duplicate conversation_ids across corpora: {duplicates[:10]}")

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
        "policy": "merge canonical records first; perform a single group-aware split afterward",
    }
    os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
    with open(args.stats_output, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    print(json.dumps(stats, indent=2))
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
