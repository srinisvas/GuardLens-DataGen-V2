#!/usr/bin/env python3
"""Merge independently authored GuardLens frontier source corpora fail-closed.

This operates only at Stage 0 (user trajectories). It never changes labels,
authoring metadata, pivots, supervision, or training eligibility. The merged
artifact is suitable for the existing Qwen rollout pipeline after a strict
source audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from frontier_common import assert_frontier_source_record, load_jsonl, write_jsonl


def source_key(record: Dict) -> str:
    metadata = record.get("metadata", {}) or {}
    return str(
        metadata.get("corpus_version")
        or metadata.get("generator")
        or record.get("seed_source")
        or "unknown_source"
    )


def generator_name(record: Dict) -> str:
    return str((record.get("metadata", {}) or {}).get("generator", "unknown"))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def normalized_sequence(record: Dict) -> Tuple[str, ...]:
    return tuple(
        normalize_text(t.get("text", ""))
        for t in record.get("turns", [])
        if str(t.get("role", "")).lower() == "user"
    )


def sequence_hash(record: Dict) -> str:
    payload = json.dumps(normalized_sequence(record), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def n_user(record: Dict) -> int:
    return sum(str(t.get("role", "")).lower() == "user" for t in record.get("turns", []))


def auc_from_scores(labels: List[int], scores: List[float]) -> float:
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def summarize(records: List[Dict]) -> Dict:
    pairs = {str(r.get("pair_id")) for r in records if r.get("pair_id") not in (None, "")}
    scenarios = {
        str((r.get("metadata", {}) or {}).get("scenario_family", ""))
        for r in records
    }
    paired = [r for r in records if r.get("pair_id") not in (None, "")]
    return {
        "records": len(records),
        "labels": dict(Counter(str(r.get("label")) for r in records)),
        "pairs": len(pairs),
        "standalone": sum(r.get("pair_id") in (None, "") for r in records),
        "scenario_families": len(scenarios),
        "generators": dict(Counter(generator_name(r) for r in records)),
        "corpus_versions": dict(Counter(source_key(r) for r in records)),
        "user_turn_histogram": dict(Counter(n_user(r) for r in records)),
        "paired_length_auc": auc_from_scores(
            [int(r.get("label")) for r in paired],
            [n_user(r) for r in paired],
        ),
        "full_length_auc": auc_from_scores(
            [int(r.get("label")) for r in records],
            [n_user(r) for r in records],
        ),
    }


def validate_one(path: str, expected_records: int | None) -> List[Dict]:
    records = load_jsonl(path)
    if expected_records is not None and len(records) != expected_records:
        raise RuntimeError(f"{path}: expected {expected_records} records, got {len(records)}")

    ids = set()
    pair_members = defaultdict(list)
    scenarios = defaultdict(list)
    seq_hashes = set()
    for record in records:
        assert_frontier_source_record(record)
        cid = str(record.get("conversation_id", ""))
        if cid in ids:
            raise RuntimeError(f"{path}: duplicate conversation_id {cid}")
        ids.add(cid)
        pair_id = record.get("pair_id")
        if pair_id not in (None, ""):
            pair_members[str(pair_id)].append(record)
        scenario = str((record.get("metadata", {}) or {}).get("scenario_family", ""))
        scenarios[scenario].append(record)
        h = sequence_hash(record)
        if h in seq_hashes:
            raise RuntimeError(f"{path}: duplicate normalized complete user trajectory at {cid}")
        seq_hashes.add(h)

    for pair_id, group in pair_members.items():
        labels = Counter(r.get("label") for r in group)
        if len(group) != 2 or labels != Counter({0: 1, 1: 1}):
            raise RuntimeError(
                f"{path}: invalid pair {pair_id}: n={len(group)} labels={dict(labels)}"
            )

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--expected-records-per-input", type=int, default=1500)
    parser.add_argument("--expected-total", type=int, default=3000)
    parser.add_argument("--max-primary-length-auc", type=float, default=0.65)
    args = parser.parse_args()

    if len(args.inputs) < 2:
        raise ValueError("merge requires at least two independently authored source files")

    all_records: List[Dict] = []
    input_stats = {}
    owner_by_id = {}
    owner_by_pair = {}
    owner_by_scenario = {}
    owner_by_sequence = {}

    for path in args.inputs:
        records = validate_one(path, args.expected_records_per_input)
        input_stats[os.path.basename(path)] = summarize(records)
        for record in records:
            owner = source_key(record)
            cid = str(record.get("conversation_id", ""))
            pair_id = record.get("pair_id")
            scenario = str((record.get("metadata", {}) or {}).get("scenario_family", ""))
            seq = sequence_hash(record)

            if cid in owner_by_id:
                raise RuntimeError(
                    f"cross-source conversation_id collision {cid}: {owner_by_id[cid]} vs {owner}"
                )
            owner_by_id[cid] = owner

            if pair_id not in (None, ""):
                key = str(pair_id)
                prior = owner_by_pair.setdefault(key, owner)
                if prior != owner:
                    raise RuntimeError(
                        f"cross-source pair_id collision {key}: {prior} vs {owner}"
                    )

            prior = owner_by_scenario.setdefault(scenario, owner)
            if prior != owner:
                raise RuntimeError(
                    f"cross-source scenario_family collision {scenario}: {prior} vs {owner}"
                )

            if seq in owner_by_sequence:
                raise RuntimeError(
                    f"cross-source normalized trajectory duplicate {cid}: "
                    f"{owner_by_sequence[seq]} vs {owner}"
                )
            owner_by_sequence[seq] = owner
            all_records.append(record)

    if len(all_records) != args.expected_total:
        raise RuntimeError(
            f"merged total expected {args.expected_total}, got {len(all_records)}"
        )

    merged_stats = summarize(all_records)
    if merged_stats["paired_length_auc"] > args.max_primary_length_auc:
        raise RuntimeError(
            f"merged paired primary length AUC={merged_stats['paired_length_auc']:.4f} "
            f"exceeds {args.max_primary_length_auc:.4f}"
        )

    # Preserve each input's original record order and preserve the input-file
    # ordering. No fields are added or rewritten here.
    write_jsonl(all_records, args.output)
    stats = {
        "inputs": input_stats,
        "merged": merged_stats,
        "cross_source_checks": {
            "conversation_id_collisions": 0,
            "pair_id_collisions": 0,
            "scenario_family_collisions": 0,
            "normalized_complete_trajectory_duplicates": 0,
        },
        "policy": {
            "records_mutated": False,
            "authoring_intent_is_ground_truth": False,
            "primary_length_auc_gate": args.max_primary_length_auc,
            "output_order": "input-file order, then original within-file order",
            "provenance_source": "existing metadata.corpus_version / metadata.generator; merge adds no model-visible fields",
        },
    }
    os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
    with open(args.stats_output, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)

    print(json.dumps(stats, indent=2))
    print(f"Merged {len(all_records)} source records -> {args.output}")


if __name__ == "__main__":
    main()
