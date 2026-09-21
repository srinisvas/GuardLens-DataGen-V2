#!/usr/bin/env python3
"""Fail-closed merge of parallel Dataset A twin-bridge checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List

from judge_stored_twin_responses import (
    BRIDGE_PROTOCOL,
    RESTORATION_POLICY_VERSION,
    bridge_shard_assignments,
    final_malicious_candidate,
    is_original_benign_twin,
    load_checkpoint,
    load_jsonl,
    turn_text_hash,
)


TERMINAL_BRIDGE_STATUSES = {"validated", "ambiguous", "rejected"}


def atomic_write_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument(
        "--shard-checkpoint",
        action="append",
        required=True,
        help="Pass once per shard, in shard-index order.",
    )
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("num-shards must be positive")
    if len(args.shard_checkpoint) != args.num_shards:
        raise RuntimeError(
            f"expected {args.num_shards} shard checkpoints, "
            f"got {len(args.shard_checkpoint)}"
        )

    records = load_jsonl(args.input)
    final_malicious = [r for r in records if final_malicious_candidate(r)]
    final_pair_ids = [str(r.get("pair_id", "")) for r in final_malicious]
    if len(final_pair_ids) != len(set(final_pair_ids)):
        raise RuntimeError("final malicious pair IDs are not unique")
    pair_set = set(final_pair_ids)

    bridge_twins = [
        r
        for r in records
        if is_original_benign_twin(r)
        and str(r.get("pair_id", "")) in pair_set
    ]
    if len(bridge_twins) != len(final_malicious):
        raise RuntimeError(
            f"bridge population mismatch: malicious={len(final_malicious)} "
            f"twins={len(bridge_twins)}"
        )

    by_cid = {
        str(record.get("conversation_id", "")): record
        for record in bridge_twins
    }
    if len(by_cid) != len(bridge_twins) or "" in by_cid:
        raise RuntimeError("bridge twins have missing/duplicate conversation IDs")

    assignments, record_loads, assistant_loads = bridge_shard_assignments(
        bridge_twins, args.num_shards
    )

    merged: Dict[str, Dict] = {}
    shard_counts = []

    for shard_idx, checkpoint_name in enumerate(args.shard_checkpoint):
        path = Path(checkpoint_name)
        if not path.is_file():
            raise RuntimeError(
                f"missing shard {shard_idx} checkpoint: {path}"
            )

        rows = load_checkpoint(str(path))
        local: Dict[str, Dict] = {}
        for record in rows:
            cid = str(record.get("conversation_id", ""))
            if cid not in by_cid:
                raise RuntimeError(
                    f"shard {shard_idx}: unexpected bridge conversation_id {cid!r}"
                )
            if assignments[cid] != shard_idx:
                raise RuntimeError(
                    f"{cid}: found in shard {shard_idx}, deterministic assignment "
                    f"is shard {assignments[cid]}"
                )
            if cid in local:
                raise RuntimeError(
                    f"shard {shard_idx}: duplicate conversation_id {cid}"
                )
            if cid in merged:
                raise RuntimeError(
                    f"{cid}: duplicated across shard checkpoints"
                )
            if turn_text_hash(record) != turn_text_hash(by_cid[cid]):
                raise RuntimeError(
                    f"{cid}: checkpoint changed observable stored trajectory"
                )

            validation = record.get("stored_target_validation", {}) or {}
            if validation.get("validated") is not True:
                raise RuntimeError(
                    f"{cid}: shard checkpoint lacks completed bridge validation"
                )
            if validation.get("protocol") != BRIDGE_PROTOCOL:
                raise RuntimeError(
                    f"{cid}: unexpected bridge protocol "
                    f"{validation.get('protocol')!r}"
                )
            if validation.get("status") not in TERMINAL_BRIDGE_STATUSES:
                raise RuntimeError(
                    f"{cid}: nonterminal bridge status "
                    f"{validation.get('status')!r}"
                )

            restoration = record.get("twin_restoration", {}) or {}
            if restoration.get("policy_version") != RESTORATION_POLICY_VERSION:
                raise RuntimeError(
                    f"{cid}: restoration policy mismatch "
                    f"{restoration.get('policy_version')!r}"
                )

            local[cid] = record
            merged[cid] = record

        expected_local = {
            cid for cid, idx in assignments.items() if idx == shard_idx
        }
        if set(local) != expected_local:
            missing = sorted(expected_local - set(local))
            extra = sorted(set(local) - expected_local)
            raise RuntimeError(
                f"shard {shard_idx} coverage mismatch: "
                f"expected={len(expected_local)} got={len(local)} "
                f"missing={missing[:10]} extra={extra[:10]}"
            )
        shard_counts.append(len(local))

    expected = set(by_cid)
    if set(merged) != expected:
        missing = sorted(expected - set(merged))
        extra = sorted(set(merged) - expected)
        raise RuntimeError(
            "merged checkpoint does not exactly cover bridge universe: "
            f"expected={len(expected)} got={len(merged)} "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    ordered = [
        merged[str(record.get("conversation_id", ""))]
        for record in bridge_twins
    ]
    output = Path(args.output_checkpoint)
    atomic_write_jsonl(ordered, output)

    statuses = Counter(
        str((r.get("stored_target_validation", {}) or {}).get("status", "missing"))
        for r in ordered
    )
    summary = {
        "records": len(ordered),
        "num_shards": args.num_shards,
        "shard_record_loads": record_loads,
        "observed_shard_records": shard_counts,
        "shard_assistant_turn_loads": assistant_loads,
        "shard_dual_rubric_request_loads": [2 * x for x in assistant_loads],
        "bridge_statuses": dict(statuses),
        "protocol": BRIDGE_PROTOCOL,
        "restoration_policy_version": RESTORATION_POLICY_VERSION,
        "output_checkpoint": str(output.resolve()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("PARALLEL BRIDGE CHECKPOINT MERGE PASSED")


if __name__ == "__main__":
    main()
