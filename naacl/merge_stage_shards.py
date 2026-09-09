#!/usr/bin/env python3
"""Fail-closed merger for parallel frontier stage outputs.

The source JSONL is the authority for record membership, shard assignment, and
final order. A shard may contain only the source records assigned by
``source_index % num_shards``. Duplicate, missing, extra, or wrong-shard records
are fatal rather than being silently overwritten during merge.
"""
from __future__ import annotations

import argparse
from typing import Dict, List

from frontier_common import load_jsonl, write_jsonl


def merge_stage_shards(
    *,
    source_records: List[Dict],
    shard_records: List[List[Dict]],
) -> List[Dict]:
    n = len(shard_records)
    if n <= 0:
        raise ValueError("at least one shard is required")

    source_ids = []
    source_pos = {}
    for idx, record in enumerate(source_records):
        cid = str(record.get("conversation_id", ""))
        if not cid:
            raise RuntimeError(f"source record at index {idx} missing conversation_id")
        if cid in source_pos:
            raise RuntimeError(f"duplicate conversation_id in source: {cid}")
        source_pos[cid] = idx
        source_ids.append(cid)

    merged = {}
    for shard_idx, records in enumerate(shard_records):
        local_seen = set()
        expected_count = sum(1 for idx in range(len(source_records)) if idx % n == shard_idx)
        if len(records) != expected_count:
            raise RuntimeError(
                f"shard {shard_idx} record count={len(records)} != expected {expected_count}"
            )
        for record in records:
            cid = str(record.get("conversation_id", ""))
            if not cid:
                raise RuntimeError(f"shard {shard_idx} contains record without conversation_id")
            if cid in local_seen:
                raise RuntimeError(f"duplicate conversation_id within shard {shard_idx}: {cid}")
            local_seen.add(cid)
            if cid not in source_pos:
                raise RuntimeError(f"unexpected conversation_id in shard {shard_idx}: {cid}")
            expected_shard = source_pos[cid] % n
            if expected_shard != shard_idx:
                raise RuntimeError(
                    f"conversation_id {cid} is in shard {shard_idx}, expected shard {expected_shard}"
                )
            if cid in merged:
                raise RuntimeError(
                    f"conversation_id appears in multiple shards: {cid}"
                )
            merged[cid] = record

    missing = [cid for cid in source_ids if cid not in merged]
    if missing:
        raise RuntimeError(
            f"missing {len(missing)} source records after shard merge: {missing[:10]}"
        )
    if len(merged) != len(source_records):
        raise RuntimeError(
            f"merged record count={len(merged)} != source count={len(source_records)}"
        )
    return [merged[cid] for cid in source_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument(
        "--shard-pattern",
        required=True,
        help="Path template containing literal {i}, e.g. /tmp/out.shard{i}.jsonl",
    )
    args = parser.parse_args()
    if args.num_shards <= 0:
        raise ValueError("num-shards must be positive")
    if "{i}" not in args.shard_pattern:
        raise ValueError("shard-pattern must contain literal {i}")

    source_records = load_jsonl(args.source)
    shard_records = [
        load_jsonl(args.shard_pattern.format(i=i))
        for i in range(args.num_shards)
    ]
    merged = merge_stage_shards(
        source_records=source_records,
        shard_records=shard_records,
    )
    write_jsonl(merged, args.output)
    print(
        f"strict shard merge passed: records={len(merged)} shards={args.num_shards} -> {args.output}"
    )


if __name__ == "__main__":
    main()
