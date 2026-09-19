#!/usr/bin/env python3
"""Attach detection-only Dataset B auxiliary outcomes to a frozen primary train split.

The primary A+B train/dev/test assignment is authoritative. Auxiliary records are
never added to dev/test. An auxiliary record may enter training only when its
scenario-family split group is already owned by primary train or is absent from
the primary corpus entirely. Auxiliary records whose family is owned by primary
dev/test are withheld, preventing family leakage while keeping primary
evaluation partitions byte-for-byte unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from typing import Dict, Iterable, List, Tuple

from audit_frontier_auxiliary import audit_record as audit_auxiliary_record
from frontier_common import load_jsonl, write_jsonl

SPLITS = ("train", "dev", "test")


def split_group(record: Dict) -> str:
    group = str((record.get("metadata", {}) or {}).get("consolidated_split_group", "")).strip()
    if not group:
        raise RuntimeError(f"{record.get('conversation_id')}: missing consolidated_split_group")
    return group


def index_primary_splits(
    train: List[Dict], dev: List[Dict], test: List[Dict]
) -> Tuple[Dict[str, str], set]:
    owners: Dict[str, str] = {}
    hash_owners: Dict[str, str] = {}
    ids = set()
    for split_name, records in (("train", train), ("dev", dev), ("test", test)):
        for record in records:
            cid = str(record.get("conversation_id", ""))
            if not cid:
                raise RuntimeError(f"{split_name}: primary record missing conversation_id")
            if cid in ids:
                raise RuntimeError(f"duplicate primary conversation_id across splits: {cid}")
            ids.add(cid)
            group = split_group(record)
            previous = owners.setdefault(group, split_name)
            if previous != split_name:
                raise RuntimeError(
                    f"primary split-group leakage: {group} appears in {previous} and {split_name}"
                )

            trajectory_hash = str(
                (record.get("metadata", {}) or {}).get(
                    "normalized_user_trajectory_hash", ""
                )
            ).strip()
            if not trajectory_hash:
                raise RuntimeError(
                    f"{cid}: primary record missing normalized_user_trajectory_hash"
                )
            previous_hash_owner = hash_owners.setdefault(
                trajectory_hash, split_name
            )
            if previous_hash_owner != split_name:
                raise RuntimeError(
                    "primary exact user-trajectory leakage: hash appears in "
                    f"{previous_hash_owner} and {split_name}"
                )
    return owners, hash_owners, ids


def attach_training_auxiliary(
    train: List[Dict],
    dev: List[Dict],
    test: List[Dict],
    auxiliary: Iterable[Dict],
):
    owners, hash_owners, primary_ids = index_primary_splits(train, dev, test)
    output_train = list(train)
    seen_ids = set(primary_ids)
    disposition = Counter()
    included_labels = Counter()
    included_groups = set()

    for record in auxiliary:
        audit_auxiliary_record(record)
        cid = str(record.get("conversation_id", ""))
        if cid in seen_ids:
            raise RuntimeError(f"auxiliary conversation_id collides with primary material: {cid}")
        seen_ids.add(cid)

        group = split_group(record)
        owner = owners.get(group)
        trajectory_hash = str(
            (record.get("metadata", {}) or {}).get(
                "normalized_user_trajectory_hash", ""
            )
        ).strip()
        hash_owner = hash_owners.get(trajectory_hash)
        if owner in {"dev", "test"}:
            disposition[f"withheld_primary_{owner}_family"] += 1
            continue
        if hash_owner in {"dev", "test"}:
            disposition[f"withheld_primary_{hash_owner}_exact_user_trajectory"] += 1
            continue

        output_train.append(record)
        included_groups.add(group)
        included_labels[int(record["detection_label"])] += 1
        disposition["included_primary_train_family" if owner == "train" else "included_aux_only_family"] += 1

    return {
        "train": output_train,
        "dev": list(dev),
        "test": list(test),
    }, {
        "primary_records": {
            "train": len(train),
            "dev": len(dev),
            "test": len(test),
        },
        "candidate_records": {
            "train": len(output_train),
            "dev": len(dev),
            "test": len(test),
        },
        "auxiliary_included": len(output_train) - len(train),
        "auxiliary_included_detection_labels": dict(sorted(included_labels.items())),
        "auxiliary_included_groups": len(included_groups),
        "auxiliary_disposition": dict(sorted(disposition.items())),
        "policy": (
            "freeze primary split; add auxiliary only to train when scenario-family and "
            "exact-user-trajectory ownership do not belong to primary dev/test"
        ),
        "dev_test_primary_only": True,
        "dev_test_copy_policy": "byte_for_byte_from_frozen_primary_inputs",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-train", required=True)
    parser.add_argument("--primary-dev", required=True)
    parser.add_argument("--primary-test", required=True)
    parser.add_argument("--auxiliary-input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    primary_train = load_jsonl(args.primary_train)
    primary_dev = load_jsonl(args.primary_dev)
    primary_test = load_jsonl(args.primary_test)
    auxiliary = load_jsonl(args.auxiliary_input)

    splits, stats = attach_training_auxiliary(
        primary_train, primary_dev, primary_test, auxiliary
    )
    os.makedirs(args.output_dir, exist_ok=True)
    write_jsonl(splits["train"], os.path.join(args.output_dir, "train.jsonl"))
    shutil.copyfile(args.primary_dev, os.path.join(args.output_dir, "dev.jsonl"))
    shutil.copyfile(args.primary_test, os.path.join(args.output_dir, "test.jsonl"))
    with open(
        os.path.join(args.output_dir, "auxiliary_attachment_metadata.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
