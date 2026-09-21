#!/usr/bin/env python3
"""Locate the authoritative frozen Dataset B artifacts by exact content hash.

The final Dataset B freeze is already known and audited. Discovery therefore
does not infer identity from filenames, row counts, or stale protocol wrappers.
It searches local roots for exact copies of:

- primary B: 1,402 records, SHA-256 875694...
- full B auxiliary input: 512 records, SHA-256 f8e19e...

Duplicate copies across repo/staging roots are collapsed by content hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List


B_PRIMARY_SHA256 = (
    "875694d3b2ba726dfc9112438b91a055c53459e961f28ab392e9334183d52b14"
)
B_AUX_SHA256 = (
    "f8e19e89dbbfcc2e41a3ff0bf560d6aa9d0d8e07a794f7306b2fc56daa00f594"
)

B_PRIMARY_RECORDS = 1402
B_PRIMARY_PAIRS = 701
B_PRIMARY_SCENARIOS = 321
B_PRIMARY_USER_HIST = Counter({6: 240, 7: 286, 8: 92, 9: 83})

B_AUX_RECORDS = 512
B_AUX_LABELS = Counter({0: 322, 1: 190})
B_AUX_AUTHORING = Counter({0: 190, 1: 322})
B_AUX_SCENARIOS = 275
B_AUX_SOURCE = "frontier_authored_v3_auxiliary"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid JSON at {path}:{line_no}: {exc}"
                ) from exc
    return rows


def n_user_turns(record: Dict) -> int:
    return sum(
        str(turn.get("role", "")).lower() == "user"
        for turn in record.get("turns", [])
    )


def scenario(record: Dict) -> str:
    return str(
        (record.get("metadata", {}) or {}).get("scenario_family", "")
    ).strip()


def validate_primary_shape(rows: List[Dict]) -> Dict:
    if len(rows) != B_PRIMARY_RECORDS:
        raise RuntimeError(
            f"frozen B primary row count changed: {len(rows)}"
        )
    if Counter(r.get("label") for r in rows) != Counter({0: 701, 1: 701}):
        raise RuntimeError("frozen B primary label counts changed")

    pairs = defaultdict(list)
    scenarios = set()
    hist = {0: Counter(), 1: Counter()}
    ids = set()

    for record in rows:
        cid = str(record.get("conversation_id", ""))
        if not cid or cid in ids:
            raise RuntimeError(f"B primary missing/duplicate ID: {cid!r}")
        ids.add(cid)

        pair_id = str(record.get("pair_id", "")).strip()
        if not pair_id:
            raise RuntimeError(f"{cid}: B primary missing pair_id")
        pairs[pair_id].append(record)

        sc = scenario(record)
        if not sc:
            raise RuntimeError(f"{cid}: B primary missing scenario_family")
        scenarios.add(sc)

        if record.get("validation_status") != "validated":
            raise RuntimeError(f"{cid}: B primary is not validated")
        if record.get("training_eligible") is not True:
            raise RuntimeError(f"{cid}: B primary is not training eligible")
        if record.get("primary_pair_complete") is not True:
            raise RuntimeError(f"{cid}: B primary lacks pair-complete flag")

        hist[int(record["label"])][n_user_turns(record)] += 1

    if len(pairs) != B_PRIMARY_PAIRS:
        raise RuntimeError(f"B primary pair count changed: {len(pairs)}")
    for pair_id, group in pairs.items():
        if (
            len(group) != 2
            or Counter(r.get("label") for r in group) != Counter({0: 1, 1: 1})
        ):
            raise RuntimeError(f"B primary pair {pair_id} is invalid")

    if len(scenarios) != B_PRIMARY_SCENARIOS:
        raise RuntimeError(f"B primary scenario count changed: {len(scenarios)}")
    for label in (0, 1):
        if hist[label] != B_PRIMARY_USER_HIST:
            raise RuntimeError(
                f"B primary label={label} user-turn histogram changed"
            )

    return {
        "records": len(rows),
        "pairs": len(pairs),
        "scenario_families": len(scenarios),
        "labels": {"0": 701, "1": 701},
    }


def validate_aux_shape(rows: List[Dict]) -> Dict:
    if len(rows) != B_AUX_RECORDS:
        raise RuntimeError(f"frozen B auxiliary row count changed: {len(rows)}")

    ids = set()
    detection = Counter()
    authoring = Counter()
    scenarios = set()

    for record in rows:
        cid = str(record.get("conversation_id", ""))
        if not cid or cid in ids:
            raise RuntimeError(f"B auxiliary missing/duplicate ID: {cid!r}")
        ids.add(cid)

        if record.get("corpus_source") != B_AUX_SOURCE:
            raise RuntimeError(f"{cid}: unexpected B auxiliary source")
        if record.get("validation_status") != "rejected":
            raise RuntimeError(f"{cid}: B auxiliary is not B2-rejected")
        if record.get("auxiliary_detection_only") is not True:
            raise RuntimeError(f"{cid}: B auxiliary flag missing")
        if record.get("use_as") != "auxiliary_detection_only":
            raise RuntimeError(f"{cid}: B auxiliary use_as mismatch")
        if float(record.get("detection_loss_weight", -1)) != 0.25:
            raise RuntimeError(f"{cid}: B auxiliary detection weight changed")

        label = record.get("detection_label")
        author_label = record.get("authoring_intent_label")
        if label not in (0, 1) or isinstance(label, bool):
            raise RuntimeError(f"{cid}: invalid B auxiliary detection label")
        if author_label not in (0, 1) or isinstance(author_label, bool):
            raise RuntimeError(f"{cid}: invalid B auxiliary authoring label")
        detection[int(label)] += 1
        authoring[int(author_label)] += 1

        sc = scenario(record)
        if not sc:
            raise RuntimeError(f"{cid}: B auxiliary missing scenario_family")
        expected_group = f"frontier::{sc}"
        if (
            (record.get("metadata", {}) or {}).get("consolidated_split_group")
            != expected_group
        ):
            raise RuntimeError(f"{cid}: B auxiliary split group changed")
        scenarios.add(sc)

    if detection != B_AUX_LABELS:
        raise RuntimeError(
            f"B auxiliary detection labels changed: {dict(detection)}"
        )
    if authoring != B_AUX_AUTHORING:
        raise RuntimeError(
            f"B auxiliary authoring labels changed: {dict(authoring)}"
        )
    if len(scenarios) != B_AUX_SCENARIOS:
        raise RuntimeError(
            f"B auxiliary scenario count changed: {len(scenarios)}"
        )

    return {
        "records": len(rows),
        "scenario_families": len(scenarios),
        "detection_labels": dict(sorted(detection.items())),
        "authoring_labels": dict(sorted(authoring.items())),
    }


def candidate_files(roots: List[str]) -> List[str]:
    files = set()
    for root in roots:
        root = os.path.expanduser(root)
        if not os.path.exists(root):
            continue
        if os.path.isfile(root):
            if root.endswith(".jsonl"):
                files.add(os.path.abspath(root))
            continue
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".jsonl"):
                    files.add(os.path.abspath(os.path.join(dirpath, filename)))
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        nargs="*",
        default=["results-naacl", os.path.expanduser("~/staging/dataset_naacl")],
    )
    args = parser.parse_args()

    matches = {
        B_PRIMARY_SHA256: {
            "kind": "B_PRIMARY",
            "expected_records": B_PRIMARY_RECORDS,
            "paths": [],
        },
        B_AUX_SHA256: {
            "kind": "B_AUX_INPUT",
            "expected_records": B_AUX_RECORDS,
            "paths": [],
        },
    }

    for path in candidate_files(args.roots):
        try:
            actual = sha256_file(path)
        except OSError:
            continue
        if actual not in matches:
            continue
        matches[actual]["paths"].append(path)

    output = []
    errors = []

    for expected_sha, item in matches.items():
        paths = sorted(set(item["paths"]))
        if not paths:
            errors.append(
                f"{item['kind']} exact hash not found: {expected_sha}"
            )
            output.append({
                **item,
                "sha256": expected_sha,
                "paths": [],
                "status": "missing",
            })
            continue

        # Every listed path is byte-identical. Validate one representative's
        # structure to catch impossible hash/contract assumptions.
        rows = load_jsonl(paths[0])
        if item["kind"] == "B_PRIMARY":
            contract = validate_primary_shape(rows)
        else:
            contract = validate_aux_shape(rows)

        output.append({
            "kind": item["kind"],
            "sha256": expected_sha,
            "records": len(rows),
            "paths": paths,
            "duplicate_copies": max(0, len(paths) - 1),
            "contract": contract,
            "status": "found",
        })

    payload = {
        "status": "passed" if not errors else "incomplete",
        "matches": output,
        "errors": errors,
        "identity_policy": "exact SHA-256 first; structural contract second",
        "authoritative_optimized_branch_paths": {
            "B_PRIMARY": (
                "results-naacl/final-data-freeze/dataset_b_primary.jsonl"
            ),
            "B_AUX_INPUT": (
                "results-naacl/final-data-freeze/dataset_b_auxiliary_512.jsonl"
            ),
        },
    }

    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
