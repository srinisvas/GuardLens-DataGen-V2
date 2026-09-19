#!/usr/bin/env python3
"""Verify the frozen 2,999-record B4 review export and companion manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from typing import Dict

from frontier_common import load_jsonl

EXPECTED_EXPORT_PROTOCOL = "raw_b4_review_export_v1"
EXPECTED_JSONL_SHA256 = "74fe911313e23fa75cf48ed195957ce71346d4ab517d99081ec3864d069af37d"
EXPECTED_CONTRACT_FINGERPRINT = "6499a9a94908bd75ae876b94402b0c145c3049d04506eee1d6fa9b830960ddef"
EXPECTED_SNAPSHOT_DIGEST = "7f219e4b78e09595bd3849306b09606111d3f6c20fe02ce98eb3a577e51c02f1"
EXPECTED_RECORDS = 2999
EXPECTED_B4_STATUSES = Counter({"complete": 877, "not_applicable": 2122})
EXPECTED_LABELS = Counter({0: 1800, 1: 1199})
EXPECTED_B2_STATUSES = Counter({"validated": 2487, "rejected": 512})
EXPECTED_MISSING_RECORD = "c1fd7a44-d06e-5f54-8234-54fa5dbf074f"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_review_export(input_path: str, manifest_path: str) -> Dict:
    actual_sha = sha256_file(input_path)
    if actual_sha != EXPECTED_JSONL_SHA256:
        raise RuntimeError(
            f"review JSONL SHA-256={actual_sha} expected={EXPECTED_JSONL_SHA256}"
        )

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    expected_manifest_fields = {
        "export_protocol": EXPECTED_EXPORT_PROTOCOL,
        "records_exported": EXPECTED_RECORDS,
        "jsonl_sha256": EXPECTED_JSONL_SHA256,
        "contract_fingerprint": EXPECTED_CONTRACT_FINGERPRINT,
        "snapshot_digest": EXPECTED_SNAPSHOT_DIGEST,
        "input_order_preserved": True,
        "record_fields_unchanged": True,
    }
    for field, expected in expected_manifest_fields.items():
        if manifest.get(field) != expected:
            raise RuntimeError(
                f"review manifest {field}={manifest.get(field)!r} expected={expected!r}"
            )

    if Counter(manifest.get("statuses", {})) != EXPECTED_B4_STATUSES:
        raise RuntimeError(
            f"review manifest B4 statuses changed: {manifest.get('statuses')!r}"
        )
    missing = [
        str(item.get("conversation_id", ""))
        for item in manifest.get("missing_records", []) or []
    ]
    if missing != [EXPECTED_MISSING_RECORD]:
        raise RuntimeError(
            f"review manifest missing-record declaration changed: {missing!r}"
        )

    records = load_jsonl(input_path)
    if len(records) != EXPECTED_RECORDS:
        raise RuntimeError(
            f"review record count={len(records)} expected={EXPECTED_RECORDS}"
        )
    ids = [str(r.get("conversation_id", "")) for r in records]
    if not all(ids) or len(ids) != len(set(ids)):
        raise RuntimeError("review export contains missing/duplicate conversation IDs")
    if EXPECTED_MISSING_RECORD in set(ids):
        raise RuntimeError("manifest-declared missing B4 record is unexpectedly present")

    b4_statuses = Counter(
        str((r.get("frontier_evidence_analysis", {}) or {}).get("status", "missing"))
        for r in records
    )
    if b4_statuses != EXPECTED_B4_STATUSES:
        raise RuntimeError(
            f"review B4 statuses={dict(b4_statuses)} expected={dict(EXPECTED_B4_STATUSES)}"
        )
    labels = Counter(r.get("label") for r in records)
    if labels != EXPECTED_LABELS:
        raise RuntimeError(
            f"review labels={dict(labels)} expected={dict(EXPECTED_LABELS)}"
        )
    b2 = Counter(str(r.get("validation_status", "missing")) for r in records)
    if b2 != EXPECTED_B2_STATUSES:
        raise RuntimeError(
            f"review B2 statuses={dict(b2)} expected={dict(EXPECTED_B2_STATUSES)}"
        )

    return {
        "status": "passed",
        "jsonl_sha256": actual_sha,
        "records": len(records),
        "labels": dict(labels),
        "b2_statuses": dict(b2),
        "b4_statuses": dict(b4_statuses),
        "missing_record": EXPECTED_MISSING_RECORD,
        "contract_fingerprint": EXPECTED_CONTRACT_FINGERPRINT,
        "snapshot_digest": EXPECTED_SNAPSHOT_DIGEST,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_review_export(args.input, args.manifest),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
