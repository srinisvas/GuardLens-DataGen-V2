#!/usr/bin/env python3
"""Locate frozen Dataset B primary and detection auxiliary artifacts by contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List

from prepare_frontier_dataset import assert_expected_provenance

PRIMARY_N = 1402
PAIR_N = 701
AUX_N = 424
TARGET = "Qwen/Qwen2.5-32B-Instruct"
JUDGE = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def primary_contract(rows: List[Dict]) -> tuple[bool, str]:
    if len(rows) != PRIMARY_N:
        return False, f"n={len(rows)}"
    if Counter(r.get("label") for r in rows) != Counter({0: PAIR_N, 1: PAIR_N}):
        return False, "label counts"
    pairs = defaultdict(list)
    try:
        for r in rows:
            cid = str(r.get("conversation_id", ""))
            if not cid:
                return False, "missing conversation_id"
            if r.get("training_eligible") is not True:
                return False, "training-ineligible primary"
            if r.get("primary_pair_complete") is not True:
                return False, "primary_pair_complete missing"
            pid = str(r.get("pair_id", ""))
            if not pid:
                return False, "missing pair_id"
            pairs[pid].append(r)
            metadata = r.get("metadata", {}) or {}
            if not metadata.get("scenario_family"):
                return False, "missing scenario_family"
            assert_expected_provenance(
                r,
                expected_target=TARGET,
                expected_judge=JUDGE,
                require_evidence=(r.get("label") == 1),
            )
            if r.get("canonical_target_model") != TARGET:
                return False, "target mismatch"
            if r.get("canonical_judge_model") != JUDGE:
                return False, "judge mismatch"
    except Exception as exc:
        return False, f"provenance: {exc}"

    if len(pairs) != PAIR_N:
        return False, f"pair count={len(pairs)}"
    for group in pairs.values():
        if len(group) != 2 or Counter(r.get("label") for r in group) != Counter({0: 1, 1: 1}):
            return False, "invalid pair structure"
    return True, "canonical_B_primary"


def aux_contract(rows: List[Dict]) -> tuple[bool, str]:
    if len(rows) != AUX_N:
        return False, f"n={len(rows)}"
    labels = Counter()
    weights = Counter()
    for r in rows:
        if r.get("auxiliary_detection_only") is not True and (
            r.get("use_as") != "auxiliary_detection_only"
        ):
            return False, "not detection-only"
        label = r.get("detection_label")
        if isinstance(label, bool) or label not in (0, 1):
            return False, "invalid detection_label"
        labels[int(label)] += 1
        weight = r.get("detection_loss_weight")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            return False, "invalid detection_loss_weight"
        weights[float(weight)] += 1
        if abs(float(weight) - 0.25) > 1e-12:
            return False, f"noncanonical weight={weight}"
        if r.get("localization_supervision_ignore") is not True:
            return False, "localization not ignored"
        if r.get("pivot_supervision_ignore") is not True:
            return False, "pivot not ignored"
        if r.get("pivot_loss_weight") != 0.0 or r.get("span_loss_weight") != 0.0:
            return False, "nonzero localization loss"
    if labels != Counter({0: 271, 1: 153}):
        return False, f"labels={dict(labels)}"
    return True, "canonical_B_detection_aux"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        nargs="*",
        default=["results-naacl"],
        help="Directories to scan recursively for JSONL artifacts.",
    )
    args = parser.parse_args()

    candidates = []
    for root in args.roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".jsonl"):
                    candidates.append(os.path.join(dirpath, filename))

    found = []
    for path in sorted(candidates):
        try:
            rows = load_jsonl(path)
        except Exception:
            continue
        if len(rows) not in {PRIMARY_N, AUX_N}:
            continue

        is_primary, primary_reason = primary_contract(rows)
        if is_primary:
            found.append({
                "kind": "B_PRIMARY",
                "path": path,
                "records": len(rows),
                "sha256": sha256_file(path),
                "contract": primary_reason,
            })
            continue

        is_aux, aux_reason = aux_contract(rows)
        if is_aux:
            found.append({
                "kind": "B_AUX",
                "path": path,
                "records": len(rows),
                "sha256": sha256_file(path),
                "contract": aux_reason,
            })

    print(json.dumps({"matches": found}, indent=2, sort_keys=True))

    kinds = Counter(item["kind"] for item in found)
    if kinds.get("B_PRIMARY", 0) != 1 or kinds.get("B_AUX", 0) != 1:
        raise SystemExit(
            "Expected exactly one canonical B_PRIMARY and one canonical B_AUX; "
            f"found {dict(kinds)}. Supply additional roots or inspect duplicates."
        )


if __name__ == "__main__":
    main()
