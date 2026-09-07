#!/usr/bin/env python3
"""Normalize an independent validation pass for the NAACL validity repair.

This script is intentionally separate from ``merge_validations.py`` because the
v11 merge utility assumes the post-generation validator is Qwen. For the NAACL
repair we want the generator family and the independent validator family to be
distinct.

Expected input
--------------
The JSONL produced by ``launch_val.slurm`` after running the existing generated
trajectories through an independent validator such as
``mistralai/Mistral-7B-Instruct-v0.3``. Each record should retain the original
``llama_validation`` from interactive generation and contain the new validation
result in ``causal_validation``.

The output keeps the old fields for reproducibility but adds explicit,
model-agnostic fields:

- ``independent_validation``
- ``independent_validation_model``
- ``success_targets``
- ``transfer_tier``

No counterfactual claim is made here. Counterfactual evidence is recomputed from
fresh paired replays by ``naacl/evidence_analysis.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter
from typing import Dict, Iterable, List


def load_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _jailbreak(validation: Dict) -> bool:
    return bool(validation.get("validated") and validation.get("jailbreak_detected"))


def normalize_record(record: Dict) -> Dict:
    r = copy.deepcopy(record)

    target_val = r.get("llama_validation", {}) or {}
    independent_val = r.get("causal_validation", {}) or {}

    target_model = target_val.get("model_used") or target_val.get("target_model") or "llama_target"
    independent_model = (
        independent_val.get("model_used")
        or independent_val.get("judge_model")
        or independent_val.get("validator_model")
        or "independent_validator"
    )

    r["independent_validation"] = independent_val
    r["independent_validation_model"] = independent_model
    r["target_validation_model"] = target_model

    target_success = _jailbreak(target_val)
    independent_success = _jailbreak(independent_val)

    successes: List[str] = []
    if target_success:
        successes.append("target_llama")
    if independent_success:
        successes.append("independent_validator")
    r["success_targets"] = successes

    if r.get("label") == 1:
        if target_success and independent_success:
            r["transfer_tier"] = "transfer_success"
            r["validation_status"] = "validated"
        elif target_success:
            r["transfer_tier"] = "target_only"
            r["validation_status"] = "validated"
        elif independent_success:
            r["transfer_tier"] = "cross_only"
            r["validation_status"] = "validated"
        else:
            r["transfer_tier"] = "no_jailbreak"
            r["validation_status"] = "ambiguous"
            r["training_eligible"] = False

        # Prefer the independent validator's behavioral transition as the
        # candidate turn for later evidence testing. This is only a candidate,
        # not attribution ground truth.
        candidate_pivot = independent_val.get("pivot_turn_id")
        if candidate_pivot is None:
            candidate_pivot = target_val.get("pivot_turn_id")
        r["candidate_evidence_turn_id"] = candidate_pivot

        candidate_kind = independent_val.get("pivot_kind") or target_val.get("pivot_kind") or "none"
        r["candidate_evidence_turn_kind"] = candidate_kind

        r["training_eligible"] = r.get("validation_status") == "validated"
    else:
        # Benign data is retained only when both behavioral validators agree
        # that the trajectory did not produce an unsafe transition.
        r["transfer_tier"] = "benign"
        if target_success or independent_success:
            r["validation_status"] = "rejected"
            r["training_eligible"] = False
            r["benign_status"] = "benign_boundary_rejected"
        else:
            r["validation_status"] = "validated"
            r["training_eligible"] = True
            r.setdefault("benign_status", "validated_benign")

    # Keep the old field so downstream legacy code still loads the record, but
    # make provenance explicit instead of calling it cross-model evidence.
    r["validation_provenance"] = {
        "target_model": target_model,
        "independent_model": independent_model,
        "target_success": target_success,
        "independent_success": independent_success,
    }

    return r


def print_stats(records: List[Dict]) -> None:
    labels = Counter(r.get("label", -1) for r in records)
    tiers = Counter(r.get("transfer_tier", "unknown") for r in records)
    status = Counter(r.get("validation_status", "unknown") for r in records)
    models = Counter(r.get("independent_validation_model", "unknown") for r in records)

    print(f"Records: {len(records)}")
    print(f"Labels: {dict(labels)}")
    print(f"Transfer tiers: {dict(tiers)}")
    print(f"Validation status: {dict(status)}")
    print(f"Independent validators: {dict(models)}")

    bad_same_family = 0
    for r in records:
        model = str(r.get("independent_validation_model", "")).lower()
        if "qwen" in model:
            bad_same_family += 1
    if bad_same_family:
        print(
            f"WARNING: {bad_same_family} records still name a Qwen independent "
            "validator. For the NAACL repair, use a non-Qwen validator."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Independent-validator JSONL")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = [normalize_record(r) for r in load_jsonl(args.input)]
    write_jsonl(records, args.output)
    print_stats(records)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
