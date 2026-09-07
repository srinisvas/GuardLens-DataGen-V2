#!/usr/bin/env python3
"""Normalize an independent validation pass for the NAACL validity repair.

This script is intentionally separate from ``merge_validations.py`` because the
v11 merge utility assumes the post-generation validator is Qwen. For the NAACL
repair we want explicit provenance and fail-closed handling of validator errors.

Expected input
--------------
The JSONL produced by ``launch_val.slurm`` after running the existing generated
trajectories through an independent validator such as
``mistralai/Mistral-7B-Instruct-v0.3``. Each record should retain the original
``llama_validation`` from interactive generation and contain the new validation
result in ``causal_validation``.

Validity policy
---------------
A record is eligible for the repaired dataset only when both the original target
trajectory and the new independent validation are structurally usable. Missing
or failed validation is never interpreted as a safe outcome. The independent
pass must use structured-judge results rather than the marker fallback.

No counterfactual claim is made here. Counterfactual evidence is recomputed from
fresh paired replays by ``naacl/evidence_analysis.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter
from typing import Dict, Iterable, List, Tuple


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
    return bool(validation.get("validated") is True and validation.get("jailbreak_detected"))


def _trajectory(validation: Dict) -> List[Dict]:
    trajectory = validation.get("compliance_trajectory", [])
    return trajectory if isinstance(trajectory, list) else []


def _target_quality(record: Dict, validation: Dict) -> Tuple[bool, str]:
    """Check that the stored target trajectory is usable, without overfitting to judge details."""
    if not isinstance(validation, dict) or validation.get("validated") is not True:
        return False, "target_validation_missing_or_unvalidated"
    if not _trajectory(validation):
        return False, "target_validation_empty_trajectory"

    # The interactive generator encodes backend failures directly into turn text.
    # Such records must not enter training merely because the fallback judge gave
    # the error string a benign-looking score.
    for turn in record.get("turns", []):
        text = str(turn.get("text", "")).strip().lower()
        if text.startswith("[target_error:"):
            return False, "target_backend_error_in_stored_turns"
        if text.startswith("[generation_error:"):
            return False, "generator_backend_error_in_stored_turns"
    return True, "ok"


def _independent_quality(validation: Dict) -> Tuple[bool, str]:
    """Fail closed on missing responses, backend errors, or marker-fallback judging."""
    if not isinstance(validation, dict) or validation.get("validated") is not True:
        return False, "independent_validation_missing_or_unvalidated"
    trajectory = _trajectory(validation)
    if not trajectory:
        return False, "independent_validation_empty_trajectory"

    for item in trajectory:
        snippet = str(item.get("response_snippet", "")).strip().lower()
        if snippet.startswith("[validation_error:"):
            return False, "independent_backend_error"
        method = item.get("method")
        if method != "structured_judge":
            return False, f"independent_non_structured_judge:{method or 'missing'}"
    return True, "ok"


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

    target_ok, target_reason = _target_quality(r, target_val)
    independent_ok, independent_reason = _independent_quality(independent_val)

    r["independent_validation"] = independent_val
    r["independent_validation_model"] = independent_model
    r["target_validation_model"] = target_model

    target_success = target_ok and _jailbreak(target_val)
    independent_success = independent_ok and _jailbreak(independent_val)

    successes: List[str] = []
    if target_success:
        successes.append("target_llama")
    if independent_success:
        successes.append("independent_validator")
    r["success_targets"] = successes

    # Both validations must be usable because the stored conversation contains
    # the original target responses and the repaired protocol explicitly asks for
    # an independent behavioral check. Missing validation is not a negative vote.
    both_usable = target_ok and independent_ok

    if r.get("label") == 1:
        if not both_usable:
            r["transfer_tier"] = "validation_incomplete"
            r["validation_status"] = "incomplete"
            r["training_eligible"] = False
        elif target_success and independent_success:
            r["transfer_tier"] = "transfer_success"
            r["validation_status"] = "validated"
            r["training_eligible"] = True
        elif target_success:
            r["transfer_tier"] = "target_only"
            r["validation_status"] = "validated"
            r["training_eligible"] = True
        elif independent_success:
            r["transfer_tier"] = "cross_only"
            r["validation_status"] = "validated"
            r["training_eligible"] = True
        else:
            r["transfer_tier"] = "no_jailbreak"
            r["validation_status"] = "ambiguous"
            r["training_eligible"] = False

        # Retain both historical candidate transitions for auditability. The
        # paired evidence script derives its actual intervention anchor from its
        # own fresh baseline replay rather than trusting either old pivot.
        r["target_candidate_evidence_turn_id"] = target_val.get("pivot_turn_id")
        r["independent_candidate_evidence_turn_id"] = independent_val.get("pivot_turn_id")
        r["candidate_evidence_turn_id"] = (
            target_val.get("pivot_turn_id")
            if target_val.get("pivot_turn_id") is not None
            else independent_val.get("pivot_turn_id")
        )
        r["candidate_evidence_turn_kind"] = (
            target_val.get("pivot_kind")
            or independent_val.get("pivot_kind")
            or "none"
        )
    else:
        r["transfer_tier"] = "benign"
        if not both_usable:
            r["validation_status"] = "incomplete"
            r["training_eligible"] = False
            r["benign_status"] = "benign_validation_incomplete"
        elif target_success or independent_success:
            r["validation_status"] = "rejected"
            r["training_eligible"] = False
            r["benign_status"] = "benign_boundary_rejected"
        else:
            r["validation_status"] = "validated"
            r["training_eligible"] = True
            r.setdefault("benign_status", "validated_benign")

    r["validation_provenance"] = {
        "target_model": target_model,
        "independent_model": independent_model,
        "target_usable": target_ok,
        "independent_usable": independent_ok,
        "target_quality_reason": target_reason,
        "independent_quality_reason": independent_reason,
        "target_success": target_success,
        "independent_success": independent_success,
    }

    return r


def print_stats(records: List[Dict]) -> None:
    labels = Counter(r.get("label", -1) for r in records)
    tiers = Counter(r.get("transfer_tier", "unknown") for r in records)
    status = Counter(r.get("validation_status", "unknown") for r in records)
    models = Counter(r.get("independent_validation_model", "unknown") for r in records)
    quality = Counter(
        (
            r.get("validation_provenance", {}).get("target_quality_reason", "missing"),
            r.get("validation_provenance", {}).get("independent_quality_reason", "missing"),
        )
        for r in records
    )

    print(f"Records: {len(records)}")
    print(f"Labels: {dict(labels)}")
    print(f"Transfer tiers: {dict(tiers)}")
    print(f"Validation status: {dict(status)}")
    print(f"Independent validators: {dict(models)}")
    print("Validation-quality pairs:")
    for pair, count in quality.most_common():
        print(f"  target={pair[0]} independent={pair[1]}: {count}")

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
