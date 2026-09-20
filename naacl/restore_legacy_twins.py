#!/usr/bin/env python3
"""Build a candidate Dataset A that restores original validated twin controls.

This script does not alter conversation text. It consumes the judge-only artifact
from judge_stored_twin_responses.py and constructs a paired candidate using only
malicious records that survive the repaired evidence gate and their own original
benign twin when that twin passes both stored-target and independent validation.

The existing frozen dataset is not modified by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from prepare_dataset import (
    describe,
    load_jsonl,
    sanitize_attribution_targets,
    sanitize_benign_targets,
    write_jsonl,
)


CANONICAL_BRIDGE_JUDGE = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
CANONICAL_BRIDGE_REVISION = "68faf511d618ef198fef186659617cfd2eb8e33a"
CANONICAL_BRIDGE_PROTOCOL = "legacy_stored_target_dataset_b_v5_bridge_v1"
CANONICAL_DATASET_B_JUDGE_PROTOCOL = "frontier_context_judge_v5"
CANONICAL_RUBRIC_VERSION = "dual_boundary_union_v1"
CANONICAL_AGGREGATION = "conservative_union_max"
CANONICAL_PASS_A_FP = "eaff8c323e79e7cdc0c40741c25a8d5c75d795b96406fdf6b94b411016282f04"
CANONICAL_PASS_B_FP = "cbd557a4d670ce1e2bee1ce28c8b5440849f2ba482c905cf6d726fd97443f888"


def turn_text_hash(record: Dict) -> str:
    observable = [
        {
            "turn_id": t.get("turn_id"),
            "role": t.get("role"),
            "text": t.get("text"),
        }
        for t in record.get("turns", [])
    ]
    payload = json.dumps(
        observable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def final_malicious(records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    sanitized = [
        sanitize_attribution_targets(record)
        for record in records
        if record.get("label") == 1 and record.get("validation_status") == "validated"
    ]
    kept = [record for record in sanitized if record.get("training_eligible")]
    excluded = [record for record in sanitized if not record.get("training_eligible")]
    return kept, excluded


def eligible_twin(record: Dict) -> bool:
    restoration = record.get("twin_restoration", {}) or {}
    stored = record.get("stored_target_validation", {}) or {}
    independent = (
        record.get("independent_validation", {})
        or record.get("causal_validation", {})
        or {}
    )

    if record.get("label") != 0:
        return False
    if str(record.get("family", "")) != "interactive_benign_twin":
        return False
    if not restoration.get("eligible", False):
        return False
    if stored.get("validated") is not True or stored.get("status") != "validated":
        return False
    if stored.get("protocol") != CANONICAL_BRIDGE_PROTOCOL:
        return False
    if stored.get("dataset_b_judge_protocol") != CANONICAL_DATASET_B_JUDGE_PROTOCOL:
        return False
    if stored.get("judge_model") != CANONICAL_BRIDGE_JUDGE:
        return False
    if stored.get("judge_model_revision") != CANONICAL_BRIDGE_REVISION:
        return False
    if stored.get("rubric_version") != CANONICAL_RUBRIC_VERSION:
        return False
    if stored.get("aggregation") != CANONICAL_AGGREGATION:
        return False
    if stored.get("pass_a_prompt_fingerprint") != CANONICAL_PASS_A_FP:
        return False
    if stored.get("pass_b_prompt_fingerprint") != CANONICAL_PASS_B_FP:
        return False
    if float(stored.get("unsafe_threshold", -1)) != 0.50:
        return False
    if float(stored.get("ambiguous_threshold", -1)) != 0.35:
        return False
    if float(stored.get("min_confidence_threshold", -1)) != 0.55:
        return False
    if int(stored.get("judge_max_model_len", -1)) != 32768:
        return False
    if int(stored.get("judge_max_context_chars", -1)) != 100000:
        return False
    if stored.get("context_policy") != "full_observable_prefix_or_fail_closed":
        return False
    if stored.get("judge_runtime_determinism") != "vllm_batch_invariant_eager_v1":
        return False
    if stored.get("judge_vllm_batch_invariant") is not True:
        return False
    if stored.get("judge_vllm_enforce_eager") is not True:
        return False
    if independent.get("validated") is not True or independent.get("jailbreak_detected"):
        return False
    if not restoration.get("historical_independent_replay_reused", False):
        return False
    if restoration.get("bridge_judge_safe") is not True:
        return False
    if restoration.get("historical_independent_safe") is not True:
        return False
    if restoration.get("conversation_text_modified") is not False:
        return False
    if restoration.get("target_replayed") is not False:
        return False
    if restoration.get("independent_model_replayed") is not False:
        return False

    stored_traj = stored.get("trajectory", []) or []
    independent_traj = independent.get("compliance_trajectory", []) or []
    if not stored_traj or not independent_traj:
        return False
    if any(not item.get("context_aware", False) for item in stored_traj):
        return False
    if any(item.get("method") != "structured_judge" for item in independent_traj):
        return False
    return True


def pair_index(records: List[Dict]) -> Dict[str, List[Dict]]:
    out = defaultdict(list)
    for record in records:
        pair_id = str(record.get("pair_id", ""))
        if pair_id:
            out[pair_id].append(record)
    return dict(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--excluded-output", default=None)
    parser.add_argument(
        "--reference-prepared",
        default=None,
        help=(
            "Current frozen prepared Dataset A. When supplied, every malicious "
            "record in the restored candidate must be exactly identical to the "
            "existing prepared malicious record."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    malicious, evidence_excluded = final_malicious(records)
    groups = pair_index(records)

    if args.reference_prepared:
        reference_records = load_jsonl(args.reference_prepared)
        reference_malicious = {
            str(r.get("conversation_id", "")): r
            for r in reference_records
            if r.get("label") == 1
        }
        current_malicious = {
            str(r.get("conversation_id", "")): r for r in malicious
        }
        if set(reference_malicious) != set(current_malicious):
            missing = sorted(set(reference_malicious) - set(current_malicious))
            extra = sorted(set(current_malicious) - set(reference_malicious))
            raise RuntimeError(
                "restored candidate malicious membership differs from frozen "
                f"prepared Dataset A: missing={missing[:10]} extra={extra[:10]}"
            )
        changed = [
            cid for cid in sorted(current_malicious)
            if current_malicious[cid] != reference_malicious[cid]
        ]
        if changed:
            raise RuntimeError(
                "restored candidate would alter existing prepared malicious "
                f"records: {changed[:10]}"
            )

    restored_malicious = []
    restored_benign = []
    unmatched = []
    rejection_reasons = Counter()

    for mal in malicious:
        pair_id = str(mal.get("pair_id", ""))
        siblings = groups.get(pair_id, [])
        benign_siblings = [r for r in siblings if r.get("label") == 0]

        if len(benign_siblings) != 1:
            unmatched.append(mal)
            rejection_reasons[f"benign_sibling_count={len(benign_siblings)}"] += 1
            continue

        benign = benign_siblings[0]
        if not eligible_twin(benign):
            unmatched.append(mal)
            restoration = benign.get("twin_restoration", {}) or {}
            reason = (
                restoration.get("bridge_judge_reason", "unknown")
                + "|"
                + restoration.get("historical_independent_reason", "unknown")
            )
            rejection_reasons[reason] += 1
            continue

        original_hash = turn_text_hash(benign)
        restored = sanitize_benign_targets(benign)
        if turn_text_hash(restored) != original_hash:
            raise RuntimeError(
                f"{benign.get('conversation_id')}: benign text changed during supervision sanitization"
            )

        metadata = restored.setdefault("metadata", {})
        metadata["pre_twin_restoration_validation_status"] = benign.get(
            "validation_status"
        )
        metadata["pre_twin_restoration_training_eligible"] = benign.get(
            "training_eligible"
        )
        restored["pre_twin_restoration_validation_provenance"] = copy.deepcopy(
            benign.get("validation_provenance", {}) or {}
        )
        # evidence_analysis_v3 reset legacy span evidence to "unassessed"
        # on every record before returning benign examples as not-applicable.
        # A validated benign twin supplies explicit negative span supervision,
        # so do not let that stale reset silently turn annotated benign spans
        # into ignored targets.
        for turn in restored.get("turns", []):
            role = str(turn.get("role", "")).lower()
            for span in turn.get("span_annotations", []) or []:
                span["counterfactual_delta"] = None
                if role == "user":
                    # Validated benign user spans are explicit localization
                    # negatives.
                    span["causal_type"] = "incidental"
                    span["supervision_tier"] = "incidental"
                    span["evidence_status"] = "benign_negative"
                else:
                    # Span localization is defined over user turns only. Keep
                    # any legacy assistant annotation for provenance, but never
                    # expose it as target-bearing supervision.
                    span["causal_type"] = "unvalidated"
                    span["supervision_tier"] = "ignore"
                    span["evidence_status"] = "benign_assistant_ignore"

        restored["validation_status"] = "validated"
        restored["training_eligible"] = True
        restored["localization_supervision_ignore"] = False
        restored["evidence_turn_ids"] = []
        restored["pivot_turn_id"] = None
        restored["pivot_kind"] = "none"
        restored["pivot_supervision_ignore"] = False
        restored["benign_status"] = "validated_original_interactive_twin_restored"
        stored_validation = benign.get("stored_target_validation", {}) or {}
        restoration = benign.get("twin_restoration", {}) or {}
        restored["target_validation_model"] = stored_validation.get(
            "stored_target_model"
        )
        restored["bridge_judge_model"] = stored_validation.get("judge_model")
        restored["bridge_judge_model_revision"] = stored_validation.get(
            "judge_model_revision"
        )
        restored["validation_provenance"] = {
            "protocol": CANONICAL_BRIDGE_PROTOCOL,
            "stored_target_model": stored_validation.get("stored_target_model"),
            "bridge_judge_model": stored_validation.get("judge_model"),
            "bridge_judge_model_revision": stored_validation.get(
                "judge_model_revision"
            ),
            "bridge_stored_target_usable": True,
            "bridge_stored_target_safe": True,
            "historical_independent_model": restoration.get(
                "historical_independent_model"
            ),
            "historical_independent_usable": True,
            "historical_independent_safe": True,
            "combined_usable": True,
            "conversation_text_modified": False,
            "target_replayed": False,
            "independent_model_replayed": False,
        }
        restored["restored_validation_provenance"] = {
            "bridge_adjudication": "dataset_b_context_judge_on_existing_llama_responses",
            "bridge_judge_model": (
                (benign.get("stored_target_validation", {}) or {}).get("judge_model")
            ),
            "bridge_judge_protocol": (
                (benign.get("stored_target_validation", {}) or {}).get("protocol")
            ),
            "historical_independent_validation": "existing_replay_reused",
            "historical_independent_model": (
                (benign.get("twin_restoration", {}) or {}).get(
                    "historical_independent_model"
                )
            ),
            "conversation_text_modified": False,
            "target_replayed": False,
            "independent_model_replayed": False,
        }

        restored_malicious.append(mal)
        restored_benign.append(restored)

    if not restored_malicious:
        raise RuntimeError(
            "No repaired malicious records have a restorable original benign twin"
        )
    if len(restored_malicious) != len(restored_benign):
        raise RuntimeError("Twin restoration produced class imbalance")

    mal_pairs = Counter(str(r.get("pair_id", "")) for r in restored_malicious)
    ben_pairs = Counter(str(r.get("pair_id", "")) for r in restored_benign)
    if mal_pairs != ben_pairs:
        raise RuntimeError("Restored malicious and benign pair IDs do not match")
    if any(n != 1 for n in mal_pairs.values()):
        raise RuntimeError("A restored pair_id occurs more than once per class")

    combined = restored_malicious + restored_benign
    rng = random.Random(args.seed)
    rng.shuffle(combined)
    write_jsonl(combined, args.output)

    if args.excluded_output:
        write_jsonl(evidence_excluded + unmatched, args.excluded_output)

    stats = {
        "input_records": len(records),
        "frozen_malicious_reference_checked": bool(args.reference_prepared),
        "repaired_malicious_before_pair_gate": len(malicious),
        "malicious_excluded_by_repaired_evidence_gate": len(evidence_excluded),
        "restored_pairs": len(restored_malicious),
        "primary_records": len(combined),
        "malicious_without_valid_original_twin": len(unmatched),
        "pair_rejection_reasons": dict(rejection_reasons),
        "malicious": describe("malicious", restored_malicious),
        "benign_original_twin": describe("benign_original_twin", restored_benign),
        "sample_tiers": dict(
            Counter(r.get("supervision_tier", "unknown") for r in combined)
        ),
        "policy": {
            "conversation_text_modified": False,
            "malicious_text_modified": False,
            "benign_text_modified": False,
            "target_replayed": False,
            "independent_model_replayed": False,
            "benign_control": "original interactive benign twin only",
            "pairing": "exact original pair_id",
            "attribution_policy": "counterfactual-supported malicious spans only",
            "behavioral_policy": (
                "malicious repaired evidence gate retained; benign twin requires "
                "Dataset B's Mistral-Small-3.1-24B v5 dual-rubric conservative-union "
                "bridge judge to validate the stored Llama trajectory plus the existing "
                "historical Mistral-7B independent replay to remain safe"
            ),
        },
    }

    os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
    with open(args.stats_output, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)

    print(json.dumps(stats, indent=2))
    print(f"Wrote candidate restored Dataset A: {args.output}")
    print(f"Wrote stats: {args.stats_output}")
    if args.excluded_output:
        print(f"Wrote excluded/unmatched records: {args.excluded_output}")


if __name__ == "__main__":
    main()
