#!/usr/bin/env python3
"""Stage B5: convert evidence-analyzed frontier records to canonical GuardLens records.

Primary Dataset B is pair-complete by construction: a paired malicious/benign
twin enters only when both sides pass their respective gates. Standalone hard
benign records are retained as a separately validated stress set rather than
entering primary training, preventing their distinct construction distribution
from becoming a label/length shortcut.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
from collections import Counter, defaultdict
from typing import Dict, List

from frontier_common import (
    DEFAULT_JUDGE_MAX_CONTEXT_CHARS,
    DEFAULT_JUDGE_MAX_MODEL_LEN,
    DEFAULT_TARGET_MAX_MODEL_LEN,
    DEFAULT_TARGET_MAX_TOKENS,
    load_jsonl,
    write_jsonl,
)

LOSS_WEIGHTS = {
    "cf_strong": 1.00,
    "cf_weak": 0.70,
    "llm_confirmed": 0.60,
    "benign_validated": 1.00,
    "ignore": 0.00,
}
DEFAULT_TARGET = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_JUDGE = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
ROLLOUT_PROTOCOL = "frontier_fixed_user_rollout_v2"
VALIDATION_PROTOCOL = "frontier_context_judge_v3"
EVIDENCE_PROTOCOL = "frontier_context_paired_counterfactual_v4"
COMPLETION_CONTRACT = "finish_reason=stop and completion_tokens recorded"
CONTEXT_POLICY = "full_observable_prefix_or_fail_closed"
EXPECTED_TARGET_MAX_TOKENS = DEFAULT_TARGET_MAX_TOKENS
EXPECTED_TARGET_MAX_MODEL_LEN = DEFAULT_TARGET_MAX_MODEL_LEN
EXPECTED_JUDGE_MAX_MODEL_LEN = DEFAULT_JUDGE_MAX_MODEL_LEN
EXPECTED_JUDGE_MAX_CONTEXT_CHARS = DEFAULT_JUDGE_MAX_CONTEXT_CHARS


def iter_spans(record: Dict):
    for turn in record.get("turns", []):
        for span in turn.get("span_annotations", []):
            yield span


def user_turn_count(record: Dict) -> int:
    return sum(
        str(t.get("role", "")).lower() == "user"
        for t in record.get("turns", [])
    )


def assert_expected_provenance(
    record: Dict,
    *,
    expected_target: str,
    expected_judge: str,
    require_evidence: bool,
) -> None:
    """Require the exact reviewed B1→B4 protocol chain for every prepared record."""
    cid = str(record.get("conversation_id", ""))
    rollout = record.get("rollout_provenance", {}) or {}
    validation = record.get("frontier_behavioral_validation", {}) or {}
    analysis = record.get("frontier_evidence_analysis", {}) or {}

    if rollout.get("protocol") != ROLLOUT_PROTOCOL:
        raise ValueError(f"{cid}: unsupported rollout protocol {rollout.get('protocol')!r}")
    if rollout.get("completion_contract") != COMPLETION_CONTRACT:
        raise ValueError(f"{cid}: rollout completion contract mismatch")
    if rollout.get("target_model") != expected_target:
        raise ValueError(
            f"{cid}: rollout target {rollout.get('target_model')!r} != expected {expected_target!r}"
        )
    if int(rollout.get("max_tokens", -1)) != EXPECTED_TARGET_MAX_TOKENS:
        raise ValueError(
            f"{cid}: rollout max_tokens={rollout.get('max_tokens')!r} != "
            f"expected {EXPECTED_TARGET_MAX_TOKENS}"
        )
    if int(rollout.get("max_model_len", -1)) != EXPECTED_TARGET_MAX_MODEL_LEN:
        raise ValueError(
            f"{cid}: rollout max_model_len={rollout.get('max_model_len')!r} != "
            f"expected {EXPECTED_TARGET_MAX_MODEL_LEN}"
        )
    if rollout.get("authoring_metadata_exposed_to_target") is not False:
        raise ValueError(f"{cid}: target metadata-exposure provenance is not fail-closed")

    if validation.get("protocol") != VALIDATION_PROTOCOL:
        raise ValueError(
            f"{cid}: unsupported validation protocol {validation.get('protocol')!r}"
        )
    if validation.get("judge_model") != expected_judge:
        raise ValueError(
            f"{cid}: validation judge {validation.get('judge_model')!r} != expected {expected_judge!r}"
        )
    if int(validation.get("judge_max_model_len", -1)) != EXPECTED_JUDGE_MAX_MODEL_LEN:
        raise ValueError(
            f"{cid}: judge runtime context={validation.get('judge_max_model_len')!r} != "
            f"expected {EXPECTED_JUDGE_MAX_MODEL_LEN}"
        )
    if int(validation.get("judge_max_context_chars", -1)) != EXPECTED_JUDGE_MAX_CONTEXT_CHARS:
        raise ValueError(
            f"{cid}: judge context budget={validation.get('judge_max_context_chars')!r} != "
            f"expected {EXPECTED_JUDGE_MAX_CONTEXT_CHARS}"
        )
    if validation.get("context_policy") != CONTEXT_POLICY:
        raise ValueError(f"{cid}: validation context policy mismatch")
    if validation.get("authoring_metadata_exposed_to_judge") is not False:
        raise ValueError(f"{cid}: judge metadata-exposure provenance is not fail-closed")

    if analysis.get("protocol") != EVIDENCE_PROTOCOL:
        raise ValueError(f"{cid}: unsupported evidence protocol {analysis.get('protocol')!r}")
    if analysis.get("target_model") != expected_target:
        raise ValueError(f"{cid}: evidence target differs from primary target")
    if analysis.get("judge_model") != expected_judge:
        raise ValueError(f"{cid}: evidence judge differs from primary judge")
    if int(analysis.get("max_tokens", -1)) != EXPECTED_TARGET_MAX_TOKENS:
        raise ValueError(f"{cid}: evidence target token cap mismatch")
    if int(analysis.get("target_max_model_len", EXPECTED_TARGET_MAX_MODEL_LEN)) != EXPECTED_TARGET_MAX_MODEL_LEN:
        raise ValueError(f"{cid}: evidence target runtime context mismatch")
    if int(analysis.get("judge_max_model_len", -1)) != EXPECTED_JUDGE_MAX_MODEL_LEN:
        raise ValueError(f"{cid}: evidence judge runtime context mismatch")
    if int(analysis.get("judge_max_context_chars", -1)) != EXPECTED_JUDGE_MAX_CONTEXT_CHARS:
        raise ValueError(f"{cid}: evidence judge context budget mismatch")
    if analysis.get("context_policy") != CONTEXT_POLICY:
        raise ValueError(f"{cid}: evidence context policy mismatch")

    if require_evidence:
        if analysis.get("status") != "complete":
            raise ValueError(f"{cid}: malicious evidence analysis is not complete")
        if analysis.get("baseline_reproduced_stored_rollout") is not True:
            raise ValueError(f"{cid}: paired baseline did not reproduce stored rollout")
    else:
        if analysis.get("status") != "not_applicable":
            raise ValueError(
                f"{cid}: benign evidence envelope status={analysis.get('status')!r}, "
                "expected not_applicable"
            )


def sanitize_malicious(
    record: Dict,
    *,
    expected_target: str,
    expected_judge: str,
) -> Dict:
    r = copy.deepcopy(record)
    analysis = r.get("frontier_evidence_analysis", {}) or {}
    if r.get("validation_status") != "validated":
        raise ValueError("malicious record is not behaviorally validated")
    if analysis.get("status") != "complete":
        raise ValueError(f"malicious evidence status is {analysis.get('status', 'missing')}")
    assert_expected_provenance(
        r,
        expected_target=expected_target,
        expected_judge=expected_judge,
        require_evidence=True,
    )

    has_strong = False
    has_weak = False
    for span in iter_spans(r):
        status = str(span.get("evidence_status", "unassessed"))
        if status == "supported_strong":
            has_strong = True
            span["causal_type"] = "causal"
            span["supervision_tier"] = "cf_strong"
        elif status == "supported_weak":
            has_weak = True
            span["causal_type"] = "causal"
            span["supervision_tier"] = "cf_weak"
        elif status == "negative_control_supported":
            span["causal_type"] = "incidental"
            span["supervision_tier"] = "incidental"
        else:
            span.setdefault("original_label", span.get("label", ""))
            span["label"] = "EVIDENCE_CANDIDATE"
            span["causal_type"] = "unvalidated"
            span["supervision_tier"] = "ignore"
            if status.startswith("not_assessable") or status == "unassessed":
                span["counterfactual_delta"] = None

    if has_strong:
        tier = "cf_strong"
    elif has_weak:
        tier = "cf_weak"
    else:
        tier = "llm_confirmed"
    r["supervision_tier"] = tier
    r["loss_weight"] = LOSS_WEIGHTS[tier]
    r["training_eligible"] = True

    evidence_turns = sorted({int(x) for x in r.get("evidence_turn_ids", [])})
    r["evidence_turn_ids"] = evidence_turns
    r["pivot_turn_id"] = evidence_turns[0] if evidence_turns else None
    r["pivot_supervision_ignore"] = not bool(evidence_turns)
    if len(evidence_turns) > 1:
        r["pivot_kind"] = "distributed"
    elif len(evidence_turns) == 1:
        r["pivot_kind"] = "contextual_pivot"
    else:
        r["pivot_kind"] = "none"

    r["repaired_detection_evidence"] = {
        "fresh_target_unsafe": True,
        "independent_context_judge_success": True,
        "eligible": True,
        "protocol": analysis.get("protocol"),
    }
    r["canonical_target_model"] = expected_target
    r["canonical_judge_model"] = expected_judge
    return r


def sanitize_benign(
    record: Dict,
    *,
    expected_target: str,
    expected_judge: str,
) -> Dict:
    r = copy.deepcopy(record)
    if r.get("validation_status") != "validated":
        raise ValueError("benign record is not behaviorally validated")
    assert_expected_provenance(
        r,
        expected_target=expected_target,
        expected_judge=expected_judge,
        require_evidence=False,
    )
    for span in iter_spans(r):
        span["causal_type"] = "incidental"
        span["supervision_tier"] = "incidental"
        span["evidence_status"] = span.get("evidence_status", "benign_negative")
        span["counterfactual_delta"] = None
    r["supervision_tier"] = "benign_validated"
    r["loss_weight"] = LOSS_WEIGHTS["benign_validated"]
    r["training_eligible"] = True
    r["pivot_turn_id"] = None
    r["pivot_kind"] = "none"
    r["pivot_supervision_ignore"] = False
    r["evidence_turn_ids"] = []
    r["canonical_target_model"] = expected_target
    r["canonical_judge_model"] = expected_judge
    return r


def describe(records: List[Dict]) -> Dict:
    user_counts = [user_turn_count(r) for r in records]
    by_label_lengths = {
        str(label): Counter(user_turn_count(r) for r in records if r.get("label") == label)
        for label in (0, 1)
    }
    return {
        "n": len(records),
        "labels": dict(Counter(r.get("label") for r in records)),
        "validation": dict(Counter(r.get("validation_status") for r in records)),
        "supervision_tiers": dict(Counter(r.get("supervision_tier") for r in records)),
        "pivot_modes": {
            "supported_malicious": sum(r.get("label") == 1 and bool(r.get("evidence_turn_ids")) for r in records),
            "unknown_malicious": sum(r.get("label") == 1 and bool(r.get("pivot_supervision_ignore")) for r in records),
            "benign_true_no_pivot": sum(r.get("label") == 0 and not r.get("pivot_supervision_ignore", False) for r in records),
        },
        "user_turns": {
            "mean": statistics.mean(user_counts) if user_counts else 0.0,
            "min": min(user_counts) if user_counts else 0,
            "max": max(user_counts) if user_counts else 0,
            "histogram": dict(Counter(user_counts)),
            "by_label": {
                label: dict(hist) for label, hist in by_label_lengths.items()
            },
        },
        "scenario_families": len({(r.get("metadata", {}) or {}).get("scenario_family") for r in records}),
        "mechanism_families": len({(r.get("metadata", {}) or {}).get("mechanism_family") for r in records}),
        "pair_ids": len({r.get("pair_id") for r in records if r.get("pair_id") not in (None, "")}),
    }


def add_common_provenance(record: Dict) -> Dict:
    record["authoring_intent_label"] = record.get("label")
    record["corpus_source"] = "frontier_authored_v3"
    return record


def excluded_copy(record: Dict, reason: str) -> Dict:
    r = add_common_provenance(copy.deepcopy(record))
    r["training_eligible"] = False
    r["exclusion_reason"] = reason
    r["use_as"] = "excluded_from_primary_frontier_corpus"
    return r


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benign-stress-output", required=True)
    parser.add_argument("--excluded-output", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--expected-target-model", default=DEFAULT_TARGET)
    parser.add_argument("--expected-judge-model", default=DEFAULT_JUDGE)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    prepared: List[Dict] = []
    benign_stress: List[Dict] = []
    excluded: List[Dict] = []

    pair_groups = defaultdict(list)
    standalone = []
    for record in records:
        pair_id = record.get("pair_id")
        if pair_id in (None, ""):
            standalone.append(record)
        else:
            pair_groups[str(pair_id)].append(record)

    retained_pairs = 0
    for pair_id, group in pair_groups.items():
        labels = Counter(r.get("label") for r in group)
        if len(group) != 2 or labels != Counter({0: 1, 1: 1}):
            reason = f"pair {pair_id} is structurally invalid: n={len(group)} labels={dict(labels)}"
            excluded.extend(excluded_copy(r, reason) for r in group)
            continue

        malicious = next(r for r in group if r.get("label") == 1)
        benign = next(r for r in group if r.get("label") == 0)
        mal_out = ben_out = None
        failures = []
        try:
            mal_out = sanitize_malicious(
                add_common_provenance(copy.deepcopy(malicious)),
                expected_target=args.expected_target_model,
                expected_judge=args.expected_judge_model,
            )
        except Exception as exc:
            failures.append(f"malicious twin failed: {exc}")
        try:
            ben_out = sanitize_benign(
                add_common_provenance(copy.deepcopy(benign)),
                expected_target=args.expected_target_model,
                expected_judge=args.expected_judge_model,
            )
        except Exception as exc:
            failures.append(f"benign twin failed: {exc}")

        if failures:
            reason = f"pair {pair_id} excluded as incomplete primary pair; " + " | ".join(failures)
            excluded.append(excluded_copy(malicious, reason))
            excluded.append(excluded_copy(benign, reason))
            continue

        if user_turn_count(mal_out) != user_turn_count(ben_out):
            reason = f"pair {pair_id} lost user-turn length symmetry"
            excluded.append(excluded_copy(malicious, reason))
            excluded.append(excluded_copy(benign, reason))
            continue

        for out in (mal_out, ben_out):
            out["source_stage"] = "canonical_training_record"
            out["use_as"] = "eligible_for_consolidated_merge"
            out["primary_pair_complete"] = True
        prepared.extend([mal_out, ben_out])
        retained_pairs += 1

    for record in standalone:
        if record.get("label") != 0:
            excluded.append(excluded_copy(record, "standalone non-benign record is unsupported"))
            continue
        try:
            out = sanitize_benign(
                add_common_provenance(copy.deepcopy(record)),
                expected_target=args.expected_target_model,
                expected_judge=args.expected_judge_model,
            )
            out["training_eligible"] = False
            out["benign_status"] = "frontier_standalone_hard_benign_stress"
            out["source_stage"] = "canonical_stress_record"
            out["use_as"] = "benign_stress_evaluation_only"
            out["primary_pair_complete"] = False
            benign_stress.append(out)
        except Exception as exc:
            excluded.append(excluded_copy(record, f"standalone benign stress validation failed: {exc}"))

    labels = Counter(r.get("label") for r in prepared)
    if labels.get(0, 0) != labels.get(1, 0):
        raise RuntimeError(f"pair-complete primary corpus is not label-balanced: {dict(labels)}")
    mal_hist = Counter(user_turn_count(r) for r in prepared if r.get("label") == 1)
    ben_hist = Counter(user_turn_count(r) for r in prepared if r.get("label") == 0)
    if mal_hist != ben_hist:
        raise RuntimeError(
            f"pair-complete primary corpus lost exact user-turn histogram matching: mal={dict(mal_hist)} ben={dict(ben_hist)}"
        )

    write_jsonl(prepared, args.output)
    write_jsonl(benign_stress, args.benign_stress_output)
    write_jsonl(excluded, args.excluded_output)
    stats = {
        "prepared": describe(prepared),
        "pair_retention": {
            "source_pairs": len(pair_groups),
            "retained_complete_pairs": retained_pairs,
            "retained_records": 2 * retained_pairs,
            "policy": "both malicious and benign twins must pass; no orphan twin enters primary corpus",
        },
        "benign_stress": describe(benign_stress),
        "excluded": {
            "n": len(excluded),
            "by_label": dict(Counter(r.get("label") for r in excluded)),
            "by_validation": dict(Counter(r.get("validation_status") for r in excluded)),
            "by_reason": dict(Counter(r.get("exclusion_reason") for r in excluded)),
        },
        "method": {
            "authoring_intent_is_ground_truth": False,
            "malicious_gate": "validated Qwen rollout AND reproducible fresh paired evidence baseline remains unsafe",
            "benign_gate": "independent full-prefix context-aware Mistral judge validates safe Qwen trajectory",
            "primary_pair_policy": "retain complete validated malicious/benign twins only",
            "standalone_hard_benign_policy": "validated standalones are evaluation-only stress records",
            "pivot_policy": "only counterfactually supported evidence turns populate pivot_turn_id",
            "span_policy": "only counterfactually supported spans receive positive attribution supervision",
            "expected_target_model": args.expected_target_model,
            "expected_judge_model": args.expected_judge_model,
            "rollout_protocol": ROLLOUT_PROTOCOL,
            "validation_protocol": VALIDATION_PROTOCOL,
            "evidence_protocol": EVIDENCE_PROTOCOL,
            "target_max_tokens": EXPECTED_TARGET_MAX_TOKENS,
            "target_max_model_len": EXPECTED_TARGET_MAX_MODEL_LEN,
            "judge_max_model_len": EXPECTED_JUDGE_MAX_MODEL_LEN,
            "judge_max_context_chars": EXPECTED_JUDGE_MAX_CONTEXT_CHARS,
            "judge_context_policy": CONTEXT_POLICY,
        },
    }
    os.makedirs(os.path.dirname(args.stats_output) or ".", exist_ok=True)
    with open(args.stats_output, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    print(json.dumps(stats, indent=2))
    print(f"Wrote prepared primary corpus: {args.output}")
    print(f"Wrote benign stress set: {args.benign_stress_output}")
    print(f"Wrote excluded records: {args.excluded_output}")


if __name__ == "__main__":
    main()
