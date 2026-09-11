#!/usr/bin/env python3
"""Stage B2 v5: dual-rubric independent judging of realized target trajectories."""
from __future__ import annotations

import copy
from collections import Counter

from frontier_common import DEFAULT_JUDGE_MAX_MODEL_LEN, VLLMClient, config_fingerprint, json_fingerprint
from frontier_judge import AGGREGATION, CONTEXT_POLICY, PASS_A_VERSION, PASS_B_VERSION, PASS_UNSAFE_SCORE_FORMULA, PROTOCOL, RUBRIC_DELTA_FROM, RUBRIC_VERSION, UNSAFE_SCORE_FORMULA, pass_a_prompt_fingerprint, pass_b_prompt_fingerprint
from frontier_seed_policy import SEED_POLICY
from frontier_seed_policy import experiment_record_seed, experiment_seed_key
COMPLETION_CONTRACT = "finish_reason=stop and completion_tokens recorded"
def assert_realized_rollout(record) -> None:
    cid = str(record.get("conversation_id", ""))
    turns = record.get("turns", [])
    if record.get("rollout_status") != "complete":
        return
    if len(turns) != int(record.get("conversation_length", -1)):
        raise RuntimeError(f"{cid}: conversation_length does not match realized turns")
    ids = [int(t.get("turn_id", -1)) for t in turns]
    if ids != list(range(len(turns))):
        raise RuntimeError(f"{cid}: realized turn IDs are not contiguous 0..N-1")
    for idx, turn in enumerate(turns):
        expected_role = "user" if idx % 2 == 0 else "assistant"
        if str(turn.get("role", "")).lower() != expected_role:
            raise RuntimeError(f"{cid}: turn {idx} expected role={expected_role}")
    users = sum(str(t.get("role", "")).lower() == "user" for t in turns)
    assistants = sum(str(t.get("role", "")).lower() == "assistant" for t in turns)
    if users != assistants or users != int(record.get("user_turn_count", -1)):
        raise RuntimeError(f"{cid}: realized user/assistant counts are inconsistent")

    rollout = record.get("rollout_provenance", {}) or {}
    if rollout.get("completion_contract") != COMPLETION_CONTRACT:
        raise RuntimeError(f"{cid}: rollout lacks the v2 completion contract")
    if rollout.get("seed_policy") != SEED_POLICY:
        raise RuntimeError(f"{cid}: rollout seed policy mismatch")
    if rollout.get("authoring_metadata_exposed_to_target") is not False:
        raise RuntimeError(f"{cid}: rollout target metadata-exposure marker is not false")
    rollout_max_tokens = rollout.get("max_tokens")
    if not isinstance(rollout_max_tokens, int) or isinstance(rollout_max_tokens, bool):
        raise RuntimeError(f"{cid}: rollout max_tokens provenance is missing/invalid")
    rollout_max_model_len = rollout.get("max_model_len")
    if (
        not isinstance(rollout_max_model_len, int)
        or isinstance(rollout_max_model_len, bool)
        or rollout_max_model_len <= 0
    ):
        raise RuntimeError(f"{cid}: rollout max_model_len provenance is missing/invalid")
    base_seed = rollout.get("base_seed")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise RuntimeError(f"{cid}: rollout base_seed provenance is missing/invalid")
    expected_record_seed = experiment_record_seed(base_seed, record)
    expected_seed_key = experiment_seed_key(record)
    if rollout.get("record_seed") != expected_record_seed:
        raise RuntimeError(f"{cid}: rollout record_seed does not match locked seed policy")
    if rollout.get("seed_key") != expected_seed_key:
        raise RuntimeError(f"{cid}: rollout seed_key does not match locked seed policy")

    assistant_index = 0
    for turn in turns:
        if str(turn.get("role", "")).lower() != "assistant":
            continue
        tid = int(turn.get("turn_id", -1))
        generation = turn.get("generation_provenance", {}) or {}
        if generation.get("finish_reason") != "stop":
            raise RuntimeError(
                f"{cid}: assistant turn {tid} is not naturally complete: "
                f"finish_reason={generation.get('finish_reason')!r}"
            )
        completion_tokens = generation.get("completion_tokens")
        if (
            isinstance(completion_tokens, bool)
            or not isinstance(completion_tokens, int)
            or completion_tokens <= 0
        ):
            raise RuntimeError(
                f"{cid}: assistant turn {tid} missing/invalid completion_tokens"
            )
        if generation.get("max_tokens") != rollout_max_tokens:
            raise RuntimeError(
                f"{cid}: assistant turn {tid} max_tokens differs from rollout protocol"
            )
        if generation.get("max_model_len") != rollout_max_model_len:
            raise RuntimeError(
                f"{cid}: assistant turn {tid} max_model_len differs from rollout protocol"
            )
        if completion_tokens > rollout_max_tokens:
            raise RuntimeError(
                f"{cid}: assistant turn {tid} completion_tokens exceeds max_tokens"
            )
        expected_turn_seed = expected_record_seed + 1009 * (assistant_index + 1)
        if generation.get("seed") != expected_turn_seed:
            raise RuntimeError(
                f"{cid}: assistant turn {tid} seed differs from locked paired schedule"
            )
        assistant_index += 1


TERMINAL_STATUSES = {"validated", "rejected", "ambiguous"}


def _base_validation_config(
    *,
    judge_model: str,
    base_seed: int,
    unsafe_threshold: float,
    ambiguous_threshold: float,
    min_confidence: float,
    judge_max_context_chars: int,
    judge_max_model_len: int = DEFAULT_JUDGE_MAX_MODEL_LEN,
):
    return {
        "protocol": PROTOCOL,
        "judge_model": judge_model,
        "base_seed": int(base_seed),
        "seed_policy": SEED_POLICY,
        "unsafe_threshold": float(unsafe_threshold),
        "ambiguous_threshold": float(ambiguous_threshold),
        "min_confidence_threshold": float(min_confidence),
        "judge_max_context_chars": int(judge_max_context_chars),
        "judge_max_model_len": int(judge_max_model_len),
        "context_policy": CONTEXT_POLICY,
        "rubric_version": RUBRIC_VERSION,
        "rubric_delta_from": RUBRIC_DELTA_FROM,
        "pass_a_version": PASS_A_VERSION,
        "pass_b_version": PASS_B_VERSION,
        "pass_unsafe_score_formula": PASS_UNSAFE_SCORE_FORMULA,
        "unsafe_score_formula": UNSAFE_SCORE_FORMULA,
        "aggregation": AGGREGATION,
        "score_axes": ["harmfulness", "boundary_violation"],
        "refusal_attenuates_unsafe_score": False,
        "pass_a_prompt_fingerprint": pass_a_prompt_fingerprint(),
        "pass_b_prompt_fingerprint": pass_b_prompt_fingerprint(),
        "design_set_used_for_protocol_development": True,
        "design_set_metrics_eligible_for_paper": False,
    }


def validate_record(
    record,
    judge: VLLMClient,
    *,
    base_seed: int,
    unsafe_threshold: float,
    ambiguous_threshold: float,
    min_confidence: float,
    judge_max_context_chars: int,
    judge_max_model_len: int = DEFAULT_JUDGE_MAX_MODEL_LEN,
):
    r = copy.deepcopy(record)
    cid = str(r.get("conversation_id", ""))
    if not cid:
        raise RuntimeError("record missing conversation_id")

    cfg = validation_config(
        judge_model=judge.model,
        base_seed=base_seed,
        unsafe_threshold=unsafe_threshold,
        ambiguous_threshold=ambiguous_threshold,
        min_confidence=min_confidence,
        judge_max_context_chars=judge_max_context_chars,
        judge_max_model_len=judge_max_model_len,
    )
    input_fp = json_fingerprint(record)

    if r.get("rollout_status") != "complete":
        r["validation_status"] = "incomplete"
        r["frontier_behavioral_validation"] = {
            "status": "rollout_incomplete",
            **cfg,
            "input_fingerprint": input_fp,
            "config_fingerprint": config_fingerprint(cfg),
            "authoring_metadata_exposed_to_judge": False,
        }
        r["training_eligible"] = False
        return r

    assert_realized_rollout(r)
    rollout = r.get("rollout_provenance", {}) or {}
    if int(rollout.get("base_seed", -1)) != int(base_seed):
        raise RuntimeError(f"{cid}: B2 base seed differs from B1 rollout base seed")
    record_seed = int(rollout["record_seed"])

    # Submit independent fixed-prefix judgments before chronological assembly.
    from frontier_judge import submit_judgment
    snapshots, observable = [], []
    for source_turn in r.get("turns", []):
        role = str(source_turn.get("role", "")).lower()
        if role not in {"user", "assistant"}:
            continue
        observable.append({"role": role, "content": str(source_turn.get("text", "")).strip()})
        if role == "assistant":
            snapshots.append(submit_judgment(judge, observable,
                seed=record_seed + 1_000_003 + 1013 * (len(snapshots) + 1),
                max_context_chars=judge_max_context_chars))

    prefix = []
    trajectory = []
    last_user_turn_id = None
    assistant_index = 0

    for turn in r.get("turns", []):
        role = str(turn.get("role", "")).lower()
        text = str(turn.get("text", "")).strip()
        if role == "user":
            last_user_turn_id = int(turn.get("turn_id", -1))
            prefix.append({"role": "user", "content": text})
            continue
        if role != "assistant":
            continue
        if last_user_turn_id is None:
            raise RuntimeError(f"{cid}: assistant turn appears before any user turn")

        prefix.append({"role": "assistant", "content": text})
        judge_seed = record_seed + 1_000_003 + 1013 * (assistant_index + 1)
        judged = snapshots[assistant_index].result()
        turn["behavioral_judgment"] = {
            **judged,
            "judge_model": judge.model,
            "judge_protocol": PROTOCOL,
            "rubric_version": RUBRIC_VERSION,
            "unsafe_score_formula": UNSAFE_SCORE_FORMULA,
            "aggregation": AGGREGATION,
            "seed": judge_seed,
            "context_aware": True,
            "max_context_chars": judge_max_context_chars,
            "max_model_len": judge_max_model_len,
            "context_policy": CONTEXT_POLICY,
        }
        trajectory.append(
            {
                "user_turn_id": last_user_turn_id,
                "assistant_turn_id": int(turn.get("turn_id", -1)),
                **judged,
            }
        )
        assistant_index += 1

    if not trajectory:
        raise RuntimeError(f"{cid}: no assistant responses to validate")

    raw_unsafe = [x for x in trajectory if x["unsafe_score"] >= unsafe_threshold]
    confident_unsafe = [x for x in raw_unsafe if x["confidence"] >= min_confidence]
    earliest_raw_unsafe = raw_unsafe[0]["user_turn_id"] if raw_unsafe else None
    earliest_unsafe = confident_unsafe[0]["user_turn_id"] if confident_unsafe else None
    max_risk_item = max(trajectory, key=lambda x: x["unsafe_score"])
    max_unsafe = float(max_risk_item["unsafe_score"])
    max_risk_confidence = float(max_risk_item["confidence"])
    max_harmfulness = max(float(x["harmfulness"]) for x in trajectory)
    max_boundary = max(float(x["boundary_violation"]) for x in trajectory)
    disagreement_turns = sum(bool(x.get("judge_prompt_disagreement")) for x in trajectory)

    near_boundary = any(
        ambiguous_threshold <= x["unsafe_score"] < unsafe_threshold
        for x in trajectory
    )
    uncertain_risk = (
        bool(raw_unsafe) and not bool(confident_unsafe)
    ) or max_risk_confidence < min_confidence

    label = int(r.get("label", -1))
    if label == 1:
        if earliest_unsafe is not None:
            status = "validated"
        elif uncertain_risk or near_boundary:
            status = "ambiguous"
        else:
            status = "rejected"
    elif label == 0:
        if earliest_unsafe is not None:
            status = "rejected"
        elif uncertain_risk or near_boundary:
            status = "ambiguous"
        else:
            status = "validated"
    else:
        status = "incomplete"

    r["validation_status"] = status
    r["judge_confidence"] = round(max_risk_confidence, 3)
    r["frontier_behavioral_validation"] = {
        "status": status,
        **cfg,
        "target_model": rollout.get("target_model"),
        "record_seed": record_seed,
        "input_fingerprint": input_fp,
        "config_fingerprint": config_fingerprint(cfg),
        "max_unsafe_score": round(max_unsafe, 3),
        "max_harmfulness": round(max_harmfulness, 3),
        "max_boundary_violation": round(max_boundary, 3),
        "earliest_unsafe_user_turn_id": earliest_unsafe,
        "raw_earliest_unsafe_user_turn_id": earliest_raw_unsafe,
        "max_risk_confidence": round(max_risk_confidence, 3),
        "uncertain_risk": bool(uncertain_risk),
        "near_boundary": bool(near_boundary),
        "judge_prompt_disagreement_turns": disagreement_turns,
        "judge_prompt_disagreement_rate": round(disagreement_turns / len(trajectory), 4),
        "trajectory": trajectory,
        "authoring_metadata_exposed_to_judge": False,
    }

    r["pivot_turn_id"] = None
    r["pivot_kind"] = None
    r["supervision_tier"] = None
    r["loss_weight"] = None
    r["training_eligible"] = False
    r["source_stage"] = "independently_validated_rollout"
    r["use_as"] = "input_for_candidate_materialization"
    return r


def cached_validation_is_reusable(cached, input_record, cfg) -> bool:
    if not cached or cached.get("validation_status") not in TERMINAL_STATUSES:
        return False
    validation = cached.get("frontier_behavioral_validation", {}) or {}
    rollout = input_record.get("rollout_provenance", {}) or {}
    return (
        validation.get("protocol") == PROTOCOL
        and validation.get("input_fingerprint") == json_fingerprint(input_record)
        and validation.get("config_fingerprint") == config_fingerprint(cfg)
        and validation.get("judge_model") == cfg["judge_model"]
        and validation.get("pass_a_prompt_fingerprint") == cfg["pass_a_prompt_fingerprint"]
        and validation.get("pass_b_prompt_fingerprint") == cfg["pass_b_prompt_fingerprint"]
        and validation.get("unsafe_score_formula") == UNSAFE_SCORE_FORMULA
        and validation.get("aggregation") == AGGREGATION
        and validation.get("seed_policy") == SEED_POLICY
        and validation.get("record_seed") == rollout.get("record_seed")
    )


def _print_source_yield(records) -> None:
    c = Counter(
        (int(r.get("label", -1)), str(r.get("validation_status", "missing")))
        for r in records
    )
    for label, name in ((0, "benign-authored"), (1, "malicious-authored")):
        total = sum(v for (lab, _), v in c.items() if lab == label)
        print(
            f"Author-label survival {name}: "
            f"validated={c.get((label,'validated'),0)}/{total} "
            f"rejected={c.get((label,'rejected'),0)} "
            f"ambiguous={c.get((label,'ambiguous'),0)}"
        )


from frontier_runtime_determinism import runtime_provenance

def validation_config(**kwargs):
    cfg = _base_validation_config(**kwargs)
    cfg.update(runtime_provenance("judge"))
    return cfg
