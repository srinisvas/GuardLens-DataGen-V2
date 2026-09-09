#!/usr/bin/env python3
"""Throughput-preserving production wrapper for Stage B4 evidence analysis.

Scientific semantics remain those of ``frontier_evidence_analysis.py``. The
wrapper adds only fail-closed execution/provenance controls:

1. B4 record seeds are taken from the already-audited B1 rollout provenance, so
   malicious/benign twins retain the locked pair-shared seed policy.
2. The target/judge runtime context windows and execution optimization are added
   to the B4 configuration fingerprint.
3. A fresh full baseline still runs first. Prefix reuse is enabled only when the
   fresh baseline response fingerprints exactly match stored B1 assistant text;
   otherwise counterfactuals fall back to the original full replay.
"""
from __future__ import annotations

import argparse
import copy
import sys
from typing import Dict, Optional, Sequence

import frontier_evidence_analysis as fea
from frontier_seed_policy import (
    SEED_POLICY,
    experiment_record_seed,
    experiment_seed_key,
)

EXECUTION_OPTIMIZATION = "verified_identical_prefix_reuse_v1"
RECORD_SEED_SOURCE = "stage_b1_rollout_provenance"


class PrefixReuseEvidenceValidator(fea.FrontierEvidenceValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._baseline_key: Optional[str] = None
        self._baseline: Optional[Dict] = None
        self._baseline_prefix_safe: bool = False

    @staticmethod
    def _turns_key(turns: Sequence[Dict]) -> str:
        observable = [
            {
                "turn_id": int(t.get("turn_id", -1)),
                "role": str(t.get("role", "")).lower(),
                "text": str(t.get("text", "")),
            }
            for t in turns
            if str(t.get("role", "")).lower() in {"user", "assistant"}
        ]
        return fea.json_fingerprint(observable)

    @staticmethod
    def _stored_prefix_matches_fresh_baseline(
        turns: Sequence[Dict], baseline: Dict
    ) -> bool:
        users = [
            t for t in turns
            if str(t.get("role", "")).lower() == "user"
        ]
        assistants = [
            t for t in turns
            if str(t.get("role", "")).lower() == "assistant"
        ]
        trajectory = list(baseline.get("trajectory", []))
        if not (len(users) == len(assistants) == len(trajectory)):
            return False
        for user, assistant, item in zip(users, assistants, trajectory):
            user_tid = int(user.get("turn_id", -1))
            if int(item.get("user_turn_id", -1)) != user_tid:
                return False
            assistant_text = str(assistant.get("text", ""))
            if item.get("response_fingerprint") != fea.json_fingerprint(assistant_text):
                return False
        return True

    def replay(self, turns, *, seed: int, replacements=None):
        replacements = replacements or {}
        key = self._turns_key(turns)

        if not replacements:
            baseline = super().replay(turns, seed=seed, replacements=None)
            self._baseline_key = key
            self._baseline = copy.deepcopy(baseline)
            self._baseline_prefix_safe = self._stored_prefix_matches_fresh_baseline(
                turns, baseline
            )
            return baseline

        if (
            len(replacements) != 1
            or self._baseline is None
            or self._baseline_key != key
            or not self._baseline_prefix_safe
        ):
            return super().replay(turns, seed=seed, replacements=replacements)

        turn_id, replacement_text = next(iter(replacements.items()))
        return self._replay_suffix(
            turns,
            baseline=self._baseline,
            seed=seed,
            intervention_turn_id=int(turn_id),
            replacement_text=str(replacement_text),
        )

    def _replay_suffix(
        self,
        turns: Sequence[Dict],
        *,
        baseline: Dict,
        seed: int,
        intervention_turn_id: int,
        replacement_text: str,
    ) -> Dict:
        users = [
            t for t in turns
            if str(t.get("role", "")).lower() == "user"
        ]
        assistants = {
            int(t.get("turn_id", -1)): t
            for t in turns
            if str(t.get("role", "")).lower() == "assistant"
        }
        baseline_by_user = {
            int(item.get("user_turn_id", -1)): item
            for item in baseline.get("trajectory", [])
        }

        messages = []
        trajectory = []
        intervention_seen = False

        for user_index, user_turn in enumerate(users):
            tid = int(user_turn.get("turn_id", -1))
            original_text = str(user_turn.get("text", "")).strip()
            if not original_text:
                raise RuntimeError(f"empty user turn at turn_id={tid}")

            if tid < intervention_turn_id:
                assistant = assistants.get(tid + 1)
                baseline_item = baseline_by_user.get(tid)
                if assistant is None or baseline_item is None:
                    raise RuntimeError(
                        f"cannot reuse baseline prefix at user turn {tid}: "
                        "missing stored assistant or fresh baseline judgment"
                    )
                assistant_text = str(assistant.get("text", "")).strip()
                if not assistant_text:
                    raise RuntimeError(f"empty stored assistant turn at turn_id={tid+1}")
                if baseline_item.get("response_fingerprint") != fea.json_fingerprint(
                    assistant_text
                ):
                    raise RuntimeError(
                        f"baseline prefix identity lost at assistant turn {tid+1}"
                    )
                messages.append({"role": "user", "content": original_text})
                messages.append({"role": "assistant", "content": assistant_text})
                trajectory.append(copy.deepcopy(baseline_item))
                continue

            if tid == intervention_turn_id:
                intervention_seen = True
                text = replacement_text.strip()
                if not text:
                    raise RuntimeError(
                        f"empty intervention replacement at turn_id={intervention_turn_id}"
                    )
            else:
                text = original_text

            messages.append({"role": "user", "content": text})
            response_seed = seed + 1009 * (user_index + 1)
            response = self.target.chat(
                messages,
                seed=response_seed,
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
            messages.append({"role": "assistant", "content": response})

            judge_seed = seed + 1_000_003 + 1013 * (user_index + 1)
            judged = fea.judge_assistant_response(
                self.judge,
                messages,
                seed=judge_seed,
                max_context_chars=self.judge_max_context_chars,
            )
            trajectory.append({
                "user_turn_id": tid,
                "unsafe_score": judged["unsafe_score"],
                "behavior": judged["behavior"],
                "harmfulness": judged["harmfulness"],
                "refusal_strength": judged["refusal_strength"],
                "confidence": judged["confidence"],
                "response_fingerprint": fea.json_fingerprint(response),
                "response_snippet": response[:240],
            })

        if not intervention_seen:
            raise RuntimeError(
                f"intervention turn {intervention_turn_id} is not a user turn"
            )
        if not trajectory:
            raise RuntimeError("counterfactual replay produced no trajectory")

        return {
            "trajectory": trajectory,
            "max_unsafe_score": round(
                max(x["unsafe_score"] for x in trajectory), 3
            ),
            "final_unsafe_score": round(trajectory[-1]["unsafe_score"], 3),
        }


def build_record_seed_map(records, *, expected_base_seed: int) -> Dict[str, int]:
    seed_map: Dict[str, int] = {}
    for record in records:
        cid = str(record.get("conversation_id", ""))
        if not cid or cid in seed_map:
            raise RuntimeError(f"invalid/duplicate conversation_id in B4 input: {cid!r}")
        rollout = record.get("rollout_provenance", {}) or {}
        if rollout.get("seed_policy") != SEED_POLICY:
            raise RuntimeError(f"{cid}: B4 input seed policy mismatch")
        if int(rollout.get("base_seed", -1)) != int(expected_base_seed):
            raise RuntimeError(f"{cid}: B4 seed differs from B1 base seed")
        expected = experiment_record_seed(expected_base_seed, record)
        if rollout.get("record_seed") != expected:
            raise RuntimeError(f"{cid}: B1 record seed violates paired seed policy")
        if rollout.get("seed_key") != experiment_seed_key(record):
            raise RuntimeError(f"{cid}: B1 seed key violates paired seed policy")
        validation = record.get("frontier_behavioral_validation", {}) or {}
        if validation.get("seed_policy") != SEED_POLICY:
            raise RuntimeError(f"{cid}: B2 seed policy mismatch")
        if validation.get("record_seed") != expected:
            raise RuntimeError(f"{cid}: B2 record seed differs from B1")
        seed_map[cid] = expected
    return seed_map


def _cli_value(args, flag: str, default=None):
    for idx, value in enumerate(args):
        if value == flag:
            if idx + 1 >= len(args):
                raise RuntimeError(f"missing value for {flag}")
            return args[idx + 1]
        prefix = flag + "="
        if value.startswith(prefix):
            return value[len(prefix):]
    return default


def main() -> None:
    # Parse wrapper-only runtime provenance flags, then remove them before handing
    # the remaining CLI to the original evidence engine.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--target-max-model-len", type=int, default=16384)
    parser.add_argument("--judge-max-model-len", type=int, default=16384)
    runtime, remaining = parser.parse_known_args(sys.argv[1:])
    if runtime.target_max_model_len <= 0 or runtime.judge_max_model_len <= 0:
        raise ValueError("runtime model context windows must be positive")

    input_path = _cli_value(remaining, "--input")
    if not input_path:
        raise RuntimeError("B4 wrapper requires --input")
    base_seed = int(_cli_value(remaining, "--seed", 42))
    records = fea.load_jsonl(input_path)
    seed_map = build_record_seed_map(records, expected_base_seed=base_seed)

    original_seed_fn = fea.stable_record_seed
    original_config_fn = fea.evidence_config

    def rollout_bound_seed(requested_base_seed: int, conversation_id: str) -> int:
        if int(requested_base_seed) != base_seed:
            raise RuntimeError(
                f"B4 requested base_seed={requested_base_seed} != audited B1 seed={base_seed}"
            )
        cid = str(conversation_id)
        if cid not in seed_map:
            raise RuntimeError(f"B4 seed requested for unknown conversation_id={cid}")
        return seed_map[cid]

    def production_evidence_config(**kwargs):
        cfg = original_config_fn(**kwargs)
        cfg.update(
            {
                "seed_policy": SEED_POLICY,
                "record_seed_source": RECORD_SEED_SOURCE,
                "target_max_model_len": int(runtime.target_max_model_len),
                "judge_max_model_len": int(runtime.judge_max_model_len),
                "execution_optimization": EXECUTION_OPTIMIZATION,
            }
        )
        return cfg

    fea.stable_record_seed = rollout_bound_seed
    fea.evidence_config = production_evidence_config
    fea.FrontierEvidenceValidator = PrefixReuseEvidenceValidator
    sys.argv = [sys.argv[0], *remaining]
    try:
        fea.main()
    finally:
        fea.stable_record_seed = original_seed_fn
        fea.evidence_config = original_config_fn


if __name__ == "__main__":
    main()
