#!/usr/bin/env python3
"""Throughput-preserving wrapper for Stage B4 evidence analysis.

This wrapper does not change the scientific counterfactual. Stage B4 still runs a
fresh full baseline first and requires exact B1/B2 reproducibility. After that
baseline has passed, counterfactual replays reuse the already-proven identical
conversation prefix before the intervention turn and regenerate/judge only the
intervention turn and its downstream suffix.

The optimization removes redundant target decoding and judge calls for turns that
occur strictly before the intervention and therefore cannot have changed. Prefix
reuse is additionally guarded inside this class: if fresh baseline target-response
fingerprints do not exactly match the stored B1 assistant texts, optimized replay
is disabled and the implementation falls back to a full counterfactual replay.
"""
from __future__ import annotations

import copy
from typing import Dict, Optional, Sequence

import frontier_evidence_analysis as fea


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


def main() -> None:
    # The original main() resolves FrontierEvidenceValidator from its module
    # globals at runtime. Patch only for executable use, not on import, so unit
    # tests and other modules do not receive an unexpected global side effect.
    fea.FrontierEvidenceValidator = PrefixReuseEvidenceValidator
    fea.main()


if __name__ == "__main__":
    main()
