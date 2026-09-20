#!/usr/bin/env python3
"""CPU-only preflight for Dataset A original-twin restoration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from typing import Dict, List

from prepare_dataset import sanitize_attribution_targets


def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return out


def validate_stored_trajectory(record: Dict) -> None:
    cid = str(record.get("conversation_id", ""))
    turns = record.get("turns", [])
    if not isinstance(turns, list) or not turns:
        raise RuntimeError(f"{cid}: empty stored trajectory")

    expected = "user"
    pairs = 0
    turn_ids = []
    for turn in turns:
        role = str(turn.get("role", "")).lower()
        text = str(turn.get("text", "")).strip().lower()
        tid = turn.get("turn_id")
        if isinstance(tid, bool) or not isinstance(tid, int):
            raise RuntimeError(f"{cid}: invalid turn_id {tid!r}")
        turn_ids.append(tid)
        if text.startswith(("[target_error:", "[generation_error:")):
            raise RuntimeError(f"{cid}: stored backend error text")
        if role != expected:
            raise RuntimeError(
                f"{cid}: non-alternating stored trajectory; expected {expected}, got {role}"
            )
        if role == "assistant":
            pairs += 1
        expected = "assistant" if expected == "user" else "user"

    if expected != "user":
        raise RuntimeError(f"{cid}: stored trajectory ends with unmatched user turn")
    if pairs == 0:
        raise RuntimeError(f"{cid}: no stored user/assistant pairs")
    if len(turn_ids) != len(set(turn_ids)):
        raise RuntimeError(f"{cid}: duplicate stored turn IDs")
    expected_ids = list(range(len(turns)))
    if turn_ids != expected_ids:
        raise RuntimeError(
            f"{cid}: stored turn IDs differ from original contiguous 0..N-1 "
            f"contract; got {turn_ids[:20]}"
        )
    declared_length = record.get("conversation_length")
    if declared_length is not None and int(declared_length) != len(turns):
        raise RuntimeError(
            f"{cid}: conversation_length={declared_length} differs from "
            f"physical turns={len(turns)}"
        )


def assert_original_twin_semantics(malicious: Dict, benign: Dict) -> None:
    pair_id = str(benign.get("pair_id", ""))
    if malicious.get("pair_id") != benign.get("pair_id"):
        raise RuntimeError(f"{pair_id}: pair_id mismatch between twins")
    if malicious.get("target_domain") != benign.get("target_domain"):
        raise RuntimeError(
            f"{pair_id}: target_domain mismatch "
            f"{malicious.get('target_domain')!r} vs {benign.get('target_domain')!r}"
        )
    if malicious.get("style") != benign.get("style"):
        raise RuntimeError(
            f"{pair_id}: style mismatch "
            f"{malicious.get('style')!r} vs {benign.get('style')!r}"
        )

    mal_turns = malicious.get("turns", [])
    ben_turns = benign.get("turns", [])
    shared_setup = []
    for turn in mal_turns:
        if (
            str(turn.get("role", "")).lower() == "user"
            and str(turn.get("semantic_role", "")) != "setup"
        ):
            break
        shared_setup.append(turn)

    if not shared_setup:
        raise RuntimeError(f"{pair_id}: malicious twin has no shared setup prefix")
    if len(ben_turns) < len(shared_setup):
        raise RuntimeError(
            f"{pair_id}: benign twin shorter than malicious shared setup prefix"
        )

    for idx, (mal_turn, ben_turn) in enumerate(zip(shared_setup, ben_turns)):
        for field in ("turn_id", "role", "text"):
            if mal_turn.get(field) != ben_turn.get(field):
                raise RuntimeError(
                    f"{pair_id}: shared setup differs at index {idx} field={field}"
                )


def observable_turn_hash(record: Dict) -> str:
    payload = [
        {
            "turn_id": turn.get("turn_id"),
            "role": turn.get("role"),
            "text": turn.get("text"),
        }
        for turn in record.get("turns", [])
    ]
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_source_hashes(path: str) -> Dict[str, str]:
    hashes = {}
    for record in load_jsonl(path):
        cid = str(record.get("conversation_id", ""))
        if not cid:
            raise RuntimeError(f"{path}: source record missing conversation_id")
        if cid in hashes:
            raise RuntimeError(f"{path}: duplicate source conversation_id {cid}")
        hashes[cid] = observable_turn_hash(record)
    return hashes


def independent_snapshot(record: Dict) -> Dict:
    return (
        record.get("independent_validation", {})
        or record.get("causal_validation", {})
        or {}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--original-source",
        default=None,
        help=(
            "Optional pre-independent-validation artifact. When supplied, every "
            "original benign twin must have byte-equivalent observable turn "
            "IDs/roles/text in the restoration input."
        ),
    )
    parser.add_argument(
        "--frozen-prepared",
        default=None,
        help=(
            "Current frozen prepared Dataset A. When supplied, reconstructed "
            "eligible malicious records must match it exactly before any GPU run."
        ),
    )
    parser.add_argument("--expected-benign-twins", type=int, default=545)
    parser.add_argument("--expected-validated-malicious", type=int, default=545)
    parser.add_argument("--expected-final-malicious-candidates", type=int, default=526)
    parser.add_argument(
        "--expected-target-model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--expected-independent-model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
    )
    args = parser.parse_args()

    records = load_jsonl(args.input)
    errors = []
    ids = Counter(str(r.get("conversation_id", "")) for r in records)
    dup_ids = [cid for cid, n in ids.items() if cid and n > 1]
    if dup_ids:
        errors.append(f"duplicate conversation IDs: {dup_ids[:10]}")

    pairs = defaultdict(list)
    for record in records:
        pair_id = str(record.get("pair_id", ""))
        if pair_id:
            pairs[pair_id].append(record)

    twins = [
        r for r in records
        if r.get("label") == 0
        and r.get("family") == "interactive_benign_twin"
        and r.get("pair_id")
    ]

    independent_models = Counter()
    independent_status = Counter()
    target_models = Counter()
    malformed = 0
    bad_pair_structure = 0
    total_stored_assistant_turns = 0
    max_observable_transcript_chars = 0

    for twin in twins:
        cid = str(twin.get("conversation_id", ""))
        pair_id = str(twin.get("pair_id", ""))
        siblings = pairs.get(pair_id, [])
        malicious = [r for r in siblings if r.get("label") == 1]
        benign = [r for r in siblings if r.get("label") == 0]

        if len(malicious) != 1 or len(benign) != 1:
            bad_pair_structure += 1
            errors.append(
                f"{cid}: pair {pair_id} has malicious={len(malicious)} benign={len(benign)}"
            )
        else:
            try:
                assert_original_twin_semantics(malicious[0], benign[0])
            except Exception as exc:
                errors.append(str(exc))

            target_validation = malicious[0].get("llama_validation", {}) or {}
            target_model = (
                target_validation.get("model_used")
                or target_validation.get("target_model")
            )
            target_models[str(target_model or "UNKNOWN")] += 1
            if not target_model:
                errors.append(
                    f"{cid}: malicious sibling lacks original target-model provenance"
                )

        try:
            validate_stored_trajectory(twin)
            assistant_turns = sum(
                str(turn.get("role", "")).lower() == "assistant"
                for turn in twin.get("turns", [])
            )
            total_stored_assistant_turns += assistant_turns
            chunks = [
                f"[{str(turn.get('role', '')).upper()}]\n{str(turn.get('text', '')).strip()}"
                for turn in twin.get("turns", [])
            ]
            transcript_chars = len("\n\n".join(chunks))
            max_observable_transcript_chars = max(
                max_observable_transcript_chars, transcript_chars
            )
            if transcript_chars > 100000:
                errors.append(
                    f"{cid}: observable transcript chars={transcript_chars} "
                    "exceed frozen Dataset B judge budget=100000"
                )
        except Exception as exc:
            malformed += 1
            errors.append(str(exc))

        validation = independent_snapshot(twin)
        model = (
            twin.get("independent_validation_model")
            or validation.get("model_used")
        )
        independent_models[str(model or "UNKNOWN")] += 1

        if validation.get("validated") is not True:
            independent_status["missing_or_unvalidated"] += 1
        elif not validation.get("compliance_trajectory"):
            independent_status["empty_trajectory"] += 1
        elif any(
            item.get("method") != "structured_judge"
            for item in validation.get("compliance_trajectory", [])
        ):
            independent_status["non_structured_judge"] += 1
        elif validation.get("jailbreak_detected"):
            independent_status["unsafe"] += 1
        else:
            independent_status["safe"] += 1

    malicious_validated = [
        r for r in records
        if r.get("label") == 1
        and r.get("family") == "interactive_adversarial"
        and r.get("validation_status") == "validated"
    ]

    final_malicious_candidates = [
        r for r in records
        if r.get("label") == 1
        and r.get("validation_status") == "validated"
        and str((r.get("evidence_analysis", {}) or {}).get("status", "missing"))
            not in {"error", "missing"}
        and (
            str((r.get("evidence_analysis", {}) or {}).get("status", "missing"))
                == "complete"
            or bool(
                (r.get("validation_provenance", {}) or {}).get(
                    "independent_success", False
                )
            )
        )
    ]

    source_lineage_checked = False
    source_lineage_missing = 0
    source_lineage_mismatch = 0
    if args.original_source:
        source_hashes = load_source_hashes(args.original_source)
        source_lineage_checked = True
        for twin in twins:
            cid = str(twin.get("conversation_id", ""))
            expected_hash = source_hashes.get(cid)
            if expected_hash is None:
                source_lineage_missing += 1
                errors.append(f"{cid}: absent from original source artifact")
            elif observable_turn_hash(twin) != expected_hash:
                source_lineage_mismatch += 1
                errors.append(
                    f"{cid}: observable stored trajectory differs from original source"
                )

    frozen_malicious_reference_checked = False
    frozen_malicious_reference_mismatch = 0
    if args.frozen_prepared:
        frozen_records = load_jsonl(args.frozen_prepared)
        frozen_malicious = {
            str(r.get("conversation_id", "")): r
            for r in frozen_records
            if r.get("label") == 1
        }
        reconstructed = {}
        for record in records:
            if record.get("label") != 1 or record.get("validation_status") != "validated":
                continue
            try:
                sanitized = sanitize_attribution_targets(record)
            except Exception as exc:
                errors.append(
                    f"{record.get('conversation_id')}: malicious sanitization failed: {exc}"
                )
                continue
            if sanitized.get("training_eligible"):
                reconstructed[str(sanitized.get("conversation_id", ""))] = sanitized

        frozen_malicious_reference_checked = True
        if set(frozen_malicious) != set(reconstructed):
            missing = sorted(set(frozen_malicious) - set(reconstructed))
            extra = sorted(set(reconstructed) - set(frozen_malicious))
            frozen_malicious_reference_mismatch += len(missing) + len(extra)
            errors.append(
                "reconstructed malicious membership differs from frozen prepared A: "
                f"missing={missing[:10]} extra={extra[:10]}"
            )
        changed = [
            cid for cid in sorted(set(frozen_malicious) & set(reconstructed))
            if frozen_malicious[cid] != reconstructed[cid]
        ]
        if changed:
            frozen_malicious_reference_mismatch += len(changed)
            errors.append(
                "reconstructed malicious records differ from frozen prepared A: "
                f"{changed[:10]}"
            )

    summary = {
        "records": len(records),
        "original_benign_twins": len(twins),
        "validated_interactive_malicious": len(malicious_validated),
        "final_malicious_candidates": len(final_malicious_candidates),
        "pair_groups": len(pairs),
        "bad_pair_structure": bad_pair_structure,
        "malformed_stored_benign_trajectories": malformed,
        "total_stored_assistant_turns": total_stored_assistant_turns,
        "dual_rubric_judge_requests": 2 * total_stored_assistant_turns,
        "max_observable_transcript_chars": max_observable_transcript_chars,
        "independent_models": dict(independent_models),
        "independent_status": dict(independent_status),
        "target_models": dict(target_models),
        "source_lineage_checked": source_lineage_checked,
        "source_lineage_missing": source_lineage_missing,
        "source_lineage_mismatch": source_lineage_mismatch,
        "frozen_malicious_reference_checked": frozen_malicious_reference_checked,
        "frozen_malicious_reference_mismatch": frozen_malicious_reference_mismatch,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    known_models = [m for m in independent_models if m != "UNKNOWN"]
    if len(known_models) != 1:
        errors.append(
            f"expected exactly one known independent-validation model, got {known_models}"
        )
    if independent_models.get("UNKNOWN", 0):
        errors.append(
            f"{independent_models['UNKNOWN']} benign twins lack independent-model provenance"
        )
    known_target_models = [m for m in target_models if m != "UNKNOWN"]
    if len(known_target_models) != 1:
        errors.append(
            f"expected exactly one original Dataset A target model, got {known_target_models}"
        )
    if target_models.get("UNKNOWN", 0):
        errors.append(
            f"{target_models['UNKNOWN']} twin pairs lack target-model provenance"
        )
    if dict(target_models) != {args.expected_target_model: len(twins)}:
        errors.append(
            f"unexpected target-model distribution {dict(target_models)}; "
            f"expected only {args.expected_target_model!r}"
        )
    if dict(independent_models) != {args.expected_independent_model: len(twins)}:
        errors.append(
            f"unexpected independent-model distribution {dict(independent_models)}; "
            f"expected only {args.expected_independent_model!r}"
        )

    if args.expected_benign_twins is not None and len(twins) != args.expected_benign_twins:
        errors.append(
            f"expected {args.expected_benign_twins} original benign twins, found {len(twins)}"
        )
    if (
        args.expected_validated_malicious is not None
        and len(malicious_validated) != args.expected_validated_malicious
    ):
        errors.append(
            f"expected {args.expected_validated_malicious} validated interactive "
            f"malicious records, found {len(malicious_validated)}"
        )
    if (
        args.expected_final_malicious_candidates is not None
        and len(final_malicious_candidates) != args.expected_final_malicious_candidates
    ):
        errors.append(
            f"expected {args.expected_final_malicious_candidates} final malicious "
            f"candidates, found {len(final_malicious_candidates)}"
        )

    if not twins:
        errors.append("no original interactive benign twins found")

    if errors:
        print("TWIN RESTORATION INPUT AUDIT FAILED", file=sys.stderr)
        for error in errors[:100]:
            print(f"ERROR: {error}", file=sys.stderr)
        if len(errors) > 100:
            print(f"... {len(errors)-100} additional errors", file=sys.stderr)
        sys.exit(2)

    print("TWIN RESTORATION INPUT AUDIT PASSED")


if __name__ == "__main__":
    main()
