#!/usr/bin/env python3
"""CPU-only preflight for Dataset A original-twin restoration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from typing import Dict, List


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
    for turn in turns:
        role = str(turn.get("role", "")).lower()
        text = str(turn.get("text", "")).strip().lower()
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
    parser.add_argument("--expected-benign-twins", type=int, default=545)
    parser.add_argument("--expected-validated-malicious", type=int, default=545)
    parser.add_argument("--expected-final-malicious-candidates", type=int, default=526)
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
    malformed = 0
    bad_pair_structure = 0

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

        try:
            validate_stored_trajectory(twin)
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

    summary = {
        "records": len(records),
        "original_benign_twins": len(twins),
        "validated_interactive_malicious": len(malicious_validated),
        "final_malicious_candidates": len(final_malicious_candidates),
        "pair_groups": len(pairs),
        "bad_pair_structure": bad_pair_structure,
        "malformed_stored_benign_trajectories": malformed,
        "independent_models": dict(independent_models),
        "independent_status": dict(independent_status),
        "source_lineage_checked": source_lineage_checked,
        "source_lineage_missing": source_lineage_missing,
        "source_lineage_mismatch": source_lineage_mismatch,
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
