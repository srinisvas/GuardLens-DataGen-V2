#!/usr/bin/env python3
"""CPU-only audit for a length-aligned common-horizon view of restored Dataset A.

This script never writes or modifies dataset records. For each exact restored
malicious/benign pair, it computes the largest shared user-turn horizon and asks
whether the repaired malicious baseline is already unsafe by that horizon.

Purpose:
  * quantify the turn-count shortcut in the restored natural twins;
  * determine whether a no-new-generation paired common-horizon view is viable;
  * report how much fresh counterfactual evidence remains inside that horizon.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return out


def user_turns(record: Dict) -> List[Dict]:
    return [
        turn
        for turn in record.get("turns", [])
        if str(turn.get("role", "")).lower() == "user"
    ]


def prefix_through_user_count(record: Dict, count: int) -> List[Dict]:
    if count <= 0:
        return []
    kept = []
    users = 0
    reached = False
    for turn in record.get("turns", []):
        role = str(turn.get("role", "")).lower()
        if role == "user":
            if reached:
                break
            users += 1
            kept.append(turn)
            if users >= count:
                reached = True
        else:
            kept.append(turn)
    if reached and kept:
        # Preserve the assistant response immediately following the final kept
        # user turn, if one exists.
        original = record.get("turns", [])
        last_tid = kept[-1].get("turn_id")
        for idx, turn in enumerate(original):
            if turn.get("turn_id") == last_tid:
                if idx + 1 < len(original):
                    nxt = original[idx + 1]
                    if str(nxt.get("role", "")).lower() == "assistant":
                        kept.append(nxt)
                break
    return kept


def total_user_chars(turns: List[Dict]) -> int:
    return sum(
        len(str(turn.get("text", "")))
        for turn in turns
        if str(turn.get("role", "")).lower() == "user"
    )


def summary(values: List[float]) -> Dict:
    if not values:
        return {"n": 0}
    xs = sorted(float(x) for x in values)
    def pct(q: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = (len(xs) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        frac = pos - lo
        return xs[lo] * (1 - frac) + xs[hi] * frac
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "min": xs[0],
        "max": xs[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    groups = defaultdict(list)
    for record in records:
        pair_id = str(record.get("pair_id", ""))
        if pair_id:
            groups[pair_id].append(record)

    errors = []
    rows: List[Tuple[Dict, Dict]] = []
    for pair_id, items in groups.items():
        mal = [r for r in items if r.get("label") == 1]
        ben = [r for r in items if r.get("label") == 0]
        if len(mal) != 1 or len(ben) != 1:
            errors.append(
                f"{pair_id}: expected one malicious + one benign, got "
                f"{len(mal)} + {len(ben)}"
            )
            continue
        rows.append((mal[0], ben[0]))

    if errors:
        raise RuntimeError("; ".join(errors[:10]))
    if not rows:
        raise RuntimeError("no paired records found")

    full_turn_rule_tp = 0
    full_turn_rule_tn = 0
    common_horizon_positive = 0
    common_horizon_negative_or_unknown = 0
    common_horizon_supported_evidence = 0
    common_horizon_no_supported_evidence = 0
    malicious_shorter_than_benign = 0
    exact_equal_user_counts = 0

    full_mal_users = []
    full_ben_users = []
    common_users = []
    common_mal_chars = []
    common_ben_chars = []
    anchor_ordinals = []
    evidence_within_counts = Counter()

    for mal, ben in rows:
        mal_users = user_turns(mal)
        ben_users = user_turns(ben)
        nm = len(mal_users)
        nb = len(ben_users)
        full_mal_users.append(nm)
        full_ben_users.append(nb)

        if nm > 10:
            full_turn_rule_tp += 1
        if nb <= 10:
            full_turn_rule_tn += 1

        if nm < nb:
            malicious_shorter_than_benign += 1
        if nm == nb:
            exact_equal_user_counts += 1

        horizon = min(nm, nb)
        common_users.append(horizon)
        mal_prefix = prefix_through_user_count(mal, horizon)
        ben_prefix = prefix_through_user_count(ben, horizon)
        common_mal_chars.append(total_user_chars(mal_prefix))
        common_ben_chars.append(total_user_chars(ben_prefix))

        if horizon <= 0:
            common_horizon_negative_or_unknown += 1
            common_horizon_no_supported_evidence += 1
            continue

        cutoff_turn_id = mal_users[horizon - 1].get("turn_id")
        if isinstance(cutoff_turn_id, bool) or not isinstance(cutoff_turn_id, int):
            raise RuntimeError(
                f"{mal.get('conversation_id')}: invalid common-horizon user turn id"
            )

        analysis = mal.get("evidence_analysis", {}) or {}
        anchor = (
            (analysis.get("baseline", {}) or {}).get("earliest_unsafe_turn")
            if analysis
            else None
        )
        if anchor is None:
            anchor = analysis.get("fresh_anchor_turn_id")
        if isinstance(anchor, int) and not isinstance(anchor, bool):
            anchor_ordinals.append(anchor)
            if anchor <= cutoff_turn_id:
                common_horizon_positive += 1
            else:
                common_horizon_negative_or_unknown += 1
        else:
            common_horizon_negative_or_unknown += 1

        evidence_ids = [
            x
            for x in mal.get("evidence_turn_ids", []) or []
            if isinstance(x, int) and not isinstance(x, bool)
        ]
        inside = sum(x <= cutoff_turn_id for x in evidence_ids)
        evidence_within_counts[inside] += 1
        if inside:
            common_horizon_supported_evidence += 1
        else:
            common_horizon_no_supported_evidence += 1

    n = len(rows)
    # This deliberately mirrors the trivial shortcut exposed by the observed
    # histograms: predict malicious iff full-conversation user-turn count > 10.
    shortcut_tp = full_turn_rule_tp
    shortcut_fn = n - shortcut_tp
    shortcut_tn = full_turn_rule_tn
    shortcut_fp = n - shortcut_tn
    tpr = shortcut_tp / n
    tnr = shortcut_tn / n
    balanced_accuracy = 0.5 * (tpr + tnr)

    result = {
        "pairs": n,
        "full_length_turn_count_shortcut": {
            "rule": "predict malicious iff n_user_turns > 10",
            "tp": shortcut_tp,
            "fn": shortcut_fn,
            "tn": shortcut_tn,
            "fp": shortcut_fp,
            "tpr": tpr,
            "tnr": tnr,
            "balanced_accuracy": balanced_accuracy,
        },
        "full_user_turns": {
            "malicious": summary(full_mal_users),
            "benign": summary(full_ben_users),
        },
        "common_horizon": {
            "definition": "min(malicious_user_turns, benign_user_turns) within each exact pair",
            "user_turns": summary(common_users),
            "malicious_shorter_than_benign_pairs": malicious_shorter_than_benign,
            "exact_equal_user_count_pairs": exact_equal_user_counts,
            "fresh_baseline_unsafe_by_horizon": common_horizon_positive,
            "fresh_baseline_not_unsafe_by_horizon_or_missing": common_horizon_negative_or_unknown,
            "fresh_baseline_unsafe_retention": common_horizon_positive / n,
            "pairs_with_supported_evidence_turn_by_horizon": common_horizon_supported_evidence,
            "pairs_without_supported_evidence_turn_by_horizon": common_horizon_no_supported_evidence,
            "supported_evidence_turn_count_within_horizon": dict(evidence_within_counts),
            "projected_total_user_chars": {
                "malicious": summary(common_mal_chars),
                "benign": summary(common_ben_chars),
            },
        },
        "writes_dataset_records": False,
        "generates_new_text": False,
    }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
