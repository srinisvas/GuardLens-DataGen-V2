#!/usr/bin/env python3
"""Compare turn-count shortcut strength across GuardLens dataset generations.

CPU-only diagnostic. Reads JSONL files and reports:
  * class counts
  * user-turn histograms by label
  * mean user-turn count by label
  * univariate ROC AUC using user-turn count
  * best threshold balanced accuracy for rule n_user_turns > threshold
  * fixed threshold >10 for continuity with restored Dataset A discussion

Supports constructing the current A+B population by combining a restored
Dataset A candidate with only the frontier-authored rows extracted from a
previous canonical A+B artifact. No records are modified or written.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, Iterable, List, Tuple


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def label_of(record: Dict) -> int:
    value = record.get("detection_label", record.get("label"))
    if value not in (0, 1):
        raise RuntimeError(
            f"{record.get('conversation_id', '<missing>')}: invalid binary label {value!r}"
        )
    return int(value)


def n_user_turns(record: Dict) -> int:
    explicit = record.get("user_turn_count")
    if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit >= 0:
        # Prefer physical turns if assistant responses have since been attached.
        physical = sum(
            str(turn.get("role", "")).lower() == "user"
            for turn in record.get("turns", [])
        )
        return physical if physical else explicit
    return sum(
        str(turn.get("role", "")).lower() == "user"
        for turn in record.get("turns", [])
    )


def rank_auc(negative: List[int], positive: List[int]) -> float:
    """Mann-Whitney ROC AUC with 0.5 credit for ties."""
    if not negative or not positive:
        return float("nan")
    wins = 0.0
    neg_counts = Counter(negative)
    for p in positive:
        wins += sum(n for x, n in neg_counts.items() if p > x)
        wins += 0.5 * neg_counts.get(p, 0)
    return wins / (len(negative) * len(positive))


def threshold_metrics(negative: List[int], positive: List[int], threshold: int) -> Dict:
    tp = sum(x > threshold for x in positive)
    fn = len(positive) - tp
    fp = sum(x > threshold for x in negative)
    tn = len(negative) - fp
    tpr = tp / len(positive) if positive else float("nan")
    tnr = tn / len(negative) if negative else float("nan")
    bal = 0.5 * (tpr + tnr)
    return {
        "threshold": threshold,
        "rule": f"predict malicious iff n_user_turns > {threshold}",
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "tpr": tpr,
        "tnr": tnr,
        "balanced_accuracy": bal,
    }


def summarize(name: str, records: List[Dict]) -> Dict:
    by_label = {0: [], 1: []}
    for record in records:
        by_label[label_of(record)].append(n_user_turns(record))

    if not by_label[0] or not by_label[1]:
        raise RuntimeError(
            f"{name}: both labels are required, got "
            f"0={len(by_label[0])} 1={len(by_label[1])}"
        )

    thresholds = sorted(set(by_label[0] + by_label[1]))
    candidates = [
        threshold_metrics(by_label[0], by_label[1], threshold)
        for threshold in thresholds
    ]
    best = max(
        candidates,
        key=lambda x: (x["balanced_accuracy"], -x["threshold"]),
    )

    return {
        "name": name,
        "records": len(records),
        "labels": {
            "0": len(by_label[0]),
            "1": len(by_label[1]),
        },
        "user_turns": {
            "0": {
                "mean": sum(by_label[0]) / len(by_label[0]),
                "min": min(by_label[0]),
                "max": max(by_label[0]),
                "histogram": dict(sorted(Counter(by_label[0]).items())),
            },
            "1": {
                "mean": sum(by_label[1]) / len(by_label[1]),
                "min": min(by_label[1]),
                "max": max(by_label[1]),
                "histogram": dict(sorted(Counter(by_label[1]).items())),
            },
        },
        "turn_count_auc": rank_auc(by_label[0], by_label[1]),
        "best_turn_count_threshold": best,
        "fixed_threshold_gt_10": threshold_metrics(
            by_label[0], by_label[1], 10
        ),
    }


def parse_named_dataset(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("expected nonempty NAME=PATH")
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_named_dataset,
        default=[],
        help="Audit one dataset as NAME=PATH. May be repeated.",
    )
    parser.add_argument(
        "--restored-a",
        default=None,
        help="Restored Dataset A candidate for constructing current A+B.",
    )
    parser.add_argument(
        "--previous-ab",
        default=None,
        help=(
            "Previous canonical A+B artifact. Only corpus_source="
            "frontier_authored_v3 rows are extracted as frozen Dataset B."
        ),
    )
    parser.add_argument(
        "--current-ab-name",
        default="current_restored_A_plus_frozen_B",
    )
    args = parser.parse_args()

    results = []

    for name, path in args.dataset:
        results.append(summarize(name, load_jsonl(path)))

    if bool(args.restored_a) != bool(args.previous_ab):
        raise RuntimeError(
            "--restored-a and --previous-ab must be supplied together"
        )

    if args.restored_a and args.previous_ab:
        restored_a = load_jsonl(args.restored_a)
        previous_ab = load_jsonl(args.previous_ab)
        frontier = [
            record
            for record in previous_ab
            if str(record.get("corpus_source", "")) == "frontier_authored_v3"
        ]
        if not frontier:
            raise RuntimeError(
                "--previous-ab contains no corpus_source=frontier_authored_v3 rows"
            )

        frontier_summary = summarize("frozen_dataset_B", frontier)
        current_ab_summary = summarize(
            args.current_ab_name,
            restored_a + frontier,
        )
        results.append(frontier_summary)
        results.append(current_ab_summary)

    print(json.dumps({"datasets": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
