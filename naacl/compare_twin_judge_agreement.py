#!/usr/bin/env python3
"""Compare 7B and 24B judgments on the same stored Dataset A twin responses."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from typing import Dict, List


def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
    return out


def validation_of(record: Dict) -> Dict:
    return (
        record.get("validation", {})
        or record.get("stored_target_validation", {})
        or {}
    )


def kappa(labels_a: List[str], labels_b: List[str]) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        return float("nan")
    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    classes = set(ca) | set(cb)
    expected = sum((ca[c] / n) * (cb[c] / n) for c in classes)
    if math.isclose(1.0 - expected, 0.0):
        return 1.0 if math.isclose(observed, 1.0) else float("nan")
    return (observed - expected) / (1.0 - expected)


def pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-24b", required=True)
    parser.add_argument("--comparison-7b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ref_records = load_jsonl(args.reference_24b)
    cmp_records = load_jsonl(args.comparison_7b)

    ref = {
        str(r.get("conversation_id")): r
        for r in ref_records
        if r.get("stored_target_validation")
    }
    cmp = {str(r.get("conversation_id")): r for r in cmp_records}

    ids = sorted(set(cmp))
    missing = [cid for cid in ids if cid not in ref]
    if missing:
        raise RuntimeError(
            f"{len(missing)} sampled records missing from 24B reference: {missing[:10]}"
        )

    status24, status7 = [], []
    binary24, binary7 = [], []
    scores24, scores7 = [], []
    confusion = Counter()
    disagreements = []

    for cid in ids:
        r24, r7 = ref[cid], cmp[cid]
        h24 = (r24.get("stored_target_validation", {}) or {}).get(
            "source_turn_text_sha256"
        )
        h7 = r7.get("source_turn_text_sha256")
        if h24 != h7:
            raise RuntimeError(f"{cid}: stored-response hash differs between judges")

        v24 = validation_of(r24)
        v7 = validation_of(r7)
        s24 = str(v24.get("status", "missing"))
        s7 = str(v7.get("status", "missing"))
        b24 = "safe" if s24 == "validated" else "not_safe"
        b7 = "safe" if s7 == "validated" else "not_safe"

        status24.append(s24)
        status7.append(s7)
        binary24.append(b24)
        binary7.append(b7)
        confusion[(s24, s7)] += 1

        x = float(v24.get("max_unsafe_score", 0.0))
        y = float(v7.get("max_unsafe_score", 0.0))
        scores24.append(x)
        scores7.append(y)

        if s24 != s7:
            disagreements.append({
                "conversation_id": cid,
                "pair_id": r7.get("pair_id"),
                "status_24b": s24,
                "status_7b": s7,
                "max_unsafe_24b": x,
                "max_unsafe_7b": y,
            })

    n = len(ids)
    reference_models = sorted({
        str((validation_of(r) or {}).get("judge_model", "unknown"))
        for r in ref_records
        if r.get("stored_target_validation")
    })
    comparison_models = sorted({
        str(r.get("judge_model", "unknown")) for r in cmp_records
    })
    comparison_revisions = sorted({
        str(r.get("judge_model_revision", "unknown")) for r in cmp_records
    })

    result = {
        "n": n,
        "reference_24b_models": reference_models,
        "comparison_7b_models": comparison_models,
        "comparison_7b_revisions": comparison_revisions,
        "status_agreement": sum(a == b for a, b in zip(status24, status7)) / n,
        "status_cohen_kappa": kappa(status24, status7),
        "binary_safe_agreement": sum(a == b for a, b in zip(binary24, binary7)) / n,
        "binary_safe_cohen_kappa": kappa(binary24, binary7),
        "max_unsafe_score_pearson": pearson(scores24, scores7),
        "max_unsafe_score_mean_abs_diff": (
            sum(abs(a - b) for a, b in zip(scores24, scores7)) / n
        ),
        "status_counts_24b": dict(Counter(status24)),
        "status_counts_7b": dict(Counter(status7)),
        "confusion_24b_by_7b": {
            f"{a}|{b}": count for (a, b), count in sorted(confusion.items())
        },
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
