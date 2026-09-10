#!/usr/bin/env python3
"""Create the precommitted held-out judge evaluation sample from B1 rollouts only.

The sampler is deliberately score-blind. It rejects any input containing B2/B4
judgments and excludes the 20-record design/calibration set. Sampling allocation is
stratified by pair_hardness and authoring label; scenario_family is treated as a
grouping unit so, when possible, at most one trajectory per family is selected.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict

from frontier_common import load_jsonl

DEFAULT_SEED = 20260910
DEFAULT_SAMPLE_SIZE = 100


def load_excluded_ids(path):
    if not path:
        return set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "records" in data:
        rows = data["records"]
    elif isinstance(data, list):
        rows = data
    else:
        raise RuntimeError("exclude manifest must be a list or contain records[]")
    return {str(x["conversation_id"]) for x in rows}


def assert_score_blind_b1(record):
    cid = str(record.get("conversation_id", ""))
    if record.get("rollout_status") != "complete":
        raise RuntimeError(f"{cid}: holdout source must be complete B1 rollout")
    if record.get("frontier_behavioral_validation") is not None:
        raise RuntimeError(f"{cid}: judged B2/B4 file supplied; use score-blind B1 input")
    for turn in record.get("turns", []):
        if "behavioral_judgment" in turn:
            raise RuntimeError(f"{cid}: behavioral judgment present; use B1 input")


def stratum(record):
    intended = record.get("intended_structure", {}) or {}
    hardness = str(intended.get("pair_hardness") or "unknown")
    label = int(record.get("label", -1))
    return hardness, label


def largest_remainder_alloc(counts, n):
    total = sum(counts.values())
    if n > total:
        raise RuntimeError(f"sample size {n} exceeds eligible population {total}")
    quotas = {k: n * v / total for k, v in counts.items()}
    alloc = {k: min(counts[k], int(math.floor(q))) for k, q in quotas.items()}
    remaining = n - sum(alloc.values())
    order = sorted(
        counts,
        key=lambda k: (quotas[k] - math.floor(quotas[k]), counts[k], str(k)),
        reverse=True,
    )
    while remaining:
        progressed = False
        for k in order:
            if alloc[k] < counts[k]:
                alloc[k] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("unable to allocate requested sample")
    return alloc


def choose_with_family_diversity(records, n, rng):
    by_family = defaultdict(list)
    for r in records:
        fam = str((r.get("metadata", {}) or {}).get("scenario_family") or r["conversation_id"])
        by_family[fam].append(r)
    families = list(by_family)
    rng.shuffle(families)
    selected = []
    leftovers = []
    for fam in families:
        rows = list(by_family[fam])
        rng.shuffle(rows)
        selected.append(rows[0])
        leftovers.extend(rows[1:])
        if len(selected) == n:
            return selected
    rng.shuffle(leftovers)
    selected.extend(leftovers[: n - len(selected)])
    if len(selected) != n:
        raise RuntimeError("insufficient records after family-aware selection")
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="B1 rollout JSONL only; no judge fields")
    p.add_argument("--output-manifest", required=True)
    p.add_argument("--output-blind-packet", required=True)
    p.add_argument("--output-annotation-template", required=True)
    p.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--exclude-manifest",
        default="naacl/frontier_judge_smoke_adjudication_v4.json",
        help="development/calibration IDs that must never enter paper holdout",
    )
    args = p.parse_args()

    records = load_jsonl(args.input)
    excluded = load_excluded_ids(args.exclude_manifest)
    eligible = []
    for r in records:
        assert_score_blind_b1(r)
        if str(r.get("conversation_id", "")) not in excluded:
            eligible.append(r)
    if not eligible:
        raise RuntimeError("no eligible score-blind B1 records")

    counts = Counter(stratum(r) for r in eligible)
    alloc = largest_remainder_alloc(counts, args.sample_size)
    rng = random.Random(args.seed)
    by_stratum = defaultdict(list)
    for r in eligible:
        by_stratum[stratum(r)].append(r)

    chosen = []
    for key in sorted(by_stratum, key=str):
        rows = by_stratum[key]
        local_rng = random.Random(f"{args.seed}:{key[0]}:{key[1]}")
        chosen.extend(choose_with_family_diversity(rows, alloc[key], local_rng))
    rng.shuffle(chosen)

    # Hard guarantee: excluded development IDs never appear.
    chosen_ids = [str(r["conversation_id"]) for r in chosen]
    leaked = sorted(set(chosen_ids) & excluded)
    if leaked:
        raise RuntimeError(f"design/calibration contamination in holdout: {leaked}")
    if len(chosen_ids) != len(set(chosen_ids)):
        raise RuntimeError("duplicate conversation IDs in holdout")

    manifest_rows = []
    blind_rows = []
    annotation_rows = []
    for i, r in enumerate(chosen, 1):
        item_id = f"H{i:03d}"
        meta = r.get("metadata", {}) or {}
        intended = r.get("intended_structure", {}) or {}
        manifest_rows.append({
            "item_id": item_id,
            "conversation_id": r["conversation_id"],
            "scenario_family": meta.get("scenario_family"),
            "mechanism_family": meta.get("mechanism_family"),
            "pair_hardness": intended.get("pair_hardness"),
            "authoring_label": r.get("label"),
            "generator": meta.get("generator"),
        })
        blind_rows.append({
            "item_id": item_id,
            "turns": [
                {"role": t.get("role"), "text": t.get("text")}
                for t in r.get("turns", [])
            ],
        })
        annotation_rows.append({
            "item_id": item_id,
            "realized_behavior": None,
            "adjudicator_confidence": None,
            "rationale": "",
        })

    manifest = {
        "protocol": "frontier_judge_holdout_v1",
        "sample_seed": args.seed,
        "sample_size": args.sample_size,
        "source_requirement": "score_blind_B1_rollout_only",
        "excluded_design_manifest": args.exclude_manifest,
        "stratification": "proportional_pair_hardness_x_authoring_label",
        "scenario_family_policy": "at_most_one_per_family_when_possible_within_stratum",
        "paper_metric_eligible": True,
        "records": manifest_rows,
    }
    with open(args.output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(args.output_blind_packet, "w", encoding="utf-8") as f:
        for row in blind_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.output_annotation_template, "w", encoding="utf-8") as f:
        for row in annotation_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    family_count = len({x["scenario_family"] for x in manifest_rows})
    print("=== Held-out judge sample created ===")
    print(f"Eligible population: {len(eligible)}")
    print(f"Excluded design/calibration IDs: {len(excluded)}")
    print(f"Sample: {len(chosen)} seed={args.seed}")
    print(f"Distinct scenario families: {family_count}")
    print(f"Stratum counts: {dict(Counter((x['pair_hardness'], x['authoring_label']) for x in manifest_rows))}")
    print("Sampling used no B2/B4 judgments or judge scores.")
    print(f"Manifest: {args.output_manifest}")
    print(f"Blind packet: {args.output_blind_packet}")
    print(f"Annotation template: {args.output_annotation_template}")


if __name__ == "__main__":
    main()
