"""
merge_validations.py

Merges validation results from multiple models into a single dataset
with per-record transferability labels.

Input sources:
  1. Interactive generation output (contains llama_validation from generation)
  2. Qwen-7B validation output (from launch_val.slurm)

Output: merged dataset with:
  - llama_validation: from generation (target model outcome)
  - qwen7b_validation: from post-generation validation
  - success_targets: ["llama8b", "qwen7b"] etc.
  - transfer_tier: transfer_success | target_only | no_jailbreak
  - Updated supervision_tier based on cross-model evidence

Usage:
    python merge_validations.py \\
        --interactive output/interactive_validated_by_qwen7b.jsonl \\
        --output output/merged_final.jsonl

    # If you also have Qwen-14B validation:
    python merge_validations.py \\
        --interactive output/interactive_validated_by_qwen7b.jsonl \\
        --qwen14b output/qwen14b_validated.jsonl \\
        --output output/merged_final.jsonl
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_semantic_datasetv11 as gen


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def merge_dataset(
    interactive_records: List[Dict],
    qwen14b_records: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Merge validation results into unified records.

    The interactive_records already contain:
      - llama_validation (from generation)
      - causal_validation (from Qwen-7B validation pass via launch_val.slurm)

    If qwen14b_records provided, merge those results too.
    """

    # Index qwen14b by conversation_id if provided
    qwen14b_map = {}
    if qwen14b_records:
        for r in qwen14b_records:
            cid = r.get("conversation_id", "")
            qwen14b_map[cid] = r

    merged = []
    for record in interactive_records:
        cid = record.get("conversation_id", "")

        # Extract validation results
        llama_val = record.get("llama_validation", {})
        qwen7b_val = record.get("causal_validation", {})

        llama_jailbreak = llama_val.get("jailbreak_detected", False)
        qwen7b_jailbreak = qwen7b_val.get("jailbreak_detected", False)

        # Build success_targets list
        success_targets = []
        if llama_jailbreak:
            success_targets.append("llama8b")
        if qwen7b_jailbreak:
            success_targets.append("qwen7b")

        # Check qwen14b if available
        qwen14b_jailbreak = False
        if cid in qwen14b_map:
            q14_val = qwen14b_map[cid].get("causal_validation", {})
            qwen14b_jailbreak = q14_val.get("jailbreak_detected", False)
            record["qwen14b_validation"] = q14_val
            if qwen14b_jailbreak:
                success_targets.append("qwen14b")

        # Rename causal_validation to qwen7b_validation for clarity
        if qwen7b_val:
            record["qwen7b_validation"] = qwen7b_val

        record["success_targets"] = success_targets

        # Assign transfer tier
        if record.get("label") != 1:
            # Benign records
            record["transfer_tier"] = "benign"
            # Benign twins that were not validated by Qwen-7B get lower weight
            if not qwen7b_val.get("validated"):
                record["validation_status"] = "unvalidated"
                record["loss_weight"] = 0.30  # Lower confidence than validated benign
            elif qwen7b_val.get("jailbreak_detected"):
                # Benign record that triggered jailbreak = false alarm
                record["validation_status"] = "rejected"
                record["training_eligible"] = False
        elif len(success_targets) >= 2:
            record["transfer_tier"] = "transfer_success"
        elif len(success_targets) == 1:
            if "llama8b" in success_targets:
                record["transfer_tier"] = "target_only"
            else:
                record["transfer_tier"] = "cross_only"
        else:
            record["transfer_tier"] = "no_jailbreak"

        # Update validation_status based on cross-model evidence
        if record.get("label") == 1:
            if len(success_targets) >= 2:
                record["validation_status"] = "validated"
            elif len(success_targets) == 1:
                record["validation_status"] = "validated"
            else:
                record["validation_status"] = "ambiguous"

        # Update pivot info: prefer qwen7b pivot if available (independent signal)
        if qwen7b_jailbreak and qwen7b_val.get("pivot_turn_id") is not None:
            record["pivot_turn_id"] = qwen7b_val["pivot_turn_id"]
            if qwen7b_val.get("pivot_kind"):
                record["pivot_kind"] = qwen7b_val["pivot_kind"]
        elif llama_jailbreak and llama_val.get("pivot_turn_id") is not None:
            record["pivot_turn_id"] = llama_val["pivot_turn_id"]
            if llama_val.get("pivot_kind"):
                record["pivot_kind"] = llama_val["pivot_kind"]

        # Update judge confidence (average across available validators)
        confidences = []
        if llama_val.get("avg_judge_confidence"):
            confidences.append(llama_val["avg_judge_confidence"])
        if qwen7b_val.get("avg_judge_confidence"):
            confidences.append(qwen7b_val["avg_judge_confidence"])
        if confidences:
            record["judge_confidence"] = round(sum(confidences) / len(confidences), 3)

        # Reassign supervision tiers with cross-model bonus
        sample = gen._dict_to_sample(record)
        sample = gen.assign_sample_tier(sample)

        # Boost tier for transfer_success records
        if record["transfer_tier"] == "transfer_success":
            if sample.supervision_tier == "construction":
                sample.supervision_tier = "llm_confirmed"
                sample.loss_weight = gen.get_loss_weight("llm_confirmed")

        record["supervision_tier"] = sample.supervision_tier
        record["loss_weight"] = sample.loss_weight

        # Compute training_eligible from the updated record fields directly
        # (not from sample, which may have stale validation_status)
        record["training_eligible"] = (
            not record.get("is_external_test", False)
            and record.get("validation_status") in {"validated", "unvalidated"}
            and record.get("supervision_tier") != "ignore"
        )

        merged.append(record)

    return merged


def print_merge_stats(records: List[Dict]):
    """Print comprehensive merge statistics."""
    total = len(records)
    mal = [r for r in records if r.get("label") == 1]
    ben = [r for r in records if r.get("label") == 0]

    print(f"\n{'='*60}")
    print(f"  Merged Dataset Statistics")
    print(f"{'='*60}")
    print(f"  Total: {total} ({len(mal)} malicious, {len(ben)} benign)")

    # Transfer tiers
    tiers = Counter(r.get("transfer_tier", "unknown") for r in records)
    print(f"\n  Transfer tiers:")
    for tier, count in tiers.most_common():
        pct = count / max(total, 1) * 100
        print(f"    {tier}: {count} ({pct:.1f}%)")

    # Success targets for malicious records
    print(f"\n  Jailbreak success (malicious only):")
    llama_jb = sum(1 for r in mal if "llama8b" in r.get("success_targets", []))
    qwen7b_jb = sum(1 for r in mal if "qwen7b" in r.get("success_targets", []))
    qwen14b_jb = sum(1 for r in mal if "qwen14b" in r.get("success_targets", []))
    both_jb = sum(1 for r in mal if len(r.get("success_targets", [])) >= 2)
    any_jb = sum(1 for r in mal if len(r.get("success_targets", [])) >= 1)

    print(f"    Llama-8B (target):  {llama_jb}/{len(mal)} ({llama_jb/max(len(mal),1)*100:.1f}%)")
    print(f"    Qwen-7B (transfer): {qwen7b_jb}/{len(mal)} ({qwen7b_jb/max(len(mal),1)*100:.1f}%)")
    if qwen14b_jb:
        print(f"    Qwen-14B:           {qwen14b_jb}/{len(mal)} ({qwen14b_jb/max(len(mal),1)*100:.1f}%)")
    print(f"    Both (transfer):    {both_jb}/{len(mal)} ({both_jb/max(len(mal),1)*100:.1f}%)")
    print(f"    Any validator:      {any_jb}/{len(mal)} ({any_jb/max(len(mal),1)*100:.1f}%)")

    # False alarm check
    ben_fa = sum(1 for r in ben if r.get("validation_status") == "rejected")
    print(f"\n  False alarms (benign):")
    print(f"    Rejected: {ben_fa}/{len(ben)} ({ben_fa/max(len(ben),1)*100:.1f}%)")

    # Supervision tiers
    stiers = Counter(r.get("supervision_tier", "unknown") for r in records)
    print(f"\n  Supervision tiers:")
    for tier, count in stiers.most_common():
        print(f"    {tier}: {count}")

    # Pivot kinds
    pivots = Counter(r.get("pivot_kind", "none") for r in mal)
    print(f"\n  Pivot kinds (malicious):")
    for pk, count in pivots.most_common():
        print(f"    {pk}: {count}")

    # Training eligible
    eligible = sum(1 for r in records if r.get("training_eligible", True))
    print(f"\n  Training eligible: {eligible}/{total}")

    # Strategy breakdown (for interactive records)
    strategies = Counter(
        r.get("metadata", r.get("attack_metadata", {})).get("strategy", "unknown")
        for r in mal
    )
    if any(s != "unknown" for s in strategies):
        print(f"\n  Attack strategies:")
        for s, count in strategies.most_common():
            jb_count = sum(
                1 for r in mal
                if r.get("metadata", r.get("attack_metadata", {})).get("strategy") == s
                and len(r.get("success_targets", [])) >= 1
            )
            print(f"    {s}: {count} total, {jb_count} jailbreaks ({jb_count/max(count,1)*100:.0f}%)")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multi-model validation results"
    )
    parser.add_argument("--interactive", type=str, required=True,
                        help="Interactive generation output validated by Qwen-7B")
    parser.add_argument("--qwen14b", type=str, default=None,
                        help="Optional Qwen-14B validation output")
    parser.add_argument("--output", type=str, required=True,
                        help="Merged output JSONL")

    args = parser.parse_args()

    # Load
    interactive = load_jsonl(args.interactive)
    print(f"Loaded {len(interactive)} interactive records")

    qwen14b = None
    if args.qwen14b and os.path.exists(args.qwen14b):
        qwen14b = load_jsonl(args.qwen14b)
        print(f"Loaded {len(qwen14b)} Qwen-14B records")

    # Merge
    merged = merge_dataset(interactive, qwen14b)

    # Write
    gen.write_jsonl(merged, args.output)
    print(f"Wrote {len(merged)} merged records to {args.output}")

    # Stats
    print_merge_stats(merged)


if __name__ == "__main__":
    main()
