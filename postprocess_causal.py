"""
postprocess_causal.py

Post-processing cleanup for the causal analysis output.

1. Relabel causal BENIGN_CONTEXT/DECOY spans → CONTEXT_BRIDGE
2. Add distributed_context tier for distributed-causal records
3. Validate supervision tier ladder consistency
4. Print final dataset statistics for paper

Usage:
    python postprocess_causal.py \
        --input causal_analyzed_all_final.jsonl \
        --output final_dataset_malicious.jsonl
"""

import argparse
import json
import os
import sys
from collections import Counter
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_semantic_datasetv11 as gen


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# =========================================================
# Step 1: Relabel causal BENIGN_CONTEXT/DECOY spans
# =========================================================

def relabel_causal_benign_spans(records: List[Dict]) -> List[Dict]:
    """
    If a span labeled BENIGN_CONTEXT or DECOY was proven causal
    by counterfactual analysis, relabel it as CONTEXT_BRIDGE.
    Store original label for auditability.
    """
    relabeled = 0
    for record in records:
        for turn in record.get("turns", []):
            for span in turn.get("span_annotations", []):
                if (span.get("label") in {"BENIGN_CONTEXT", "DECOY"}
                        and span.get("causal_type") == "causal"
                        and span.get("counterfactual_delta", 0.0) >= 0.15):
                    span["original_label"] = span["label"]
                    span["label"] = "CONTEXT_BRIDGE"
                    relabeled += 1

    print(f"  Relabeled {relabeled} causal BENIGN_CONTEXT/DECOY spans -> CONTEXT_BRIDGE")
    return records


# =========================================================
# Step 2: Add distributed_context tier
# =========================================================

def assign_distributed_context_tier(records: List[Dict]) -> List[Dict]:
    """
    For records where:
      - pivot-turn CF = distributed_or_unclear
      - transfer_tier in {transfer_success, target_only, cross_only}

    Set supervision_tier to llm_confirmed (0.60 weight).
    These are useful for detection training but not for
    token-level attribution claims.
    """
    upgraded = 0
    for record in records:
        if record.get("label") != 1:
            continue

        pivot_cf = record.get("causal_analysis", {}).get("pivot_turn_cf", {})
        turn_label = pivot_cf.get("label", "")
        transfer = record.get("transfer_tier", "no_jailbreak")

        if (turn_label == "distributed_or_unclear"
                and transfer in {"transfer_success", "target_only", "cross_only"}
                and record.get("supervision_tier") == "construction"):
            record["supervision_tier"] = "llm_confirmed"
            record["loss_weight"] = gen.get_loss_weight("llm_confirmed")
            record["causal_note"] = "distributed_context_upgrade"
            upgraded += 1

    print(f"  Upgraded {upgraded} distributed-causal records: construction -> llm_confirmed")
    return records


# =========================================================
# Step 3: Validate supervision tier consistency
# =========================================================

def validate_tier_consistency(records: List[Dict]) -> List[Dict]:
    """Ensure supervision tiers and loss weights are consistent."""
    fixed = 0
    for record in records:
        tier = record.get("supervision_tier", "construction")
        expected_weight = gen.get_loss_weight(tier)
        actual_weight = record.get("loss_weight", 0.5)

        if abs(actual_weight - expected_weight) > 0.01:
            record["loss_weight"] = expected_weight
            fixed += 1

        # training_eligible consistency
        expected_eligible = (
            not record.get("is_external_test", False)
            and record.get("validation_status") in {"validated", "unvalidated"}
            and record.get("supervision_tier") != "ignore"
        )
        if record.get("training_eligible") != expected_eligible:
            record["training_eligible"] = expected_eligible
            fixed += 1

    if fixed:
        print(f"  Fixed {fixed} tier/weight/eligibility inconsistencies")
    else:
        print(f"  All tiers consistent")
    return records


# =========================================================
# Step 4: Print paper-ready statistics
# =========================================================

def print_final_stats(records: List[Dict]):
    """Print comprehensive statistics for paper methods section."""
    total = len(records)
    mal = [r for r in records if r.get("label") == 1]
    ben = [r for r in records if r.get("label") == 0]
    eligible = [r for r in records if r.get("training_eligible")]

    print(f"\n{'=' * 70}")
    print(f"  FINAL DATASET STATISTICS (for paper)")
    print(f"{'=' * 70}")

    print(f"\n  Total records: {total}")
    print(f"  Malicious: {len(mal)}")
    print(f"  Benign: {len(ben)}")
    print(f"  Training eligible: {len(eligible)}")

    # Transfer tiers
    print(f"\n  Transfer tiers (malicious):")
    ttiers = Counter(r.get("transfer_tier", "unknown") for r in mal)
    for t, c in ttiers.most_common():
        print(f"    {t}: {c}")

    # Supervision tiers
    print(f"\n  Supervision tiers (all records):")
    stiers = Counter(r.get("supervision_tier", "unknown") for r in records)
    for t, c in stiers.most_common():
        w = gen.get_loss_weight(t)
        print(f"    {t} (w={w:.2f}): {c}")

    # Pivot-turn CF
    print(f"\n  Pivot-turn counterfactual (malicious with pivots):")
    ptcf = Counter(
        r.get("causal_analysis", {}).get("pivot_turn_cf", {}).get("label", "none")
        for r in mal
    )
    for l, c in ptcf.most_common():
        print(f"    {l}: {c}")

    # Span statistics
    causal = incidental = unvalidated = 0
    causal_by_label = Counter()
    incidental_by_label = Counter()
    for r in records:
        for t in r.get("turns", []):
            for s in t.get("span_annotations", []):
                ct = s.get("causal_type", "unvalidated")
                label = s.get("label", "?")
                if ct == "causal":
                    causal += 1
                    causal_by_label[label] += 1
                elif ct == "incidental":
                    incidental += 1
                    incidental_by_label[label] += 1
                else:
                    unvalidated += 1

    print(f"\n  Span-level attribution:")
    print(f"    Causal spans: {causal}")
    if causal_by_label:
        for l, c in causal_by_label.most_common():
            print(f"      {l}: {c}")
    print(f"    Incidental spans: {incidental}")
    if incidental_by_label:
        for l, c in incidental_by_label.most_common():
            print(f"      {l}: {c}")
    print(f"    Unvalidated spans: {unvalidated}")

    # Relabeled spans
    relabeled = 0
    for r in records:
        for t in r.get("turns", []):
            for s in t.get("span_annotations", []):
                if s.get("original_label"):
                    relabeled += 1
    if relabeled:
        print(f"    Relabeled (BENIGN_CONTEXT/DECOY -> CONTEXT_BRIDGE): {relabeled}")

    # Loss weight distribution
    print(f"\n  Loss weight distribution (eligible only):")
    weights = [r.get("loss_weight", 0.5) for r in eligible]
    if weights:
        buckets = Counter()
        for w in weights:
            if w >= 0.9:
                buckets[">=0.9 (cf_strong / clean benign)"] += 1
            elif w >= 0.7:
                buckets["0.7-0.9 (cf_weak / hard benign)"] += 1
            elif w >= 0.5:
                buckets["0.5-0.7 (llm_confirmed)"] += 1
            elif w >= 0.2:
                buckets["0.2-0.5 (construction / llm_only)"] += 1
            else:
                buckets["<0.2 (incidental / ignored)"] += 1
        for b, c in sorted(buckets.items(), reverse=True):
            print(f"    {b}: {c}")
        print(f"    Mean: {sum(weights) / len(weights):.3f}")

    # Attack strategies (for interactive records)
    strategies = Counter(
        r.get("metadata", r.get("attack_metadata", {})).get("strategy", "unknown")
        for r in mal
    )
    if any(s != "unknown" for s in strategies):
        print(f"\n  Attack strategies (malicious):")
        for s, count in strategies.most_common():
            jb = sum(
                1 for r in mal
                if r.get("metadata", r.get("attack_metadata", {})).get("strategy") == s
                and (r.get("transfer_tier", "") in {"transfer_success", "target_only", "cross_only"})
            )
            print(f"    {s}: {count} total, {jb} jailbreaks ({jb / max(count, 1) * 100:.0f}%)")

    # Generator model breakdown
    gen_models = Counter()
    for r in mal:
        llama_val = r.get("llama_validation", {})
        judge_model = llama_val.get("judge_model", "unknown")
        if "14B" in judge_model or "14b" in judge_model:
            gen_models["14B"] += 1
        elif "7B" in judge_model or "7b" in judge_model:
            gen_models["7B"] += 1
        else:
            gen_models["unknown"] += 1
    if gen_models:
        print(f"\n  Generator model breakdown:")
        for m, c in gen_models.most_common():
            print(f"    {m}: {c}")

    # Conversation length
    mal_lengths = [r.get("conversation_length", 0) for r in mal]
    ben_lengths = [r.get("conversation_length", 0) for r in ben]
    if mal_lengths:
        print(f"\n  Conversation length:")
        print(f"    Malicious: mean={sum(mal_lengths)/len(mal_lengths):.1f}, "
              f"min={min(mal_lengths)}, max={max(mal_lengths)}")
    if ben_lengths:
        print(f"    Benign: mean={sum(ben_lengths)/len(ben_lengths):.1f}, "
              f"min={min(ben_lengths)}, max={max(ben_lengths)}")

    # Distributed context upgrades
    dist_upgraded = sum(1 for r in records if r.get("causal_note") == "distributed_context_upgrade")
    if dist_upgraded:
        print(f"\n  Distributed-context upgrades: {dist_upgraded}")

    print(f"\n{'=' * 70}")


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-process causal analysis output"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    print("\nStep 1: Relabel causal BENIGN_CONTEXT/DECOY spans")
    records = relabel_causal_benign_spans(records)

    print("\nStep 2: Assign distributed-context tier")
    records = assign_distributed_context_tier(records)

    print("\nStep 3: Validate tier consistency")
    records = validate_tier_consistency(records)

    # Write output
    gen.write_jsonl(records, args.output)
    print(f"\nWrote {len(records)} records to {args.output}")

    # Copy to persistent storage
    results_dir = os.path.expanduser("~/work/results/dataset_gen")
    os.makedirs(results_dir, exist_ok=True)
    import shutil
    persistent = os.path.join(results_dir, os.path.basename(args.output))
    shutil.copy2(args.output, persistent)
    print(f"Persistent: {persistent}")

    # Print paper statistics
    print_final_stats(records)


if __name__ == "__main__":
    main()
