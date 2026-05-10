"""
split_dataset.py

Train/dev/test splitting for the v11 dataset.

Splitting constraints:
  1. Pair linkage: paired twins (same pair_id) stay in the same split.
  2. Paraphrase linkage: original + paraphrases (same pair_id, or
     conversation_id prefix match) stay in the same split.
  3. MHJ / external records -> test split only (is_external_test=True).
  4. Rejected records (validation_status="rejected") excluded entirely.
  5. Stratification by family, difficulty, and pivot_kind.
  6. Human benchmark selection: 100 records for annotation, 50 double.

Usage:
    python split_dataset.py --input output/validated.jsonl \\
                            --output-dir output/splits/

    python split_dataset.py --input output/validated.jsonl \\
                            --mhj-input output/mhj.jsonl \\
                            --output-dir output/splits/ \\
                            --human-benchmark 100 --double-annotated 50
"""

import argparse
import json
import os
import random
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def group_by_pair(records: List[Dict]) -> Dict[str, List[Dict]]:
    """Group records by pair_id. Records sharing a pair_id must stay together."""
    groups = defaultdict(list)
    for r in records:
        pid = r.get("pair_id", r.get("conversation_id", ""))
        groups[pid].append(r)
    return dict(groups)


def stratification_key(records: List[Dict]) -> str:
    """
    Compute a stratification key for a group of records.
    Uses the first record's family + difficulty + label + pivot_kind.
    """
    if not records:
        return "unknown"
    r = records[0]
    family = r.get("family", "unknown")
    diff = r.get("difficulty", "unknown")
    label = r.get("label", -1)
    pivot = r.get("pivot_kind", "none")
    return f"{family}_{diff}_{label}_{pivot}"


def split_groups(
    groups: Dict[str, List[Dict]],
    train_frac: float = 0.70,
    dev_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split groups into train/dev/test with stratification.
    Each group (pair_id) goes entirely to one split.
    """
    rng = random.Random(seed)

    # Stratify groups
    strata = defaultdict(list)
    for pid, records in groups.items():
        key = stratification_key(records)
        strata[key].append((pid, records))

    train, dev, test = [], [], []

    for key, group_list in strata.items():
        rng.shuffle(group_list)
        n = len(group_list)
        n_train = max(1, int(n * train_frac))
        n_dev = max(0, int(n * dev_frac))
        # Rest goes to test
        n_test = n - n_train - n_dev

        for pid, records in group_list[:n_train]:
            train.extend(records)
        for pid, records in group_list[n_train:n_train + n_dev]:
            dev.extend(records)
        for pid, records in group_list[n_train + n_dev:]:
            test.extend(records)

    return train, dev, test


def select_human_benchmark(
    records: List[Dict],
    n_total: int = 100,
    n_double: int = 50,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Select records for human annotation benchmark.

    Returns dict with keys:
      - 'single': records for single annotation
      - 'double': records for double annotation (subset of single)
      - 'ids': list of conversation_ids selected

    Selection criteria:
      - Balanced by label (50/50 malicious/benign target)
      - Diverse families and difficulties
      - Prefer validated records
      - Prefer records with counterfactual data
    """
    rng = random.Random(seed)

    # Separate by label
    malicious = [r for r in records if r.get("label") == 1]
    benign = [r for r in records if r.get("label") == 0]

    # Prefer validated, then with counterfactual
    def sort_key(r):
        has_val = 1 if r.get("validation_status") == "validated" else 0
        has_cf = 1 if r.get("causal_validation", {}).get("jailbreak_detected") else 0
        return (has_val, has_cf, rng.random())

    malicious.sort(key=sort_key, reverse=True)
    benign.sort(key=sort_key, reverse=True)

    n_mal = min(n_total // 2, len(malicious))
    n_ben = min(n_total - n_mal, len(benign))

    selected = malicious[:n_mal] + benign[:n_ben]
    rng.shuffle(selected)

    # Double-annotated subset
    double = selected[:min(n_double, len(selected))]

    return {
        "single": selected,
        "double": double,
        "ids": [r.get("conversation_id", "") for r in selected],
    }


def print_split_stats(name: str, records: List[Dict]):
    """Print statistics for a split."""
    if not records:
        print(f"  {name}: 0 records")
        return

    labels = Counter(r.get("label", -1) for r in records)
    families = Counter(r.get("family", "unknown") for r in records)
    tiers = Counter(r.get("supervision_tier", "unknown") for r in records)
    statuses = Counter(r.get("validation_status", "unknown") for r in records)
    pivots = Counter(r.get("pivot_kind", "none") for r in records)
    eligible = sum(1 for r in records if r.get("training_eligible", True))
    external = sum(1 for r in records if r.get("is_external_test", False))

    print(f"  {name}: {len(records)} records")
    print(f"    Labels: {dict(labels)}")
    print(f"    Training eligible: {eligible}")
    print(f"    External test: {external}")
    print(f"    Validation: {dict(statuses)}")
    print(f"    Supervision tiers: {dict(tiers)}")
    print(f"    Pivot kinds: {dict(pivots)}")
    print(f"    Families (top 5): {dict(families.most_common(5))}")


def main():
    parser = argparse.ArgumentParser(
        description="Split v11 dataset into train/dev/test"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Validated JSONL from batch 2")
    parser.add_argument("--mhj-input", type=str, default=None,
                        help="MHJ JSONL from mhj_loader.py (goes to test)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for splits")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--dev-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--human-benchmark", type=int, default=100,
                        help="Number of records for human benchmark")
    parser.add_argument("--double-annotated", type=int, default=50,
                        help="Number of double-annotated records")
    parser.add_argument("--exclude-rejected", action="store_true", default=True,
                        help="Exclude validation_status=rejected records")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load main dataset
    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    # Exclude rejected records
    if args.exclude_rejected:
        before = len(records)
        records = [r for r in records if r.get("validation_status") != "rejected"]
        excluded = before - len(records)
        if excluded:
            print(f"Excluded {excluded} rejected records")

    # Separate external test records
    external_test = [r for r in records if r.get("is_external_test", False)]
    internal = [r for r in records if not r.get("is_external_test", False)]
    print(f"Internal records: {len(internal)}, External test: {len(external_test)}")

    # Load MHJ if provided
    if args.mhj_input and os.path.exists(args.mhj_input):
        mhj_records = load_jsonl(args.mhj_input)
        print(f"Loaded {len(mhj_records)} MHJ records from {args.mhj_input}")
        external_test.extend(mhj_records)

    # Group internal records by pair_id
    groups = group_by_pair(internal)
    print(f"Grouped into {len(groups)} pair groups")

    # Split
    train, dev, test = split_groups(
        groups,
        train_frac=args.train_frac,
        dev_frac=args.dev_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    # Add external test records to test split
    test.extend(external_test)

    # Write splits
    write_jsonl(train, os.path.join(args.output_dir, "train.jsonl"))
    write_jsonl(dev, os.path.join(args.output_dir, "dev.jsonl"))
    write_jsonl(test, os.path.join(args.output_dir, "test.jsonl"))

    print(f"\nSplit statistics:")
    print_split_stats("Train", train)
    print_split_stats("Dev", dev)
    print_split_stats("Test", test)

    # Human benchmark selection (from dev + test)
    benchmark_pool = dev + test
    benchmark = select_human_benchmark(
        benchmark_pool,
        n_total=args.human_benchmark,
        n_double=args.double_annotated,
        seed=args.seed,
    )

    write_jsonl(
        benchmark["single"],
        os.path.join(args.output_dir, "human_benchmark.jsonl"),
    )
    write_jsonl(
        benchmark["double"],
        os.path.join(args.output_dir, "human_benchmark_double.jsonl"),
    )

    # Write benchmark IDs
    with open(os.path.join(args.output_dir, "human_benchmark_ids.json"), "w") as f:
        json.dump(benchmark["ids"], f, indent=2)

    print(f"\nHuman benchmark:")
    print(f"  Single-annotated: {len(benchmark['single'])} records")
    print(f"  Double-annotated: {len(benchmark['double'])} records")

    # Write split metadata
    meta = {
        "total_records": len(records) + len(external_test),
        "train": len(train),
        "dev": len(dev),
        "test": len(test),
        "external_test_in_test": len(external_test),
        "rejected_excluded": args.exclude_rejected,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "dev_frac": args.dev_frac,
        "test_frac": args.test_frac,
        "human_benchmark": len(benchmark["single"]),
        "double_annotated": len(benchmark["double"]),
    }
    with open(os.path.join(args.output_dir, "split_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nAll splits written to {args.output_dir}/")


if __name__ == "__main__":
    main()
