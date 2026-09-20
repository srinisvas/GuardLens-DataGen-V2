#!/usr/bin/env python3
"""Final fail-closed audit for frozen NAACL data-preparation artifacts.

This audit is intentionally model-agnostic. It verifies corpus membership,
expected counts, exact-content hashing, grouped split isolation, source/length
shortcut controls, and the optional detection-only auxiliary attachment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, List

from audit_frontier_auxiliary import audit as audit_auxiliary
from audit_review_export import audit_review_export
from audit_semantic_turn_consistency import (
    CASE_A,
    CASE_B,
    inspect_splits as inspect_semantic_turns,
)
from frontier_common import json_fingerprint, load_jsonl
from merge_training_corpora import (
    EXPECTED_COMBINED_PER_LABEL,
    EXPECTED_COMBINED_RECORDS,
    EXPECTED_FRONTIER_RECORDS,
    EXPECTED_LEGACY_RECORDS,
    assert_source_shortcut_invariants,
    user_trajectory_hash,
)
from semantic_span_policy import CONSTRUCTION_META_LEXICON
from split_consolidated import assert_no_leakage

SPLITS = ("train", "dev", "test")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_ids(records: Iterable[Dict], where: str) -> set:
    ids = []
    for record in records:
        cid = str(record.get("conversation_id", "")).strip()
        if not cid:
            raise RuntimeError(f"{where}: record missing conversation_id")
        ids.append(cid)
    duplicates = [cid for cid, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise RuntimeError(f"{where}: duplicate conversation_ids: {duplicates[:10]}")
    return set(ids)


def audit_hashes(records: Iterable[Dict], where: str) -> None:
    for record in records:
        cid = str(record.get("conversation_id", ""))
        metadata = record.get("metadata", {}) or {}
        stored = str(metadata.get("normalized_user_trajectory_hash", "")).strip()
        if not stored:
            raise RuntimeError(f"{where}: {cid} missing normalized_user_trajectory_hash")
        recomputed = user_trajectory_hash(record)
        if stored != recomputed:
            raise RuntimeError(f"{where}: {cid} normalized user-trajectory hash mismatch")


def casefold_user_trajectory_hash(record: Dict) -> str:
    texts = []
    for turn in record.get("turns", []):
        if str(turn.get("role", "")).lower() != "user":
            continue
        text = unicodedata.normalize("NFKC", str(turn.get("text", ""))).casefold()
        texts.append(" ".join(text.split()))
    normalized = "\n<USER_TURN>\n".join(texts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def audit_casefold_trajectory_duplicates(records: Iterable[Dict]) -> None:
    owners = defaultdict(set)
    examples = defaultdict(list)
    for record in records:
        metadata = record.get("metadata", {}) or {}
        group = str(metadata.get("consolidated_split_group", "")).strip()
        if not group:
            raise RuntimeError(
                f"{record.get('conversation_id')}: missing consolidated_split_group"
            )
        digest = casefold_user_trajectory_hash(record)
        owners[digest].add(group)
        if len(examples[digest]) < 5:
            examples[digest].append(str(record.get("conversation_id", "")))
    collisions = [digest for digest, groups in owners.items() if len(groups) > 1]
    if collisions:
        sample = [
            {
                "hash": digest,
                "groups": sorted(owners[digest]),
                "conversation_ids": examples[digest],
            }
            for digest in collisions[:10]
        ]
        raise RuntimeError(
            "NFKC/casefold-normalized user trajectories occur across independent "
            f"split groups: {sample}"
        )


def split_source_label_report(splits: Dict[str, List[Dict]]) -> Dict:
    report = {}
    for split_name, records in splits.items():
        counts = Counter(
            (str(r.get("corpus_source", "unknown")), int(r.get("label")))
            for r in records
        )
        sources = sorted({source for source, _ in counts})
        for source in sources:
            missing = [label for label in (0, 1) if counts[(source, label)] == 0]
            if missing:
                raise RuntimeError(
                    f"{split_name}: source {source} is missing labels {missing}"
                )
        label_totals = Counter(int(r.get("label")) for r in records)
        source_share_gap = {}
        for source in sources:
            benign_share = counts[(source, 0)] / max(1, label_totals[0])
            malicious_share = counts[(source, 1)] / max(1, label_totals[1])
            source_share_gap[source] = {
                "label_0_count": counts[(source, 0)],
                "label_1_count": counts[(source, 1)],
                "p_source_given_label_0": benign_share,
                "p_source_given_label_1": malicious_share,
                "absolute_share_gap": abs(benign_share - malicious_share),
            }
        report[split_name] = source_share_gap
    return report


def audit_internal_identifier_leakage(records: Iterable[Dict]) -> None:
    leaks = []
    for record in records:
        cid = str(record.get("conversation_id", ""))
        pair_id = str(record.get("pair_id", "") or "")
        needles = [value for value in (cid, pair_id) if len(value) >= 8]
        if not needles:
            continue
        visible = "\n".join(str(t.get("text", "")) for t in record.get("turns", []))
        for needle in needles:
            if needle in visible:
                leaks.append((cid, needle))
                break
    if leaks:
        raise RuntimeError(
            "internal conversation/pair identifiers leaked into model-visible text: "
            f"{leaks[:10]}"
        )


def expected_frontier_membership(raw_records: Iterable[Dict]):
    pair_groups = defaultdict(list)
    standalone = []
    for record in raw_records:
        pair_id = record.get("pair_id")
        if pair_id in (None, ""):
            standalone.append(record)
        else:
            pair_groups[str(pair_id)].append(record)

    primary_ids = set()
    for pair_id, group in pair_groups.items():
        labels = Counter(r.get("label") for r in group)
        if len(group) != 2 or labels != Counter({0: 1, 1: 1}):
            continue
        malicious = next(r for r in group if r.get("label") == 1)
        benign = next(r for r in group if r.get("label") == 0)
        scenarios = {
            str((r.get("metadata", {}) or {}).get("scenario_family", "")).strip()
            for r in group
        }
        if "" in scenarios or len(scenarios) != 1:
            continue
        if malicious.get("validation_status") != "validated":
            continue
        if (
            (malicious.get("frontier_evidence_analysis", {}) or {}).get("status")
            != "complete"
        ):
            continue
        if benign.get("validation_status") != "validated":
            continue
        if (
            (benign.get("frontier_evidence_analysis", {}) or {}).get("status")
            != "not_applicable"
        ):
            continue
        primary_ids.update(str(r.get("conversation_id", "")) for r in group)

    stress_ids = {
        str(r.get("conversation_id", ""))
        for r in standalone
        if r.get("label") == 0
        and r.get("validation_status") == "validated"
        and (r.get("frontier_evidence_analysis", {}) or {}).get("status")
        == "not_applicable"
    }
    raw_ids = {str(r.get("conversation_id", "")) for r in raw_records}
    excluded_ids = raw_ids - primary_ids - stress_ids
    auxiliary_ids = {
        str(r.get("conversation_id", ""))
        for r in raw_records
        if r.get("validation_status") == "rejected"
    }
    return primary_ids, stress_ids, excluded_ids, auxiliary_ids


def audit_frontier_stress_contract(records: Iterable[Dict]) -> None:
    for record in records:
        cid = str(record.get("conversation_id", ""))
        if record.get("label") != 0:
            raise RuntimeError(f"frontier stress: {cid} is not benign")
        if record.get("pair_id") not in (None, ""):
            raise RuntimeError(f"frontier stress: {cid} unexpectedly has pair_id")
        if record.get("training_eligible") is not False:
            raise RuntimeError(f"frontier stress: {cid} is training eligible")
        if record.get("primary_pair_complete") is not False:
            raise RuntimeError(f"frontier stress: {cid} claims primary pair membership")
        if record.get("use_as") != "benign_stress_evaluation_only":
            raise RuntimeError(f"frontier stress: {cid} has unexpected use_as")


def audit_frontier_excluded_contract(records: Iterable[Dict]) -> None:
    for record in records:
        cid = str(record.get("conversation_id", ""))
        if record.get("training_eligible") is not False:
            raise RuntimeError(f"frontier excluded: {cid} is training eligible")
        if record.get("use_as") != "excluded_from_primary_frontier_corpus":
            raise RuntimeError(f"frontier excluded: {cid} has unexpected use_as")
        if not str(record.get("exclusion_reason", "")).strip():
            raise RuntimeError(f"frontier excluded: {cid} lacks exclusion_reason")


def construction_language_report(records: Iterable[Dict]) -> Dict:
    counts = Counter()
    examples = defaultdict(list)
    lowered_phrases = tuple(p.lower() for p in CONSTRUCTION_META_LEXICON)
    for record in records:
        text = "\n".join(
            str(turn.get("text", "")) for turn in record.get("turns", [])
        ).lower()
        hits = [phrase for phrase in lowered_phrases if phrase in text]
        if not hits:
            continue
        key = f"{record.get('corpus_source')}|label={record.get('label')}"
        counts[key] += 1
        if len(examples[key]) < 5:
            examples[key].append(str(record.get("conversation_id", "")))
    return {
        "records_with_reviewed_construction_language": dict(sorted(counts.items())),
        "example_conversation_ids": dict(sorted(examples.items())),
    }


def load_split_dir(path: str) -> Dict[str, List[Dict]]:
    return {
        name: load_jsonl(os.path.join(path, f"{name}.jsonl"))
        for name in SPLITS
    }


def assert_partition_exactly(
    parent: List[Dict],
    splits: Dict[str, List[Dict]],
    *,
    where: str,
) -> None:
    parent_ids = unique_ids(parent, f"{where} parent")
    split_ids = {}
    for name, records in splits.items():
        split_ids[name] = unique_ids(records, f"{where} {name}")

    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                raise RuntimeError(
                    f"{where}: conversation IDs overlap between {left} and {right}: "
                    f"{sorted(overlap)[:10]}"
                )

    union = set().union(*(split_ids[name] for name in SPLITS))
    if union != parent_ids:
        missing = sorted(parent_ids - union)
        extra = sorted(union - parent_ids)
        raise RuntimeError(
            f"{where}: split membership does not exactly partition parent; "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    parent_fingerprints = {
        str(record["conversation_id"]): json_fingerprint(record)
        for record in parent
    }
    for split_name, records in splits.items():
        for record in records:
            cid = str(record["conversation_id"])
            observed = json_fingerprint(record)
            expected = parent_fingerprints[cid]
            if observed != expected:
                raise RuntimeError(
                    f"{where}: {cid} content changed in {split_name} after splitting"
                )
    assert_no_leakage(splits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-input", required=True)
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--legacy-input", required=True)
    parser.add_argument("--frontier-primary", required=True)
    parser.add_argument("--frontier-stress", required=True)
    parser.add_argument("--frontier-excluded", required=True)
    parser.add_argument("--merged-input", required=True)
    parser.add_argument("--primary-split-dir", required=True)
    parser.add_argument("--auxiliary-input")
    parser.add_argument("--auxiliary-candidate-split-dir")
    parser.add_argument("--expect-final-naacl-counts", action="store_true")
    parser.add_argument("--report-output")
    args = parser.parse_args()

    review_report = audit_review_export(args.review_input, args.review_manifest)
    raw_review = load_jsonl(args.review_input)
    legacy = load_jsonl(args.legacy_input)
    frontier = load_jsonl(args.frontier_primary)
    frontier_stress = load_jsonl(args.frontier_stress)
    frontier_excluded = load_jsonl(args.frontier_excluded)
    merged = load_jsonl(args.merged_input)

    raw_review_ids = unique_ids(raw_review, "raw frontier review")
    legacy_ids = unique_ids(legacy, "legacy")
    frontier_ids = unique_ids(frontier, "frontier primary")
    frontier_stress_ids = unique_ids(frontier_stress, "frontier stress")
    frontier_excluded_ids = unique_ids(frontier_excluded, "frontier excluded")
    merged_ids = unique_ids(merged, "merged primary")

    for left_name, left_ids, right_name, right_ids in (
        ("primary", frontier_ids, "stress", frontier_stress_ids),
        ("primary", frontier_ids, "excluded", frontier_excluded_ids),
        ("stress", frontier_stress_ids, "excluded", frontier_excluded_ids),
    ):
        overlap = left_ids & right_ids
        if overlap:
            raise RuntimeError(
                f"frontier preparation overlap between {left_name} and {right_name}: "
                f"{sorted(overlap)[:10]}"
            )
    prepared_union = frontier_ids | frontier_stress_ids | frontier_excluded_ids
    if prepared_union != raw_review_ids:
        missing = sorted(raw_review_ids - prepared_union)
        extra = sorted(prepared_union - raw_review_ids)
        raise RuntimeError(
            "frontier primary/stress/excluded do not exactly partition the raw review; "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    (
        expected_primary_ids,
        expected_stress_ids,
        expected_excluded_ids,
        expected_auxiliary_ids,
    ) = expected_frontier_membership(raw_review)
    for name, observed, expected in (
        ("primary", frontier_ids, expected_primary_ids),
        ("stress", frontier_stress_ids, expected_stress_ids),
        ("excluded", frontier_excluded_ids, expected_excluded_ids),
    ):
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise RuntimeError(
                f"frontier {name} membership differs from raw-derived admission rule; "
                f"missing={missing[:10]} extra={extra[:10]}"
            )

    audit_frontier_stress_contract(frontier_stress)
    audit_frontier_excluded_contract(frontier_excluded)

    if legacy_ids & frontier_ids:
        raise RuntimeError(
            "legacy and frontier primary corpora share conversation IDs"
        )
    if merged_ids != legacy_ids | frontier_ids:
        raise RuntimeError("merged corpus membership differs from A union B")

    if args.expect_final_naacl_counts:
        if len(legacy) != EXPECTED_LEGACY_RECORDS:
            raise RuntimeError(
                f"legacy count={len(legacy)} expected={EXPECTED_LEGACY_RECORDS}"
            )
        if len(frontier) != EXPECTED_FRONTIER_RECORDS:
            raise RuntimeError(
                f"frontier count={len(frontier)} expected={EXPECTED_FRONTIER_RECORDS}"
            )
        if len(merged) != EXPECTED_COMBINED_RECORDS:
            raise RuntimeError(
                f"merged count={len(merged)} expected={EXPECTED_COMBINED_RECORDS}"
            )
        labels = Counter(r.get("label") for r in merged)
        expected = Counter(
            {0: EXPECTED_COMBINED_PER_LABEL, 1: EXPECTED_COMBINED_PER_LABEL}
        )
        if labels != expected:
            raise RuntimeError(
                f"merged labels={dict(labels)} expected={dict(expected)}"
            )

    source_counts = Counter(str(r.get("corpus_source")) for r in merged)
    if source_counts != Counter(
        {
            "legacy_repaired": len(legacy),
            "frontier_authored_v3": len(frontier),
        }
    ):
        raise RuntimeError(
            f"merged corpus_source counts are inconsistent: {dict(source_counts)}"
        )

    audit_hashes(merged, "merged primary")
    audit_casefold_trajectory_duplicates(merged)
    assert_source_shortcut_invariants(merged, "legacy_repaired")
    assert_source_shortcut_invariants(merged, "frontier_authored_v3")
    audit_internal_identifier_leakage(merged)

    primary_splits = load_split_dir(args.primary_split_dir)
    assert_partition_exactly(
        merged,
        primary_splits,
        where="primary A+B split",
    )

    # The semantic repair inspection is deliberately restricted to train/dev.
    # The held-out test remains outside semantic inspection; exact test-byte
    # preservation is proved separately by the repair-delta audit.
    semantic_turn_report = inspect_semantic_turns(
        {
            "train": primary_splits["train"],
            "dev": primary_splits["dev"],
        },
        enforce_reviewed_counts=args.expect_final_naacl_counts,
    )
    if semantic_turn_report["silent_fallback_case_B_turns"] != 0:
        raise RuntimeError(
            "primary train/dev retain orphan semantic-masked evidence-turn targets"
        )
    if semantic_turn_report["unrepaired_case_A_turns"] != 0:
        raise RuntimeError(
            "primary train/dev retain unrepaired semantic-masked turn contradictions"
        )
    if args.expect_final_naacl_counts:
        if semantic_turn_report["repaired_case_A_turns"] != 4:
            raise RuntimeError(
                "semantic turn repair count changed: "
                f"{semantic_turn_report['repaired_case_A_turns']} != 4"
            )
        if semantic_turn_report["case_counts_by_turn"].get(CASE_A, 0) != 4:
            raise RuntimeError("expected exactly four semantically repaired case-A turns")
        if semantic_turn_report["case_counts_by_turn"].get(CASE_B, 0) != 0:
            raise RuntimeError("unexpected semantic case-B turns")

    split_source_labels = split_source_label_report(primary_splits)

    report = {
        "status": "passed",
        "review_export": review_report,
        "artifact_sha256": {
            "review_input": file_sha256(args.review_input),
            "review_manifest": file_sha256(args.review_manifest),
            "legacy_input": file_sha256(args.legacy_input),
            "frontier_primary": file_sha256(args.frontier_primary),
            "frontier_stress": file_sha256(args.frontier_stress),
            "frontier_excluded": file_sha256(args.frontier_excluded),
            "merged_input": file_sha256(args.merged_input),
            "primary_train": file_sha256(os.path.join(args.primary_split_dir, "train.jsonl")),
            "primary_dev": file_sha256(os.path.join(args.primary_split_dir, "dev.jsonl")),
            "primary_test": file_sha256(os.path.join(args.primary_split_dir, "test.jsonl")),
        },
        "counts": {
            "legacy": len(legacy),
            "frontier_primary": len(frontier),
            "frontier_stress": len(frontier_stress),
            "frontier_excluded": len(frontier_excluded),
            "frontier_review_partition_total": (
                len(frontier) + len(frontier_stress) + len(frontier_excluded)
            ),
            "merged_primary": len(merged),
            "primary_splits": {
                name: len(rows) for name, rows in primary_splits.items()
            },
        },
        "source_counts": dict(source_counts),
        "source_shortcut_controls": {
            "per_source_label_balance": "passed",
            "per_source_user_turn_histogram_match": "passed",
            "per_source_total_turn_histogram_match": "passed",
            "nfkc_casefold_exact_trajectory_duplicates_across_groups": 0,
            "internal_identifier_leakage": "passed",
            "split_source_label_composition": split_source_labels,
        },
        "construction_language_visibility": construction_language_report(merged),
        "frontier_review_partition": "passed",
        "frontier_membership_reconstructed_from_raw": {
            "primary": len(expected_primary_ids),
            "stress": len(expected_stress_ids),
            "excluded": len(expected_excluded_ids),
            "auxiliary_rejected_candidates": len(expected_auxiliary_ids),
        },
        "frontier_stress_contract": "passed",
        "frontier_excluded_contract": "passed",
        "semantic_turn_supervision": semantic_turn_report,
        "primary_split_leakage": "passed",
    }

    if args.auxiliary_input:
        auxiliary = load_jsonl(args.auxiliary_input)
        auxiliary_ids = unique_ids(auxiliary, "auxiliary")
        if auxiliary_ids != expected_auxiliary_ids:
            missing = sorted(expected_auxiliary_ids - auxiliary_ids)
            extra = sorted(auxiliary_ids - expected_auxiliary_ids)
            raise RuntimeError(
                "auxiliary membership differs from the complete raw B2-rejected set; "
                f"missing={missing[:10]} extra={extra[:10]}"
            )
        aux_report = audit_auxiliary(
            auxiliary,
            expect_full_review_export=args.expect_final_naacl_counts,
            validate_provenance=True,
        )
        report["auxiliary"] = {
            **aux_report,
            "raw_rejected_membership_exact": True,
        }
        report["artifact_sha256"]["auxiliary_input"] = file_sha256(args.auxiliary_input)

        if args.auxiliary_candidate_split_dir:
            candidate = load_split_dir(args.auxiliary_candidate_split_dir)
            primary_train_ids = unique_ids(primary_splits["train"], "primary train")
            aux_ids = unique_ids(auxiliary, "auxiliary")
            candidate_train_ids = unique_ids(candidate["train"], "candidate train")
            candidate_dev_ids = unique_ids(candidate["dev"], "candidate dev")
            candidate_test_ids = unique_ids(candidate["test"], "candidate test")

            if not primary_train_ids <= candidate_train_ids:
                raise RuntimeError("auxiliary candidate dropped primary training records")
            if candidate_dev_ids != unique_ids(primary_splits["dev"], "primary dev"):
                raise RuntimeError("auxiliary candidate dev membership changed")
            if candidate_test_ids != unique_ids(primary_splits["test"], "primary test"):
                raise RuntimeError("auxiliary candidate test membership changed")
            if (candidate_train_ids - primary_train_ids) - aux_ids:
                raise RuntimeError("auxiliary candidate train contains unknown extra records")
            if candidate_dev_ids & aux_ids or candidate_test_ids & aux_ids:
                raise RuntimeError("auxiliary records entered dev/test")

            primary_dev_path = os.path.join(args.primary_split_dir, "dev.jsonl")
            primary_test_path = os.path.join(args.primary_split_dir, "test.jsonl")
            candidate_dev_path = os.path.join(
                args.auxiliary_candidate_split_dir, "dev.jsonl"
            )
            candidate_test_path = os.path.join(
                args.auxiliary_candidate_split_dir, "test.jsonl"
            )
            if file_sha256(primary_dev_path) != file_sha256(candidate_dev_path):
                raise RuntimeError("auxiliary candidate dev is not byte-identical")
            if file_sha256(primary_test_path) != file_sha256(candidate_test_path):
                raise RuntimeError("auxiliary candidate test is not byte-identical")

            assert_no_leakage(candidate)
            report["artifact_sha256"].update({
                "auxiliary_candidate_train": file_sha256(
                    os.path.join(args.auxiliary_candidate_split_dir, "train.jsonl")
                ),
                "auxiliary_candidate_dev": file_sha256(candidate_dev_path),
                "auxiliary_candidate_test": file_sha256(candidate_test_path),
            })
            report["auxiliary_candidate"] = {
                "train_records": len(candidate["train"]),
                "auxiliary_records_in_train": len(
                    candidate_train_ids - primary_train_ids
                ),
                "dev_records": len(candidate["dev"]),
                "test_records": len(candidate["test"]),
                "dev_byte_identical_to_primary": True,
                "test_byte_identical_to_primary": True,
                "leakage_check": "passed",
            }
    elif args.auxiliary_candidate_split_dir:
        raise RuntimeError(
            "--auxiliary-candidate-split-dir requires --auxiliary-input"
        )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report_output:
        os.makedirs(os.path.dirname(args.report_output) or ".", exist_ok=True)
        with open(args.report_output, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.write("\n")
    print(rendered)


if __name__ == "__main__":
    main()
