#!/usr/bin/env python3
"""Build the restored-A + frozen-B GuardLens data freeze.

Scientific order
----------------
1. Verify exact frozen input hashes and structural contracts.
2. Merge PRIMARY A (516 semantic pairs) with frozen PRIMARY B (701 pairs).
3. Split PRIMARY records only. A pair IDs and B scenario families are indivisible.
4. Add all independently generated A detection auxiliaries to train only.
5. Attach B's full 512-record auxiliary pool using the frozen B policy:
   include only when its scenario group is owned by primary train or absent
   from primary, and withhold when its group or exact user trajectory is owned
   by primary dev/test.
6. Copy primary dev/test byte-for-byte into the auxiliary training variant.
7. Emit source-stratified shortcut diagnostics, hashes, and the exact freeze
   report schema consumed by GuardLens-Transformer.

This script never truncates or rewrites conversation text. It refuses to write
into a non-empty output directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


# -------------------------------
# Frozen input identities
# -------------------------------

A_PRIMARY_SHA256 = (
    "f9021672150696b2c3a367b1a1c67edafa4ce2f1a6a83fc1293870eaa7e6c34f"
)
A_AUX_SHA256 = (
    "9d17f3b094957ba2626184e98c33dde81a153a55e91fbb94799fbe7a0ae577ad"
)
B_PRIMARY_SHA256 = (
    "875694d3b2ba726dfc9112438b91a055c53459e961f28ab392e9334183d52b14"
)
B_AUX_SHA256 = (
    "f8e19e89dbbfcc2e41a3ff0bf560d6aa9d0d8e07a794f7306b2fc56daa00f594"
)

A_PRIMARY_RECORDS = 1032
A_PRIMARY_PAIRS = 516
A_AUX_RECORDS = 721
A_PRIMARY_FAMILIES = Counter({
    "interactive_adversarial": 516,
    "interactive_benign_twin": 516,
})
A_PRIMARY_TIERS = Counter({
    "benign_validated": 516,
    "cf_strong": 3,
    "cf_weak": 2,
    "llm_confirmed": 511,
})

B_PRIMARY_RECORDS = 1402
B_PRIMARY_PAIRS = 701
B_PRIMARY_SCENARIOS = 321
B_PRIMARY_LABELS = Counter({0: 701, 1: 701})
B_PRIMARY_USER_HIST_PER_LABEL = Counter({6: 240, 7: 286, 8: 92, 9: 83})
B_PRIMARY_TIERS = Counter({
    "benign_validated": 701,
    "cf_strong": 629,
    "cf_weak": 4,
    "llm_confirmed": 68,
})

B_AUX_RECORDS = 512
B_AUX_DETECTION_LABELS = Counter({0: 322, 1: 190})
B_AUX_AUTHORING_LABELS = Counter({0: 190, 1: 322})
B_AUX_SCENARIOS = 275
B_AUX_WEIGHT = 0.25
B_AUX_SOURCE = "frontier_authored_v3_auxiliary"

B_TARGET = "Qwen/Qwen2.5-32B-Instruct"
B_JUDGE = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

SPLITS = ("train", "dev", "test")
DEFAULT_MAX_TURNS = 64
EXPECTED_DATAGEN_BRANCH = "naacl-validity-repair"


# -------------------------------
# Generic file helpers
# -------------------------------

def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def verify_code_checkout() -> Dict:
    branch = git_output("rev-parse", "--abbrev-ref", "HEAD")
    if branch != EXPECTED_DATAGEN_BRANCH:
        raise RuntimeError(
            f"expected DataGen branch {EXPECTED_DATAGEN_BRANCH}, got {branch}"
        )
    commit = git_output("rev-parse", "HEAD")

    tracked_dirty = subprocess.run(
        ["git", "diff", "--quiet"],
        check=False,
    ).returncode != 0 or subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        check=False,
    ).returncode != 0
    if tracked_dirty:
        raise RuntimeError(
            "tracked DataGen working-tree changes detected; commit or stash "
            "before creating a scientific freeze"
        )
    return {
        "branch": branch,
        "commit": commit,
        "tracked_working_tree_clean": True,
    }


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid JSON at {path}:{line_no}: {exc}"
                ) from exc
    return rows


def write_jsonl(records: Iterable[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_json(payload: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(path: str, expected: str, name: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{name} SHA-256 mismatch: expected={expected} actual={actual}"
        )
    return actual


def ensure_fresh_output_dir(path: str) -> None:
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise RuntimeError(f"output path exists and is not a directory: {path}")
        entries = os.listdir(path)
        if entries:
            raise RuntimeError(
                f"output directory is not empty: {path}; existing entries={entries[:10]}"
            )
    else:
        os.makedirs(path, exist_ok=False)


# -------------------------------
# Observable-content identities
# -------------------------------

def n_user_turns(record: Dict) -> int:
    return sum(
        str(turn.get("role", "")).lower() == "user"
        for turn in record.get("turns", [])
    )


def normalized_user_hash(record: Dict) -> str:
    """Frozen B-compatible whitespace-normalized user-trajectory hash."""
    texts = [
        str(turn.get("text", "")).strip()
        for turn in record.get("turns", [])
        if str(turn.get("role", "")).lower() == "user"
    ]
    normalized = "\n<USER_TURN>\n".join(
        " ".join(text.split()) for text in texts
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def nfkc_casefold_user_hash(record: Dict) -> str:
    """Stronger exact-content leakage hash used in addition to frozen B hash."""
    texts = []
    for turn in record.get("turns", []):
        if str(turn.get("role", "")).lower() != "user":
            continue
        text = unicodedata.normalize("NFKC", str(turn.get("text", "")))
        texts.append(" ".join(text.casefold().split()))
    normalized = "\n<USER_TURN>\n".join(texts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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


def validate_realized_trajectory(record: Dict, *, source: str, max_turns: int) -> None:
    cid = str(record.get("conversation_id", "")) or "<missing>"
    turns = record.get("turns")
    if not isinstance(turns, list) or not turns:
        raise RuntimeError(f"{source}:{cid}: empty trajectory")
    if len(turns) > max_turns:
        raise RuntimeError(
            f"{source}:{cid}: {len(turns)} physical turns exceed max_turns={max_turns}"
        )

    expected_role = "user"
    for index, turn in enumerate(turns):
        if turn.get("turn_id") != index:
            raise RuntimeError(
                f"{source}:{cid}: turn_id={turn.get('turn_id')!r} "
                f"at physical index={index}"
            )
        role = str(turn.get("role", "")).lower()
        if role != expected_role:
            raise RuntimeError(
                f"{source}:{cid}: expected role={expected_role}, got={role!r} "
                f"at turn {index}"
            )
        text = turn.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(
                f"{source}:{cid}: empty/non-string text at turn {index}"
            )
        expected_role = "assistant" if expected_role == "user" else "user"

    if expected_role != "user":
        raise RuntimeError(f"{source}:{cid}: trajectory ends with unmatched user turn")

    declared = record.get("conversation_length")
    if declared is not None and int(declared) != len(turns):
        raise RuntimeError(
            f"{source}:{cid}: conversation_length={declared} "
            f"!= physical turns={len(turns)}"
        )


# -------------------------------
# Dataset A contracts
# -------------------------------

def validate_a_primary(records: Sequence[Dict], max_turns: int) -> None:
    if len(records) != A_PRIMARY_RECORDS:
        raise RuntimeError(
            f"A primary expected {A_PRIMARY_RECORDS}, got {len(records)}"
        )
    labels = Counter(record.get("label") for record in records)
    if labels != Counter({0: A_PRIMARY_PAIRS, 1: A_PRIMARY_PAIRS}):
        raise RuntimeError(f"A primary label counts invalid: {dict(labels)}")
    families = Counter(str(record.get("family")) for record in records)
    if families != A_PRIMARY_FAMILIES:
        raise RuntimeError(f"A primary family counts changed: {dict(families)}")
    tiers = Counter(str(record.get("supervision_tier")) for record in records)
    if tiers != A_PRIMARY_TIERS:
        raise RuntimeError(f"A primary supervision tiers changed: {dict(tiers)}")

    ids = set()
    pairs = defaultdict(list)
    for record in records:
        validate_realized_trajectory(record, source="A-primary", max_turns=max_turns)
        cid = str(record.get("conversation_id", ""))
        if not cid or cid in ids:
            raise RuntimeError(f"A primary missing/duplicate conversation_id: {cid!r}")
        ids.add(cid)
        if record.get("validation_status") != "validated":
            raise RuntimeError(f"{cid}: A primary is not validation_status=validated")
        if record.get("training_eligible") is not True:
            raise RuntimeError(f"{cid}: A primary is not training eligible")

        pair_id = str(record.get("pair_id", "")).strip()
        if not pair_id:
            raise RuntimeError(f"{cid}: A primary missing pair_id")
        pairs[pair_id].append(record)

    if len(pairs) != A_PRIMARY_PAIRS:
        raise RuntimeError(
            f"A primary expected {A_PRIMARY_PAIRS} pair groups, got {len(pairs)}"
        )
    for pair_id, group in pairs.items():
        if (
            len(group) != 2
            or Counter(record.get("label") for record in group)
            != Counter({0: 1, 1: 1})
        ):
            raise RuntimeError(
                f"A primary pair {pair_id} is not exactly one benign + one malicious"
            )


def validate_a_aux(records: Sequence[Dict], max_turns: int) -> None:
    if len(records) != A_AUX_RECORDS:
        raise RuntimeError(f"A auxiliary expected {A_AUX_RECORDS}, got {len(records)}")

    ids = set()
    for record in records:
        validate_realized_trajectory(record, source="A-aux", max_turns=max_turns)
        cid = str(record.get("conversation_id", ""))
        if not cid or cid in ids:
            raise RuntimeError(f"A auxiliary missing/duplicate conversation_id: {cid!r}")
        ids.add(cid)

        if record.get("label") != 0 or record.get("detection_label") != 0:
            raise RuntimeError(f"{cid}: A auxiliary must be benign detection label 0")
        if record.get("validation_status") != "validated":
            raise RuntimeError(f"{cid}: A auxiliary is not validated")
        if record.get("auxiliary_detection_only") is not True:
            raise RuntimeError(f"{cid}: A auxiliary missing auxiliary_detection_only")
        if record.get("use_as") != "auxiliary_detection_only":
            raise RuntimeError(f"{cid}: A auxiliary use_as mismatch")
        if record.get("supervision_tier") != "auxiliary_detection":
            raise RuntimeError(f"{cid}: A auxiliary supervision tier mismatch")
        if float(record.get("detection_loss_weight", -1)) != 1.0:
            raise RuntimeError(f"{cid}: A auxiliary detection weight must be 1.0")
        if record.get("localization_supervision_ignore") is not True:
            raise RuntimeError(f"{cid}: A auxiliary localization not masked")
        if record.get("pivot_supervision_ignore") is not True:
            raise RuntimeError(f"{cid}: A auxiliary pivot not masked")
        if record.get("pivot_loss_weight") != 0.0:
            raise RuntimeError(f"{cid}: A auxiliary pivot loss must be 0")
        if record.get("span_loss_weight") != 0.0:
            raise RuntimeError(f"{cid}: A auxiliary span loss must be 0")


# -------------------------------
# Dataset B contracts
# -------------------------------

def b_scenario(record: Dict) -> str:
    return str(
        (record.get("metadata", {}) or {}).get("scenario_family", "")
    ).strip()


def b_group(record: Dict) -> str:
    scenario = b_scenario(record)
    return f"frontier::{scenario}" if scenario else ""


def validate_b_primary(records: Sequence[Dict], max_turns: int) -> None:
    """Validate the exact already-frozen B artifact structurally.

    Full v4/v5/v8 production provenance was audited before the pinned file hash
    was frozen. This stage does not reinterpret historical protocol wrappers.
    """
    if len(records) != B_PRIMARY_RECORDS:
        raise RuntimeError(
            f"B primary expected {B_PRIMARY_RECORDS}, got {len(records)}"
        )

    labels = Counter(record.get("label") for record in records)
    if labels != B_PRIMARY_LABELS:
        raise RuntimeError(f"B primary labels invalid: {dict(labels)}")

    tiers = Counter(str(record.get("supervision_tier")) for record in records)
    if tiers != B_PRIMARY_TIERS:
        raise RuntimeError(
            f"B primary supervision tiers changed: {dict(tiers)}"
        )

    ids = set()
    pairs = defaultdict(list)
    scenarios = set()
    per_label_hist = {0: Counter(), 1: Counter()}

    for record in records:
        validate_realized_trajectory(record, source="B-primary", max_turns=max_turns)
        cid = str(record.get("conversation_id", ""))
        if not cid or cid in ids:
            raise RuntimeError(f"B primary missing/duplicate conversation_id: {cid!r}")
        ids.add(cid)

        if record.get("validation_status") != "validated":
            raise RuntimeError(f"{cid}: B primary not validation_status=validated")
        if record.get("training_eligible") is not True:
            raise RuntimeError(f"{cid}: B primary is not training eligible")
        if record.get("primary_pair_complete") is not True:
            raise RuntimeError(f"{cid}: B primary missing primary_pair_complete=true")
        if record.get("canonical_target_model") != B_TARGET:
            raise RuntimeError(f"{cid}: B canonical target model changed")
        if record.get("canonical_judge_model") != B_JUDGE:
            raise RuntimeError(f"{cid}: B canonical judge model changed")

        pair_id = str(record.get("pair_id", "")).strip()
        if not pair_id:
            raise RuntimeError(f"{cid}: B primary missing pair_id")
        pairs[pair_id].append(record)

        scenario = b_scenario(record)
        if not scenario:
            raise RuntimeError(f"{cid}: B primary missing scenario_family")
        scenarios.add(scenario)

        label = int(record["label"])
        per_label_hist[label][n_user_turns(record)] += 1

    if len(pairs) != B_PRIMARY_PAIRS:
        raise RuntimeError(
            f"B primary expected {B_PRIMARY_PAIRS} pair IDs, got {len(pairs)}"
        )
    for pair_id, group in pairs.items():
        if (
            len(group) != 2
            or Counter(record.get("label") for record in group)
            != Counter({0: 1, 1: 1})
        ):
            raise RuntimeError(f"B primary pair {pair_id} is structurally invalid")

    if len(scenarios) != B_PRIMARY_SCENARIOS:
        raise RuntimeError(
            f"B primary expected {B_PRIMARY_SCENARIOS} scenarios, got {len(scenarios)}"
        )
    for label in (0, 1):
        if per_label_hist[label] != B_PRIMARY_USER_HIST_PER_LABEL:
            raise RuntimeError(
                f"B primary label={label} user-turn histogram changed: "
                f"{dict(per_label_hist[label])}"
            )


def validate_b_aux(records: Sequence[Dict], max_turns: int) -> None:
    if len(records) != B_AUX_RECORDS:
        raise RuntimeError(f"B auxiliary expected {B_AUX_RECORDS}, got {len(records)}")

    ids = set()
    detection = Counter()
    authoring = Counter()
    groups = set()
    hash_groups = defaultdict(set)

    for record in records:
        validate_realized_trajectory(record, source="B-aux", max_turns=max_turns)
        cid = str(record.get("conversation_id", ""))
        if not cid or cid in ids:
            raise RuntimeError(f"B auxiliary missing/duplicate conversation_id: {cid!r}")
        ids.add(cid)

        if record.get("corpus_source") != B_AUX_SOURCE:
            raise RuntimeError(
                f"{cid}: B auxiliary corpus_source={record.get('corpus_source')!r}"
            )
        if record.get("validation_status") != "rejected":
            raise RuntimeError(f"{cid}: B auxiliary must be B2-rejected")
        if record.get("training_eligible") is not True:
            raise RuntimeError(f"{cid}: B auxiliary not training eligible")
        if record.get("auxiliary_detection_only") is not True:
            raise RuntimeError(f"{cid}: B auxiliary flag missing")
        if record.get("use_as") != "auxiliary_detection_only":
            raise RuntimeError(f"{cid}: B auxiliary use_as mismatch")
        if record.get("primary_pair_complete") is not False:
            raise RuntimeError(f"{cid}: B auxiliary cannot claim primary-pair membership")
        if record.get("supervision_tier") != "auxiliary_detection":
            raise RuntimeError(f"{cid}: B auxiliary supervision tier mismatch")
        if float(record.get("loss_weight", -1)) != B_AUX_WEIGHT:
            raise RuntimeError(f"{cid}: B auxiliary loss_weight must be {B_AUX_WEIGHT}")
        if float(record.get("detection_loss_weight", -1)) != B_AUX_WEIGHT:
            raise RuntimeError(
                f"{cid}: B auxiliary detection_loss_weight must be {B_AUX_WEIGHT}"
            )
        if record.get("pivot_loss_weight") != 0.0:
            raise RuntimeError(f"{cid}: B auxiliary pivot loss must be 0")
        if record.get("span_loss_weight") != 0.0:
            raise RuntimeError(f"{cid}: B auxiliary span loss must be 0")
        if record.get("pivot_supervision_ignore") is not True:
            raise RuntimeError(f"{cid}: B auxiliary pivot supervision not masked")
        if record.get("localization_supervision_ignore") is not True:
            raise RuntimeError(f"{cid}: B auxiliary localization not masked")
        if record.get("pivot_turn_id") is not None:
            raise RuntimeError(f"{cid}: B auxiliary carries pivot target")
        if record.get("evidence_turn_ids") != []:
            raise RuntimeError(f"{cid}: B auxiliary carries evidence-turn targets")

        label = record.get("detection_label")
        author_label = record.get("authoring_intent_label")
        observed = record.get("observed_behavior_label")
        if isinstance(label, bool) or label not in (0, 1):
            raise RuntimeError(f"{cid}: invalid B auxiliary detection label")
        if observed != label:
            raise RuntimeError(f"{cid}: observed behavior label != detection label")
        if author_label != record.get("label"):
            raise RuntimeError(f"{cid}: authoring label provenance changed")
        detection[int(label)] += 1
        authoring[int(author_label)] += 1

        scenario = b_scenario(record)
        group = b_group(record)
        metadata = record.get("metadata", {}) or {}
        if not scenario or metadata.get("consolidated_split_group") != group:
            raise RuntimeError(f"{cid}: B auxiliary split group is not frontier::<scenario>")
        groups.add(group)

        expected_hash = normalized_user_hash(record)
        stored_hash = str(metadata.get("normalized_user_trajectory_hash", ""))
        if stored_hash != expected_hash:
            raise RuntimeError(f"{cid}: B auxiliary normalized user hash mismatch")
        hash_groups[expected_hash].add(group)

    if detection != B_AUX_DETECTION_LABELS:
        raise RuntimeError(
            f"B auxiliary detection labels changed: {dict(detection)}"
        )
    if authoring != B_AUX_AUTHORING_LABELS:
        raise RuntimeError(
            f"B auxiliary authoring labels changed: {dict(authoring)}"
        )
    if len(groups) != B_AUX_SCENARIOS:
        raise RuntimeError(
            f"B auxiliary expected {B_AUX_SCENARIOS} scenarios, got {len(groups)}"
        )
    cross_group = [h for h, owners in hash_groups.items() if len(owners) > 1]
    if cross_group:
        raise RuntimeError(
            "B auxiliary exact user trajectories occur across independent "
            f"scenario groups: {cross_group[:10]}"
        )


# -------------------------------
# Primary canonicalization
# -------------------------------

def canonicalize_a_primary(record: Dict) -> Dict:
    out = copy.deepcopy(record)
    pair_id = str(out.get("pair_id", "")).strip()
    out["corpus_source"] = "legacy_restored_primary"
    metadata = out.setdefault("metadata", {})
    metadata["consolidated_split_group"] = f"legacy::pair::{pair_id}"
    metadata["normalized_user_trajectory_hash"] = normalized_user_hash(out)
    metadata["nfkc_casefold_user_trajectory_hash"] = nfkc_casefold_user_hash(out)
    metadata["observable_turn_hash"] = observable_turn_hash(out)
    return out


def canonicalize_b_primary(record: Dict) -> Dict:
    out = copy.deepcopy(record)
    out["corpus_source"] = "frontier_authored_v3"
    metadata = out.setdefault("metadata", {})
    expected_group = b_group(out)
    existing = str(metadata.get("consolidated_split_group", "")).strip()
    if existing and existing != expected_group:
        raise RuntimeError(
            f"{out.get('conversation_id')}: B primary split group "
            f"{existing!r} != {expected_group!r}"
        )
    metadata["consolidated_split_group"] = expected_group
    metadata["normalized_user_trajectory_hash"] = normalized_user_hash(out)
    metadata["nfkc_casefold_user_trajectory_hash"] = nfkc_casefold_user_hash(out)
    metadata["observable_turn_hash"] = observable_turn_hash(out)
    return out


def assert_primary_cross_corpus_integrity(records: Sequence[Dict]) -> None:
    ids = Counter(str(record.get("conversation_id", "")) for record in records)
    duplicates = [cid for cid, count in ids.items() if not cid or count > 1]
    if duplicates:
        raise RuntimeError(f"primary missing/duplicate conversation IDs: {duplicates[:10]}")

    hash_groups = defaultdict(set)
    strong_hash_groups = defaultdict(set)
    for record in records:
        metadata = record.get("metadata", {}) or {}
        trajectory_hash = str(metadata.get("normalized_user_trajectory_hash", ""))
        strong_hash = nfkc_casefold_user_hash(record)
        group = str(metadata.get("consolidated_split_group", ""))
        if not trajectory_hash or not group:
            raise RuntimeError(
                f"{record.get('conversation_id')}: missing primary hash/group"
            )
        hash_groups[trajectory_hash].add(group)
        strong_hash_groups[strong_hash].add(group)

    cross_group = [h for h, groups in hash_groups.items() if len(groups) > 1]
    if cross_group:
        raise RuntimeError(
            "exact normalized primary user trajectories cross independent groups: "
            f"{cross_group[:10]}"
        )
    strong_cross_group = [
        h for h, groups in strong_hash_groups.items() if len(groups) > 1
    ]
    if strong_cross_group:
        raise RuntimeError(
            "NFKC/casefold-equivalent primary user trajectories cross independent "
            f"groups: {strong_cross_group[:10]}"
        )


# -------------------------------
# Primary-only split
# -------------------------------

def effective_label(record: Dict) -> int:
    value = (
        record.get("detection_label")
        if (
            record.get("auxiliary_detection_only") is True
            or record.get("use_as") == "auxiliary_detection_only"
        )
        else record.get("label")
    )
    if isinstance(value, bool) or value not in (0, 1):
        raise RuntimeError(
            f"{record.get('conversation_id')}: invalid effective detection label"
        )
    return int(value)


def is_frontier(record: Dict) -> bool:
    return str(record.get("corpus_source", "")).startswith("frontier_authored_v3")


def frontier_author(record: Dict) -> str:
    metadata = record.get("metadata", {}) or {}
    return str(
        metadata.get("corpus_version")
        or metadata.get("generator")
        or record.get("seed_source")
        or "unknown_source"
    )


def group_records(records: Sequence[Dict]) -> Dict[str, List[Dict]]:
    groups = defaultdict(list)
    for record in records:
        group = str(
            (record.get("metadata", {}) or {}).get("consolidated_split_group", "")
        ).strip()
        if not group:
            raise RuntimeError(
                f"{record.get('conversation_id')}: missing consolidated_split_group"
            )
        groups[group].append(record)
    return dict(groups)


def group_signature(group: Sequence[Dict]) -> Counter:
    signature = Counter()
    for record in group:
        source = str(record.get("corpus_source", "unknown"))
        label = str(effective_label(record))
        difficulty = str(record.get("difficulty", "unknown"))
        user_len = str(n_user_turns(record))

        signature[("label", label)] += 1
        signature[("source", source)] += 1
        signature[("source_label", source, label)] += 1
        signature[("source_difficulty", source, difficulty)] += 1
        signature[("source_label_user_turns", source, label, user_len)] += 1

        if is_frontier(record):
            metadata = record.get("metadata", {}) or {}
            intended = record.get("intended_structure", {}) or {}
            author = frontier_author(record)
            signature[("frontier_author", author)] += 1
            signature[("frontier_author_label", author, label)] += 1
            signature[("frontier_domain", str(record.get("target_domain", "unknown")))] += 1
            signature[("frontier_slice_role", str(metadata.get("slice_role", "unknown")))] += 1
            signature[("frontier_pair_hardness", str(intended.get("pair_hardness", "none")))] += 1
            signature[("frontier_trajectory_family", str(intended.get("trajectory_family", "unknown")))] += 1
            signature[("frontier_mechanism_family", str(metadata.get("mechanism_family", "unknown")))] += 1
            signature[("frontier_style", str(record.get("style", "unknown")))] += 1
        else:
            signature[("legacy_family", str(record.get("family", "unknown")))] += 1
    return signature


def split_primary(
    groups: Dict[str, List[Dict]],
    *,
    fractions: Dict[str, float],
    seed: int,
) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    total_records = sum(len(group) for group in groups.values())
    target_total = {
        split_name: total_records * fractions[split_name]
        for split_name in SPLITS
    }

    global_signature = Counter()
    for group in groups.values():
        global_signature.update(group_signature(group))
    target_signature = {
        split_name: {
            key: value * fractions[split_name]
            for key, value in global_signature.items()
        }
        for split_name in SPLITS
    }

    assigned = {split_name: [] for split_name in SPLITS}
    counts = {split_name: 0 for split_name in SPLITS}
    signature_counts = {split_name: Counter() for split_name in SPLITS}

    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)

    for group_id, group in items:
        group_sig = group_signature(group)
        best_split = None
        best_score = None

        for split_name in SPLITS:
            total_fill = (
                counts[split_name] + len(group)
            ) / max(target_total[split_name], 1.0)

            signature_fills = []
            for key, amount in group_sig.items():
                target = target_signature[split_name].get(key, 0.0)
                if target > 0:
                    signature_fills.append(
                        (signature_counts[split_name][key] + amount) / target
                    )
            signature_fill = (
                sum(signature_fills) / len(signature_fills)
                if signature_fills else total_fill
            )

            score = (
                0.72 * total_fill
                + 0.28 * signature_fill
                + rng.random() * 1e-9
            )
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name

        assigned[best_split].append((group_id, group))
        counts[best_split] += len(group)
        signature_counts[best_split].update(group_sig)

    output: Dict[str, List[Dict]] = {}
    for split_name in SPLITS:
        output[split_name] = [
            record
            for _, group in assigned[split_name]
            for record in group
        ]
        rng.shuffle(output[split_name])
    return output


def assert_primary_split_integrity(
    splits: Dict[str, List[Dict]],
    fractions: Dict[str, float],
    max_group_size: int,
) -> None:
    group_owner = {}
    pair_owner = {}
    scenario_owner = {}
    hash_owner = {}
    strong_hash_owner = {}
    ids = set()

    for split_name, records in splits.items():
        if not records:
            raise RuntimeError(f"primary split {split_name} is empty")

        for record in records:
            cid = str(record.get("conversation_id", ""))
            if cid in ids:
                raise RuntimeError(
                    f"duplicate primary conversation_id across splits: {cid}"
                )
            ids.add(cid)

            metadata = record.get("metadata", {}) or {}
            group = str(metadata.get("consolidated_split_group", "")).strip()
            prior = group_owner.setdefault(group, split_name)
            if prior != split_name:
                raise RuntimeError(
                    f"primary group leakage: {group} in {prior} and {split_name}"
                )

            trajectory_hash = str(
                metadata.get("normalized_user_trajectory_hash", "")
            ).strip()
            prior = hash_owner.setdefault(trajectory_hash, split_name)
            if prior != split_name:
                raise RuntimeError(
                    "exact primary user-trajectory leakage across splits"
                )
            strong_hash = nfkc_casefold_user_hash(record)
            prior = strong_hash_owner.setdefault(strong_hash, split_name)
            if prior != split_name:
                raise RuntimeError(
                    "NFKC/casefold primary user-trajectory leakage across splits"
                )

            pair_id = str(record.get("pair_id", "")).strip()
            pair_key = (str(record.get("corpus_source")), pair_id)
            prior = pair_owner.setdefault(pair_key, split_name)
            if prior != split_name:
                raise RuntimeError(
                    f"primary pair leakage: {pair_key} in {prior} and {split_name}"
                )

            if is_frontier(record):
                scenario = b_scenario(record)
                prior = scenario_owner.setdefault(scenario, split_name)
                if prior != split_name:
                    raise RuntimeError(
                        f"B scenario leakage: {scenario} in {prior} and {split_name}"
                    )

    total = sum(len(records) for records in splits.values())
    tolerance = max_group_size / total + 0.005
    for split_name in SPLITS:
        actual = len(splits[split_name]) / total
        if abs(actual - fractions[split_name]) > tolerance:
            raise RuntimeError(
                f"{split_name} primary ratio={actual:.4f} differs from target "
                f"{fractions[split_name]:.4f} beyond tolerance={tolerance:.4f}"
            )


# -------------------------------
# Auxiliary attachment
# -------------------------------

def index_primary_ownership(
    splits: Dict[str, List[Dict]],
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], set]:
    group_owner: Dict[str, str] = {}
    hash_owner: Dict[str, str] = {}
    strong_hash_owner: Dict[str, str] = {}
    ids = set()

    for split_name, records in splits.items():
        for record in records:
            cid = str(record.get("conversation_id", ""))
            if cid in ids:
                raise RuntimeError(f"duplicate primary ID while indexing: {cid}")
            ids.add(cid)

            metadata = record.get("metadata", {}) or {}
            group = str(metadata.get("consolidated_split_group", "")).strip()
            trajectory_hash = str(
                metadata.get("normalized_user_trajectory_hash", "")
            ).strip()
            if not group or not trajectory_hash:
                raise RuntimeError(f"{cid}: primary group/hash missing")

            prior = group_owner.setdefault(group, split_name)
            if prior != split_name:
                raise RuntimeError(
                    f"primary group ownership conflict for {group}"
                )
            prior = hash_owner.setdefault(trajectory_hash, split_name)
            if prior != split_name:
                raise RuntimeError(
                    "primary exact trajectory ownership conflict"
                )
            strong_hash = nfkc_casefold_user_hash(record)
            prior = strong_hash_owner.setdefault(strong_hash, split_name)
            if prior != split_name:
                raise RuntimeError(
                    "primary NFKC/casefold trajectory ownership conflict"
                )

    return group_owner, hash_owner, strong_hash_owner, ids


def canonicalize_a_aux(record: Dict) -> Dict:
    out = copy.deepcopy(record)
    out["corpus_source"] = "legacy_detection_aux"
    metadata = out.setdefault("metadata", {})
    metadata["consolidated_split_group"] = (
        f"legacy_aux::conversation::{out.get('conversation_id')}"
    )
    metadata["normalized_user_trajectory_hash"] = normalized_user_hash(out)
    metadata["nfkc_casefold_user_trajectory_hash"] = nfkc_casefold_user_hash(out)
    metadata["observable_turn_hash"] = observable_turn_hash(out)
    return out


def attach_auxiliary(
    splits: Dict[str, List[Dict]],
    a_aux: Sequence[Dict],
    b_aux: Sequence[Dict],
) -> Tuple[Dict[str, List[Dict]], List[Dict], List[Dict], Dict]:
    group_owner, hash_owner, strong_hash_owner, primary_ids = index_primary_ownership(splits)

    # A auxiliary is independently generated and intentionally train-only.
    # Any exact overlap with primary is unexpected and therefore fail-closed.
    seen_ids = set(primary_ids)
    primary_hashes = set(hash_owner)
    primary_strong_hashes = set(strong_hash_owner)
    a_aux_hashes = set()
    a_aux_strong_hashes = set()
    included_a: List[Dict] = []

    for original in a_aux:
        record = canonicalize_a_aux(original)
        cid = str(record.get("conversation_id", ""))
        trajectory_hash = normalized_user_hash(record)
        strong_hash = nfkc_casefold_user_hash(record)

        if cid in seen_ids:
            raise RuntimeError(
                f"A auxiliary conversation_id overlaps primary material: {cid}"
            )
        if trajectory_hash in primary_hashes:
            raise RuntimeError(
                f"A auxiliary exact user trajectory overlaps primary material: {cid}"
            )
        if strong_hash in primary_strong_hashes:
            raise RuntimeError(
                f"A auxiliary NFKC/casefold trajectory overlaps primary material: {cid}"
            )
        if trajectory_hash in a_aux_hashes or strong_hash in a_aux_strong_hashes:
            raise RuntimeError(
                f"A auxiliary contains duplicate/equivalent user trajectory: {cid}"
            )
        seen_ids.add(cid)
        a_aux_hashes.add(trajectory_hash)
        a_aux_strong_hashes.add(strong_hash)
        included_a.append(record)

    # B auxiliary follows the frozen optimized-branch attachment policy.
    included_b: List[Dict] = []
    withheld_b: List[Dict] = []
    disposition = Counter()
    included_b_labels = Counter()
    included_b_groups = set()

    for original in b_aux:
        record = copy.deepcopy(original)
        cid = str(record.get("conversation_id", ""))
        if cid in seen_ids:
            raise RuntimeError(
                f"B auxiliary conversation_id collides with primary/A auxiliary: {cid}"
            )
        seen_ids.add(cid)

        metadata = record.get("metadata", {}) or {}
        group = str(metadata.get("consolidated_split_group", "")).strip()
        trajectory_hash = str(
            metadata.get("normalized_user_trajectory_hash", "")
        ).strip()
        strong_hash = nfkc_casefold_user_hash(record)
        if not group or not trajectory_hash:
            raise RuntimeError(f"{cid}: B auxiliary missing frozen group/hash")

        # Cross-A/B exact auxiliary duplication is an independent-corpus defect.
        if trajectory_hash in a_aux_hashes or strong_hash in a_aux_strong_hashes:
            raise RuntimeError(
                f"B auxiliary exact/equivalent trajectory duplicates A auxiliary: {cid}"
            )

        owner = group_owner.get(group)
        exact_owner = hash_owner.get(trajectory_hash)
        strong_owner = strong_hash_owner.get(strong_hash)
        reason = None
        if owner in {"dev", "test"}:
            reason = f"primary_{owner}_family"
        elif exact_owner in {"dev", "test"}:
            reason = f"primary_{exact_owner}_exact_user_trajectory"
        elif strong_owner in {"dev", "test"}:
            reason = f"primary_{strong_owner}_nfkc_casefold_user_trajectory"

        if reason is not None:
            withheld = copy.deepcopy(record)
            withheld["auxiliary_withheld_reason"] = reason
            withheld_b.append(withheld)
            disposition[f"withheld_{reason}"] += 1
            continue

        included_b.append(record)
        included_b_labels[int(record["detection_label"])] += 1
        included_b_groups.add(group)
        disposition[
            "included_primary_train_family"
            if owner == "train"
            else "included_aux_only_family"
        ] += 1

    train_with_aux = (
        list(splits["train"])
        + included_a
        + included_b
    )

    return (
        {
            "train": train_with_aux,
            "dev": list(splits["dev"]),
            "test": list(splits["test"]),
        },
        included_b,
        withheld_b,
        {
            "primary_records": {
                split_name: len(splits[split_name])
                for split_name in SPLITS
            },
            "a_auxiliary_input": len(a_aux),
            "a_auxiliary_included": len(included_a),
            "b_auxiliary_input": len(b_aux),
            "b_auxiliary_included": len(included_b),
            "b_auxiliary_withheld": len(withheld_b),
            "b_auxiliary_included_detection_labels": dict(
                sorted(included_b_labels.items())
            ),
            "b_auxiliary_included_groups": len(included_b_groups),
            "b_auxiliary_disposition": dict(sorted(disposition.items())),
            "policy": (
                "freeze primary split; add all non-overlapping independent A "
                "auxiliary to train; add B auxiliary only when scenario-family "
                "and exact-user-trajectory ownership do not belong to primary "
                "dev/test"
            ),
            "dev_test_primary_only": True,
            "dev_test_copy_policy": "byte_for_byte_from_frozen_primary_inputs",
        },
    )


# -------------------------------
# Shortcut diagnostics
# -------------------------------

def detection_label(record: Dict) -> int:
    if (
        record.get("auxiliary_detection_only") is True
        or record.get("use_as") == "auxiliary_detection_only"
    ):
        value = record.get("detection_label")
    else:
        value = record.get("label")
    if isinstance(value, bool) or value not in (0, 1):
        raise RuntimeError(
            f"{record.get('conversation_id')}: invalid effective detection label"
        )
    return int(value)


def detection_weight(record: Dict) -> float:
    if (
        record.get("auxiliary_detection_only") is True
        or record.get("use_as") == "auxiliary_detection_only"
    ):
        value = record.get("detection_loss_weight")
    else:
        value = 1.0
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise RuntimeError(
            f"{record.get('conversation_id')}: invalid detection weight={value!r}"
        )
    return float(value)


def turn_auc(records: Sequence[Dict], *, weighted: bool) -> float:
    positives = []
    negatives = []
    for record in records:
        item = (
            n_user_turns(record),
            detection_weight(record) if weighted else 1.0,
        )
        if detection_label(record) == 1:
            positives.append(item)
        else:
            negatives.append(item)

    if not positives or not negatives:
        raise RuntimeError("turn-count AUC requires both detection classes")
    pos_mass = sum(weight for _, weight in positives)
    neg_mass = sum(weight for _, weight in negatives)

    credit = 0.0
    for p_value, p_weight in positives:
        for n_value, n_weight in negatives:
            if p_value > n_value:
                pair_credit = 1.0
            elif p_value == n_value:
                pair_credit = 0.5
            else:
                pair_credit = 0.0
            credit += pair_credit * p_weight * n_weight

    return credit / (pos_mass * neg_mass)


def threshold_diagnostic(
    records: Sequence[Dict],
    threshold: int,
    *,
    weighted: bool,
) -> Dict:
    pos_mass = neg_mass = tp = tn = 0.0

    for record in records:
        label = detection_label(record)
        weight = detection_weight(record) if weighted else 1.0
        pred = int(n_user_turns(record) > threshold)

        if label == 1:
            pos_mass += weight
            if pred == 1:
                tp += weight
        else:
            neg_mass += weight
            if pred == 0:
                tn += weight

    if pos_mass <= 0 or neg_mass <= 0:
        raise RuntimeError("threshold diagnostic requires both classes")

    tpr = tp / pos_mass
    tnr = tn / neg_mass
    return {
        "rule": f"predict malicious iff n_user_turns > {threshold}",
        "threshold": threshold,
        "tpr": tpr,
        "tnr": tnr,
        "balanced_accuracy": 0.5 * (tpr + tnr),
    }


def source_family(record: Dict) -> str:
    source = str(record.get("corpus_source", ""))
    if source in {"legacy_restored_primary", "legacy_detection_aux"}:
        return "A"
    if source in {B_AUX_SOURCE, "frontier_authored_v3"}:
        return "B"
    return source or "unknown"


def diagnostic_block(records: Sequence[Dict]) -> Dict:
    labels = Counter(detection_label(record) for record in records)
    result = {
        "records": len(records),
        "labels": dict(sorted(labels.items())),
        "mean_user_turns": {
            str(label): (
                sum(
                    n_user_turns(record)
                    for record in records
                    if detection_label(record) == label
                )
                / labels[label]
            )
            for label in sorted(labels)
        },
        "max_physical_turns": max(len(record.get("turns", [])) for record in records),
    }

    if labels.get(0, 0) and labels.get(1, 0):
        result.update({
            "turn_count_auc_unweighted": turn_auc(records, weighted=False),
            "turn_count_auc_detection_weighted": turn_auc(records, weighted=True),
            "gt_10_unweighted": threshold_diagnostic(
                records, 10, weighted=False
            ),
            "gt_10_detection_weighted": threshold_diagnostic(
                records, 10, weighted=True
            ),
        })

    return result


def describe_training_view(records: Sequence[Dict]) -> Dict:
    result = diagnostic_block(records)
    result["corpus_source"] = dict(
        Counter(str(record.get("corpus_source", "unknown")) for record in records)
    )
    result["auxiliary_records"] = sum(
        1
        for record in records
        if (
            record.get("auxiliary_detection_only") is True
            or record.get("use_as") == "auxiliary_detection_only"
        )
    )
    result["source_diagnostics"] = {
        family: diagnostic_block(
            [record for record in records if source_family(record) == family]
        )
        for family in sorted({source_family(record) for record in records})
    }
    return result


def describe_primary_test_structure(records: Sequence[Dict]) -> Dict:
    """Structural provenance only. Do not compute shortcut metrics on held-out test."""
    return {
        "records": len(records),
        "labels": dict(
            Counter(str(record.get("label")) for record in records)
        ),
        "corpus_source": dict(
            Counter(str(record.get("corpus_source", "unknown")) for record in records)
        ),
        "source_label": dict(
            Counter(
                f"{record.get('corpus_source')}|{record.get('label')}"
                for record in records
            )
        ),
        "groups": len({
            (record.get("metadata", {}) or {}).get("consolidated_split_group")
            for record in records
        }),
        "shortcut_diagnostics_computed": False,
        "semantic_inspection_performed": False,
    }


def describe_primary_split(records: Sequence[Dict]) -> Dict:
    result = diagnostic_block(records)
    result["corpus_source"] = dict(
        Counter(str(record.get("corpus_source", "unknown")) for record in records)
    )
    result["source_label"] = dict(
        Counter(
            f"{record.get('corpus_source')}|{record.get('label')}"
            for record in records
        )
    )
    result["groups"] = len({
        (record.get("metadata", {}) or {}).get("consolidated_split_group")
        for record in records
    })
    result["source_diagnostics"] = {
        family: diagnostic_block(
            [record for record in records if source_family(record) == family]
        )
        for family in sorted({source_family(record) for record in records})
    }
    return result


# -------------------------------
# Main
# -------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-primary", required=True)
    parser.add_argument("--a-aux", required=True)
    parser.add_argument("--b-primary", required=True)
    parser.add_argument("--b-aux", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--dev-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    args = parser.parse_args()

    fractions = {
        "train": args.train_frac,
        "dev": args.dev_frac,
        "test": args.test_frac,
    }
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
        raise RuntimeError("train/dev/test fractions must sum to 1")
    if min(fractions.values()) <= 0:
        raise RuntimeError("all split fractions must be positive")
    if args.max_turns <= 0:
        raise RuntimeError("--max-turns must be positive")

    # Verify code and immutable input identities before parsing or writing anything.
    code_provenance = verify_code_checkout()
    input_hashes = {
        "a_primary": require_sha(args.a_primary, A_PRIMARY_SHA256, "A primary"),
        "a_aux": require_sha(args.a_aux, A_AUX_SHA256, "A auxiliary"),
        "b_primary": require_sha(args.b_primary, B_PRIMARY_SHA256, "B primary"),
        "b_aux": require_sha(args.b_aux, B_AUX_SHA256, "B auxiliary"),
    }

    a_primary = load_jsonl(args.a_primary)
    a_aux = load_jsonl(args.a_aux)
    b_primary = load_jsonl(args.b_primary)
    b_aux = load_jsonl(args.b_aux)

    validate_a_primary(a_primary, args.max_turns)
    validate_a_aux(a_aux, args.max_turns)
    validate_b_primary(b_primary, args.max_turns)
    validate_b_aux(b_aux, args.max_turns)

    primary = [
        canonicalize_a_primary(record) for record in a_primary
    ] + [
        canonicalize_b_primary(record) for record in b_primary
    ]

    if len(primary) != 2434:
        raise RuntimeError(f"consolidated primary expected 2434, got {len(primary)}")
    labels = Counter(record.get("label") for record in primary)
    if labels != Counter({0: 1217, 1: 1217}):
        raise RuntimeError(
            f"consolidated primary must be 1217/1217 balanced, got {dict(labels)}"
        )

    assert_primary_cross_corpus_integrity(primary)
    groups = group_records(primary)
    splits = split_primary(groups, fractions=fractions, seed=args.seed)
    assert_primary_split_integrity(
        splits,
        fractions,
        max_group_size=max(len(group) for group in groups.values()),
    )

    candidate_splits, included_b_aux, withheld_b_aux, attachment = attach_auxiliary(
        splits,
        a_aux,
        b_aux,
    )

    # Deterministic training order after attachment.
    rng = random.Random(args.seed)
    rng.shuffle(candidate_splits["train"])

    ensure_fresh_output_dir(args.output_dir)
    primary_dir = os.path.join(args.output_dir, "splits_primary")
    candidate_dir = os.path.join(
        args.output_dir, "splits_primary_plus_train_auxiliary"
    )
    os.makedirs(primary_dir, exist_ok=False)
    os.makedirs(candidate_dir, exist_ok=False)

    primary_all_path = os.path.join(args.output_dir, "primary_all.jsonl")
    primary_train_path = os.path.join(primary_dir, "train.jsonl")
    primary_dev_path = os.path.join(primary_dir, "dev.jsonl")
    primary_test_path = os.path.join(primary_dir, "test.jsonl")

    candidate_train_path = os.path.join(candidate_dir, "train.jsonl")
    candidate_dev_path = os.path.join(candidate_dir, "dev.jsonl")
    candidate_test_path = os.path.join(candidate_dir, "test.jsonl")

    a_aux_path = os.path.join(args.output_dir, "a_detection_aux_721.jsonl")
    b_aux_input_path = os.path.join(args.output_dir, "b_detection_aux_512.jsonl")
    b_aux_included_path = os.path.join(
        args.output_dir, "b_detection_aux_included_train.jsonl"
    )
    b_aux_withheld_path = os.path.join(
        args.output_dir, "b_detection_aux_withheld.jsonl"
    )

    write_jsonl(primary, primary_all_path)
    write_jsonl(splits["train"], primary_train_path)
    write_jsonl(splits["dev"], primary_dev_path)
    write_jsonl(splits["test"], primary_test_path)

    write_jsonl(candidate_splits["train"], candidate_train_path)
    shutil.copyfile(primary_dev_path, candidate_dev_path)
    shutil.copyfile(primary_test_path, candidate_test_path)

    write_jsonl(
        [canonicalize_a_aux(record) for record in a_aux],
        a_aux_path,
    )
    write_jsonl(b_aux, b_aux_input_path)
    write_jsonl(included_b_aux, b_aux_included_path)
    write_jsonl(withheld_b_aux, b_aux_withheld_path)

    if sha256_file(primary_dev_path) != sha256_file(candidate_dev_path):
        raise RuntimeError("candidate dev is not byte-identical to primary dev")
    if sha256_file(primary_test_path) != sha256_file(candidate_test_path):
        raise RuntimeError("candidate test is not byte-identical to primary test")

    artifact_paths = {
        "primary_all": primary_all_path,
        "primary_train": primary_train_path,
        "primary_dev": primary_dev_path,
        "primary_test": primary_test_path,
        "auxiliary_candidate_train": candidate_train_path,
        "auxiliary_candidate_dev": candidate_dev_path,
        "auxiliary_candidate_test": candidate_test_path,
        "a_auxiliary": a_aux_path,
        "b_auxiliary_input": b_aux_input_path,
        "b_auxiliary_included": b_aux_included_path,
        "b_auxiliary_withheld": b_aux_withheld_path,
    }
    artifact_hashes = {
        name: sha256_file(path) for name, path in artifact_paths.items()
    }

    primary_descriptions = {
        "train": describe_primary_split(splits["train"]),
        "dev": describe_primary_split(splits["dev"]),
        "test": describe_primary_test_structure(splits["test"]),
    }
    training_description = describe_training_view(candidate_splits["train"])

    manifest = {
        "status": "passed",
        "code_provenance": code_provenance,
        "input_sha256": input_hashes,
        "counts": {
            "primary_all": len(primary),
            "a_primary": len(a_primary),
            "b_primary": len(b_primary),
            "a_auxiliary_input": len(a_aux),
            "b_auxiliary_input": len(b_aux),
            "primary_splits": {
                split_name: len(splits[split_name])
                for split_name in SPLITS
            },
            "candidate_train": len(candidate_splits["train"]),
            "b_auxiliary_included": len(included_b_aux),
            "b_auxiliary_withheld": len(withheld_b_aux),
        },
        "primary_splits": primary_descriptions,
        "train_with_auxiliary": training_description,
        "auxiliary_attachment": attachment,
        "policy": {
            "primary_split_before_auxiliary_attachment": True,
            "a_primary_group": "legacy::pair::<generation-time pair_id>",
            "b_primary_group": "frontier::<scenario_family>",
            "a_auxiliary": "all 721 independent validated benign records train-only after exact-overlap rejection",
            "b_auxiliary": (
                "all 512 frozen rejected-outcome records considered; withhold "
                "records owned by primary dev/test scenario family or exact "
                "user trajectory; include primary-train-family and aux-only-family "
                "records in train"
            ),
            "dev_test_primary_only": True,
            "dev_test_candidate_copy": "byte-for-byte",
            "max_turns": args.max_turns,
            "seed": args.seed,
            "fractions": fractions,
            "length_shortcut_policy": "train_dev_diagnostic_only_no_arbitrary_auc_gate",
            "held_out_test_shortcut_diagnostics_computed": False,
            "held_out_test_semantically_inspected": False,
        },
        "artifact_sha256": artifact_hashes,
    }

    # Detailed review artifact.
    write_json(manifest, os.path.join(args.output_dir, "freeze_manifest.json"))

    # Exact schema consumed by GuardLens-Transformer/guardlens.data.verify_freeze.
    freeze_report = {
        "status": "passed",
        "artifact_sha256": {
            "primary_train": artifact_hashes["primary_train"],
            "primary_dev": artifact_hashes["primary_dev"],
            "primary_test": artifact_hashes["primary_test"],
            "auxiliary_candidate_train": artifact_hashes["auxiliary_candidate_train"],
            "auxiliary_candidate_dev": artifact_hashes["auxiliary_candidate_dev"],
            "auxiliary_candidate_test": artifact_hashes["auxiliary_candidate_test"],
        },
        "counts": {
            "primary_splits": {
                split_name: len(splits[split_name])
                for split_name in SPLITS
            },
            "primary_all": len(primary),
        },
        "auxiliary_candidate": {
            "train_records": len(candidate_splits["train"]),
            "dev_records": len(splits["dev"]),
            "test_records": len(splits["test"]),
            "a_auxiliary_records_in_train": len(a_aux),
            "b_auxiliary_input_records": len(b_aux),
            "b_auxiliary_records_in_train": len(included_b_aux),
            "b_auxiliary_records_withheld": len(withheld_b_aux),
            "dev_byte_identical_to_primary": True,
            "test_byte_identical_to_primary": True,
            "attachment_metadata": attachment,
        },
        "code_provenance": code_provenance,
        "input_sha256": input_hashes,
        "shortcut_diagnostics": {
            "primary_train": primary_descriptions["train"],
            "primary_dev": primary_descriptions["dev"],
            "train_with_auxiliary": training_description,
            "held_out_test": {
                "computed": False,
                "semantic_inspection_performed": False,
            },
        },
        "policy": manifest["policy"],
    }
    write_json(
        freeze_report,
        os.path.join(args.output_dir, "data_prep_freeze_report.json"),
    )
    write_json(
        attachment,
        os.path.join(candidate_dir, "auxiliary_attachment_metadata.json"),
    )
    with open(
        os.path.join(args.output_dir, "data_prep_code_commit.txt"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(code_provenance["commit"] + "\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
