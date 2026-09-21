#!/usr/bin/env python3
"""Locate frozen Dataset B artifacts by contract, not by filename.

The B primary is discovered as the canonical 1,402-row / 701-pair frontier
artifact. B auxiliary may exist either as a standalone 424-row file or only
embedded in the historically frozen 2,144-row primary+auxiliary train file.
Identical copies across roots are deduplicated by SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List

from prepare_frontier_dataset import assert_expected_provenance


PRIMARY_N = 1402
PAIR_N = 701
AUX_N = 424
TARGET = "Qwen/Qwen2.5-32B-Instruct"
JUDGE = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

# Previously frozen A+B primary train and the optional B-auxiliary candidate
# documented in GuardLens-Transformer/NAACL_TRAINING.md.
FROZEN_OLD_PRIMARY_TRAIN_N = 1720
FROZEN_OLD_PRIMARY_TRAIN_SHA256 = (
    "7ba829d3acd0699b76012c477426504301694e215a7755e97341b196d5bf9011"
)
FROZEN_OLD_AUX_TRAIN_N = 2144
FROZEN_OLD_AUX_TRAIN_SHA256 = (
    "3533198efdb55fe33087170c50d62d301969b841422e283c5ebdc71f10c7490f"
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str) -> List[Dict]:
    rows = []
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


def primary_contract(rows: List[Dict]) -> tuple[bool, str]:
    if len(rows) != PRIMARY_N:
        return False, f"n={len(rows)}"
    if Counter(r.get("label") for r in rows) != Counter({0: PAIR_N, 1: PAIR_N}):
        return False, "label counts"

    pairs = defaultdict(list)
    try:
        for record in rows:
            cid = str(record.get("conversation_id", ""))
            if not cid:
                return False, "missing conversation_id"
            if record.get("training_eligible") is not True:
                return False, "training-ineligible primary"
            if record.get("primary_pair_complete") is not True:
                return False, "primary_pair_complete missing"

            pair_id = str(record.get("pair_id", ""))
            if not pair_id:
                return False, "missing pair_id"
            pairs[pair_id].append(record)

            metadata = record.get("metadata", {}) or {}
            if not metadata.get("scenario_family"):
                return False, "missing scenario_family"

            assert_expected_provenance(
                record,
                expected_target=TARGET,
                expected_judge=JUDGE,
                require_evidence=(record.get("label") == 1),
            )
            if record.get("canonical_target_model") != TARGET:
                return False, "target mismatch"
            if record.get("canonical_judge_model") != JUDGE:
                return False, "judge mismatch"
    except Exception as exc:
        return False, f"provenance: {exc}"

    if len(pairs) != PAIR_N:
        return False, f"pair count={len(pairs)}"
    for group in pairs.values():
        if (
            len(group) != 2
            or Counter(r.get("label") for r in group)
            != Counter({0: 1, 1: 1})
        ):
            return False, "invalid pair structure"
    return True, "canonical_B_primary"


def aux_contract(rows: List[Dict]) -> tuple[bool, str]:
    if len(rows) != AUX_N:
        return False, f"n={len(rows)}"

    labels = Counter()
    ids = set()
    for record in rows:
        cid = str(record.get("conversation_id", ""))
        if not cid:
            return False, "missing conversation_id"
        if cid in ids:
            return False, "duplicate conversation_id"
        ids.add(cid)

        if record.get("auxiliary_detection_only") is not True and (
            record.get("use_as") != "auxiliary_detection_only"
        ):
            return False, "not detection-only"

        label = record.get("detection_label")
        if isinstance(label, bool) or label not in (0, 1):
            return False, "invalid detection_label"
        labels[int(label)] += 1

        weight = record.get("detection_loss_weight")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            return False, "invalid detection_loss_weight"
        if abs(float(weight) - 0.25) > 1e-12:
            return False, f"noncanonical weight={weight}"

        if record.get("localization_supervision_ignore") is not True:
            return False, "localization not ignored"
        if record.get("pivot_supervision_ignore") is not True:
            return False, "pivot not ignored"
        if (
            record.get("pivot_loss_weight") != 0.0
            or record.get("span_loss_weight") != 0.0
        ):
            return False, "nonzero localization loss"

    if labels != Counter({0: 271, 1: 153}):
        return False, f"labels={dict(labels)}"
    return True, "canonical_B_detection_aux"


def discover_files(roots: List[str]) -> List[str]:
    candidates = []
    for root in roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".jsonl"):
                    candidates.append(os.path.join(dirpath, filename))
    return sorted(set(candidates))


def dedupe_matches(matches: List[Dict]) -> List[Dict]:
    grouped = {}
    for item in matches:
        key = (item["kind"], item["sha256"])
        existing = grouped.setdefault(
            key,
            {
                "kind": item["kind"],
                "sha256": item["sha256"],
                "records": item["records"],
                "contract": item["contract"],
                "paths": [],
            },
        )
        existing["paths"].append(item["path"])
    for item in grouped.values():
        item["paths"] = sorted(set(item["paths"]))
    return sorted(
        grouped.values(),
        key=lambda item: (item["kind"], item["sha256"]),
    )


def extract_embedded_aux(
    primary_train_path: str,
    candidate_train_path: str,
) -> List[Dict]:
    primary = load_jsonl(primary_train_path)
    candidate = load_jsonl(candidate_train_path)

    if len(primary) != FROZEN_OLD_PRIMARY_TRAIN_N:
        raise RuntimeError("historical primary train count mismatch")
    if len(candidate) != FROZEN_OLD_AUX_TRAIN_N:
        raise RuntimeError("historical auxiliary-candidate train count mismatch")

    primary_by_id = {
        str(record.get("conversation_id", "")): record for record in primary
    }
    if len(primary_by_id) != len(primary):
        raise RuntimeError("historical primary train has duplicate/missing IDs")

    candidate_by_id = {
        str(record.get("conversation_id", "")): record for record in candidate
    }
    if len(candidate_by_id) != len(candidate):
        raise RuntimeError("historical candidate train has duplicate/missing IDs")

    missing = sorted(set(primary_by_id) - set(candidate_by_id))
    if missing:
        raise RuntimeError(
            f"auxiliary candidate does not contain all frozen primary train rows: "
            f"{missing[:10]}"
        )

    changed = [
        cid for cid, record in primary_by_id.items()
        if candidate_by_id[cid] != record
    ]
    if changed:
        raise RuntimeError(
            "primary rows changed inside historical auxiliary candidate: "
            f"{changed[:10]}"
        )

    aux_ids = sorted(set(candidate_by_id) - set(primary_by_id))
    auxiliary = [candidate_by_id[cid] for cid in aux_ids]
    ok, reason = aux_contract(auxiliary)
    if not ok:
        raise RuntimeError(
            f"424-row complement fails B auxiliary contract: {reason}"
        )
    return auxiliary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        nargs="*",
        default=["results-naacl"],
        help="Directories to scan recursively for JSONL artifacts.",
    )
    parser.add_argument(
        "--extract-b-aux",
        default="",
        help=(
            "If no standalone 424-row B auxiliary exists but the exact frozen "
            "1720/2144 train pair is found, write the verified 424-row complement "
            "to this path."
        ),
    )
    args = parser.parse_args()

    files = discover_files(args.roots)
    raw_matches = []
    old_primary_train_paths = []
    old_aux_train_paths = []

    for path in files:
        try:
            size = sum(
                1 for line in open(path, "r", encoding="utf-8") if line.strip()
            )
        except Exception:
            continue

        if size not in {
            PRIMARY_N,
            AUX_N,
            FROZEN_OLD_PRIMARY_TRAIN_N,
            FROZEN_OLD_AUX_TRAIN_N,
        }:
            continue

        sha = sha256_file(path)

        if size == FROZEN_OLD_PRIMARY_TRAIN_N and sha == FROZEN_OLD_PRIMARY_TRAIN_SHA256:
            old_primary_train_paths.append(path)
            continue
        if size == FROZEN_OLD_AUX_TRAIN_N and sha == FROZEN_OLD_AUX_TRAIN_SHA256:
            old_aux_train_paths.append(path)
            continue

        try:
            rows = load_jsonl(path)
        except Exception:
            continue

        is_primary, primary_reason = primary_contract(rows)
        if is_primary:
            raw_matches.append({
                "kind": "B_PRIMARY",
                "path": path,
                "records": len(rows),
                "sha256": sha,
                "contract": primary_reason,
            })
            continue

        is_aux, aux_reason = aux_contract(rows)
        if is_aux:
            raw_matches.append({
                "kind": "B_AUX",
                "path": path,
                "records": len(rows),
                "sha256": sha,
                "contract": aux_reason,
            })

    matches = dedupe_matches(raw_matches)
    primary_matches = [x for x in matches if x["kind"] == "B_PRIMARY"]
    aux_matches = [x for x in matches if x["kind"] == "B_AUX"]

    embedded = None
    if not aux_matches and old_primary_train_paths and old_aux_train_paths:
        # All matching copies are exact frozen hashes. One representative pair
        # is enough to reconstruct and verify the complement.
        auxiliary = extract_embedded_aux(
            sorted(old_primary_train_paths)[0],
            sorted(old_aux_train_paths)[0],
        )
        embedded = {
            "kind": "B_AUX_EMBEDDED_IN_FROZEN_TRAIN",
            "records": len(auxiliary),
            "contract": "canonical_B_detection_aux",
            "source_primary_train_paths": sorted(old_primary_train_paths),
            "source_aux_candidate_train_paths": sorted(old_aux_train_paths),
        }
        if args.extract_b_aux:
            write_jsonl(auxiliary, args.extract_b_aux)
            embedded["extracted_path"] = args.extract_b_aux
            embedded["sha256"] = sha256_file(args.extract_b_aux)
            # Treat verified extraction as the canonical standalone match.
            aux_matches = [{
                "kind": "B_AUX",
                "records": len(auxiliary),
                "contract": "canonical_B_detection_aux_extracted_from_frozen_train",
                "paths": [args.extract_b_aux],
                "sha256": embedded["sha256"],
            }]

    payload = {
        "matches": primary_matches + aux_matches,
        "embedded_auxiliary": embedded,
        "duplicate_copies_deduplicated_by_sha256": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if len(primary_matches) != 1:
        raise SystemExit(
            "Expected exactly one unique canonical B_PRIMARY content hash; "
            f"found {len(primary_matches)}."
        )
    if len(aux_matches) != 1:
        if embedded and not args.extract_b_aux:
            raise SystemExit(
                "Canonical B auxiliary exists only inside the frozen 2144-row "
                "train candidate. Rerun with --extract-b-aux PATH to materialize "
                "the verified 424-row complement."
            )
        raise SystemExit(
            "Expected exactly one unique canonical B_AUX content hash; "
            f"found {len(aux_matches)}."
        )


if __name__ == "__main__":
    main()
