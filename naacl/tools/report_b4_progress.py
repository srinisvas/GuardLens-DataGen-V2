#!/usr/bin/env python3
"""Report committed B4 supervision after an allocation, without changing execution state.

Standard-library only. Kept outside naacl/*.py to preserve executor code hashes.
Snapshots describe reporting intervals, not inferred record completion timestamps.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def option(command, name):
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Saved executor command lacks {name}") from exc


def metrics(record):
    analysis = record["frontier_evidence_analysis"]
    status = analysis["status"]
    if status not in {"complete", "not_applicable", "baseline_not_unsafe"}:
        raise ValueError("Nonterminal B4 record in completed results")
    result = Counter(records=1)
    result[status] += 1
    if status == "not_applicable":
        return dict(result)
    tier = record.get("supervision_tier")
    if tier not in {"cf_strong", "cf_weak", "llm_confirmed", "ignore"}:
        raise ValueError(f"Unknown record supervision tier: {tier!r}")
    result[tier + "_records"] += 1
    result["training_eligible_records"] += record.get("training_eligible") is True
    result["supported_evidence_turn_records"] += bool(record.get("evidence_turn_ids"))
    context_count = 0
    for item in analysis.get("turn_interventions", []):
        status = item.get("status")
        if status:
            result["turn_status/" + status] += 1
        context_count += status == "not_assessable_context_envelope"
    controls = 0
    for turn in record.get("turns", []):
        for span in turn.get("span_annotations", []):
            tier = span.get("supervision_tier")
            if tier in {"cf_strong", "cf_weak"}:
                result[tier + "_spans"] += 1
            status = span.get("evidence_status")
            if status:
                result["span_status/" + status] += 1
            context_count += status == "not_assessable_context_envelope"
            controls += status == "negative_control_violated"
    result["negative_control_violations"] += controls
    result["records_with_control_violations"] += bool(controls)
    result["context_unassessable_interventions"] += context_count
    result["records_with_context_unassessable"] += bool(context_count)
    return dict(result)


def summarize(records):
    totals = Counter({name: 0 for name in (
        "records", "complete", "not_applicable", "baseline_not_unsafe",
        "cf_strong_records", "cf_weak_records", "llm_confirmed_records", "ignore_records",
        "training_eligible_records", "supported_evidence_turn_records",
        "cf_strong_spans", "cf_weak_spans", "negative_control_violations",
        "records_with_control_violations", "context_unassessable_interventions",
        "records_with_context_unassessable")})
    for entry in records.values():
        totals.update(entry["metrics"])
    return dict(sorted(totals.items()))


def build_report(state, job_id, previous=None):
    """Caller must hold both existing executor/allocation locks throughout."""
    allocation_dir = state / ("allocation-" + job_id)
    command = read_json(allocation_dir / "executor-command.json")
    allocations = {}
    for path in state.glob("allocation-*/allocation.json"):
        value = read_json(path)
        allocations[str(value["job"])] = value["start_time"]
    if job_id not in allocations or any(t > allocations[job_id] for t in allocations.values()):
        raise ValueError("Report the latest allocation only; earlier batch state cannot be reconstructed")
    if Path(option(command, "--state-dir")).resolve() != state:
        raise ValueError("Executor command belongs to another state directory")
    inputs = read_rows(Path(option(command, "--input")))
    lookup = {r["conversation_id"]: r for r in inputs}
    if len(lookup) != len(inputs) or not inputs:
        raise ValueError("Input IDs must be unique and nonempty")
    keys = {fingerprint(dict(kind="record", id=cid, input=fingerprint(r))): cid
            for cid, r in lookup.items()}
    records, failures = {}, {}
    subtask_count = 0
    db = sqlite3.connect((state / "progress.sqlite").as_uri() + "?mode=ro", uri=True)
    try:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        contract = json.loads(db.execute("SELECT value FROM meta WHERE key='contract'").fetchone()[0])
        if contract["stage"] != "b4" or contract["input_fingerprint"] != fingerprint(inputs):
            raise ValueError("State is not B4 or input fingerprint differs from its contract")
        cfg_fp = fingerprint(contract["scientific_config"])
        for key, encoded, digest in db.execute("SELECT key,value,digest FROM results"):
            value = json.loads(encoded)
            if fingerprint(value) != digest:
                raise ValueError(f"Result digest mismatch: {key}")
            if key not in keys:
                if isinstance(value, dict) and "frontier_evidence_analysis" in value:
                    raise ValueError("Record result has an unknown input identity")
                subtask_count += 1
                continue
            cid = keys[key]
            analysis = value.get("frontier_evidence_analysis", {})
            if (value.get("conversation_id") != cid
                    or analysis.get("input_fingerprint") != fingerprint(lookup[cid])
                    or analysis.get("config_fingerprint") != cfg_fp):
                raise ValueError(f"Record identity/config mismatch: {cid}")
            records[cid] = dict(digest=digest, metrics=metrics(value))
        for key, encoded, digest in db.execute("SELECT key,value,digest FROM failures"):
            value = json.loads(encoded)
            if fingerprint(value) != digest or key not in keys or value.get("conversation_id") != keys[key]:
                raise ValueError(f"Failure digest/identity mismatch: {key}")
            if keys[key] not in records:
                failures[keys[key]] = value
    finally:
        db.close()

    # Imports are not new GPU work. Verify them against committed results, even
    # if a later allocation's command no longer mentions the original import.
    imports = set()
    for saved in state.glob("allocation-*/executor-command.json"):
        args = read_json(saved)
        if "--import-checkpoints" not in args:
            continue
        patterns = args[args.index("--import-checkpoints") + 1:]
        for pattern in patterns:
            if pattern.startswith("--"):
                break
            paths = sorted(glob.glob(pattern))
            if not paths:
                raise ValueError(f"Import source unavailable for accounting: {pattern}")
            for path in paths:
                latest = {r["conversation_id"]: r for r in read_rows(Path(path))}
                for cid, value in latest.items():
                    if cid in records:
                        if records[cid]["digest"] != fingerprint(value):
                            raise ValueError(f"Imported record changed: {cid}")
                        imports.add(cid)

    contract_fp = fingerprint(contract)
    previous_records = {}
    prior_allocations = {}
    if previous is not None:
        if (previous.get("protocol") != "b4_progress_snapshot_v1"
                or previous.get("state_dir") != str(state)
                or previous.get("contract_fingerprint") != contract_fp):
            raise ValueError("Previous snapshot belongs to another state/contract")
        if previous["job_id"] == job_id:
            raise ValueError("Previous snapshot must precede this allocation")
        previous_records = previous["records"]
        prior_allocations = previous["allocations"]
        if job_id in prior_allocations or any(allocations.get(k) != v for k, v in prior_allocations.items()):
            raise ValueError("Previous allocation history is incompatible")
        for cid, entry in previous_records.items():
            if records.get(cid) != entry:
                raise ValueError(f"Previously reported record missing or changed: {cid}")
    added = {cid: entry for cid, entry in records.items() if cid not in previous_records}
    newly_imported = {cid: entry for cid, entry in added.items() if cid in imports}
    newly_computed = {cid: entry for cid, entry in added.items() if cid not in imports}
    eligible_ids = {cid for cid, r in lookup.items()
                    if r.get("label") == 1 and r.get("validation_status") == "validated"}
    return dict(
        protocol="b4_progress_snapshot_v1", state_dir=str(state), job_id=job_id,
        contract_fingerprint=contract_fp, allocations=allocations,
        previous_job_id=previous["job_id"] if previous else None,
        interval_allocation_ids=sorted(set(allocations) - set(prior_allocations), key=allocations.get),
        cumulative=summarize(records), added_since_previous=summarize(added),
        newly_computed=summarize(newly_computed), newly_imported=summarize(newly_imported),
        input_records=len(inputs), eligible_input_records=len(eligible_ids),
        records_without_complete_result=len(inputs) - len(records),
        eligible_records_without_complete_result=len(eligible_ids - records.keys()),
        pending_without_persisted_failure=len(set(lookup) - records.keys() - failures.keys()),
        unresolved_failures=failures, cached_nonrecord_results=subtask_count,
        records=records,
        notes=["Counts cover committed whole records; partial work has no final supervision tier.",
               "Nonrecord cache entries include work for completed records; they are not a count of partial records.",
               "Snapshot differences cover all listed allocations, not inferred completion timestamps.",
               "Digest and identity checks are not a replacement for the final scientific publication audit."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--previous", type=Path, help="Previous snapshot, taken before this allocation began")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]+", args.job_id):
        parser.error("--job-id must be a numeric Slurm job ID")
    state = Path(args.state_dir).resolve(strict=True)
    with ExitStack() as stack:
        for name in ("allocation.lock", "owner.lock"):
            handle = stack.enter_context((state / name).open("rb"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Allocation/executor is active; report after it exits") from exc
        previous = None
        if args.previous:
            envelope = read_json(args.previous)
            previous = envelope["report"]
            if fingerprint(previous) != envelope["digest"]:
                raise ValueError("Previous snapshot digest mismatch")
        report = build_report(state, args.job_id, previous)
        destination = state / "reports" / ("b4-progress-" + args.job_id + ".json")
        envelope = dict(report=report, digest=fingerprint(report))
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            if read_json(destination) != envelope:
                raise ValueError("Snapshot already exists with different contents; refusing overwrite")
        else:
            # Publish a complete snapshot without overwriting an existing one.
            # The temporary file is on the same filesystem as the report.
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                             dir=destination.parent) as handle:
                json.dump(envelope, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                os.link(handle.name, destination)
                fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2, sort_keys=True))
        print(f"Snapshot: {destination}")


if __name__ == "__main__":
    main()
