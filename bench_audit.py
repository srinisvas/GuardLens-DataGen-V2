"""Prepare a blinded human audit or render a derived report without model calls."""

import argparse
import hashlib
import json
from pathlib import Path

from bench_measurement import (
    MANUAL_FIELDS, MEASUREMENT_VERSION, measure_record,
    record_digest, record_key, validate_annotation,
)
from model_bench import Configuration, RoleSpec, render_report, summarize_configuration


def load_raw(path):
    payload = path.read_bytes()
    records = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    keys = [record_key(r) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate strategy/objective/seed identities in raw data")
    return records, hashlib.sha256(payload).hexdigest()


def prepare(raw_path, output_dir):
    records, digest = load_raw(raw_path)
    # Validate all records before creating any output.
    measured = [measure_record(r) for r in records]
    output_dir.mkdir(parents=True, exist_ok=False)
    template = {"measurement_version": MEASUREMENT_VERSION,
                "source_sha256": digest, "records": {}}
    # No judge scores, response text, or success labels in adherence worksheet.
    # Trajectory length can reveal early termination, so blinding is partial.
    with (output_dir / "adherence_review.jsonl").open("x") as adherence, \
         (output_dir / "response_review.jsonl").open("x") as responses:
        for raw, row in zip(records, measured):
            key = record_key(raw)
            template["records"][key] = {
                "record_sha256": record_digest(raw), "audit_status": "pending",
                **{field: None for field in MANUAL_FIELDS},
            }
            identity = {"record_key": key, "objective_id": raw["objective_id"],
                        "objective": raw["objective"], "strategy": raw["strategy"]}
            adherence.write(json.dumps({
                **identity, "strategy_eligible_turn": row["strategy_eligible_turn"],
                "objective_required_by_turn": row["objective_required_by_turn"],
                "user_turns": [{"turn": s["turn"], "audit_phase": s["audit_phase"],
                                "user_turn": s["user_turn"]} for s in row["trajectory"]],
            }) + "\n")
            responses.write(json.dumps({**identity, "trajectory": row["trajectory"]}) + "\n")
    (output_dir / "annotations.json").write_text(json.dumps(template, indent=2) + "\n")
    return template


def apply_annotations(records, digest, annotations):
    if annotations.get("source_sha256") != digest:
        raise ValueError("Audit file does not match the raw file SHA-256")
    if annotations.get("measurement_version") != MEASUREMENT_VERSION:
        raise ValueError("Audit rubric version mismatch")
    indexed = annotations.get("records", {})
    if set(indexed) != {record_key(r) for r in records}:
        raise ValueError("Audit keys must match every raw record exactly")
    out = []
    for raw in records:
        annotation = indexed[record_key(raw)]
        validate_annotation(annotation, raw)
        row = measure_record(raw)
        row.update({field: annotation.get(field) for field in MANUAL_FIELDS})
        row["audit_status"] = annotation["audit_status"]
        out.append(measure_record(row))
    return out


def report(raw_path, metrics_path, annotations_path, output_dir):
    records, digest = load_raw(raw_path)
    if annotations_path:
        records = apply_annotations(records, digest, json.loads(annotations_path.read_text()))
    else:
        records = [measure_record(r) for r in records]
    original = json.loads(metrics_path.read_text())
    if original.get("n_attempts") != len(records):
        raise ValueError("Stored metrics attempt count does not match the raw file")
    # Summary/reporting only needs role labels. No HTTP or inference is used.
    config = Configuration(original["config_name"], *[
        RoleSpec("stored", original[role], "offline") for role in ("generator", "target", "judge")
    ])
    metrics = summarize_configuration(config, records)
    for role in ("generator", "target", "judge"):
        metrics[role] = original[role]
    metrics["source_sha256"] = digest
    metrics["source_metrics_sha256"] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    metrics["source_measurement_version"] = original.get("measurement_version", "legacy_v5")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    (output_dir / "bench_report.md").write_text(render_report([{"metrics": metrics}], {}) + "\n")
    with (output_dir / "annotated_conversations.jsonl").open("x") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "report"))
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            if args.metrics or args.annotations:
                parser.error("prepare does not accept --metrics or --annotations")
            prepare(args.raw, args.output_dir)
        else:
            if not args.metrics:
                parser.error("report requires --metrics from the same configuration")
            report(args.raw, args.metrics, args.annotations, args.output_dir)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Saved {args.mode} outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
