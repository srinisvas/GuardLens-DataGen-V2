"""Operational accounting must survive resume without double-counting GPU work."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "report_b4_progress.py"
spec = importlib.util.spec_from_file_location("b4_progress_reporter", SCRIPT)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)
fp = reporter.fingerprint


class B4ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name).resolve()
        self.rows = [dict(conversation_id=str(i), label=int(i != 0),
                          validation_status="validated") for i in range(5)]
        self.input = self.state / "input.jsonl"
        self.input.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        self.cfg = {"protocol": "fixture"}
        self.contract = dict(stage="b4", input_fingerprint=fp(self.rows), scientific_config=self.cfg)
        self.db = sqlite3.connect(self.state / "progress.sqlite")
        self.addCleanup(self.db.close)
        self.db.executescript("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);"
                             "CREATE TABLE results(key TEXT PRIMARY KEY,value TEXT,digest TEXT);"
                             "CREATE TABLE failures(key TEXT PRIMARY KEY,value TEXT,digest TEXT);")
        self.db.execute("INSERT INTO meta VALUES ('contract',?)", (json.dumps(self.contract),))
        self.db.commit()
        for name in ("allocation.lock", "owner.lock"):
            (self.state / name).touch()
        self.allocation("1", 100)

    def allocation(self, job, start, imports=None):
        root = self.state / ("allocation-" + job)
        root.mkdir()
        (root / "allocation.json").write_text(json.dumps(dict(job=job, start_time=start)))
        command = ["python", "run_stage.py", "b4", "--input", str(self.input),
                   "--state-dir", str(self.state)]
        if imports:
            command += ["--import-checkpoints", str(imports)]
        (root / "executor-command.json").write_text(json.dumps(command))

    def key(self, i):
        return fp(dict(kind="record", id=str(i), input=fp(self.rows[i])))

    def put(self, key, value, table="results"):
        self.db.execute(f"INSERT OR REPLACE INTO {table} VALUES (?,?,?)",
                        (key, json.dumps(value), fp(value)))
        self.db.commit()

    def record(self, i, tier="cf_strong"):
        row = self.rows[i]
        spans = [] if not i else [dict(supervision_tier=tier, evidence_status="supported_strong"),
                                  dict(supervision_tier="ignore", evidence_status="negative_control_violated")]
        value = dict(row, supervision_tier=tier, training_eligible=bool(i),
                     evidence_turn_ids=[0] if i else [], turns=[dict(span_annotations=spans)],
                     frontier_evidence_analysis=dict(status="complete" if i else "not_applicable",
                         input_fingerprint=fp(row), config_fingerprint=fp(self.cfg), turn_interventions=[]))
        self.put(self.key(i), value)
        return value

    def test_imports_cumulative_and_resume_interval(self):
        imported = self.record(1)
        source = self.state / "import.jsonl"
        source.write_text(json.dumps(imported) + "\n")
        command = self.state / "allocation-1" / "executor-command.json"
        command.write_text(json.dumps(json.loads(command.read_text()) + ["--import-checkpoints", str(source)]))
        self.record(0)
        self.record(2, "cf_weak")
        self.put("subtask", dict(content="cached model response"))
        self.put(self.key(4), dict(conversation_id="4", error="length failure"), "failures")
        first = reporter.build_report(self.state, "1")
        self.assertEqual(first["cumulative"]["records"], 3)
        self.assertEqual(first["cumulative"]["cf_strong_records"], 1)
        self.assertEqual(first["newly_computed"]["cf_strong_records"], 0)
        self.assertEqual(first["newly_imported"]["cf_strong_records"], 1)
        self.assertEqual(first["newly_computed"]["cf_weak_records"], 1)
        self.assertEqual(first["cached_nonrecord_results"], 1)
        self.assertEqual(first["pending_without_persisted_failure"], 1)
        self.assertEqual(first["eligible_records_without_complete_result"], 2)
        self.allocation("2", 200)
        self.record(3, "llm_confirmed")
        second = reporter.build_report(self.state, "2", first)
        self.assertEqual(second["interval_allocation_ids"], ["2"])
        self.assertEqual(second["newly_computed"]["records"], 1)
        self.assertEqual(second["newly_computed"]["llm_confirmed_records"], 1)
        self.assertEqual(second["newly_imported"]["records"], 0)
        # Missing a report changes the declared interval, never the denominator.
        self.allocation("3", 300)
        third = reporter.build_report(self.state, "3", first)
        self.assertEqual(third["interval_allocation_ids"], ["2", "3"])

    def test_corrupt_subtask_is_rejected(self):
        self.put("subtask", {"ok": True})
        self.db.execute("UPDATE results SET digest='bad'")
        self.db.commit()
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            reporter.build_report(self.state, "1")

    def test_input_and_record_identity_rejected(self):
        value = self.record(1)
        value["frontier_evidence_analysis"]["config_fingerprint"] = "bad"
        self.put(self.key(1), value)
        with self.assertRaisesRegex(ValueError, "identity/config mismatch"):
            reporter.build_report(self.state, "1")
        self.input.write_text(json.dumps(self.rows[0]) + "\n")
        with self.assertRaisesRegex(ValueError, "input fingerprint"):
            reporter.build_report(self.state, "1")

    def test_old_job_or_changed_previous_rejected(self):
        self.record(1)
        previous = reporter.build_report(self.state, "1")
        self.allocation("2", 200)
        with self.assertRaisesRegex(ValueError, "latest allocation"):
            reporter.build_report(self.state, "1")
        self.record(1, "cf_weak")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            reporter.build_report(self.state, "2", previous)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), "--state-dir", str(self.state),
                               "--job-id", "1", *args], capture_output=True, text=True)

    def test_read_only_and_idempotent_snapshot(self):
        self.record(1)
        before = (self.state / "progress.sqlite").read_bytes()
        self.assertEqual(self.run_cli().returncode, 0)
        path = self.state / "reports" / "b4-progress-1.json"
        saved = path.read_bytes()
        self.assertEqual(self.run_cli().returncode, 0)
        self.assertEqual(path.read_bytes(), saved)
        self.assertEqual((self.state / "progress.sqlite").read_bytes(), before)
        self.record(2)
        self.assertNotEqual(self.run_cli().returncode, 0)
        self.assertEqual(path.read_bytes(), saved)

    def test_live_allocation_and_executor_are_rejected(self):
        import fcntl
        for name in ("allocation.lock", "owner.lock"):
            with (self.state / name).open("rb") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = self.run_cli()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("is active", result.stderr)
        self.assertFalse((self.state / "reports").exists())

    def test_previous_snapshot_digest_is_checked(self):
        bad = self.state / "bad.json"
        bad.write_text(json.dumps(dict(report={}, digest="bad")))
        result = self.run_cli("--previous", str(bad))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("snapshot digest mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
