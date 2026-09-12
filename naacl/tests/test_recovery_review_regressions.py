"""Regressions for the recovery-rollout review of adaptive B1/B4 state."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import test_optimized_pipeline as pipeline
from test_optimized_contract import run_active

ADAPTIVE = "length_retry_2048_4096_8192_v1"


class RecoveryPipelineReviewTests(unittest.TestCase):
    setUpClass = classmethod(pipeline.OptimizedPipelineTests.setUpClass.__func__)
    tearDownClass = classmethod(pipeline.OptimizedPipelineTests.tearDownClass.__func__)
    tearDown = pipeline.OptimizedPipelineTests.tearDown
    run_cli = pipeline.OptimizedPipelineTests.run_cli
    optimized = pipeline.OptimizedPipelineTests.optimized

    def _adaptive_evidence(self, root: Path):
        source = root / "source.jsonl"
        source.write_text(json.dumps(pipeline.source("p1-m", 1, "NEEDS4096")) + "\n")
        b1, _ = self.optimized(root, "b1", source, budget_policy=ADAPTIVE)
        b2, _ = self.optimized(root, "b2", b1, budget_policy=ADAPTIVE)
        b3 = root / "b3.jsonl"
        self.run_cli(
            "materialize_frontier_candidates.py",
            "--input",
            b2,
            "--output",
            b3,
            "--max-turn-candidates",
            "4",
            "--spans-per-turn",
            "2",
            "--controls",
            "2",
        )
        b4, _ = self.optimized(root, "b4", b3, budget_policy=ADAPTIVE)
        return b1, b4

    def test_public_b4_audit_binds_adaptive_baseline_to_b1(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, b4 = self._adaptive_evidence(root)
            rows = [json.loads(line) for line in b4.read_text().splitlines()]
            baseline_generation = rows[0]["frontier_evidence_analysis"]["baseline"][
                "trajectory"
            ][0]["target_generation"]
            self.assertGreaterEqual(len(baseline_generation["budget_attempts"]), 2)
            # Keep the attempt syntactically valid while making it inconsistent
            # with the immutable B1 generation_provenance.
            baseline_generation["budget_attempts"][0]["response_fingerprint"] = "f" * 64
            corrupt = root / "baseline-fingerprint-drift.jsonl"
            corrupt.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = self.run_cli(
                "audit_frontier_evidence.py", "--input", corrupt, success=False
            )
            self.assertIn(
                "target baseline budget/usage provenance drift",
                result.stdout + result.stderr,
            )

    def test_public_b4_audit_rejects_scored_span_without_complete_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, b4 = self._adaptive_evidence(root)
            rows = [json.loads(line) for line in b4.read_text().splitlines()]
            assessed = None
            for row in rows:
                for turn in row.get("turns", []):
                    for span in turn.get("span_annotations", []):
                        if (
                            span.get("evidence_cf_post_unsafe") is not None
                            and span.get("counterfactual_post_trajectory")
                        ):
                            assessed = span
                            break
                    if assessed is not None:
                        break
                if assessed is not None:
                    break
            self.assertIsNotNone(assessed, "fixture produced no assessed adaptive span")
            assessed["counterfactual_post_trajectory"] = []
            corrupt = root / "missing-span-trace.jsonl"
            corrupt.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = self.run_cli(
                "audit_frontier_evidence.py", "--input", corrupt, success=False
            )
            self.assertIn(
                "assessed counterfactual has no complete trace",
                result.stdout + result.stderr,
            )

    def test_gate_zero_replays_adaptive_4096_attempt_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(json.dumps(pipeline.source("p1-m", 1, "NEEDS4096")) + "\n")
            b1, _ = self.optimized(root, "b1", source, budget_policy=ADAPTIVE)
            pipeline.Oracle.requests_seen = []
            self.run_cli(
                "probe_runtime.py",
                "--input",
                b1,
                "--target-urls",
                self.url,
                "--concurrency",
                "1",
                "2",
                "--repeats",
                "1",
                "--output",
                root / "gate0.json",
            )
            target_budgets = [
                request["max_tokens"]
                for request in pipeline.Oracle.requests_seen
                if request.get("model") == pipeline.TARGET
            ]
            self.assertIn(2048, target_budgets)
            self.assertIn(4096, target_budgets)
            self.assertNotIn(8192, target_budgets)
            self.assertEqual(json.loads((root / "gate0.json").read_text())["status"], "passed")

    def test_gate_zero_scales_timeout_with_adaptive_budget(self):
        # Run against the active naacl module path, matching the other isolated
        # contract tests. unittest discovery adds naacl/tests, not naacl itself,
        # to this process's import path.
        run_active(
            r'''
            from unittest.mock import Mock, patch
            from probe_runtime import ProbeVLLMClient
            from run_stage import TARGET

            response = Mock(status_code=200)
            response.json.return_value = {
                "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 12},
            }
            response.raise_for_status.return_value = None
            client = ProbeVLLMClient(TARGET, "http://fixture")
            with patch("probe_runtime.requests.post", return_value=response) as post:
                client.chat_result(
                    [{"role": "user", "content": "fixture"}],
                    seed=42,
                    max_tokens=4096,
                    require_stop=True,
                    require_usage=True,
                )
            assert post.call_args.kwargs["timeout"] == 1200
            '''
        )


class RecoveryJournalReviewTests(unittest.TestCase):
    def test_incompatible_checkpoint_rejection_is_database_read_only(self):
        run_active(
            r'''
            import json
            from pathlib import Path
            import sqlite3
            import tempfile
            from execution import Journal

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                root.mkdir(exist_ok=True)
                db_path = root / "progress.sqlite"
                db = sqlite3.connect(db_path)
                db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
                db.execute("CREATE TABLE results (key TEXT PRIMARY KEY, value TEXT, digest TEXT)")
                db.execute(
                    "INSERT INTO meta VALUES ('contract', ?)",
                    (json.dumps({'source':'old'}, sort_keys=True),),
                )
                db.commit()
                db.close()
                before = db_path.read_bytes()

                try:
                    Journal(root, {'source':'new'})
                except RuntimeError as exc:
                    assert 'checkpoint contract mismatch' in str(exc)
                else:
                    raise AssertionError('accepted incompatible checkpoint contract')

                assert db_path.read_bytes() == before, 'rejected checkpoint database was modified'
                db = sqlite3.connect(db_path)
                tables = {
                    row[0]
                    for row in db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                db.close()
                assert tables == {'meta', 'results'}, tables
            '''
        )


if __name__ == "__main__":
    unittest.main()
