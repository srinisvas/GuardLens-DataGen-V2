#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(THIS_DIR, "frontier_judge_smoke_adjudication_v4.json")


class JudgeSmokeManifestTests(unittest.TestCase):
    def test_manifest_is_frozen_20_record_realized_behavior_set(self):
        with open(MANIFEST, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        records = manifest.get("records")
        self.assertIsInstance(records, dict)
        self.assertEqual(len(records), 20)

        expected = [item.get("expected_realized_behavior") for item in records.values()]
        self.assertEqual(expected.count("unsafe"), 9)
        self.assertEqual(expected.count("safe"), 11)
        self.assertEqual(
            manifest.get("expected_counts"),
            {"unsafe": 9, "safe": 11},
        )

        for cid, item in records.items():
            self.assertTrue(cid)
            self.assertIn(item.get("expected_realized_behavior"), {"safe", "unsafe"})
            self.assertTrue(str(item.get("subtype", "")).strip())
            self.assertTrue(str(item.get("rationale", "")).strip())

    def test_manifest_is_explicitly_smoke_only(self):
        with open(MANIFEST, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        scope = str(manifest.get("scope", "")).lower()
        self.assertIn("smoke-only", scope)
        self.assertIn("never use", scope)
        self.assertIn("training", scope)
        self.assertIn("production filtering", scope)


if __name__ == "__main__":
    unittest.main(verbosity=2)
