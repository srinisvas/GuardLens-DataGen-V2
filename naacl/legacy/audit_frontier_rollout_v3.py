#!/usr/bin/env python3
"""Fail-closed Stage-B1 audit for deterministic rollout v3."""
from __future__ import annotations

import argparse

import audit_frontier_rollout as base
from frontier_common import load_jsonl
from frontier_runtime_determinism import assert_runtime_fields
from rollout_frontier_source_v3 import PROTOCOL, rollout_config


def precheck(path: str) -> None:
    for record in load_jsonl(path):
        cid = str(record.get("conversation_id", ""))
        prov = record.get("rollout_provenance", {}) or {}
        if prov.get("protocol") != PROTOCOL:
            raise RuntimeError(
                f"{cid}: rollout protocol={prov.get('protocol')!r} != {PROTOCOL!r}"
            )
        try:
            assert_runtime_fields(prov)
        except Exception as exc:
            raise RuntimeError(f"{cid}: deterministic runtime provenance invalid: {exc}") from exc


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--input", required=True)
    args, _ = p.parse_known_args()
    precheck(args.input)
    base.ROLLOUT_PROTOCOL = PROTOCOL
    base.rollout_config = rollout_config
    base.main()


if __name__ == "__main__":
    main()
