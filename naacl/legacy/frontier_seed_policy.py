#!/usr/bin/env python3
"""Locked seed policy for frontier experiments.

Paired malicious/benign twins share the same record-level seed so the branch text
is the only designed difference in generation/judging. Standalone records use
their conversation ID. The actual model turn seeds remain deterministic offsets
from this record-level seed.
"""
from __future__ import annotations

from typing import Dict

from frontier_common import stable_record_seed

SEED_POLICY = "pair_id_if_present_else_conversation_id_v1"


def experiment_seed_key(record: Dict) -> str:
    pair_id = record.get("pair_id")
    if pair_id not in (None, ""):
        return f"pair:{pair_id}"
    cid = str(record.get("conversation_id", ""))
    if not cid:
        raise ValueError("record missing conversation_id for seed derivation")
    return f"conversation:{cid}"


def experiment_record_seed(base_seed: int, record: Dict) -> int:
    return stable_record_seed(base_seed, experiment_seed_key(record))
