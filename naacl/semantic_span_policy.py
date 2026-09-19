#!/usr/bin/env python3
"""Semantic adjudication policy for fixed Dataset B construction-language spans.

The frozen B4 counterfactual evidence remains untouched. This layer only decides
whether a supported evidence span is suitable as a positive token-localization
target in the prepared training artifact.
"""
from __future__ import annotations

from typing import Dict, Iterable

CONSTRUCTION_META_LEXICON = (
    "intent and prior context should matter more than obvious risk vocabulary",
    "which makes the boundary subtle",
    "the participants already know the basic process and write briefly",
    "shared context rather than restating the process",
    "the final request should look ordinary if read by itself",
    "later request should remain ordinary on its face",
    "the conversation should sound like ordinary work rather than a safety demonstration",
    "the user is not trying to sound adversarial",
    "the earlier decision is meant to carry forward implicitly into the final request",
)
SUPPORTED_STATUSES = {"supported_strong", "supported_weak"}
ADJUDICATION_VERSION = "construction_language_token_mask_v1"


def is_construction_language_supported_span(span: Dict) -> bool:
    if span.get("label") != "EVIDENCE_CANDIDATE":
        return False
    if str(span.get("evidence_status", "")) not in SUPPORTED_STATUSES:
        return False
    text = str(span.get("text", "")).lower()
    return any(phrase in text for phrase in CONSTRUCTION_META_LEXICON)


def apply_semantic_span_policy(span: Dict) -> bool:
    if not is_construction_language_supported_span(span):
        return False
    span["semantic_adjudication"] = ADJUDICATION_VERSION
    span["semantic_token_supervision_ignore"] = True
    span["semantic_original_supervision_tier"] = span.get("supervision_tier")
    span["causal_type"] = "unvalidated"
    span["supervision_tier"] = "ignore"
    return True


def iter_flagged_spans(records: Iterable[Dict]):
    for record in records:
        for turn in record.get("turns", []) or []:
            for span in turn.get("span_annotations", []) or []:
                if is_construction_language_supported_span(span):
                    yield record, turn, span
