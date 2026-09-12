"""Versioned target completion budgets. Judge budgets are unchanged."""
from __future__ import annotations

import copy
from frontier_common import json_fingerprint

FIXED = 'fixed_2048_v1'
ADAPTIVE = 'length_retry_2048_4096_8192_v1'
POLICIES = (FIXED, ADAPTIVE)
BUDGETS = (2048, 4096, 8192)
ROLLOUT_V4 = 'frontier_fixed_user_rollout_v4'
EVIDENCE_V7 = 'frontier_context_paired_counterfactual_v7'


class RecordCompletionError(RuntimeError):
    """A terminal, record-local completion failure, never a successful result."""
    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = details or {}


def policy_fields(policy):
    if policy not in POLICIES:
        raise ValueError(f'unknown target budget policy: {policy}')
    return {} if policy == FIXED else dict(target_budget_policy=policy, target_token_budgets=list(BUDGETS))


def policy_from(mapping):
    policy = mapping.get('target_budget_policy', FIXED)
    expected = policy_fields(policy)
    if policy == FIXED and ('target_budget_policy' in mapping or 'target_token_budgets' in mapping):
        raise RuntimeError('fixed protocol must not carry adaptive budget provenance')
    for key, value in expected.items():
        if mapping.get(key) != value:
            raise RuntimeError(f'invalid {key} provenance')
    return policy


def target_completion(client, messages, *, seed, max_tokens, policy=FIXED):
    if policy == FIXED:
        return client.chat_result(messages, seed=seed, temperature=0.0, max_tokens=max_tokens,
                                  require_stop=False, require_usage=False)
    policy_fields(policy)
    if max_tokens != BUDGETS[0]:
        raise ValueError('adaptive completion must start at 2048 tokens')
    attempts = []
    prompt = copy.deepcopy(messages)
    for budget in BUDGETS:
        # Every attempt starts from the identical observable prompt, not from
        # the truncated response. Request cache keys include the actual budget.
        attempt_call = getattr(client, "budget_attempt", client.chat_result)
        result = attempt_call(copy.deepcopy(prompt), seed=seed, temperature=0.0,
                                    max_tokens=budget, require_stop=False, require_usage=False)
        tokens = result.get('completion_tokens')
        attempt = dict(max_tokens=budget, finish_reason=result.get('finish_reason'),
                       completion_tokens=tokens, response_fingerprint=json_fingerprint(result['content']))
        attempts.append(attempt)
        if type(tokens) is not int or not 0 < tokens <= budget:
            raise RecordCompletionError('invalid target completion usage', details=dict(attempts=attempts))
        if result.get('finish_reason') == 'stop':
            return {**result, 'max_tokens':budget, **policy_fields(policy), 'budget_attempts':attempts}
        if result.get('finish_reason') != 'length' or tokens != budget:
            raise RecordCompletionError('target stopped without natural completion or exhausted output budget',
                                        details=dict(attempts=attempts))
    raise RecordCompletionError('target exhausted all completion budgets', details=dict(attempts=attempts))


def generation_budget_fields(result, policy):
    if policy == FIXED:
        return {}
    return {**policy_fields(policy), 'max_tokens':result['max_tokens'],
            'budget_attempts':copy.deepcopy(result['budget_attempts'])}


def assert_budget_generation(generation, policy, *, initial_max_tokens=2048, response_fingerprint=None):
    if policy == FIXED:
        if policy_from(generation) != FIXED or generation.get('max_tokens') != initial_max_tokens:
            raise RuntimeError('fixed target generation budget mismatch')
        return
    if initial_max_tokens != BUDGETS[0] or policy_from(generation) != policy:
        raise RuntimeError('adaptive target generation policy mismatch')
    attempts = generation.get('budget_attempts')
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= len(BUDGETS):
        raise RuntimeError('missing or invalid target budget attempt trace')
    for i, attempt in enumerate(attempts):
        budget = BUDGETS[i]
        tokens = attempt.get('completion_tokens')
        if attempt.get('max_tokens') != budget or type(tokens) is not int or not 0 < tokens <= budget:
            raise RuntimeError('invalid target budget attempt ordering or usage')
        fp = attempt.get('response_fingerprint')
        if not isinstance(fp, str) or len(fp) != 64 or any(c not in '0123456789abcdef' for c in fp):
            raise RuntimeError('missing target attempt response fingerprint')
        if i < len(attempts)-1 and (attempt.get('finish_reason') != 'length' or tokens != budget):
            raise RuntimeError('budget escalated without exhausting previous output limit')
    last = attempts[-1]
    for key in ('max_tokens','completion_tokens','finish_reason'):
        if generation.get(key) != last[key]:
            raise RuntimeError(f'final target attempt {key} mismatch')
    if last['finish_reason'] != 'stop':
        raise RuntimeError('target budget trace is not naturally complete')
    if response_fingerprint is not None and last['response_fingerprint'] != response_fingerprint:
        raise RuntimeError('final target response fingerprint mismatch')
