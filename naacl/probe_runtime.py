#!/usr/bin/env python3
"""Gate 0: compare solo and mixed-load outputs on real observable B1 prefixes.

Run on the allocated compute node against idle deterministic servers. This is a
measurement utility, not a replacement for the full B1/B2/B4 equivalence smoke.
Adaptive B1 artifacts are replayed through the shared completion policy, including
the complete 2048 -> 4096 -> 8192 attempt history when escalation occurred.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import requests

from completion_policy import FIXED, assert_budget_generation, policy_from, target_completion
from frontier_common import (
    VLLMClient,
    json_fingerprint,
    load_jsonl,
    parse_chat_completion_data,
)
from frontier_judge import judge_assistant_response_v5
from run_stage import TARGET, JUDGE, audit


class ProbeVLLMClient(VLLMClient):
    """Direct probe client with the same output-budget timeout scaling as execution.py."""

    def chat_result(
        self,
        messages,
        *,
        seed,
        temperature=0.0,
        max_tokens=2048,
        require_stop=True,
        require_usage=False,
    ):
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": float(temperature),
            "top_p": 1.0,
            "max_tokens": int(max_tokens),
            "seed": int(seed),
        }
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=600 * max(1, (int(max_tokens) + 2047) // 2048),
        )
        response.raise_for_status()
        return parse_chat_completion_data(
            response.json(),
            require_stop=require_stop,
            require_usage=require_usage,
        )


def _generation_signature(generation, policy, initial_max_tokens):
    signature = {
        "finish_reason": generation.get("finish_reason"),
        "completion_tokens": generation.get("completion_tokens"),
        "max_tokens": generation.get("max_tokens", initial_max_tokens),
    }
    if policy != FIXED:
        signature.update(
            target_budget_policy=generation.get("target_budget_policy"),
            target_token_budgets=generation.get("target_token_budgets"),
            budget_attempts=copy.deepcopy(generation.get("budget_attempts")),
        )
    return signature


def replay_target_case(client, case):
    result = target_completion(
        client,
        case["messages"],
        seed=case["seed"],
        max_tokens=case["initial_max_tokens"],
        policy=case["policy"],
    )
    actual_generation = {
        "finish_reason": result.get("finish_reason"),
        "completion_tokens": result.get("completion_tokens"),
        "max_tokens": result.get("max_tokens", case["initial_max_tokens"]),
    }
    if case["policy"] != FIXED:
        actual_generation.update(
            target_budget_policy=result.get("target_budget_policy"),
            target_token_budgets=result.get("target_token_budgets"),
            budget_attempts=copy.deepcopy(result.get("budget_attempts")),
        )
    assert_budget_generation(
        actual_generation,
        case["policy"],
        initial_max_tokens=case["initial_max_tokens"],
        response_fingerprint=json_fingerprint(result["content"]),
    )
    if result["content"] != case["expected"]:
        raise RuntimeError("GATE 0 FAILED: solo/mixed target differs from stored B1")
    expected_signature = _generation_signature(
        case["expected_generation"], case["policy"], case["initial_max_tokens"]
    )
    actual_signature = _generation_signature(
        actual_generation, case["policy"], case["initial_max_tokens"]
    )
    if actual_signature != expected_signature:
        raise RuntimeError(
            "GATE 0 FAILED: target adaptive attempt trace differs from stored B1"
        )
    return result


def requests_for(rows):
    candidates = []
    judge = []

    class Recorder:
        def chat(self, messages, **kwargs):
            judge.append(dict(messages=copy.deepcopy(messages), kwargs=copy.deepcopy(kwargs)))
            return (
                '{"assistant_behavior":"SAFE_HELP","harmfulness":0,'
                '"boundary_violation":0,"refusal_strength":0,"confidence":1}'
            )

    for row in rows:
        messages = []
        index = 0
        rollout = row["rollout_provenance"]
        seed = rollout["record_seed"]
        policy = policy_from(rollout)
        initial_max_tokens = int(rollout.get("max_tokens", 2048))
        for turn in row["turns"]:
            if turn["role"] == "assistant":
                generation = copy.deepcopy(turn.get("generation_provenance", {}) or {})
                response = turn["text"]
                target_seed = seed + 1009 * (index + 1)
                judge_seed = seed + 1000003 + 1013 * (index + 1)
                assert_budget_generation(
                    generation,
                    policy,
                    initial_max_tokens=initial_max_tokens,
                    response_fingerprint=json_fingerprint(response),
                )
                if generation.get("seed") != target_seed or generation.get("temperature") != 0.0:
                    raise RuntimeError("GATE 0 input has target seed/temperature provenance drift")
                candidates.append(
                    dict(
                        prefix_chars=sum(len(message["content"]) for message in messages),
                        messages=copy.deepcopy(messages),
                        seed=target_seed,
                        judge_seed=judge_seed,
                        expected=response,
                        expected_generation=generation,
                        policy=policy,
                        initial_max_tokens=initial_max_tokens,
                    )
                )
                index += 1
            messages.append(
                {
                    "role": turn["role"],
                    "content": turn["text"].strip() if turn["role"] == "user" else turn["text"],
                }
            )

    if not candidates:
        raise RuntimeError("no realized assistant prefixes")
    candidates.sort(key=lambda item: item["prefix_chars"])
    selected = [
        candidates[i]
        for i in sorted({0, len(candidates) // 3, 2 * len(candidates) // 3, len(candidates) - 1})
    ]
    target = []
    for case in selected:
        target.append(copy.deepcopy(case))
        judge_assistant_response_v5(
            Recorder(),
            case["messages"] + [{"role": "assistant", "content": case["expected"]}],
            seed=case["judge_seed"],
            max_context_chars=100000,
        )
    return {"target": target, "judge": judge}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Complete deterministic B1 (or B2) JSONL")
    parser.add_argument("--target-urls", nargs="*", default=[])
    parser.add_argument("--judge-urls", nargs="*", default=[])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not (args.target_urls or args.judge_urls) or min(args.concurrency + [args.repeats]) < 1:
        parser.error("provide endpoints and positive concurrency/repeats")

    rows = load_jsonl(args.input)
    audit("b1", args.input, len(rows))
    workload = requests_for(rows)
    report = []
    for role, urls in [("target", args.target_urls), ("judge", args.judge_urls)]:
        model = TARGET if role == "target" else JUDGE
        cases = workload[role]
        for url in urls:
            client = ProbeVLLMClient(
                model, url, api_key=os.environ.get("VLLM_API_KEY", "EMPTY")
            )
            if not client.health_check():
                raise RuntimeError(f"{url} unavailable")
            solo = []
            for case in cases:
                if role == "target":
                    value = replay_target_case(client, case)
                else:
                    value = client.chat_result(
                        case["messages"],
                        require_stop=True,
                        require_usage=True,
                        **case["kwargs"],
                    )
                solo.append(value)

            for level in args.concurrency:
                start = time.monotonic()
                tokens = 0
                for repeat in range(args.repeats):
                    # Start a full wave together, rotating to include every case.
                    jobs = [i % len(cases) for i in range(max(level, len(cases)))]
                    if repeat % 2:
                        jobs.reverse()

                    def call(i):
                        case = cases[i]
                        current = ProbeVLLMClient(
                            model, url, api_key=os.environ.get("VLLM_API_KEY", "EMPTY")
                        )
                        if role == "target":
                            return i, replay_target_case(current, case)
                        return i, current.chat_result(
                            case["messages"],
                            require_stop=True,
                            require_usage=True,
                            **case["kwargs"],
                        )

                    with ThreadPoolExecutor(max_workers=level) as pool:
                        for i, result in pool.map(call, jobs):
                            if result != solo[i]:
                                raise RuntimeError(
                                    f"GATE 0 FAILED: {role} {url} concurrency={level} case={i}"
                                )
                            tokens += result["completion_tokens"]
                elapsed = time.monotonic() - start
                row = dict(
                    role=role,
                    server=url,
                    concurrency=level,
                    repeats=args.repeats,
                    seconds=elapsed,
                    completion_tokens=tokens,
                    completion_tokens_per_second=tokens / elapsed,
                    request_fingerprints=[json_fingerprint(case) for case in cases],
                    response_fingerprints=[json_fingerprint(value) for value in solo],
                )
                report.append(row)
                print(json.dumps(row), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(status="passed", measurements=report), indent=2) + "\n")
    print("GATE 0 PASSED: exact solo/mixed outputs on the tested prefixes and concurrency levels")


if __name__ == "__main__":
    main()
