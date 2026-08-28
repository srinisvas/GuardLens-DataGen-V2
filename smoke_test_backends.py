"""
smoke_test_backends.py

Pre-flight verification that vLLM servers are healthy AND expose the exact
response schema that model_bench.py + causal_analysis_v12_3.py depend on.

Runs after servers are up but BEFORE submitting the Phase 1 benchmark, so
schema issues surface in 5 minutes instead of 4 hours into a run.

What it checks per unique server (deduplicated from the YAML):
  1. Health endpoint responds (200)
  2. /v1/models lists the expected model
  3. /v1/chat/completions accepts a request with the exact payload shape
     model_bench.py issues (messages, temperature, max_tokens, top_p, seed)
  4. Response has choices[0].message.content as a non-empty string
  5. Response has usage.prompt_tokens as an integer
  6. Response has usage.completion_tokens as an integer
  7. BONUS: same seed twice produces byte-identical content (seed determinism)

Prints a color-coded pass/fail table at the end. Exit code 0 if all servers
pass, 1 if any fail.

Usage:
    # From a compute node after servers are launched:
    python smoke_test_backends.py \\
        --config-file model_bench_configs.yaml

    # Or test a subset:
    python smoke_test_backends.py \\
        --config-file model_bench_configs.yaml \\
        --config-name config_a

    # Skip seed determinism if you know the server was launched without
    # deterministic settings (still checks schema):
    python smoke_test_backends.py \\
        --config-file model_bench_configs.yaml \\
        --skip-seed-check
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests
import yaml


# ANSI colors (fallback to empty strings if not a TTY)
if sys.stdout.isatty():
    GREEN, RED, YELLOW, BOLD, RESET = (
        "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[0m"
    )
else:
    GREEN = RED = YELLOW = BOLD = RESET = ""

PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
WARN = f"{YELLOW}WARN{RESET}"


# =========================================================
# Config parsing (mirrors model_bench.py)
# =========================================================

def parse_yaml_configs(path: str, filter_name: Optional[str] = None) -> List[Dict]:
    """Return a list of {name, generator, target, judge} dicts."""
    with open(path) as f:
        data = yaml.safe_load(f)
    out = []
    for conf in data.get("configurations", []):
        if filter_name is not None and conf.get("name") != filter_name:
            continue
        out.append({
            "name": conf["name"],
            "generator": conf["generator"],
            "target": conf["target"],
            "judge": conf["judge"],
        })
    return out


def unique_servers(configs: List[Dict]) -> List[Tuple[str, str, List[Tuple[str, str]]]]:
    """
    Deduplicate (model, base_url) across all roles in all configs.

    Returns list of (model, base_url, [(config_name, role), ...]) where the
    last field records which (config, role) pairs use this server. That way
    the report shows "Qwen/Qwen3.5-27B on port 8002: config_a/judge,
    config_b/generator".
    """
    seen: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for conf in configs:
        for role in ("generator", "target", "judge"):
            r = conf[role]
            key = (r["model"], r["base_url"].rstrip("/"))
            seen.setdefault(key, []).append((conf["name"], role))
    return [(model, url, uses) for (model, url), uses in seen.items()]


# =========================================================
# Individual checks
# =========================================================

def check_health(base_url: str, timeout: float = 5.0) -> Tuple[bool, str]:
    """GET /health. vLLM returns 200 with empty body when ready."""
    try:
        r = requests.get(f"{base_url}/health", timeout=timeout)
        if r.status_code == 200:
            return True, "200 OK"
        return False, f"HTTP {r.status_code}: {r.text[:100]}"
    except requests.RequestException as e:
        return False, f"connection error: {str(e)[:100]}"


def check_models_listed(base_url: str, expected_model: str,
                         timeout: float = 5.0) -> Tuple[bool, str]:
    """GET /v1/models. Should list expected_model in data[].id."""
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        ids = [m.get("id") for m in data.get("data", [])]
        if expected_model in ids:
            return True, f"listed: {expected_model}"
        return False, f"expected {expected_model!r} not in {ids}"
    except requests.RequestException as e:
        return False, f"connection error: {str(e)[:100]}"
    except (KeyError, ValueError) as e:
        return False, f"malformed response: {str(e)[:100]}"


def check_chat_completion_schema(
    base_url: str, model: str, seed: int = 42, timeout: float = 60.0,
) -> Tuple[bool, str, Optional[Dict]]:
    """
    POST /v1/chat/completions with the exact payload shape model_bench.py
    issues. Verify response has every field the benchmark reads.

    Returns (ok, message, raw_response_or_None).
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": "You are a helpful assistant. Reply concisely."},
            {"role": "user",
             "content": "Say hello in exactly five words."},
        ],
        "temperature": 0.7,
        "max_tokens": 30,
        "top_p": 0.92,
        "seed": seed,
    }
    try:
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer EMPTY"},
            json=payload, timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        return False, f"request error: {str(e)[:150]}", None
    except ValueError as e:
        return False, f"json parse error: {str(e)[:100]}", None

    problems = []

    # Check choices[0].message.content
    try:
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            problems.append(f"choices[0].message.content empty or non-string")
    except (KeyError, IndexError, TypeError):
        problems.append("missing choices[0].message.content")

    # Check usage.prompt_tokens
    try:
        pt = data["usage"]["prompt_tokens"]
        if not isinstance(pt, int):
            problems.append(f"usage.prompt_tokens is {type(pt).__name__}, not int")
    except (KeyError, TypeError):
        problems.append("missing usage.prompt_tokens")

    # Check usage.completion_tokens
    try:
        ct = data["usage"]["completion_tokens"]
        if not isinstance(ct, int):
            problems.append(f"usage.completion_tokens is {type(ct).__name__}, not int")
    except (KeyError, TypeError):
        problems.append("missing usage.completion_tokens")

    if problems:
        return False, "; ".join(problems), data
    return True, "schema OK", data


def check_seed_determinism(
    base_url: str, model: str, seed: int = 42, timeout: float = 60.0,
) -> Tuple[bool, str]:
    """Two requests with same seed and temperature=0.0 should produce
    byte-identical content. Some vLLM configs disable this for perf; if
    so, report as WARN not FAIL."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "List 3 primary colors."}],
        "temperature": 0.0,
        "max_tokens": 20,
        "seed": seed,
    }
    try:
        outputs = []
        for _ in range(2):
            r = requests.post(
                f"{base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer EMPTY"},
                json=payload, timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            outputs.append(data["choices"][0]["message"]["content"])
        if outputs[0] == outputs[1]:
            return True, "deterministic"
        # Non-deterministic isn't strictly a bug — just a note
        return False, (f"differ: {outputs[0][:40]!r} vs {outputs[1][:40]!r}"
                       f" (vLLM may not enforce full determinism at this temp)")
    except requests.RequestException as e:
        return False, f"request error: {str(e)[:100]}"


# =========================================================
# Server test runner
# =========================================================

def test_server(
    model: str, base_url: str, uses: List[Tuple[str, str]],
    skip_seed_check: bool = False,
) -> Dict:
    """Run all checks against one server. Return result dict."""
    print(f"\n{BOLD}Testing {model} @ {base_url}{RESET}")
    print(f"  Used by: {', '.join(f'{c}/{r}' for c, r in uses)}")

    result = {
        "model": model,
        "base_url": base_url,
        "uses": uses,
        "checks": {},
        "overall_pass": True,
    }

    # 1. Health
    ok, msg = check_health(base_url)
    result["checks"]["health"] = {"pass": ok, "message": msg}
    print(f"  {'[' + PASS + ']' if ok else '[' + FAIL + ']'} health: {msg}")
    if not ok:
        result["overall_pass"] = False
        print(f"  {YELLOW}Skipping remaining checks — server not reachable.{RESET}")
        return result

    # 2. Model listed
    ok, msg = check_models_listed(base_url, model)
    result["checks"]["models_listed"] = {"pass": ok, "message": msg}
    print(f"  {'[' + PASS + ']' if ok else '[' + FAIL + ']'} models listed: {msg}")
    if not ok:
        result["overall_pass"] = False

    # 3. Chat completion schema
    ok, msg, raw = check_chat_completion_schema(base_url, model)
    result["checks"]["chat_schema"] = {"pass": ok, "message": msg}
    print(f"  {'[' + PASS + ']' if ok else '[' + FAIL + ']'} chat schema: {msg}")
    if not ok:
        result["overall_pass"] = False
        if raw is not None:
            print(f"  {YELLOW}Raw response for debugging:{RESET}")
            print(f"  {json.dumps(raw, indent=2)[:400]}")

    # 4. Seed determinism (bonus, WARN not FAIL)
    if not skip_seed_check:
        ok, msg = check_seed_determinism(base_url, model)
        marker = f"[{PASS}]" if ok else f"[{WARN}]"
        result["checks"]["seed_determinism"] = {"pass": ok, "message": msg}
        print(f"  {marker} seed determinism: {msg}")
        # Don't flip overall_pass on this — it's advisory

    return result


# =========================================================
# Report
# =========================================================

def print_summary(results: List[Dict]) -> bool:
    print(f"\n{'=' * 70}")
    print(f"  Summary")
    print(f"{'=' * 70}")
    all_pass = True
    for r in results:
        marker = PASS if r["overall_pass"] else FAIL
        uses_str = ", ".join(f"{c}/{ro}" for c, ro in r["uses"])
        print(f"  [{marker}] {r['model']} @ {r['base_url']}")
        print(f"         used by: {uses_str}")
        for check_name, check in r["checks"].items():
            if not check["pass"] and check_name != "seed_determinism":
                print(f"         → {check_name}: {check['message']}")
        if not r["overall_pass"]:
            all_pass = False
    print(f"\n{BOLD}Overall: {PASS if all_pass else FAIL}{RESET}")
    if all_pass:
        print(f"  All servers ready for Phase 1 benchmark.")
    else:
        print(f"  Fix failing servers before submitting the SLURM job.")
    return all_pass


# =========================================================
# CLI
# =========================================================

def main():
    p = argparse.ArgumentParser(description="Pre-flight check for vLLM servers")
    p.add_argument("--config-file", required=True,
                   help="Path to model_bench_configs.yaml")
    p.add_argument("--config-name", default=None,
                   help="Test only servers used by this config name")
    p.add_argument("--skip-seed-check", action="store_true",
                   help="Skip the seed-determinism bonus check")
    p.add_argument("--output-json", default=None,
                   help="Also write full results to this JSON path")
    args = p.parse_args()

    configs = parse_yaml_configs(os.path.expanduser(args.config_file),
                                  filter_name=args.config_name)
    if not configs:
        print(f"ERROR: no configurations matched --config-name={args.config_name}",
              file=sys.stderr)
        sys.exit(2)

    servers = unique_servers(configs)
    print(f"{BOLD}Configurations: {[c['name'] for c in configs]}{RESET}")
    print(f"{BOLD}Unique servers to test: {len(servers)}{RESET}")
    for model, url, uses in servers:
        print(f"  {model} @ {url}  ({len(uses)} role(s))")

    results = []
    for model, url, uses in servers:
        r = test_server(model, url, uses, skip_seed_check=args.skip_seed_check)
        results.append(r)

    all_pass = print_summary(results)

    if args.output_json:
        with open(os.path.expanduser(args.output_json), "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"\nFull results: {args.output_json}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()