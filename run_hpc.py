"""
run_hpc.py (v11)

HPC runner for the dataset generation pipeline. Wires the inference
backend into the generator, paraphraser, annotator, and validator.

v11 changes:
  - Supports separate generation and validation modes (--mode gen|val)
  - Checkpoint/resume for validation (write-ahead per record)
  - Validates ALL records (malicious + benign + hard-negative)
  - Misleading pivot fraction parameter
  - Supervision tier assignment post-validation

Usage (generation only, batch 1):
    python run_hpc.py --mode gen --n-pairs 1000 --backend vllm

Usage (validation only, batch 2):
    python run_hpc.py --mode val --input output/raw.jsonl --backend vllm

Usage (full pipeline, single batch):
    python run_hpc.py --mode full --n-pairs 500 --backend vllm

Usage (SLURM array, generation sharding):
    python run_hpc.py --mode gen --n-pairs 1000 --backend vllm --array-mode
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_backend import create_backend, InferenceBackend
import build_semantic_datasetv11 as gen


# =========================================================
# Patch the generation classes to use a pluggable backend
# =========================================================

class BackendTurnGenerator(gen.LLMTurnGenerator):
    def __init__(self, backend: InferenceBackend, enabled: bool = True):
        self.backend = backend
        self.enabled = enabled
        self.model = getattr(backend, 'model', 'unknown')
        self.url = None

    def _call(self, prompt: str, system: str) -> str:
        return self.backend.generate(
            prompt=prompt, system=system,
            temperature=0.80, max_tokens=120,
        )


class BackendParaphraser(gen.LocalParaphraser):
    def __init__(self, backend: InferenceBackend, enabled: bool = True):
        self.backend = backend
        self.enabled = enabled
        self.model = getattr(backend, 'model', 'unknown')
        self.url = None

    def _call(self, prompt: str) -> str:
        return self.backend.generate(
            prompt=prompt, system="",
            temperature=0.65, max_tokens=150,
        )


class BackendAnnotator(gen.LLMSpanAnnotator):
    def __init__(self, backend: InferenceBackend, enabled: bool = True):
        self.backend = backend
        self.enabled = enabled
        self.model = getattr(backend, 'model', 'unknown')
        self.url = None

    def _call(self, prompt: str) -> str:
        return self.backend.generate(
            prompt=prompt, system="",
            temperature=0.1, max_tokens=300,
        )


class BackendValidator(gen.CausalValidator):
    def __init__(self, backend: InferenceBackend, enabled: bool = True,
                 use_structured_judge: bool = True):
        self.backend = backend
        self.enabled = enabled
        self.use_structured_judge = use_structured_judge
        self.model = getattr(backend, 'model', 'unknown')
        self.url = None

    def _chat(self, messages):
        try:
            return self.backend.chat(
                messages=messages, temperature=0.3, max_tokens=200,
            )
        except Exception as e:
            return f"[validation_error: {e}]"


# =========================================================
# Generation mode (Batch 1)
# =========================================================

def generate_dataset_hpc(
    backend: InferenceBackend,
    n_pairs: int = 100,
    paraphrase_variants: int = 1,
    tokenizer_name=None,
    use_llm_generator: bool = True,
    use_local_paraphraser: bool = True,
    use_llm_annotator: bool = False,
    seed_files=None,
    use_builtin_seeds: bool = True,
    dedup: bool = True,
    dedup_threshold: float = 0.70,
    random_seed=None,
    misleading_fraction: float = 0.10,
):
    """
    Generation-only pipeline. No validation (that's batch 2).
    Pre-validation dedup included.
    """
    import random as _random

    if random_seed is not None:
        _random.seed(random_seed)
        gen.random.seed(random_seed)
        print(f"  Random seed: {random_seed}")

    aligner = gen.TokenAligner(tokenizer_name=tokenizer_name)
    generator = BackendTurnGenerator(backend=backend, enabled=use_llm_generator)
    paraphraser = BackendParaphraser(backend=backend, enabled=use_local_paraphraser)
    llm_annotator = BackendAnnotator(backend=backend, enabled=use_llm_annotator)

    dataset = []

    # Phase 1: Synthetic conversations
    print(f"Phase 1: Generating {n_pairs} synthetic conversation pairs...")
    for iteration in range(n_pairs):
        base_samples = []

        mal, ben = gen.generate_paired_twin_samples(generator)
        base_samples.extend([mal, ben])

        # v11: Misleading pivot mode
        if gen.random.random() < misleading_fraction:
            base_samples.append(gen.generate_misleading_pivot(generator))

        if gen.random.random() < 0.35:
            base_samples.append(gen.generate_benign_late_guardrail(generator))
            base_samples.append(gen.generate_fragmented_attack(generator))

        if gen.random.random() < 0.35:
            base_samples.append(gen.generate_false_positive_trap(generator))
            base_samples.append(gen.generate_fragmented_attack(generator))

        for s in base_samples:
            if use_llm_annotator:
                prior = []
                for t in s.turns:
                    t = llm_annotator.annotate_turn(t, prior)
                    prior.append(t)

            gen.align_sample_spans(s, aligner)
            dataset.append(gen.sample_to_dict(s))
            for p in gen.paraphrase_sample(s, paraphraser, variants=paraphrase_variants):
                gen.align_sample_spans(p, aligner)
                dataset.append(gen.sample_to_dict(p))

        if (iteration + 1) % 10 == 0:
            print(f"  {iteration + 1}/{n_pairs} pairs ({len(dataset)} records)")

    # Phase 2: Real-world seeds
    seeds = []
    if use_builtin_seeds:
        seeds.extend(gen.SeedLoader.load_builtin())
    if seed_files:
        for sf in seed_files:
            fmt = sf.get("format", "jsonl")
            if fmt == "jsonl":
                seeds.extend(gen.SeedLoader.load_from_jsonl(
                    sf["path"], text_field=sf.get("text_field", "prompt"),
                ))
            elif fmt == "csv":
                seeds.extend(gen.SeedLoader.load_from_csv(
                    sf["path"], text_col=sf.get("text_col", "goal"),
                ))

    if seeds:
        print(f"Phase 2: Extending {len(seeds)} seeds...")
        for i, seed in enumerate(seeds):
            s = gen.SeedLoader.extend_to_multiturn(seed, generator)
            if use_llm_annotator:
                prior = []
                for t in s.turns:
                    t = llm_annotator.annotate_turn(t, prior)
                    prior.append(t)
            gen.align_sample_spans(s, aligner)
            dataset.append(gen.sample_to_dict(s))
            for p in gen.paraphrase_sample(s, paraphraser, variants=1):
                gen.align_sample_spans(p, aligner)
                dataset.append(gen.sample_to_dict(p))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(seeds)} seeds extended")

    # Phase 3: Pre-validation dedup
    if dedup:
        before = len(dataset)
        dataset = gen.deduplicate_dataset(dataset, threshold=dedup_threshold)
        print(f"Pre-validation dedup: {before} -> {len(dataset)}")

    return dataset


# =========================================================
# Validation mode (Batch 2) with checkpoint/resume
# =========================================================

def validate_dataset_hpc(
    backend: InferenceBackend,
    input_path: str,
    output_path: str,
    checkpoint_path: str = None,
    use_counterfactual: bool = True,
    use_structured_judge: bool = True,
    checkpoint_interval: int = 25,
):
    """
    Validate all records with checkpoint/resume.
    Writes each validated record immediately to checkpoint file.
    """
    from dataclasses import asdict

    # Load input
    records = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {input_path}")

    # Check for checkpoint (resume)
    if checkpoint_path is None:
        checkpoint_path = output_path + ".checkpoint"

    completed_ids = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    completed_ids.add(r.get("conversation_id", ""))
        print(f"Resuming: {len(completed_ids)} already validated")

    # Setup validator
    validator = BackendValidator(
        backend=backend, enabled=True,
        use_structured_judge=use_structured_judge,
    )
    cf_analyzer = gen.CounterfactualAnalyzer(validator=validator)

    validated = jailbreaks = benign_checked = false_alarms = 0
    start_time = time.time()

    # Open checkpoint for append
    with open(checkpoint_path, "a") as ckpt:
        for idx, record in enumerate(records):
            cid = record.get("conversation_id", "")
            if cid in completed_ids:
                continue

            sample = gen._dict_to_sample(record)
            result = validator.validate_conversation(sample)
            record["causal_validation"] = result
            record["validation_status"] = "validated"
            record["judge_confidence"] = result.get("avg_judge_confidence", 0.0)

            if result.get("pivot_turn_id") is not None:
                record["pivot_turn_id"] = result["pivot_turn_id"]
            if result.get("pivot_kind"):
                record["pivot_kind"] = result["pivot_kind"]

            validated += 1

            if record.get("label") == 1:
                if result.get("jailbreak_detected"):
                    jailbreaks += 1
                    if use_counterfactual:
                        sample.causal_validation = result
                        sample.pivot_turn_id = result.get("pivot_turn_id")
                        sample = cf_analyzer.analyze_sample(sample)
                        record["turns"] = [asdict(t) for t in sample.turns]
                else:
                    record["validation_status"] = "ambiguous"
            else:
                benign_checked += 1
                if result.get("jailbreak_detected"):
                    false_alarms += 1
                    record["validation_status"] = "rejected"
                    record["training_eligible"] = False

            # Assign supervision tiers
            sample = gen._dict_to_sample(record)
            sample = gen.assign_sample_tier(sample)
            record["supervision_tier"] = sample.supervision_tier
            record["loss_weight"] = sample.loss_weight
            record["training_eligible"] = gen.is_training_eligible(sample)
            for i, turn in enumerate(sample.turns):
                for j, span in enumerate(turn.span_annotations):
                    if i < len(record["turns"]) and j < len(record["turns"][i].get("span_annotations", [])):
                        record["turns"][i]["span_annotations"][j]["supervision_tier"] = span.supervision_tier

            # Write-ahead checkpoint
            ckpt.write(json.dumps(record, ensure_ascii=False) + "\n")
            ckpt.flush()

            if validated % checkpoint_interval == 0:
                elapsed = (time.time() - start_time) / 60
                rate = validated / max(elapsed, 0.01)
                remaining = (len(records) - idx - 1) / max(rate, 0.01)
                print(f"  {validated}/{len(records)} validated "
                      f"({jailbreaks} jailbreaks, {false_alarms} false alarms) "
                      f"[{elapsed:.0f}m elapsed, ~{remaining:.0f}m remaining]")

    print(f"\nValidation complete: {validated} validated, "
          f"{jailbreaks} jailbreaks, {benign_checked} benign, "
          f"{false_alarms} false alarms")

    # Merge checkpoint back into records (preserving order)
    validated_map = {}
    with open(checkpoint_path, "r") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                validated_map[r.get("conversation_id", "")] = r

    for i, record in enumerate(records):
        cid = record.get("conversation_id", "")
        if cid in validated_map:
            records[i] = validated_map[cid]

    # Write final output
    gen.write_jsonl(records, output_path)
    print(f"Wrote {len(records)} records to {output_path}")

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.rename(checkpoint_path, checkpoint_path + ".done")

    return records


# =========================================================
# SLURM array job support
# =========================================================

def compute_shard(n_pairs: int, array_task_id: int, array_size: int):
    base = n_pairs // array_size
    remainder = n_pairs % array_size
    if array_task_id < remainder:
        start = array_task_id * (base + 1)
        count = base + 1
    else:
        start = remainder * (base + 1) + (array_task_id - remainder) * base
        count = base
    return start, count


def main():
    parser = argparse.ArgumentParser(
        description="HPC dataset generation/validation runner (v11)"
    )
    # Mode
    parser.add_argument("--mode", type=str, default="full",
                        choices=["gen", "val", "full"],
                        help="gen=generation only, val=validation only, full=both")

    # Generation params
    parser.add_argument("--n-pairs", type=int, default=1000)
    parser.add_argument("--paraphrase-variants", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--output-name", type=str,
                        default="semantic_multiturn_v11")
    parser.add_argument("--misleading-fraction", type=float, default=0.10)

    # Validation params (for --mode val)
    parser.add_argument("--input", type=str, default=None,
                        help="Input JSONL for validation mode")
    parser.add_argument("--checkpoint-interval", type=int, default=25)

    # Backend params
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["ollama", "vllm", "hf"])
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--base-url", type=str,
                        default="http://localhost:8000")

    # Feature flags
    parser.add_argument("--llm-annotator", action="store_true", default=False)
    parser.add_argument("--counterfactual", action="store_true", default=False)
    parser.add_argument("--no-structured-judge", action="store_true", default=False)
    parser.add_argument("--no-builtin-seeds", action="store_true", default=False)
    parser.add_argument("--seed-files", type=str, nargs="*", default=None)
    parser.add_argument("--dedup-threshold", type=float, default=0.70)

    # SLURM array mode
    parser.add_argument("--array-mode", action="store_true", default=False)

    args = parser.parse_args()

    # Create backend
    backend = create_backend(
        backend_type=args.backend,
        model=args.model,
        base_url=args.base_url,
    )

    # Wait for server
    if args.backend in ("vllm", "ollama"):
        print(f"Waiting for {args.backend} server at {args.base_url}...")
        backend.wait_until_ready(timeout=600, interval=10)
        print("Server ready.")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- GENERATION MODE ----
    if args.mode in ("gen", "full"):
        n_pairs = args.n_pairs
        shard_id = None

        if args.array_mode:
            task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
            array_size = int(os.environ.get("SLURM_ARRAY_TASK_COUNT",
                             os.environ.get("SLURM_ARRAY_TASK_MAX", 1)))
            if "SLURM_ARRAY_TASK_MAX" in os.environ and "SLURM_ARRAY_TASK_COUNT" not in os.environ:
                array_size = int(os.environ["SLURM_ARRAY_TASK_MAX"]) + 1
            start, count = compute_shard(n_pairs, task_id, array_size)
            shard_id = task_id
            n_pairs = count
            print(f"SLURM array task {task_id}/{array_size}: "
                  f"generating shard of {count} pairs (offset {start})")
            args.seed = args.seed + task_id * 10000

        seed_file_configs = None
        if args.seed_files:
            seed_file_configs = []
            for sf in args.seed_files:
                fmt = "csv" if sf.endswith(".csv") else "jsonl"
                seed_file_configs.append({"path": sf, "format": fmt})

        data = generate_dataset_hpc(
            backend=backend,
            n_pairs=n_pairs,
            paraphrase_variants=args.paraphrase_variants,
            use_llm_generator=True,
            use_local_paraphraser=True,
            use_llm_annotator=args.llm_annotator,
            seed_files=seed_file_configs,
            use_builtin_seeds=not args.no_builtin_seeds,
            dedup=True,
            dedup_threshold=args.dedup_threshold,
            random_seed=args.seed,
            misleading_fraction=args.misleading_fraction,
        )

        if shard_id is not None:
            out_path = os.path.join(
                args.output_dir,
                f"{args.output_name}_shard{shard_id:04d}.jsonl",
            )
        else:
            out_path = os.path.join(
                args.output_dir, f"{args.output_name}_raw.jsonl",
            )

        gen.write_jsonl(data, out_path)
        print(f"Wrote {len(data)} samples to {out_path}")

        if args.mode == "gen":
            return

        # For full mode, set input to generated output
        args.input = out_path

    # ---- VALIDATION MODE ----
    if args.mode in ("val", "full"):
        if args.input is None:
            print("ERROR: --input required for validation mode")
            sys.exit(1)

        out_path = os.path.join(
            args.output_dir, f"{args.output_name}_validated.jsonl",
        )

        validate_dataset_hpc(
            backend=backend,
            input_path=args.input,
            output_path=out_path,
            use_counterfactual=args.counterfactual,
            use_structured_judge=not args.no_structured_judge,
            checkpoint_interval=args.checkpoint_interval,
        )


if __name__ == "__main__":
    main()
