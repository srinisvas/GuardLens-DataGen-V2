"""
contamination_check.py

Two-stage contamination detector for v12 dataset generation.

Stage 1 — LEXICAL (always on, HARD GATE):
  5-gram MinHash + LSH Jaccard against reference user turns. Fast,
  deterministic; catches verbatim and near-duplicate lexical reuse.
  Lexical hits → DROP AND REGENERATE.

Stage 2 — SEMANTIC (opt-in, QUARANTINE ONLY until calibrated):
  Sentence embeddings + cosine similarity via BAAI/bge-m3 (8192 token
  context, 24K char per chunk, sliding-window for longer records).
  Semantic-only hits → QUARANTINE for manual review. Do NOT automatically
  drop, because two independently authored contemporary conversations
  about the same broad topic (phishing, malware) can legitimately score
  high cosine similarity without contamination. After threshold
  calibration on real data, semantic can be promoted to a hard gate.

Both stages run at BOTH per-turn AND trajectory-level granularity.

Input record shapes supported by user-turn extraction:
  - v11/v12 pipeline: record["turns"] = [{role, text, turn_id}]
  - MHJ: record["turns"] = [{role, content}]  (role="human"/"user")
  - HarmBench: record["behavior"] = str
  - JailbreakBench: record["prompt"] = str
  - bench_objectives.jsonl: record["objective"] = str  (v4)

Setup:
    pip install datasketch
    pip install 'sentence-transformers>=2.7.0'    # for --semantic-check
    # bge-m3 auto-downloads on first use (~2.3GB)

Exit codes:
  0 -- clean (or only semantic hits, no lexical)  [permissive default]
  1 -- lexical hits (DROP AND REGENERATE required)
  3 -- semantic-only hits (QUARANTINE FOR REVIEW; no lexical hits)
  2 -- setup / config error

  Use --semantic-hard-gate to promote semantic hits to exit 1 (drop)
  once the threshold is calibrated. Off by default.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    print("ERROR: datasketch not installed. Run: pip install datasketch",
          file=sys.stderr)
    sys.exit(2)


# =========================================================
# Text extraction
# =========================================================

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _normalize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return _WORD_RE.findall(text.lower())


def _shingles(tokens: List[str], k: int) -> List[str]:
    if len(tokens) < k:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)]


def _user_turns(record: Dict) -> List[Tuple[int, str]]:
    """Extract (turn_id, text) for every text-bearing field.

    Supports v11/v12, MHJ, HarmBench, JailbreakBench, and objective-bank
    (record['objective']) shapes so contamination can be checked at
    every stage of the pipeline including the seed objectives.
    """
    out = []
    if "turns" in record and isinstance(record["turns"], list):
        for i, t in enumerate(record["turns"]):
            role = (t.get("role", "") or "").lower()
            if role in ("user", "human"):
                text = t.get("text") or t.get("content") or ""
                tid = t.get("turn_id", i)
                if text.strip():
                    out.append((tid, text))
        return out
    if "prompt" in record and isinstance(record["prompt"], str):
        out.append((0, record["prompt"]))
        return out
    if "behavior" in record and isinstance(record["behavior"], str):
        out.append((0, record["behavior"]))
        return out
    # v4: support bench_objectives.jsonl (record["objective"]) so the
    # objective bank itself can be contamination-checked before use.
    if "objective" in record and isinstance(record["objective"], str):
        out.append((0, record["objective"]))
        return out
    return out


def _concatenate_user_trajectory(turns: List[Tuple[int, str]]) -> str:
    return " ".join(t for _, t in turns)


def _record_id(record: Dict, fallback: str) -> str:
    for k in ("conversation_id", "id", "behavior_id", "prompt_id",
              "objective_id"):
        v = record.get(k)
        if isinstance(v, str) and v:
            return v
    return fallback


def load_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARN: bad JSON at {path}:{line_num}: {e}",
                      file=sys.stderr)


# =========================================================
# STAGE 1: Lexical
# =========================================================

def _minhash(shingles: List[str], num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 0.0
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class LexicalIndex:
    def __init__(self, shingle_size: int, num_perm: int, lsh_threshold: float):
        self.shingle_size = shingle_size
        self.num_perm = num_perm
        self.lsh_turn = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
        self.lsh_traj = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
        self.entries_turn: Dict[str, Dict] = {}
        self.entries_traj: Dict[str, Dict] = {}

    def index_reference(self, corpus_name: str, path: str) -> int:
        if not os.path.exists(path):
            print(f"  WARN: reference not found: {corpus_name} -> {path}",
                  file=sys.stderr)
            return 0
        n_turn = n_traj = 0
        for i, record in enumerate(load_jsonl(path)):
            rid = _record_id(record, fallback=f"{corpus_name}#{i}")
            turns = _user_turns(record)
            for tid, text in turns:
                sh = _shingles(_normalize(text), self.shingle_size)
                if not sh:
                    continue
                key = f"turn|{corpus_name}|{rid}|t{tid}"
                if key in self.entries_turn:
                    continue
                mh = _minhash(sh, self.num_perm)
                self.entries_turn[key] = {
                    "corpus": corpus_name, "record_id": rid, "turn_id": tid,
                    "text": text, "shingles": sh,
                }
                self.lsh_turn.insert(key, mh)
                n_turn += 1
            traj_text = _concatenate_user_trajectory(turns)
            if traj_text.strip():
                sh_traj = _shingles(_normalize(traj_text), self.shingle_size)
                if sh_traj:
                    key = f"traj|{corpus_name}|{rid}"
                    if key not in self.entries_traj:
                        mh = _minhash(sh_traj, self.num_perm)
                        self.entries_traj[key] = {
                            "corpus": corpus_name, "record_id": rid,
                            "text": traj_text[:800], "shingles": sh_traj,
                        }
                        self.lsh_traj.insert(key, mh)
                        n_traj += 1
        return n_turn + n_traj


def stage1_check_record(
    idx: LexicalIndex, candidate_record: Dict,
    jaccard_threshold: float, max_hits: int = 5,
) -> List[Dict]:
    hits = []
    turns = _user_turns(candidate_record)
    cid = _record_id(candidate_record, fallback="candidate")

    for tid, text in turns:
        sh = _shingles(_normalize(text), idx.shingle_size)
        if not sh:
            continue
        mh = _minhash(sh, idx.num_perm)
        for key in idx.lsh_turn.query(mh):
            ref = idx.entries_turn[key]
            j = _jaccard(sh, ref["shingles"])
            if j >= jaccard_threshold:
                hits.append({
                    "stage": "lexical",
                    "granularity": "per_turn",
                    "candidate_conversation_id": cid,
                    "candidate_turn_id": tid,
                    "candidate_text": text[:400],
                    "similarity": round(j, 3),
                    "metric": "jaccard_5gram",
                    "source_corpus": ref["corpus"],
                    "source_record_id": ref["record_id"],
                    "source_turn_id": ref["turn_id"],
                    "source_text": ref["text"][:400],
                })

    traj_text = _concatenate_user_trajectory(turns)
    sh_traj = _shingles(_normalize(traj_text), idx.shingle_size)
    if sh_traj:
        mh = _minhash(sh_traj, idx.num_perm)
        for key in idx.lsh_traj.query(mh):
            ref = idx.entries_traj[key]
            j = _jaccard(sh_traj, ref["shingles"])
            if j >= jaccard_threshold:
                hits.append({
                    "stage": "lexical",
                    "granularity": "trajectory",
                    "candidate_conversation_id": cid,
                    "candidate_text": traj_text[:400],
                    "similarity": round(j, 3),
                    "metric": "jaccard_5gram_concatenated",
                    "source_corpus": ref["corpus"],
                    "source_record_id": ref["record_id"],
                    "source_text": ref["text"][:400],
                })

    hits.sort(key=lambda h: -h["similarity"])
    return hits[:max_hits]


# =========================================================
# STAGE 2: Semantic (bge-m3)
# =========================================================

_BGE_M3_MAX_CHARS = 24000
_BGE_M3_CHUNK_CHARS = 20000
_BGE_M3_CHUNK_STRIDE = 15000


class SemanticIndex:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            print("ERROR: sentence-transformers not installed. "
                  "Run: pip install 'sentence-transformers>=2.7.0'",
                  file=sys.stderr)
            sys.exit(2)
        self.np = np
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.turn_texts: List[str] = []
        self.turn_meta: List[Dict] = []
        self.turn_embs = None
        self.traj_chunk_texts: List[str] = []
        self.traj_chunk_meta: List[Dict] = []
        self.traj_chunk_embs = None

    def _chunk_long_text(self, text: str) -> List[str]:
        if len(text) <= _BGE_M3_MAX_CHARS:
            return [text]
        chunks = []
        pos = 0
        while pos < len(text):
            chunks.append(text[pos:pos + _BGE_M3_CHUNK_CHARS])
            if pos + _BGE_M3_CHUNK_CHARS >= len(text):
                break
            pos += _BGE_M3_CHUNK_STRIDE
        return chunks

    def index_reference(self, corpus_name: str, path: str) -> int:
        if not os.path.exists(path):
            return 0
        n = 0
        for i, record in enumerate(load_jsonl(path)):
            rid = _record_id(record, fallback=f"{corpus_name}#{i}")
            turns = _user_turns(record)
            for tid, text in turns:
                if not text.strip():
                    continue
                self.turn_texts.append(text[:_BGE_M3_MAX_CHARS])
                self.turn_meta.append({
                    "corpus": corpus_name, "record_id": rid, "turn_id": tid,
                    "text": text,
                })
                n += 1
            traj_text = _concatenate_user_trajectory(turns)
            if traj_text.strip():
                for ci, chunk in enumerate(self._chunk_long_text(traj_text)):
                    self.traj_chunk_texts.append(chunk)
                    self.traj_chunk_meta.append({
                        "corpus": corpus_name, "record_id": rid,
                        "chunk_index": ci, "text": traj_text,
                    })
        return n

    def finalize(self, batch_size: int = 16):
        if self.turn_texts:
            print(f"    encoding {len(self.turn_texts)} per-turn references...")
            self.turn_embs = self.model.encode(
                self.turn_texts, batch_size=batch_size,
                normalize_embeddings=True, show_progress_bar=True,
            )
        if self.traj_chunk_texts:
            print(f"    encoding {len(self.traj_chunk_texts)} trajectory chunks...")
            self.traj_chunk_embs = self.model.encode(
                self.traj_chunk_texts, batch_size=batch_size,
                normalize_embeddings=True, show_progress_bar=True,
            )

    def check_record(
        self, candidate_record: Dict,
        threshold: float, max_hits: int = 5,
    ) -> List[Dict]:
        hits = []
        turns = _user_turns(candidate_record)
        cid = _record_id(candidate_record, fallback="candidate")

        if self.turn_embs is not None and turns:
            texts = [t[:_BGE_M3_MAX_CHARS] for _, t in turns]
            embs = self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False,
            )
            sims = self.np.matmul(embs, self.turn_embs.T)
            for i, (tid, text) in enumerate(turns):
                row = sims[i]
                idxs = self.np.argsort(-row)
                for j_idx in idxs[:max_hits]:
                    s = float(row[j_idx])
                    if s < threshold:
                        break
                    ref = self.turn_meta[j_idx]
                    hits.append({
                        "stage": "semantic",
                        "granularity": "per_turn",
                        "candidate_conversation_id": cid,
                        "candidate_turn_id": tid,
                        "candidate_text": text[:400],
                        "similarity": round(s, 3),
                        "metric": f"cosine_{self.model_name.split('/')[-1]}",
                        "source_corpus": ref["corpus"],
                        "source_record_id": ref["record_id"],
                        "source_turn_id": ref["turn_id"],
                        "source_text": ref["text"][:400],
                    })

        if self.traj_chunk_embs is not None and turns:
            traj_text = _concatenate_user_trajectory(turns)
            if traj_text.strip():
                cand_chunks = self._chunk_long_text(traj_text)
                cand_embs = self.model.encode(
                    cand_chunks, normalize_embeddings=True,
                    show_progress_bar=False,
                )
                sims = self.np.matmul(cand_embs, self.traj_chunk_embs.T)
                per_ref_best: Dict[Tuple[str, str], Tuple[float, Dict]] = {}
                for ci in range(len(cand_chunks)):
                    row = sims[ci]
                    for ri in range(len(self.traj_chunk_meta)):
                        s = float(row[ri])
                        if s < threshold:
                            continue
                        ref = self.traj_chunk_meta[ri]
                        key = (ref["corpus"], ref["record_id"])
                        if key not in per_ref_best or s > per_ref_best[key][0]:
                            per_ref_best[key] = (s, ref)
                sorted_hits = sorted(per_ref_best.values(),
                                     key=lambda x: -x[0])[:max_hits]
                for s, ref in sorted_hits:
                    hits.append({
                        "stage": "semantic",
                        "granularity": "trajectory",
                        "candidate_conversation_id": cid,
                        "candidate_text": traj_text[:400],
                        "similarity": round(s, 3),
                        "metric": (f"cosine_{self.model_name.split('/')[-1]}"
                                   f"_max_over_chunks"),
                        "source_corpus": ref["corpus"],
                        "source_record_id": ref["record_id"],
                        "source_text": ref["text"][:400],
                    })

        hits.sort(key=lambda h: -h["similarity"])
        return hits[:max_hits * 2]


# =========================================================
# Pipeline
# =========================================================

def check_candidate_dataset(
    candidate_path: str,
    lex_idx: LexicalIndex,
    sem_idx: Optional[SemanticIndex],
    jaccard_threshold: float,
    semantic_threshold: float,
    max_hits_per_record: int = 10,
    semantic_hard_gate: bool = False,
) -> Tuple[List[Dict], List[Dict], Dict]:
    hits: List[Dict] = []
    per_record: List[Dict] = []
    summary = {
        "candidate_path": candidate_path,
        "n_records": 0,
        "n_records_lexical_hit": 0,
        "n_records_semantic_only_hit": 0,
        "n_records_both_hit": 0,
        "n_records_drop_and_regenerate": 0,
        "n_records_quarantine_for_review": 0,
        "n_records_pass": 0,
        "hits_by_source_corpus": Counter(),
        "hits_by_stage_granularity": Counter(),
        "jaccard_threshold": jaccard_threshold,
        "semantic_threshold": semantic_threshold,
        "semantic_stage_enabled": sem_idx is not None,
        "semantic_hard_gate": semantic_hard_gate,
        "semantic_model": sem_idx.model_name if sem_idx else None,
    }

    for record in load_jsonl(candidate_path):
        summary["n_records"] += 1
        cid = _record_id(record, fallback=f"candidate#{summary['n_records']}")

        lex_hits = stage1_check_record(lex_idx, record, jaccard_threshold,
                                        max_hits=max_hits_per_record)
        sem_hits = []
        if sem_idx is not None:
            sem_hits = sem_idx.check_record(record, semantic_threshold,
                                             max_hits=max_hits_per_record)

        all_hits = lex_hits + sem_hits
        hits.extend(all_hits)

        contaminated_lex = len(lex_hits) > 0
        contaminated_sem = len(sem_hits) > 0

        if contaminated_lex:
            summary["n_records_lexical_hit"] += 1
            if contaminated_sem:
                summary["n_records_both_hit"] += 1
            action = "drop_and_regenerate"
            summary["n_records_drop_and_regenerate"] += 1
        elif contaminated_sem:
            summary["n_records_semantic_only_hit"] += 1
            if semantic_hard_gate:
                action = "drop_and_regenerate"
                summary["n_records_drop_and_regenerate"] += 1
            else:
                action = "quarantine_for_review"
                summary["n_records_quarantine_for_review"] += 1
        else:
            action = "pass"
            summary["n_records_pass"] += 1

        if contaminated_lex or contaminated_sem:
            for h in all_hits:
                summary["hits_by_source_corpus"][h["source_corpus"]] += 1
                summary["hits_by_stage_granularity"][
                    f"{h['stage']}/{h['granularity']}"
                ] += 1

        worst_lex = max((h["similarity"] for h in lex_hits), default=0.0)
        worst_sem = max((h["similarity"] for h in sem_hits), default=0.0)
        per_record.append({
            "candidate_conversation_id": cid,
            "worst_lexical_jaccard": round(worst_lex, 3),
            "worst_semantic_cosine": round(worst_sem, 3),
            "action_required": action,
            "n_lexical_hits": len(lex_hits),
            "n_semantic_hits": len(sem_hits),
        })

    return hits, per_record, summary


# =========================================================
# CLI
# =========================================================

def parse_reference_arg(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"Need 'name:path', got: {spec}")
    name, path = spec.split(":", 1)
    return name.strip(), os.path.expanduser(path.strip())


def main():
    p = argparse.ArgumentParser(description="Two-stage contamination check")
    p.add_argument("--candidate", required=True)
    p.add_argument("--references", nargs="+", required=True,
                   type=parse_reference_arg, metavar="NAME:PATH")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shingle-size", type=int, default=5)
    p.add_argument("--num-perm", type=int, default=128)
    p.add_argument("--jaccard-threshold", type=float, default=0.6)
    p.add_argument("--lsh-threshold", type=float, default=0.55)
    p.add_argument("--semantic-check", action="store_true")
    p.add_argument("--semantic-threshold", type=float, default=0.82,
                   help="Cosine threshold for semantic contamination "
                        "(bge-m3 provisional 0.82; calibrate on real data)")
    p.add_argument("--semantic-hard-gate", action="store_true",
                   help="Promote semantic hits to drop_and_regenerate. "
                        "Off by default; use only after threshold calibration.")
    p.add_argument("--semantic-model", default="BAAI/bge-m3")
    p.add_argument("--max-hits-per-record", type=int, default=10)
    args = p.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nStage 1: Building lexical index (5-gram MinHash+LSH)...")
    lex_idx = LexicalIndex(args.shingle_size, args.num_perm, args.lsh_threshold)
    t0 = time.time()
    for corpus, path in args.references:
        n = lex_idx.index_reference(corpus, path)
        print(f"  {corpus}: {n} units")
    print(f"  lexical index built in {time.time() - t0:.1f}s")

    sem_idx = None
    if args.semantic_check:
        print(f"\nStage 2: Building semantic index ({args.semantic_model})...")
        gate_mode = ("HARD GATE (drop)" if args.semantic_hard_gate
                     else "QUARANTINE ONLY (review)")
        print(f"  semantic threshold: {args.semantic_threshold} "
              f"[mode: {gate_mode}]")
        if not args.semantic_hard_gate:
            print(f"  Semantic-only hits will be quarantined, not dropped. "
                  f"Promote to hard gate with --semantic-hard-gate after "
                  f"calibration on real v12 data.")
        sem_idx = SemanticIndex(model_name=args.semantic_model)
        t0 = time.time()
        for corpus, path in args.references:
            n = sem_idx.index_reference(corpus, path)
            print(f"  {corpus}: {n} units")
        sem_idx.finalize()
        print(f"  semantic index built in {time.time() - t0:.1f}s")
    else:
        print(f"\nStage 2 SKIPPED. Only lexical near-duplicates caught.")

    print(f"\nChecking candidate: {args.candidate}")
    t0 = time.time()
    hits, per_record, summary = check_candidate_dataset(
        os.path.expanduser(args.candidate), lex_idx, sem_idx,
        args.jaccard_threshold, args.semantic_threshold,
        args.max_hits_per_record, semantic_hard_gate=args.semantic_hard_gate,
    )
    print(f"  scan complete in {time.time() - t0:.1f}s")

    with open(os.path.join(output_dir, "contamination_hits.jsonl"), "w") as f:
        for h in hits:
            f.write(json.dumps(h) + "\n")
    with open(os.path.join(output_dir, "contamination_gate.jsonl"), "w") as f:
        for r in per_record:
            f.write(json.dumps(r) + "\n")
    summary_dict = dict(summary)
    summary_dict["hits_by_source_corpus"] = dict(summary["hits_by_source_corpus"])
    summary_dict["hits_by_stage_granularity"] = dict(
        summary["hits_by_stage_granularity"]
    )
    with open(os.path.join(output_dir, "contamination_summary.json"), "w") as f:
        json.dump(summary_dict, f, indent=2, sort_keys=True)

    print(f"\n{'=' * 60}")
    print(f"Contamination summary")
    print(f"{'=' * 60}")
    print(f"  records:                    {summary['n_records']}")
    print(f"  lexical hits (hard):        {summary['n_records_lexical_hit']}")
    print(f"  semantic-only hits:         {summary['n_records_semantic_only_hit']}")
    print(f"  both stages:                {summary['n_records_both_hit']}")
    print(f"")
    print(f"  action required:")
    print(f"    drop_and_regenerate:      {summary['n_records_drop_and_regenerate']}")
    print(f"    quarantine_for_review:    {summary['n_records_quarantine_for_review']}")
    print(f"    pass:                     {summary['n_records_pass']}")
    print(f"\n  Hits by source corpus:")
    for corpus, n in summary["hits_by_source_corpus"].most_common():
        print(f"    {corpus}: {n}")
    print(f"\n  Hits by stage/granularity:")
    for k, n in summary["hits_by_stage_granularity"].most_common():
        print(f"    {k}: {n}")

    if summary["n_records_drop_and_regenerate"] > 0:
        print(f"\n  ACTION: drop and regenerate the "
              f"{summary['n_records_drop_and_regenerate']} lexically "
              f"contaminated records. Do NOT paraphrase.")
        sys.exit(1)
    elif summary["n_records_quarantine_for_review"] > 0:
        print(f"\n  ACTION: manually review the "
              f"{summary['n_records_quarantine_for_review']} semantic-only "
              f"quarantined records. Two independently authored "
              f"contemporary conversations about the same topic may "
              f"legitimately score high cosine similarity without "
              f"contamination. Calibrate the threshold before promoting "
              f"semantic to a hard gate with --semantic-hard-gate.")
        sys.exit(3)
    else:
        print(f"\n  Clean.")
        sys.exit(0)


if __name__ == "__main__":
    main()
