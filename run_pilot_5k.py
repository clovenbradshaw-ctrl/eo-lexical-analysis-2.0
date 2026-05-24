#!/usr/bin/env python3
"""run_pilot_5k.py — Drive the 5,000-clause-per-counter-set pilot for the
question-set falsifiability test.

For each registered question set (except `eo`), this script:
  1. samples a stratified 5,000-clause subset from a base run-dir's
     classified.jsonl (stratified by the existing EO 27-cell so every cell
     gets proportional representation, capped at min(cell_size, allotment));
  2. classifies the same 5,000 ids under the counter set using Claude +
     GPT-4o (consensus), writing into a fresh `<base>/pilot/<set_name>/`
     classified.jsonl in the *nested-by-set* shape;
  3. copies (or hard-links) the matching embedding subset into
     `<base>/pilot/<set_name>/embeddings.npz`;
  4. runs falsify_3x3x3.py against that pilot dir.

The aggregator falsify_aggregate.py then reads every per-set
falsify_results.json + per-set classified.jsonl and produces the cross-set
comparison and the pre-registered verdict.

The EO baseline numbers are loaded from the base run-dir's existing
falsify/falsify_results.json (no re-classification needed for EO itself).

API keys:
  ANTHROPIC_API_KEY   — required (Claude classifier)
  OPENAI_API_KEY      — required (GPT-4o classifier)

Usage:
  python run_pilot_5k.py --run-dir run_2026-03-19_144302 \
      --sets tense,register,agency,aristotle,peirce,hegel,eo_q1_split
  python run_pilot_5k.py --run-dir run_2026-03-19_144302 --sets all --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import question_sets
from falsify_3x3x3 import labels_for_ids, load_classified


SAMPLE_PER_SET = 5000


def stratified_sample(classified: Dict[str, dict], ids_in_emb: List[str],
                      n: int, seed: int = 0) -> List[str]:
    """Stratify by the existing EO 27-cell so every cell gets a proportional
    share of the n samples, capped by cell membership."""
    rng = random.Random(seed)
    # Bucket ids by EO 27-cell using the existing classified.jsonl
    valid_ids = [cid for cid in ids_in_emb if cid in classified]
    eo_q1_vals = question_sets.get("eo").axis_value_lists["q1"]
    eo_q2_vals = question_sets.get("eo").axis_value_lists["q2"]
    eo_q3_vals = question_sets.get("eo").axis_value_lists["q3"]
    buckets: Dict[tuple, List[str]] = defaultdict(list)
    unclassified: List[str] = []
    for cid in valid_ids:
        rec = classified[cid]
        cons = rec.get("consensus")
        cls = rec.get("classifications", {}) or {}
        chosen = None
        if isinstance(cons, dict) and "q1" in cons:
            chosen = cons
        elif "claude" in cls:
            chosen = cls["claude"]
        elif "gpt4" in cls:
            chosen = cls["gpt4"]
        if not chosen:
            unclassified.append(cid)
            continue
        q1, q2, q3 = chosen.get("q1"), chosen.get("q2"), chosen.get("q3")
        # Accept both legacy "PARTICULAR" and new "ENTITY" for Q3
        if q3 == "ENTITY":
            q3 = "PARTICULAR"
        if q1 in eo_q1_vals and q2 in eo_q2_vals and q3 in {"CONDITION", "PARTICULAR", "PATTERN"}:
            buckets[(q1, q2, q3)].append(cid)
        else:
            unclassified.append(cid)
    if not buckets:
        # Fallback: uniform sample
        rng.shuffle(valid_ids)
        return valid_ids[:n]
    total_classified = sum(len(v) for v in buckets.values())
    chosen: List[str] = []
    for cell, members in buckets.items():
        share = max(1, round(n * len(members) / total_classified))
        take = min(share, len(members))
        chosen.extend(rng.sample(members, take))
    # Trim/extend to exactly n
    rng.shuffle(chosen)
    if len(chosen) > n:
        chosen = chosen[:n]
    elif len(chosen) < n and unclassified:
        # Pad from unclassified to reach n (these will be classified fresh too)
        need = n - len(chosen)
        rng.shuffle(unclassified)
        chosen.extend(unclassified[:need])
    return chosen


def load_embeddings_npz(path: Path):
    npz = np.load(path)
    vectors = npz["vectors"] if "vectors" in npz.files else npz["embeddings"]
    ids = [str(x) for x in npz["ids"]]
    return vectors, ids


def slice_embeddings(base_npz: Path, ids_keep: List[str], out_npz: Path) -> None:
    vectors, all_ids = load_embeddings_npz(base_npz)
    idx_map = {cid: i for i, cid in enumerate(all_ids)}
    idx = np.array([idx_map[cid] for cid in ids_keep if cid in idx_map], dtype=np.int64)
    if len(idx) != len(ids_keep):
        missing = [cid for cid in ids_keep if cid not in idx_map]
        print(f"  warning: {len(missing)} pilot ids missing from embeddings.npz; dropping them")
    np.savez(out_npz, vectors=vectors[idx], ids=np.array([all_ids[i] for i in idx]))
    print(f"  wrote {out_npz} ({len(idx)} embeddings)")


def classify_pilot(set_name: str, clauses: List[dict], out_jsonl: Path,
                   anthropic_key: str, openai_key: str, resume: bool = True) -> None:
    """Classify a pilot subset under `set_name`. Writes the nested-by-set
    shape: rec.classifications[set_name][model] = {q1, q2, q3, ...}."""
    import app2  # imports register the EO entry; pulls in classifiers

    qs = question_sets.get(set_name)
    # Lazy import to avoid forcing API SDKs at module load
    from anthropic import Anthropic
    from openai import OpenAI

    anth = Anthropic(api_key=anthropic_key)
    oai = OpenAI(api_key=openai_key)

    existing: Dict[str, dict] = {}
    if resume and out_jsonl.exists():
        with out_jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    existing[r["id"]] = r
                except Exception:
                    pass

    def has_set(rec: dict) -> bool:
        cls = rec.get("classifications", {}) or {}
        per_set = cls.get(set_name, {}) or {}
        return ("claude" in per_set) and ("gpt4" in per_set)

    todo = [c for c in clauses if not has_set(existing.get(c["id"], {}))]
    print(f"[{set_name}] {len(todo)} clauses to classify "
          f"({len(clauses) - len(todo)} already done)")

    t0 = time.time()
    errors = 0
    with out_jsonl.open("a", encoding="utf-8") as f:
        # Rewrite-on-update: simplest pattern is to keep the file append-only
        # per pilot run and let the loader take the latest line per id (we
        # dedupe in the aggregator). For pilots this is fine; for production
        # use atomic rewrite.
        for i, clause_data in enumerate(todo):
            cid = clause_data["id"]
            base = existing.get(cid) or {**clause_data, "classifications": {}, "consensus": {}}
            cls = base.setdefault("classifications", {})
            cons = base.setdefault("consensus", {})
            per_set_cls = cls.setdefault(set_name, {})
            text = clause_data.get("clause") or clause_data.get("text", "")

            r_claude = app2.classify_clause_anthropic(text, anth, question_set=qs)
            time.sleep(0.2)
            r_gpt = app2.classify_clause_openai(text, oai, model="gpt-4o", question_set=qs)
            time.sleep(0.1)
            if r_claude:
                per_set_cls["claude"] = r_claude
            else:
                errors += 1
            if r_gpt:
                per_set_cls["gpt4"] = r_gpt
            else:
                errors += 1
            # Per-set consensus: agree on all three axes
            if "claude" in per_set_cls and "gpt4" in per_set_cls:
                a, b = per_set_cls["claude"], per_set_cls["gpt4"]
                if all(a.get(k) == b.get(k) for k in ("q1", "q2", "q3")):
                    cons[set_name] = {k: a[k] for k in ("q1", "q2", "q3")}
            f.write(json.dumps(base, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (len(todo) - i - 1) / max(rate, 1e-6)
                print(f"  [{set_name}] {i+1}/{len(todo)}  "
                      f"{rate:.1f}/s  ETA {eta/60:.0f}m  errors {errors}")
    print(f"[{set_name}] done in {(time.time()-t0)/60:.1f}m, errors={errors}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Base run-dir with classified.jsonl + embeddings.npz")
    ap.add_argument("--sets", default="all", help="Comma-separated set names or 'all' (excludes 'eo')")
    ap.add_argument("--n", type=int, default=SAMPLE_PER_SET, help=f"Pilot sample size (default {SAMPLE_PER_SET})")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="Stop before any LLM calls; report plan + cost only")
    ap.add_argument("--skip-classification", action="store_true",
                    help="Reuse existing per-set classified.jsonl; only run falsifier + aggregator")
    args = ap.parse_args()

    base = Path(args.run_dir)
    base_classified = load_classified(base)
    emb_path = base / "embeddings.npz"
    base_ids: List[str]
    if emb_path.stat().st_size < 4096:
        # Drive pointer; fall back to classified.jsonl id ordering
        print(f"  note: {emb_path} is a placeholder ({emb_path.stat().st_size} bytes); "
              f"using classified.jsonl id ordering. Real embedding slicing will run "
              f"once the actual npz is fetched.")
        base_ids = list(base_classified.keys())
    else:
        _, base_ids = load_embeddings_npz(emb_path)
    print(f"Base run-dir: {base}   n_ids={len(base_ids)}   n_classified={len(base_classified)}")

    if args.sets == "all":
        target_sets = [n for n in question_sets.names() if n != "eo"]
    else:
        target_sets = [s.strip() for s in args.sets.split(",") if s.strip()]
        for s in target_sets:
            question_sets.get(s)  # validate

    # Stratified sample (shared across sets so cross-set ARI is meaningful)
    sample_ids = stratified_sample(base_classified, base_ids, args.n, seed=args.seed)
    print(f"Pilot sample: {len(sample_ids)} clauses (target {args.n})")
    cell_counts = Counter()
    for cid in sample_ids:
        rec = base_classified.get(cid, {})
        cons = rec.get("consensus")
        if isinstance(cons, dict) and "q1" in cons:
            cell_counts[(cons.get("q1"), cons.get("q2"), cons.get("q3"))] += 1
    print(f"  EO 27-cell cells represented in sample: {len(cell_counts)}/27")

    pilot_root = base / "pilot"
    pilot_root.mkdir(exist_ok=True)
    manifest = {
        "base_run_dir": str(base),
        "n_per_set": args.n,
        "seed": args.seed,
        "sample_ids_count": len(sample_ids),
        "sets": target_sets,
    }
    (pilot_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (pilot_root / "sample_ids.json").write_text(json.dumps(sample_ids))
    print(f"Wrote {pilot_root / 'manifest.json'} and sample_ids.json")

    # Cost estimate
    calls = len(sample_ids) * len(target_sets) * 2  # claude + gpt4
    tokens_per_call_in, tokens_per_call_out = 350, 30
    tokens_in = calls * tokens_per_call_in
    tokens_out = calls * tokens_per_call_out
    cost = tokens_in / 1e6 * 3.0 + tokens_out / 1e6 * 10.0  # blended rough estimate
    print(f"Cost estimate: {calls} calls (~{tokens_in/1e6:.1f}M in / "
          f"{tokens_out/1e6:.2f}M out)  ≈ ${cost:.0f}")

    if args.dry_run:
        print("--dry-run: stopping before classification.")
        return 0

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not args.skip_classification and not (anthropic_key and openai_key):
        print("ERROR: ANTHROPIC_API_KEY and OPENAI_API_KEY env vars required "
              "(or pass --skip-classification to reuse existing per-set jsonl).",
              file=sys.stderr)
        return 2

    # Build the clause records list (id + clause text) from raw_clauses.jsonl
    # if available; otherwise pull text from base_classified records.
    raw_path = base / "raw_clauses.jsonl"
    clause_text: Dict[str, dict] = {}
    if raw_path.exists():
        with raw_path.open() as f:
            for line in f:
                r = json.loads(line)
                clause_text[r["id"]] = r
    else:
        for cid, rec in base_classified.items():
            clause_text[cid] = {"id": cid, "clause": rec.get("clause", rec.get("text", ""))}

    clauses_for_pilot = [clause_text[cid] for cid in sample_ids if cid in clause_text]
    if len(clauses_for_pilot) != len(sample_ids):
        print(f"  warning: only {len(clauses_for_pilot)}/{len(sample_ids)} pilot ids have clause text")

    for set_name in target_sets:
        set_dir = pilot_root / set_name
        set_dir.mkdir(exist_ok=True)
        cls_file = set_dir / "classified.jsonl"
        emb_file = set_dir / "embeddings.npz"
        # Copy embedding subset for this pilot
        if not emb_file.exists():
            slice_embeddings(base / "embeddings.npz", sample_ids, emb_file)
        if not args.skip_classification:
            classify_pilot(set_name, clauses_for_pilot, cls_file,
                           anthropic_key=anthropic_key, openai_key=openai_key)
        # Run the falsifier panel on this pilot
        print(f"[{set_name}] running falsify_3x3x3.py ...")
        import subprocess
        subprocess.run([
            sys.executable, "falsify_3x3x3.py",
            "--run-dir", str(set_dir),
            "--question-set", set_name,
            "--out-dir", str(set_dir / "falsify"),
        ], check=False)

    print("\nAll pilot sets complete. Run:")
    print(f"  python falsify_aggregate.py --pilot-dir {pilot_root} --base-run-dir {base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
