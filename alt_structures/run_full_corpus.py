#!/usr/bin/env python3
"""
run_full_corpus.py -- the no-LLM-needed candidates (eo-consensus's own
structure, pca-tertile, tree, surface, verbnet-lexical) scored on the
FULL 9,221-clause consensus corpus, not a balanced subsample.

These candidates don't need new local-LLM classification (they only need
embeddings + labels already in classified.jsonl), so unlike
vendler/halliday/srl/discourse they aren't bottlenecked by CPU LLM
generation time -- just embedding time, which is minutes not hours even
at this scale. This is the test that actually matches the original
analysis's own scale (WHY-THESE-THREE.md's 9,221-clause primary corpus),
rather than the small pilot samples used elsewhere in this suite.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness
from local_embeddings import embed_cached
from candidates.blind_geometric import GEOMETRIC_CANDIDATES
from candidates.verbnet_lexical import VERBNET_CANDIDATES

ALL_CANDIDATES = {**GEOMETRIC_CANDIDATES, **VERBNET_CANDIDATES}
CLASSIFIED = HERE.parent / "run_2026-03-15_122636" / "classified.jsonl"
Q1 = ["DIFFERENTIATING", "RELATING", "GENERATING"]
Q2 = ["EXISTENCE", "STRUCTURE", "SIGNIFICANCE"]
Q3 = ["CONDITION", "ENTITY", "PATTERN"]

def load_all_consensus():
    clauses, labels = [], []
    with open(CLASSIFIED) as f:
        for line in f:
            r = json.loads(line)
            c = r.get("consensus")
            if not c:
                continue
            clauses.append(r["clause"])
            labels.append([Q1.index(c["q1"]), Q2.index(c["q2"]), Q3.index(c["q3"])])
    return clauses, np.array(labels, dtype=int)

def main():
    t0 = time.time()
    clauses, labels = load_all_consensus()
    print(f"{len(clauses):,} consensus clauses loaded", file=sys.stderr)

    print("embedding (this is the slow step at this scale, cached)...", file=sys.stderr)
    X = embed_cached(clauses, HERE / "results" / "full_corpus_embeddings.npz")
    print(f"embedded in {time.time()-t0:.0f}s, shape={X.shape}", file=sys.stderr)

    report = {"n": len(clauses), "candidates": []}

    tr, te = harness.train_test_split(len(clauses), seed=0)
    sc = harness.score_structure(X[tr], labels[tr], X[te], labels[te], [3, 3, 3])
    null = harness.shuffled_null(X[tr], labels[tr], X[te], labels[te], [3, 3, 3], n_perm=100, seed=0)
    report["candidates"].append({"name": "eo-consensus", "n": len(clauses), **sc, "unseen_null": null})
    print(f"eo-consensus done ({time.time()-t0:.0f}s elapsed)", file=sys.stderr)

    for name, (needs_text, n_lev, factory) in ALL_CANDIDATES.items():
        t1 = time.time()
        cand = factory(0)
        try:
            if needs_text:
                ctr, cte = [clauses[i] for i in tr], [clauses[i] for i in te]
                cand.fit(ctr)
                ltr, lte = cand.apply(ctr), cand.apply(cte)
            else:
                cand.fit(X[tr])
                ltr, lte = cand.apply(X[tr]), cand.apply(X[te])
            sc = harness.score_structure(X[tr], ltr, X[te], lte, n_lev)
            null = harness.shuffled_null(X[tr], ltr, X[te], lte, n_lev, n_perm=100, seed=0)
            report["candidates"].append({"name": name, "n": len(clauses), "n_lev": n_lev, **sc, "unseen_null": null})
            print(f"{name} done in {time.time()-t1:.0f}s", file=sys.stderr)
        except Exception as e:
            report["candidates"].append({"name": name, "n": len(clauses), "error": str(e)})
            print(f"{name} FAILED: {e}", file=sys.stderr)

    print(f"\n{'candidate':32} {'n':>6} {'cells':>6} {'PAST':>8} {'FUTURE':>8} {'UNSEEN':>8} {'UNSEEN_z':>9}")
    print("-" * 80)
    for c in report["candidates"]:
        if "error" in c:
            print(f"{c['name']:32} {c['n']:6d}    -- {c['error']}")
            continue
        z = c.get("unseen_null", {}).get("z", float("nan"))
        print(f"{c['name']:32} {c['n']:6d} {c['cells']:6d} {c['past']:+8.4f} {c['future']:+8.4f} {c['unseen']:+8.4f} {z:9.2f}")

    out = HERE / "results" / "full_corpus_report.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\ntotal time {time.time()-t0:.0f}s, wrote {out}", file=sys.stderr)

if __name__ == "__main__":
    main()
