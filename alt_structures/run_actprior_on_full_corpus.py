#!/usr/bin/env python3
"""Scores actprior-lexical against the SAME cached full-corpus embeddings
and train/test split run_full_corpus.py already produced, appending to
full_corpus_report.json rather than re-embedding."""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness
from local_embeddings import embed_cached
from candidates.act_prior_lexical import ACT_PRIOR_CANDIDATES

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

t0 = time.time()
clauses, labels = load_all_consensus()
X = embed_cached(clauses, HERE / "results" / "full_corpus_embeddings.npz")  # cache hit, fast
print(f"{len(clauses)} clauses, embeddings loaded from cache in {time.time()-t0:.0f}s", file=sys.stderr)

tr, te = harness.train_test_split(len(clauses), seed=0)

report_path = HERE / "results" / "full_corpus_report.json"
report = json.loads(report_path.read_text())

for name, (needs_text, n_lev, factory) in ACT_PRIOR_CANDIDATES.items():
    t1 = time.time()
    cand = factory(0)
    ctr, cte = [clauses[i] for i in tr], [clauses[i] for i in te]
    cand.fit(ctr)
    ltr, lte = cand.apply(ctr), cand.apply(cte)
    n_covered = int((ltr[:, 0] != n_lev[0]-1).sum() + (lte[:, 0] != n_lev[0]-1).sum())
    sc = harness.score_structure(X[tr], ltr, X[te], lte, n_lev)
    null = harness.shuffled_null(X[tr], ltr, X[te], lte, n_lev, n_perm=100, seed=0)
    entry = {"name": name, "n": len(clauses), "n_lev": n_lev,
              "n_covered_by_lexicon": n_covered, **sc, "unseen_null": null}
    report["candidates"] = [c for c in report["candidates"] if c.get("name") != name] + [entry]
    print(f"{name}: covered {n_covered}/{len(clauses)} ({100*n_covered/len(clauses):.1f}%), "
          f"done in {time.time()-t1:.0f}s", file=sys.stderr)
    print(f"  PAST={sc['past']:+.4f} FUTURE={sc['future']:+.4f} UNSEEN={sc['unseen']:+.4f} z={null['z']:.2f}")

report_path.write_text(json.dumps(report, indent=2, default=float))
print(f"updated {report_path}", file=sys.stderr)
