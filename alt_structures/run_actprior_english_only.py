#!/usr/bin/env python3
"""actprior-lexical, restricted to English clauses only -- act-prior-en.json
is an English lexicon; testing it against a 97%-non-English corpus (as the
first pass did, 4.5% coverage) wasn't a fair test of whether it structures
what it can actually see."""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness
from local_embeddings import embed_cached
from candidates.act_prior_lexical import ACT_PRIOR_CANDIDATES, main_verb_lemma, verb_act

CLASSIFIED = HERE.parent / "run_2026-03-15_122636" / "classified.jsonl"
Q1 = ["DIFFERENTIATING", "RELATING", "GENERATING"]
Q2 = ["EXISTENCE", "STRUCTURE", "SIGNIFICANCE"]
Q3 = ["CONDITION", "ENTITY", "PATTERN"]

clauses, labels = [], []
with open(CLASSIFIED) as f:
    for line in f:
        r = json.loads(line)
        c = r.get("consensus")
        if not c or r.get("language") != "en":
            continue
        clauses.append(r["clause"])
        labels.append([Q1.index(c["q1"]), Q2.index(c["q2"]), Q3.index(c["q3"])])
labels = np.array(labels, dtype=int)
print(f"{len(clauses)} English consensus clauses", file=sys.stderr)

t0 = time.time()
X = embed_cached(clauses, HERE / "results" / "english_only_embeddings.npz")
print(f"embedded in {time.time()-t0:.0f}s", file=sys.stderr)

# coverage check
covered = sum(1 for c in clauses if verb_act(main_verb_lemma(c)) is not None)
print(f"lexicon covers {covered}/{len(clauses)} ({100*covered/len(clauses):.1f}%) of English clauses", file=sys.stderr)

tr, te = harness.train_test_split(len(clauses), seed=0)

name, (needs_text, n_lev, factory) = list(ACT_PRIOR_CANDIDATES.items())[0]
cand = factory(0)
ctr, cte = [clauses[i] for i in tr], [clauses[i] for i in te]
cand.fit(ctr)
ltr, lte = cand.apply(ctr), cand.apply(cte)
sc = harness.score_structure(X[tr], ltr, X[te], lte, n_lev)
null = harness.shuffled_null(X[tr], ltr, X[te], lte, n_lev, n_perm=200, seed=0)
print(f"\n{name} (English-only, n={len(clauses)}):")
print(f"  cells={sc['cells']} PAST={sc['past']:+.4f} FUTURE={sc['future']:+.4f} UNSEEN={sc['unseen']:+.4f} z={null['z']:.2f}")

# also: EO-consensus on this SAME English-only subset, for a same-n comparison
sc_eo = harness.score_structure(X[tr], labels[tr], X[te], labels[te], [3,3,3])
null_eo = harness.shuffled_null(X[tr], labels[tr], X[te], labels[te], [3,3,3], n_perm=200, seed=0)
print(f"eo-consensus (English-only, same n={len(clauses)}):")
print(f"  cells={sc_eo['cells']} PAST={sc_eo['past']:+.4f} FUTURE={sc_eo['future']:+.4f} UNSEEN={sc_eo['unseen']:+.4f} z={null_eo['z']:.2f}")

out = {"n": len(clauses), "coverage": covered/len(clauses),
       "actprior": {**sc, "unseen_null": null}, "eo_consensus_same_n": {**sc_eo, "unseen_null": null_eo}}
(HERE / "results" / "actprior_english_only_report.json").write_text(json.dumps(out, indent=2, default=float))
