#!/usr/bin/env python3
"""
run_phasepost_live.py -- does the cube earn its keep INSIDE the live
pipeline, not just in isolation? PR#18's own named next step, now that
this session has eoreader7/live_priors access.

Runs eoreader7's REAL extractRelations -> phasepost.js (backed by
live_priors' REAL ActPrior@1) over the English-only corpus
(run_2026-03-19_144302: 3,595 clauses, 1,354 with claude/gpt4 consensus --
6.4x PR#18's own 213-clause English subset), then scores the result
through this suite's own harness.py, on the identical held-out split
eo-consensus/pca-tertile/the naive act_prior_lexical.py are scored on, so
every number in the printed table is a fair, apples-to-apples comparison.

Two questions, kept apart:
  1. COVERAGE -- across all 3,595 clauses (consensus not required), what
     fraction does the real pipeline even produce a typed verdict for,
     and at which standing (mechanical/copula/lexical/contested/gap/
     no_match)? This alone tests whether P56's "never coin-flip"
     discipline survives contact with a real, independent clause corpus
     the module was never tuned against.
  2. STRUCTURE -- on the 1,354-clause consensus subset (needed for a fair
     n-matched comparison against eo-consensus), does the real pipeline's
     op/cell verdict correlate with embedding geometry the way EO's own
     labels, or a naive first-candidate policy, or a blind PCA split, do?
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness
from local_embeddings import embed_cached
from candidates.blind_geometric import GEOMETRIC_CANDIDATES
from candidates.act_prior_lexical import ACT_PRIOR_CANDIDATES
from candidates.phasepost_live import make_live_candidates, coverage_report

CLASSIFIED = HERE.parent / "run_2026-03-19_144302" / "classified.jsonl"
Q1 = ["DIFFERENTIATING", "RELATING", "GENERATING"]
Q2 = ["EXISTENCE", "STRUCTURE", "SIGNIFICANCE"]
Q3 = ["CONDITION", "ENTITY", "PATTERN"]


def load_corpus():
    all_ids, all_clauses = [], []
    cons_ids, cons_clauses, cons_labels = [], [], []
    with open(CLASSIFIED) as f:
        for line in f:
            r = json.loads(line)
            if r.get("language") != "en":
                continue
            all_ids.append(r["id"])
            all_clauses.append(r["clause"])
            c = r.get("consensus")
            if c:
                cons_ids.append(r["id"])
                cons_clauses.append(r["clause"])
                cons_labels.append([Q1.index(c["q1"]), Q2.index(c["q2"]), Q3.index(c["q3"])])
    limit = os.environ.get("PHASEPOST_LIVE_LIMIT")
    if limit:
        limit = int(limit)
        all_ids, all_clauses = all_ids[:limit], all_clauses[:limit]
        cons_ids, cons_clauses, cons_labels = cons_ids[:limit], cons_clauses[:limit], cons_labels[:limit]
    return all_ids, all_clauses, cons_ids, cons_clauses, np.array(cons_labels, dtype=int)


def print_row(name, n, cells, sc, z):
    print(f"{name:38} {n:6d} {cells:6d} {sc['past']:+8.4f} {sc['future']:+8.4f} {sc['unseen']:+8.4f} {z:9.2f}")


def main():
    t0 = time.time()
    all_ids, all_clauses, cons_ids, cons_clauses, cons_labels = load_corpus()
    print(f"{len(all_clauses):,} English clauses total, {len(cons_clauses):,} with claude/gpt4 consensus "
          f"({len(cons_clauses) / 213:.1f}x PR#18's own 213-clause English subset)", file=sys.stderr)

    report = {"n_all": len(all_clauses), "n_consensus": len(cons_clauses)}

    # ---- 1. COVERAGE, over every English clause, consensus or not ----
    print("\nrunning the real live pipeline (extractRelations -> phasepost.js) over all clauses...", file=sys.stderr)
    dr45_flag = "--dr45" in sys.argv
    all_id_to_clause = dict(zip(all_ids, all_clauses))
    _, _, all_results = make_live_candidates(all_id_to_clause, dr45=dr45_flag)
    cov = coverage_report(all_results)
    total = sum(cov.values())
    print(f"\nCOVERAGE across all {total} English clauses (dr45={dr45_flag}):")
    for k, v in sorted(cov.items(), key=lambda kv: -kv[1]):
        print(f"  {k:55} {v:5d}  ({100*v/total:.1f}%)")
    report["coverage_all_english"] = cov

    # sanity: P56 never-coin-flip check -- a "contested" verdict must never
    # carry a resolved op, on every single row this ran over.
    contested_with_op = [r for r in all_results.values() if r.get("standing") == "contested" and r.get("op")]
    assert not contested_with_op, f"P56 VIOLATED: {len(contested_with_op)} contested verdicts carried a resolved op"
    print(f"\nP56 check: {cov.get('contested', 0)} contested verdicts, 0 carried a resolved op (confirmed).", file=sys.stderr)

    # ---- 2. STRUCTURE, on the consensus subset (fair n-matched comparison) ----
    print("\nembedding the consensus subset (cached)...", file=sys.stderr)
    X = embed_cached(cons_clauses, HERE / "results" / "phasepost_live_embeddings.npz")
    print(f"embedded {X.shape[0]} clauses in {time.time()-t0:.0f}s", file=sys.stderr)

    tr, te = harness.train_test_split(len(cons_clauses), seed=0)
    Xtr, Xte = X[tr], X[te]

    rows = []

    def score_and_record(name, ltr, lte, n_lev):
        sc = harness.score_structure(Xtr, ltr, Xte, lte, n_lev)
        null = harness.shuffled_null(Xtr, ltr, Xte, lte, n_lev, n_perm=200, seed=0)
        rows.append({"name": name, "n": len(cons_clauses), "n_lev": n_lev, **sc, "unseen_null": null})
        print_row(name, len(cons_clauses), sc["cells"], sc, null["z"])

    print(f"\n{'candidate':38} {'n':>6} {'cells':>6} {'PAST':>8} {'FUTURE':>8} {'UNSEEN':>8} {'UNSEEN_z':>9}")
    print("-" * 96)

    score_and_record("eo-consensus", cons_labels[tr], cons_labels[te], [3, 3, 3])

    live_cands, cell_cand, cons_results = make_live_candidates(dict(zip(cons_ids, cons_clauses)), dr45=dr45_flag)
    # id-keyed (not clause-text-keyed, see phasepost_live.py's own note on
    # this corpus's 305 duplicate-text rows): tr/te are positions into
    # cons_clauses/cons_ids, so map through cons_ids for the live candidates.
    idtr, idte = [cons_ids[i] for i in tr], [cons_ids[i] for i in te]
    ctr = [cons_clauses[i] for i in tr]  # still needed below for act_prior_lexical/pca-tertile
    cte = [cons_clauses[i] for i in te]
    for name, (_, n_lev, factory) in live_cands.items():
        c = factory()
        c.fit(idtr)
        score_and_record(name, c.apply(idtr), c.apply(idte), n_lev)
    cell_cand.fit(idtr)
    score_and_record("phasepost-live-cell(op x grain)", cell_cand.apply(idtr), cell_cand.apply(idte), cell_cand.n_lev)

    # old naive baseline (PR#18), for a direct before/after on the SAME subset
    name, (_, n_lev, factory) = list(ACT_PRIOR_CANDIDATES.items())[0]
    c = factory(0)
    c.fit(ctr)
    score_and_record(f"{name} [PR#18 baseline, same subset]", c.apply(ctr), c.apply(cte), n_lev)

    # pca-tertile, the standing "does a blind split beat this" comparator
    for name, (needs_text, n_lev, factory) in GEOMETRIC_CANDIDATES.items():
        if "pca-tertile" not in name and "tree" not in name:
            continue
        c = factory(0)
        try:
            if needs_text:
                c.fit(ctr)
                score_and_record(name, c.apply(ctr), c.apply(cte), n_lev)
            else:
                c.fit(Xtr)
                score_and_record(name, c.apply(Xtr), c.apply(Xte), n_lev)
        except Exception as e:
            print(f"{name} FAILED: {e}", file=sys.stderr)

    report["structure_rows"] = rows
    report["dr45"] = dr45_flag
    out = HERE / "results" / ("phasepost_live_report_dr45.json" if dr45_flag else "phasepost_live_report.json")
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\ntotal time {time.time()-t0:.0f}s, wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
