#!/usr/bin/env python3
"""
domain_filter.py — the question the study never asked.

CLASSIFICATION_PROMPT in app2.py opens: "Answer these three questions about the
transformation this clause describes." There is no refusal option. Every clause
is forced into one of the 27 cells, and nothing anywhere asks whether the clause
describes a transformation at all. Clause selection was syntactic (>=1 VERB,
8-30 tokens, declarative), which admits pure statives, properties, attitude
reports and directives.

In Chomsky's terms that is a stage-1 failure: the cognitive domain D was never
delimited, it was assumed. This script measures the size of the problem and
tries to repair it post hoc.

  1. AGREEMENT IS NOT A DOMAIN SIGNAL. If a clause is not a transformation the
     three questions have no determinate answer, so two labellers should
     converge at chance -- a tempting free filter. Measured against hand
     judgments it fails: Spearman r = -0.015, p = 0.91. The dose-response of
     geometry against agreement is ordinary label noise, not domain membership.

  2. A FILTER IS BUILDABLE. 162 clean hand judgments
     (data/transformation-judgments.json) train a logistic regression on the
     run's own embeddings at 5-fold AUC 0.907 / accuracy 0.833. Applied
     corpus-wide it calls 28% in-domain, against a hand-judged English rate of
     32% strict / 52% inclusive.

  3. AND THE CORPUS STILL CANNOT ANSWER THE QUESTION. Only 2,710 consensus
     clauses survive the filter, and --power shows EO's estimator crosses zero
     at n ~ 2,700: at that sample size a matched RANDOM subsample also scores
     nothing. The in-domain arm is underpowered by construction, and its cell
     occupancy is worse besides (evenness 0.664 vs 0.771, one cell holding a
     single clause). Testing the domain hypothesis needs filtering BEFORE
     labelling, not after -- a re-run of the labelling pass, not a post-hoc fix.

The power curve is a finding in its own right: EO's scores rise monotonically
with n and have not plateaued at the full 9,221. The often-quoted ~1% of
variance is a floor set by corpus size, not an estimate of the effect.

Usage:
  python domain_filter.py --emb <real.npz>            # train, apply, compare
  python domain_filter.py --emb <real.npz> --power    # the power curve alone
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import recursive_split as rs

JUDGMENTS = Path(__file__).resolve().parent / "data" / "transformation-judgments.json"


def train_filter(emb: Path, C_grid=(0.01, 0.03, 0.1, 0.3, 1.0), min_auc=0.70):
    z = np.load(emb, allow_pickle=False)
    X = z["vectors"].astype(np.float64)
    pos = {str(v): i for i, v in enumerate(z["ids"])}
    J = json.load(JUDGMENTS.open())["judgments"]
    rows = [(pos[r["id"]], r["is_transformation"]) for r in J
            if r["is_transformation"] in (0.0, 1.0) and r["id"] in pos]
    Xi = X[[i for i, _ in rows]]
    y = np.array([v for _, v in rows])
    best = None
    for C in C_grid:
        p = cross_val_predict(LogisticRegression(C=C, max_iter=4000), Xi, y,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method="predict_proba")[:, 1]
        auc, acc = roc_auc_score(y, p), accuracy_score(y, p > 0.5)
        print(f"  C={C:<5} 5-fold AUC {auc:.3f}  acc {acc:.3f}")
        if best is None or auc > best[1]:
            best = (C, auc, acc)
    print(f"  best C={best[0]}  AUC {best[1]:.3f}  acc {best[2]:.3f}")
    if best[1] < min_auc:
        raise SystemExit(f"AUC {best[1]:.3f} below the {min_auc} bar; no usable filter.")
    model = LogisticRegression(C=best[0], max_iter=4000).fit(Xi, y)
    return dict(zip([str(v) for v in z["ids"]], model.predict_proba(X)[:, 1])), best


def arm(X, lab, idx, n, seeds=10):
    A = {"past": [], "fut": [], "uns": []}
    for s in range(seeds):
        rng = np.random.default_rng(s)
        sub = rng.choice(idx, min(n, len(idx)), replace=False)
        q = rng.permutation(len(sub))
        ntr = int(len(sub) * 0.7)
        sc = rs.score(X[sub[q[:ntr]]], lab[sub[q[:ntr]]], X[sub[q[ntr:]]], lab[sub[q[ntr:]]])
        A["past"].append(sc["past_cellmean_r2"])
        A["fut"].append(sc["future_additive_r2"])
        A["uns"].append(sc["future_unseen_cell_r2"])
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in A.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--out", default="domain_filter_results.json")
    args = ap.parse_args()
    emb = Path(args.emb)
    X, lab, extra = rs.load_run(emb)
    print(f"{len(X):,} consensus clauses, dim={X.shape[1]}")

    if args.power:
        print("\npower curve - EO on RANDOM subsamples (8 seeds):")
        print(f"{'n':>7} {'PAST':>18} {'FUTURE':>18} {'UNSEEN':>18}")
        curve = {}
        for n in (1000, 2000, 2710, 4000, 6000, len(X)):
            m = arm(X, lab, np.arange(len(X)), n, seeds=8)
            curve[n] = m
            print(f"{n:7,} {m['past'][0]:+9.4f}+/-{m['past'][1]:.4f} "
                  f"{m['fut'][0]:+9.4f}+/-{m['fut'][1]:.4f} "
                  f"{m['uns'][0]:+9.4f}+/-{m['uns'][1]:.4f}")
        Path(args.out).write_text(json.dumps(
            {str(k): {kk: list(vv) for kk, vv in v.items()} for k, v in curve.items()}, indent=2))
        return

    print("\ntraining the domain filter on hand judgments:")
    scores, best = train_filter(emb)
    p = np.array([scores[str(i)] for i in extra["ids"]])
    IN, OUT = p > 0.5, p <= 0.5
    n = int(min(IN.sum(), OUT.sum()))
    print(f"\nin-domain {int(IN.sum()):,}   out-of-domain {int(OUT.sum()):,}   equal-n {n:,}")
    print(f"\n{'arm':30} {'PAST':>18} {'FUTURE':>18} {'UNSEEN':>18}")
    print("-" * 88)
    res = {}
    for name, idx in [("RANDOM subsample (control)", np.arange(len(X))),
                      ("in-domain", np.where(IN)[0]),
                      ("out-of-domain", np.where(OUT)[0])]:
        m = arm(X, lab, idx, n)
        res[name] = m
        print(f"{name:30} {m['past'][0]:+9.4f}+/-{m['past'][1]:.4f} "
              f"{m['fut'][0]:+9.4f}+/-{m['fut'][1]:.4f} "
              f"{m['uns'][0]:+9.4f}+/-{m['uns'][1]:.4f}")
    print("\nThe random control at the same n scores ~0 (see --power): this comparison")
    print("is underpowered and cannot decide the domain question either way.")
    Path(args.out).write_text(json.dumps(
        {"auc": best[1], "acc": best[2], "n_per_arm": n,
         "arms": {k: {kk: list(vv) for kk, vv in v.items()} for k, v in res.items()}}, indent=2))


if __name__ == "__main__":
    main()
