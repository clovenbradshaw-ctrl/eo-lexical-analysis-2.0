#!/usr/bin/env python3
"""
dimensionality.py — "why three?", answered with a number.

Every artifact in this repo assumes 3x3x3 and fits it. Nothing compares that
choice against 2 axes, or 2 levels, or against where the embedding space's own
optimum sits. This script does both:

  ABLATION   drop an axis, or merge two levels within an axis, and see whether
             held-out prediction degrades. If two axes predict as well as
             three, the third is redundant. If merging levels loses nothing,
             three levels is finer than the material supports.

  BLIND SWEEP  for a range of (n_axes, n_levels), tertile/quantile the top
             principal directions and score the same way. This locates the
             dimensionality the SPACE prefers, independent of EO.

Scoring is held-out throughout, so the differing parameter counts across
configurations are handled by construction rather than by a penalty term:
an additive model with A axes and L levels has 1 + A(L-1) free parameters,
and a configuration that buys its fit with parameters loses it on held-out data.

  FUTURE   predict a held-out clause's position from its labels alone
  UNSEEN   leave one whole cell out of the fit, predict it from the rest

Usage:
  python dimensionality.py --emb <real embeddings.npz>
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

Q1 = ["DIFFERENTIATING", "RELATING", "GENERATING"]
Q2 = ["EXISTENCE", "STRUCTURE", "SIGNIFICANCE"]
Q3 = ["CONDITION", "ENTITY", "PATTERN"]
AXIS_NAME = {0: "Q1 mode", 1: "Q2 domain", 2: "Q3 object"}
LEVELS = {0: Q1, 1: Q2, 2: Q3}


def load_run(p: Path, consensus_only=True):
    if p.stat().st_size < 4096:
        raise RuntimeError(f"{p} is a Drive pointer, not embeddings:\n  {p.read_text().strip()}")
    z = np.load(p, allow_pickle=False)
    lab = np.stack([
        np.array([Q1.index(v) if v in Q1 else -1 for v in z["q1"]]),
        np.array([Q2.index(v) if v in Q2 else -1 for v in z["q2"]]),
        np.array([Q3.index(v) if v in Q3 else -1 for v in z["q3"]]),
    ], axis=1)
    keep = (lab >= 0).all(1)
    if consensus_only and "consensus" in z.files:
        keep &= z["consensus"].astype(bool)
    extra = {k: z[k][keep] for k in ("language", "source", "ids") if k in z.files}
    return z["vectors"][keep].astype(np.float64), lab[keep], extra


def design(lab, n_lev):
    """One-hot over each axis. lab[:, a] in [0, n_lev[a])."""
    off, cols = [], 0
    for L in n_lev:
        off.append(cols)
        cols += L
    X = np.zeros((len(lab), cols))
    r = np.arange(len(lab))
    for a, o in enumerate(off):
        X[r, o + lab[:, a]] = 1
    return X


def score(Xtr, ltr, Xte, lte, n_lev):
    """FUTURE = held-out clauses via the additive model.
       UNSEEN = leave one whole cell out of the centroid fit."""
    mu = Xtr.mean(0)
    cells = [c for c in itertools.product(*[range(L) for L in n_lev])]
    cent, keys = [], []
    for c in cells:
        m = (ltr == np.array(c)).all(1)
        if m.sum() >= 1:
            cent.append(Xtr[m].mean(0))
            keys.append(c)
    Y = np.array(cent)
    K = np.array(keys)
    B = np.linalg.pinv(design(K, n_lev)) @ Y
    pred = design(lte, n_lev) @ B
    future = 1.0 - ((Xte - pred) ** 2).sum() / ((Xte - mu) ** 2).sum()

    D = design(K, n_lev)
    sse = sst = 0.0
    for h in range(len(keys)):
        tr = [r for r in range(len(keys)) if r != h]
        if any(len(np.unique(K[tr][:, a])) < n_lev[a] for a in range(len(n_lev))):
            continue
        Bh = np.linalg.pinv(D[tr]) @ Y[tr]
        p = D[h] @ Bh
        m = Y[tr].mean(0)
        sse += ((Y[h] - p) ** 2).sum()
        sst += ((Y[h] - m) ** 2).sum()
    unseen = 1.0 - sse / sst if sst > 0 else float("nan")
    n_par = 1 + sum(L - 1 for L in n_lev)
    return {"cells": len(keys), "params": n_par,
            "future": float(future), "unseen": float(unseen)}


def merged(lab, axis, pair):
    """Collapse two levels of one axis into one. Returns (labels, n_levels)."""
    out = lab.copy()
    lo, hi = sorted(pair)
    col = out[:, axis]
    col = np.where(col == hi, lo, col)
    col = np.where(col > hi, col - 1, col)
    out[:, axis] = col
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--out", default="dimensionality_results.json")
    args = ap.parse_args()

    X, lab, extra = load_run(Path(args.emb))
    print(f"{len(X):,} consensus clauses, dim={X.shape[1]}")
    if "language" in extra:
        print(f"languages: {len(np.unique(extra['language']))}")

    def run(fn):
        """Average a scoring closure over seeds."""
        acc = []
        for s in range(args.seeds):
            rng = np.random.default_rng(s)
            p = rng.permutation(len(X))
            n = int(len(X) * (1 - args.test_frac))
            acc.append(fn(p[:n], p[n:], s))
        return {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}

    results = {}

    # ── EO, full ────────────────────────────────────────────────────────────
    full = run(lambda tr, te, s: score(X[tr], lab[tr], X[te], lab[te], (3, 3, 3)))
    results["eo_3x3x3"] = full
    print(f"\n{'configuration':34} {'cells':>5} {'par':>4} {'FUTURE':>8} {'UNSEEN':>8}")
    print("-" * 64)
    print(f"{'EO  3x3x3 (full)':34} {full['cells']:5.0f} {full['params']:4.0f} "
          f"{full['future']:+8.4f} {full['unseen']:+8.4f}")

    # ── drop an axis ────────────────────────────────────────────────────────
    print(f"\n-- drop one axis (does the third carry independent weight?) --")
    for drop in range(3):
        keep = [a for a in range(3) if a != drop]
        r = run(lambda tr, te, s, k=keep: score(X[tr], lab[tr][:, k], X[te], lab[te][:, k], (3, 3)))
        results[f"eo_drop_{AXIS_NAME[drop].split()[0]}"] = r
        d = r["future"] - full["future"]
        print(f"{'  without ' + AXIS_NAME[drop]:34} {r['cells']:5.0f} {r['params']:4.0f} "
              f"{r['future']:+8.4f} {r['unseen']:+8.4f}   dFUTURE {d:+.4f}")

    # ── one axis alone ──────────────────────────────────────────────────────
    print(f"\n-- one axis alone --")
    for a in range(3):
        r = run(lambda tr, te, s, a=a: score(X[tr], lab[tr][:, [a]], X[te], lab[te][:, [a]], (3,)))
        results[f"eo_only_{AXIS_NAME[a].split()[0]}"] = r
        print(f"{'  ' + AXIS_NAME[a] + ' only':34} {r['cells']:5.0f} {r['params']:4.0f} "
              f"{r['future']:+8.4f} {r['unseen']:+8.4f}")

    # ── merge two levels within an axis ─────────────────────────────────────
    print(f"\n-- merge two levels of one axis (are three levels earned?) --")
    for a in range(3):
        for pair in itertools.combinations(range(3), 2):
            lv = [3, 3, 3]
            lv[a] = 2
            lm = merged(lab, a, pair)
            r = run(lambda tr, te, s, lm=lm, lv=tuple(lv):
                    score(X[tr], lm[tr], X[te], lm[te], lv))
            nm = f"  {AXIS_NAME[a].split()[0]}: {LEVELS[a][pair[0]][:5]}+{LEVELS[a][pair[1]][:5]}"
            results[f"eo_merge_{a}_{pair[0]}{pair[1]}"] = r
            d = r["future"] - full["future"]
            print(f"{nm:34} {r['cells']:5.0f} {r['params']:4.0f} "
                  f"{r['future']:+8.4f} {r['unseen']:+8.4f}   dFUTURE {d:+.4f}")

    # ── blind sweep: where does the SPACE want to sit? ──────────────────────
    print(f"\n-- blind PCA-quantile sweep (the space's own preferred shape) --")
    print(f"{'axes x levels':34} {'cells':>5} {'par':>4} {'FUTURE':>8} {'UNSEEN':>8}")
    print("-" * 64)
    blind = {}
    for A in (2, 3, 4, 5):
        for L in (2, 3, 4):
            if A * L > 20 or L ** A > 400:
                continue
            def f(tr, te, s, A=A, L=L):
                p = PCA(A, random_state=s).fit(X[tr])
                str_, ste = p.transform(X[tr]), p.transform(X[te])
                qs = [np.quantile(str_[:, k], np.arange(1, L) / L) for k in range(A)]
                ltr = np.stack([np.searchsorted(qs[k], str_[:, k]) for k in range(A)], 1)
                lte = np.stack([np.searchsorted(qs[k], ste[:, k]) for k in range(A)], 1)
                return score(X[tr], ltr, X[te], lte, tuple([L] * A))
            r = run(f)
            blind[f"{A}x{L}"] = r
            star = "  <-- EO's shape" if (A, L) == (3, 3) else ""
            print(f"{'  PCA ' + str(A) + ' axes x ' + str(L) + ' levels':34} {r['cells']:5.0f} "
                  f"{r['params']:4.0f} {r['future']:+8.4f} {r['unseen']:+8.4f}{star}")
    results["blind_sweep"] = blind

    best_f = max(blind, key=lambda k: blind[k]["future"])
    best_u = max(blind, key=lambda k: blind[k]["unseen"])
    print(f"\nblind optimum: FUTURE at {best_f} ({blind[best_f]['future']:+.4f}), "
          f"UNSEEN at {best_u} ({blind[best_u]['unseen']:+.4f})")

    Path(args.out).write_text(json.dumps(
        {"emb": Path(args.emb).name, "n": len(X), "dim": X.shape[1],
         "seeds": args.seeds, "results": results}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
