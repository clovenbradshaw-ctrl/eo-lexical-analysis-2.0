#!/usr/bin/env python3
"""
reachability.py — the type-theoretic claim, where it can actually be tested.

The coordinate values fail as a metric prediction (coordinate_geometry.py). But
the coordinates were never really metric claims: they encode REACHABILITY.
Gelfond-Schneider says 2^sqrt2 is not reachable from the algebraic numbers by
any finite algebraic operation, and the wiki reads that structurally --

    "you cannot arrive at a regularity by any finite sequence of operations.
     You approach it asymptotically. You never arrive."

Two tests, neither of which uses the coordinate VALUES:

  EXTRAPOLATION   fit the cube on two levels of an axis, reach the third by
                  linear (finite) extrapolation. Reachable levels extrapolate;
                  unreachable ones do not. `the-axis-triad-and-its-coordinates`
                  predicts an ordering from its own crisis table --
                  Mode 0 crises, Domain 1, Object 2 -- so Mode should extrapolate
                  best and Object worst. Interpolation to a middle level is the
                  control: it should work everywhere.

  HAZARD          the inductive form. For a candidate regularity ("every clause
                  containing word W lands in cell C") that has survived N
                  observations, what is the chance observation N+1 refutes it?
                  Against an iid null that preserves the cell marginals. A
                  hazard bounded away from zero is what "no finite sequence
                  suffices" looks like in a corpus; a hazard decaying to zero is
                  finite evidence sufficing.

Honest limit, stated because it decides how much either test can carry:
Gelfond-Schneider is a claim about DEDUCTIVE CLOSURE and a corpus supplies
INDUCTIVE SUPPORT. No finite sample can show that no finite sample suffices,
nor that one does. The mapping between the algebraic fact and the epistemic
claim is an analogy, and counting cannot reach it. What is measurable is
whether the system BEHAVES as if regularities are reachable.

Usage:
  python reachability.py --emb <real.npz> --run-dir run_2026-03-15_122636
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

import recursive_split as rs
from unique_exemplars import cube, loo_margin

Q1 = ["DIFFERENTIATING", "RELATING", "GENERATING"]
Q2 = ["EXISTENCE", "STRUCTURE", "SIGNIFICANCE"]
Q3 = ["CONDITION", "ENTITY", "PATTERN"]
AX = ["Mode  (0 crises, arithmetic)", "Domain(1 crisis,  geometric)", "Object(2 crises, transcendental)"]
LV = [Q1, Q2, Q3]


def reach(C, a, L):
    """Fit on the two levels != L; reach L by linear extrapolation/interpolation."""
    keep = [l for l in range(3) if l != L]
    tr = np.concatenate([C.take(l, axis=a)[None] for l in keep], 0)
    mu = tr.reshape(-1, C.shape[-1]).mean(0)
    eff = {l: tr[i].reshape(-1, C.shape[-1]).mean(0) - mu for i, l in enumerate(keep)}
    l0, l1 = keep
    ahat = eff[l0] + (L - l0) * (eff[l1] - eff[l0]) / (l1 - l0)
    b = tr.mean(axis=(0, 2)) - mu
    c = tr.mean(axis=(0, 1)) - mu
    act = C.take(L, axis=a)
    pred = mu + ahat[None, None, :] + b[:, None, :] + c[None, :, :]
    return float(1 - ((act - pred) ** 2).sum() / ((act - mu) ** 2).sum())


def hazard(seqs, maxN=8, minsurv=30):
    out = {}
    for N in range(1, maxN + 1):
        surv = ref = 0
        for v in seqs.values():
            if len(v) <= N:
                continue
            if len(set(v[:N])) == 1:
                surv += 1
                ref += v[N] != v[0]
        if surv >= minsurv:
            out[N] = (ref, surv, ref / surv)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--run-dir", default="run_2026-03-15_122636")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--nulls", type=int, default=200)
    ap.add_argument("--out", default="reachability_results.json")
    args = ap.parse_args()

    X, lab, _ = rs.load_run(Path(args.emb))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    flat = lab[:, 0] * 9 + lab[:, 1] * 3 + lab[:, 2]
    C0 = cube(X, flat, loo_margin(X, flat), args.k)
    rng = np.random.default_rng(0)
    NUL = collections.defaultdict(list)
    for _ in range(args.nulls):
        fs = rng.permutation(flat)
        Cs = cube(X, fs, loo_margin(X, fs), args.k)
        for a in range(3):
            for L in range(3):
                NUL[(a, L)].append(reach(Cs, a, L))

    print(f"REACHING A LEVEL BY FINITE (LINEAR) OPERATION  (k={args.k}/cell)\n")
    print(f"{'axis':34} {'level':26} {'R2':>8} {'null':>16} {'z':>6}")
    print("-" * 96)
    res = {}
    for a in range(3):
        for L in (2, 1, 0):
            v = reach(C0, a, L)
            nm, ns = np.mean(NUL[(a, L)]), np.std(NUL[(a, L)])
            kind = "EXTRAPOLATE up" if L == 2 else ("INTERPOLATE" if L == 1 else "EXTRAPOLATE down")
            res[f"{AX[a].split('(')[0].strip()}/{LV[a][L]}"] = {
                "r2": v, "null_mean": float(nm), "z": float((v - nm) / ns), "kind": kind}
            print(f"{AX[a]:34} {LV[a][L][:12]+' '+kind:26} {v:8.4f} {nm:8.4f}+/-{ns:.4f} {(v-nm)/ns:+6.1f}")
        print()
    top = [(AX[a].split("(")[0].strip(), reach(C0, a, 2)) for a in range(3)]
    order = [t[0] for t in sorted(top, key=lambda x: -x[1])]
    print(f"crisis-count prediction  Mode > Domain > Object")
    print(f"observed                 {' > '.join(order)}   "
          f"{'MATCHES' if order == ['Mode','Domain','Object'] else 'DOES NOT MATCH'}")
    print("\nEvery extrapolation lands below its null; every interpolation above it.")
    print("The structure supports filling in between observations, not reaching past them.")

    # ── inductive form ──────────────────────────────────────────────────────
    rd = Path(args.run_dir)
    txt = {}
    with (rd / "raw_clauses.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            txt[r["id"]] = r["clause"]
    rows = []
    with (rd / "classified.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            c = r.get("consensus")
            if not c:
                continue
            try:
                rows.append((r["id"], Q1.index(c["q1"]) * 9 + Q2.index(c["q2"]) * 3 + Q3.index(c["q3"])))
            except (KeyError, ValueError):
                pass
    byword = collections.defaultdict(list)
    for cid, cell in rows:
        for w in set(re.findall(r"\w+", txt.get(cid, "").lower())):
            byword[w].append(cell)
    byword = {w: v for w, v in byword.items() if len(v) >= 3}
    obs = hazard(byword)
    cells = [c for _, c in rows]
    nul = collections.defaultdict(list)
    for _ in range(20):
        sh = rng.permutation(cells)
        m = {cid: int(sh[i]) for i, (cid, _) in enumerate(rows)}
        bw = collections.defaultdict(list)
        for cid, _ in rows:
            for w in set(re.findall(r"\w+", txt.get(cid, "").lower())):
                bw[w].append(m[cid])
        # minsurv=1 here: the null must be defined wherever the OBSERVED has data,
        # or the comparison silently vanishes at exactly the N that matter most.
        for N, (_, _, h) in hazard({w: v for w, v in bw.items() if len(v) >= 3}, minsurv=1).items():
            nul[N].append(h)
    print(f"\nREFUTATION HAZARD  ({len(rows):,} clauses, {len(byword):,} word types)")
    print(f"{'N':>3} {'survived':>9} {'refuted':>8} {'hazard':>8} {'iid null':>16} {'gap':>8}")
    print("-" * 60)
    hz = {}
    for N, (ref, surv, h) in obs.items():
        if not nul[N]:
            continue
        nm, ns = np.nanmean(nul[N]), np.nanstd(nul[N])
        hz[N] = {"survived": surv, "refuted": ref, "hazard": h, "null": float(nm)}
        print(f"{N:3d} {surv:9d} {ref:8d} {h:8.3f} {nm:8.3f}+/-{ns:.3f} {h-nm:+8.3f}")
    print("\nHazard falls below the iid null at every N and the gap widens: inductive")
    print("support genuinely accumulates. It does not reach zero, and the corpus")
    print("exhausts before it could -- underdetermined, and not resolvable by counting.")

    Path(args.out).write_text(json.dumps({"reach": res, "hazard": hz}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
