#!/usr/bin/env python3
"""
factorization_test.py — "Why these three?"

The EO report's headline results (monotonicity, per-axis z-scores) cannot
answer that question, because they are reproduced by ANY three directions:
quantile-binning any direction in any embedding space makes distance grow
with the number of differing bins. See --arbitrary below, which produces a
monotone spread on data with no planted structure whatsoever.

This script tests the one property arbitrary directions do NOT confer:
whether the 27 cells are an ADDITIVE PRODUCT of three axes.

If Q1/Q2/Q3 are three independent dimensions, a cell's position in embedding
space should decompose as

    c(i,j,k) ~ mu + a_i + b_j + c_k

with small interaction residual. This is a strong constraint: it says the
"cost" of being GENERATING rather than DIFFERENTIATING is the same shift no
matter which domain and object you pair it with.

It then asks the predictive question: hold one whole cell out, fit on the other
26, and predict the cell you have never seen. A product can do this from axis
marginals alone. A tree or an arbitrary arrangement cannot.

The null is the sharp one. Hold the 27 centroids fixed and re-assign them to
grid positions: every alternative assignment fits the same model with the same
parameter count, so any advantage EO shows is about the ARRANGEMENT alone.

Usage:
  python factorization_test.py                    # the product test + null
  python factorization_test.py --arbitrary        # the monotonicity artifact
  python factorization_test.py --nulls 50000 --restarts 200
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np

Q1 = ["DIFFERENTIATING", "RELATING", "GENERATING"]
Q2 = ["EXISTENCE", "STRUCTURE", "SIGNIFICANCE"]
Q3 = ["CONDITION", "ENTITY", "PATTERN"]


def axis_maps(run_dirs):
    """Derive (q1,q2,q3) -> (operator, resolution, site) from classified.jsonl
    rather than hand-typing it. Returns the three pair-maps."""
    op, res, site = {}, {}, {}
    for rd in run_dirs:
        path = Path(rd) / "classified.jsonl"
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                for c in (rec.get("classifications") or {}).values():
                    if not c:
                        continue
                    a, b, d = c.get("q1"), c.get("q2"), c.get("q3")
                    if a not in Q1 or b not in Q2 or d not in Q3:
                        continue
                    op.setdefault((a, b), c.get("operator"))
                    res.setdefault((a, d), c.get("resolution"))
                    site.setdefault((b, d), c.get("site"))
    if len(op) != 9 or len(res) != 9 or len(site) != 9:
        raise RuntimeError(
            f"incomplete axis mapping from classified.jsonl "
            f"(operator={len(op)}, resolution={len(res)}, site={len(site)}; need 9 each)"
        )
    return op, res, site


def load_cube(archetypes: Path, run_dirs):
    """Returns C[3,3,3,D] of archetype centroids indexed by (q1,q2,q3)."""
    d = json.load(archetypes.open())
    if d.get("face") != "27cell":
        raise RuntimeError(f"{archetypes} is not a 27cell archetype file")
    op, res, site = axis_maps(run_dirs)
    cent = d["centroids"]
    C = np.zeros((3, 3, 3, d["dim"]), dtype=np.float64)
    for i, a in enumerate(Q1):
        for j, b in enumerate(Q2):
            for k, c in enumerate(Q3):
                key = f"{op[(a, b)]}({res[(a, c)]}, {site[(b, c)]})"
                if key not in cent:
                    raise KeyError(f"centroid {key!r} missing from {archetypes.name}")
                C[i, j, k] = np.asarray(cent[key], dtype=np.float64)
    return C, d


def additive_r2(X):
    """X[3,3,3,D] -> (R^2, [share_q1, share_q2, share_q3]).

    On a balanced full factorial the marginal means ARE the OLS fit, and the
    main effects are mutually orthogonal, so explained variance adds.
    """
    mu = X.mean(axis=(0, 1, 2))
    a = X.mean(axis=(1, 2)) - mu
    b = X.mean(axis=(0, 2)) - mu
    c = X.mean(axis=(0, 1)) - mu
    fit = mu + a[:, None, None, :] + b[None, :, None, :] + c[None, None, :, :]
    tot = ((X - mu) ** 2).sum()
    res = ((X - fit) ** 2).sum()
    ss = [9 * (a ** 2).sum(), 9 * (b ** 2).sum(), 9 * (c ** 2).sum()]
    return 1.0 - res / tot, [s / tot for s in ss]


def design_matrix(cells):
    """One-hot over the three factors: 9 columns, one per axis level."""
    X = np.zeros((len(cells), 9))
    for r, (i, j, k) in enumerate(cells):
        X[r, i] = 1
        X[r, 3 + j] = 1
        X[r, 6 + k] = 1
    return X


def loco_r2(flat):
    """Leave-one-CELL-out. Fit the additive model on 26 cells, predict the 27th.

    This is the predictive criterion: a genuine product structure can place a
    combination it has never observed, because the axis marginals carry it. A
    tree or an arbitrary arrangement has nothing to extrapolate from, and lands
    below R^2 = 0 -- worse than predicting the training mean.
    """
    cells = [(i, j, k) for i in range(3) for j in range(3) for k in range(3)]
    X = design_matrix(cells)
    sse = sst = 0.0
    per = []
    for h in range(27):
        tr = [r for r in range(27) if r != h]
        B = np.linalg.pinv(X[tr]) @ flat[tr]
        pred = X[h] @ B
        mu = flat[tr].mean(0)
        e = ((flat[h] - pred) ** 2).sum()
        t = ((flat[h] - mu) ** 2).sum()
        sse += e
        sst += t
        per.append((h, 1.0 - e / t))
    return 1.0 - sse / sst, per


def climb(flat, perm):
    """Greedy pairwise-swap ascent on the cell->grid assignment."""
    p = perm.copy()
    cur = additive_r2(flat[p].reshape(3, 3, 3, -1))[0]
    improved = True
    while improved:
        improved = False
        for a, b in itertools.combinations(range(27), 2):
            p[a], p[b] = p[b], p[a]
            v = additive_r2(flat[p].reshape(3, 3, 3, -1))[0]
            if v > cur + 1e-12:
                cur, improved = v, True
            else:
                p[a], p[b] = p[b], p[a]
    return cur, p


def principal_angles(C):
    mu = C.mean(axis=(0, 1, 2))
    eff = [C.mean(axis=(1, 2)) - mu, C.mean(axis=(0, 2)) - mu, C.mean(axis=(0, 1)) - mu]
    span = lambda M: np.linalg.qr(M.T)[0][:, :2]
    out = {}
    names = ["q1", "q2", "q3"]
    for a, b in itertools.combinations(range(3), 2):
        sv = np.linalg.svd(span(eff[a]).T @ span(eff[b]), compute_uv=False)
        out[f"{names[a]}_vs_{names[b]}"] = [
            round(float(x), 2) for x in np.degrees(np.arccos(np.clip(sv, -1, 1)))
        ]
    return out


def arbitrary_demo(seed=0, n=3000, dim=384, pairs=200_000):
    """Three ARBITRARY orthogonal directions, quantile-binned, over data with
    no planted 3x3x3 structure. Monotonicity appears anyway."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)) * (np.arange(1, dim + 1) ** -0.8)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q = np.linalg.qr(rng.standard_normal((dim, 3)))[0]
    proj = X @ Q
    lab = np.stack(
        [np.searchsorted(np.quantile(proj[:, k], [1 / 3, 2 / 3]), proj[:, k]) for k in range(3)],
        axis=1,
    )
    ii, jj = np.triu_indices(n, 1)
    sel = rng.choice(len(ii), min(pairs, len(ii)), replace=False)
    ii, jj = ii[sel], jj[sel]
    diff = (lab[ii] != lab[jj]).sum(1)
    dist = 1.0 - (X[ii] * X[jj]).sum(1)
    return {str(k): float(dist[diff == k].mean()) for k in range(4) if (diff == k).any()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archetypes", default=None, help="27cell archetype json")
    ap.add_argument("--nulls", type=int, default=20000)
    ap.add_argument("--restarts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arbitrary", action="store_true", help="only the monotonicity artifact")
    ap.add_argument("--out", default="factorization_results.json")
    args = ap.parse_args()

    if args.arbitrary:
        d = arbitrary_demo(args.seed)
        print("Three ARBITRARY orthogonal directions, structureless data:")
        for k, v in d.items():
            print(f"  {k} axes differ   mean cosine distance {v:.4f}")
        print(f"  monotone spread = {d['3'] - d['0']:+.4f}")
        print("\nMonotonicity is a property of quantile-binning, not of EO.")
        return

    root = Path(__file__).resolve().parent
    arch = Path(args.archetypes) if args.archetypes else None
    if arch is None:
        found = sorted(glob.glob(str(root / "archetypes-27-*.json")))
        if not found:
            sys.exit("no archetypes-27-*.json found; pass --archetypes")
        arch = Path(found[-1])
    runs = sorted(glob.glob(str(root / "run_*")))

    C, meta = load_cube(arch, runs)
    print(f"archetypes : {arch.name}")
    print(f"model      : {meta['model']}  dim={meta['dim']}  cells={meta['cell_count']}")

    r2_eo, shares = additive_r2(C)
    print(f"\nEO assignment: additive R^2 = {r2_eo:.4f}")
    for nm, s in zip(("Q1 mode  ", "Q2 domain", "Q3 object"), shares):
        print(f"  {nm} {s * 100:5.2f}% of between-cell variance")
    print(f"  interaction/residual  {100 * (1 - r2_eo):5.2f}%")

    flat = C.reshape(27, -1)
    rng = np.random.default_rng(args.seed)
    nulls = np.empty(args.nulls)
    idx = np.arange(27)
    for t in range(args.nulls):
        rng.shuffle(idx)
        nulls[t] = additive_r2(flat[idx].reshape(3, 3, 3, -1))[0]
    beat = int((nulls >= r2_eo).sum())
    z = float((r2_eo - nulls.mean()) / nulls.std())
    print(f"\nnull — same 27 centroids re-assigned to the grid (n={args.nulls}):")
    print(f"  mean {nulls.mean():.4f}  sd {nulls.std():.4f}  max {nulls.max():.4f}")
    print(f"  EO z = {z:+.2f}   assignments beating EO: {beat}/{args.nulls}")

    # predictive arm: leave one whole cell out
    r2_loco, per_cell = loco_r2(flat)
    lnull = np.empty(min(args.nulls, 2000))
    idx = np.arange(27)
    for t in range(len(lnull)):
        rng.shuffle(idx)
        lnull[t] = loco_r2(flat[idx])[0]
    lz = float((r2_loco - lnull.mean()) / lnull.std())
    print(f"\nPAST   in-sample additive R^2        = {r2_eo:.4f}")
    print(f"FUTURE leave-one-CELL-out  R^2       = {r2_loco:.4f}"
          f"   <- predicting a cell never observed")
    print(f"  null (n={len(lnull)}): mean {lnull.mean():+.4f}  max {lnull.max():+.4f}  "
          f"EO z = {lz:+.2f}  beating EO: {int((lnull >= r2_loco).sum())}")
    print(f"  R^2 <= 0 means the model predicts an unseen cell worse than the training mean")

    at_eo = climb(flat, np.arange(27))[0]
    best = max(climb(flat, rng.permutation(27))[0] for _ in range(args.restarts))
    print(f"\nsearch over assignments:")
    print(f"  ascent started at EO      = {at_eo:.4f}  ({'EO is a local max' if abs(at_eo - r2_eo) < 1e-9 else 'EO is NOT a local max'})")
    print(f"  best of {args.restarts} random restarts = {best:.4f}")
    if best <= r2_eo + 1e-9:
        print("  search cannot beat EO's own assignment")

    angles = principal_angles(C)
    print("\nprincipal angles between main-effect subspaces (90 = orthogonal):")
    for k, v in angles.items():
        print(f"  {k:12} {v}")

    out = {
        "archetypes": arch.name,
        "model": meta["model"],
        "eo_additive_r2": round(float(r2_eo), 6),
        "axis_variance_share": {k: round(float(s), 6) for k, s in zip(("q1", "q2", "q3"), shares)},
        "interaction_residual": round(float(1 - r2_eo), 6),
        "null": {
            "n": args.nulls,
            "mean": round(float(nulls.mean()), 6),
            "sd": round(float(nulls.std()), 6),
            "max": round(float(nulls.max()), 6),
            "z": round(z, 4),
            "n_beating_eo": beat,
        },
        "search": {"restarts": args.restarts, "best": round(float(best), 6),
                   "ascent_from_eo": round(float(at_eo), 6)},
        "loco": {
            "r2": round(float(r2_loco), 6),
            "null_mean": round(float(lnull.mean()), 6),
            "null_max": round(float(lnull.max()), 6),
            "z": round(lz, 4),
            "n_beating_eo": int((lnull >= r2_loco).sum()),
            "per_cell_r2": [round(float(v), 4) for _, v in per_cell],
        },
        "principal_angles_deg": angles,
        "arbitrary_directions_monotonicity": arbitrary_demo(args.seed),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
