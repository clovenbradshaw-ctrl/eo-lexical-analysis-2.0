#!/usr/bin/env python3
"""
coordinate_geometry.py — the canonical coordinate address, tested.

eo-wiki `the-axis-triad-and-its-coordinates` gives every form in the capacity
ground a three-coordinate address (Mode, Domain, Object):

    Mode    {0, 1, 2}        arithmetic      von Neumann ordinals
    Domain  {-1, +1, sqrt2}  geometric       Pythagorean
    Object  {2, sqrt2, 2^sqrt2}  transcendental  Gelfond-Schneider

verified here against the article's own five worked examples. The wiki already
records these predictions as "Not met" (`the-evidence`, SYNC-PUNCHLIST A1); this
re-runs them on the balanced most-unique instrument, which is stronger than what
the original test had, and confirms rather than overturns the failure.

Three tests, coarse to fine:

  MANTEL       the predicted 27x27 coordinate distance matrix against the
               observed one, permutation null over cell assignment.
  STEP RATIOS  per axis, the spacing the coordinates predict against the spacing
               observed.
  COLLINEARITY the precondition. Scalar coordinates place three levels ON A LINE:
               a degenerate triangle, largest interior angle 180 degrees. If the
               three level-centroids form a near-equilateral triangle instead,
               NO scalar assignment of any character can describe them, and the
               step-ratio question is moot before the numbers are chosen.

Also fits three composition laws to the level step, since the three maths name
three different algebras:

    arithmetic     x + t        translation
    geometric      D x          per-coordinate multiplicative (diagonal)
    transcendental s R x        scaling x rotation -- the complex exponential

Usage:
  python coordinate_geometry.py --emb <real.npz>
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import recursive_split as rs
from unique_exemplars import cube, loo_margin

R2 = math.sqrt(2)
T = 2 ** R2
MODE, DOM, OBJ = [0, 1, 2], [-1, 1, R2], [2, R2, T]
AXES = [("Mode   ARITHMETIC", MODE), ("Domain GEOMETRIC", DOM), ("Object TRANSCENDENTAL", OBJ)]
WORKED = {  # the article's own examples, as a check on this reconstruction
    "INS x Figure": ((2, -1, R2), (MODE[2], DOM[0], OBJ[1])),
    "NUL x Ground": ((0, -1, 2), (MODE[0], DOM[0], OBJ[0])),
    "REC x Pattern": ((2, R2, T), (MODE[2], DOM[2], OBJ[2])),
    "SYN x Ground": ((2, 1, 2), (MODE[2], DOM[1], OBJ[0])),
    "EVA x Pattern": ((1, R2, T), (MODE[1], DOM[2], OBJ[2])),
}


def triangle_angles(cs):
    def ang(o, p, q):
        u, v = cs[p] - cs[o], cs[q] - cs[o]
        return math.degrees(math.acos(float(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))
    return [ang(0, 1, 2), ang(1, 0, 2), ang(2, 0, 1)]


def fit_laws(C, k=6):
    """LOO prediction error per composition law, in the top-k PC subspace."""
    flat = C.reshape(27, -1)
    Z = flat - flat.mean(0)
    Vt = np.linalg.svd(Z, full_matrices=False)[2]
    Ck = (Z @ Vt[:k].T).reshape(3, 3, 3, k)
    E = {m: 0.0 for m in ("identity", "arithmetic", "geometric", "transcendental")}
    for a in range(3):
        for p, q in ((0, 1), (1, 2), (0, 2)):
            o = [x for x in range(3) if x != a]
            Xs, Ys = [], []
            for u, v in itertools.product(range(3), repeat=2):
                ip, iq = [0, 0, 0], [0, 0, 0]
                ip[a], iq[a] = p, q
                ip[o[0]] = iq[o[0]] = u
                ip[o[1]] = iq[o[1]] = v
                Xs.append(Ck[tuple(ip)])
                Ys.append(Ck[tuple(iq)])
            Xs, Ys = np.array(Xs), np.array(Ys)
            for h in range(len(Xs)):
                tr = [i for i in range(len(Xs)) if i != h]
                A, B, x, y = Xs[tr], Ys[tr], Xs[h], Ys[h]
                E["identity"] += ((y - x) ** 2).sum()
                E["arithmetic"] += ((y - (x + (B - A).mean(0))) ** 2).sum()
                d = (A * B).sum(0) / np.maximum((A * A).sum(0), 1e-12)
                E["geometric"] += ((y - d * x) ** 2).sum()
                U, Sv, Vt2 = np.linalg.svd(B.T @ A)
                R = U @ Vt2
                s = Sv.sum() / max((A * A).sum(), 1e-12)
                E["transcendental"] += ((y - s * (R @ x)) ** 2).sum()
    return {m: 1 - v / E["identity"] for m, v in E.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--out", default="coordinate_geometry_results.json")
    args = ap.parse_args()

    print("verifying the address model against the article's worked examples:")
    ok = True
    for name, (want, got) in WORKED.items():
        good = np.allclose(want, got)
        ok &= good
        print(f"   {name:14} article {tuple(round(x,3) for x in want)}  "
              f"model {tuple(round(x,3) for x in got)}  {'OK' if good else 'MISMATCH'}")
    if not ok:
        raise SystemExit("address reconstruction disagrees with the article")

    X, lab, _ = rs.load_run(Path(args.emb))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    flat = lab[:, 0] * 9 + lab[:, 1] * 3 + lab[:, 2]
    C = cube(X, flat, loo_margin(X, flat), args.k)
    F = C.reshape(27, -1)

    addr = np.array([[MODE[c // 9], DOM[(c // 3) % 3], OBJ[c % 3]] for c in range(27)])
    iu = np.triu_indices(27, 1)
    Dp = np.linalg.norm(addr[:, None] - addr[None, :], axis=2)[iu]
    Do = np.linalg.norm(F[:, None] - F[None, :], axis=2)[iu]
    r = spearmanr(Dp, Do)[0]
    rng = np.random.default_rng(0)
    nul = np.array([spearmanr(np.linalg.norm(addr[p][:, None] - addr[p][None, :], axis=2)[iu], Do)[0]
                    for p in (rng.permutation(27) for _ in range(args.perms))])
    print(f"\nMANTEL  predicted vs observed 27x27 distances")
    print(f"   Spearman r = {r:+.4f}   null {nul.mean():+.4f}+/-{nul.std():.4f}   "
          f"z = {(r-nul.mean())/nul.std():+.2f}   beaten by {(np.abs(nul)>=abs(r)).sum()}/{args.perms}")

    print(f"\nSTEP RATIOS and COLLINEARITY (k={args.k})")
    axes = {}
    for a, (nm, co) in enumerate(AXES):
        cs = [C.take(l, axis=a).reshape(-1, C.shape[-1]).mean(0) for l in range(3)]
        d = [np.linalg.norm(cs[0] - cs[1]), np.linalg.norm(cs[1] - cs[2]), np.linalg.norm(cs[0] - cs[2])]
        pr = [abs(co[0] - co[1]), abs(co[1] - co[2]), abs(co[0] - co[2])]
        mx = max(triangle_angles(cs))
        axes[nm] = {"predicted": [1, pr[1]/pr[0], pr[2]/pr[0]],
                    "observed": [1, d[1]/d[0], d[2]/d[0]], "max_angle_deg": mx}
        print(f"   {nm:22} predicted 1 : {pr[1]/pr[0]:.3f} : {pr[2]/pr[0]:.3f}"
              f"    observed 1 : {d[1]/d[0]:.3f} : {d[2]/d[0]:.3f}    max angle {mx:5.1f} deg")
    print("   collinear coordinates require ~180 deg; near-60 is an equilateral simplex,")
    print("   which no scalar assignment of any character can describe.")

    laws = fit_laws(C)
    print(f"\nCOMPOSITION LAW (LOO, top-6 PC subspace, fraction of step variance explained)")
    for m in ("arithmetic", "geometric", "transcendental"):
        print(f"   {m:16} {laws[m]:+.4f}")

    Path(args.out).write_text(json.dumps(
        {"mantel_r": float(r), "mantel_z": float((r - nul.mean()) / nul.std()),
         "mantel_null_mean": float(nul.mean()), "mantel_null_sd": float(nul.std()),
         "axes": axes, "composition_laws": laws}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
