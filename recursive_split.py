#!/usr/bin/env python3
"""
recursive_split.py — blind recursive 3-way splits vs the EO axes.

The question: if you split the embedded clauses into the three most
meaningful distinctions, then split each of those three ways again, and again,
you get 27 leaves without ever consulting EO. Are those leaves as useful as
EO's 27 cells?

"Useful" is scored on two criteria, in the order that matters:

  PAST     — does the scheme make the observed corpus understandable?
             (how much held-out clause variance its 27 groups explain)
  FUTURE   — does it make new observations predictable?
             (a) held-out clauses: predict a clause's position from its three
                 labels alone, via the additive model mu + a_i + b_j + c_k
             (b) UNSEEN CELLS: leave one whole cell out, predict it from the
                 other 26. Only a genuine product structure can do this — a
                 tree or an arbitrary partition has nothing to extrapolate from.

Criterion (b) is the discriminating one. Compression is easy: KMeans optimises
it directly and will beat EO at it. Predicting a combination you have never
observed is what a coordinate system buys you and a hierarchy does not.

Also reports the PRODUCT-vs-TREE diagnostic: at depth 2, is the split found
inside branch 1 the same split found inside branches 2 and 3? Same split
everywhere = a product (three questions that apply everywhere, EO's shape).
Different per branch = a tree, where the second question depends on the answer
to the first — which is not three dimensions, however good the leaves look.

Usage:
  python recursive_split.py --emb run_2026-03-15_122636/embeddings.npz
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

Q1 = ["DIFFERENTIATING", "RELATING", "GENERATING"]
Q2 = ["EXISTENCE", "STRUCTURE", "SIGNIFICANCE"]
Q3 = ["CONDITION", "ENTITY", "PATTERN"]


# ── data ───────────────────────────────────────────────────────────────────

def load_run(emb_path: Path, consensus_only=True):
    """The run's own embeddings.npz, which carries vectors AND labels.

    The copies committed to this repo are Drive pointers; fetch the real file
    with the URL inside them, e.g.
      curl -sSL -o embeddings.npz \
        "https://drive.usercontent.google.com/download?id=<ID>&export=download&confirm=t"
    """
    if emb_path.stat().st_size < 4096:
        raise RuntimeError(
            f"{emb_path} is a {emb_path.stat().st_size}-byte Drive pointer, not the "
            f"embeddings:\n  {emb_path.read_text().strip()}\nFetch it first (see docstring)."
        )
    z = np.load(emb_path, allow_pickle=False)
    X = z["vectors"].astype(np.float64)
    lab = np.stack([
        np.array([Q1.index(v) if v in Q1 else -1 for v in z["q1"]]),
        np.array([Q2.index(v) if v in Q2 else -1 for v in z["q2"]]),
        np.array([Q3.index(v) if v in Q3 else -1 for v in z["q3"]]),
    ], axis=1)
    keep = (lab >= 0).all(1)
    if consensus_only and "consensus" in z.files:
        keep &= z["consensus"].astype(bool)
    extra = {k: z[k][keep] for k in ("language", "source", "ids") if k in z.files}
    return X[keep], lab[keep], extra


# ── labelling schemes ───────────────────────────────────────────────────────

class Tree:
    """Recursive 3-way KMeans to depth 3. Fit on train, applies to any point."""

    def __init__(self, seed=0):
        self.seed = seed
        self.root = None

    def fit(self, X):
        def node(idx, depth):
            km = KMeans(3, n_init=10, random_state=self.seed).fit(X[idx])
            n = {"c": km.cluster_centers_, "kids": None}
            if depth < 2:
                a = km.labels_
                n["kids"] = [node(idx[a == b], depth + 1) for b in range(3)]
            return n
        self.root = node(np.arange(len(X)), 0)
        return self

    def apply(self, X):
        out = np.zeros((len(X), 3), dtype=int)
        def walk(n, idx, depth):
            if len(idx) == 0:
                return
            a = np.argmin(((X[idx][:, None, :] - n["c"][None]) ** 2).sum(-1), axis=1)
            out[idx, depth] = a
            if n["kids"]:
                for b in range(3):
                    walk(n["kids"][b], idx[a == b], depth + 1)
        walk(self.root, np.arange(len(X)), 0)
        return out

    def branch_subspaces(self):
        """Depth-2 split directions, per depth-1 branch. Each branch's 3
        centroids span a 2-d subspace after centring."""
        out = []
        for kid in self.root["kids"]:
            c = kid["c"] - kid["c"].mean(0)
            out.append(np.linalg.qr(c.T)[0][:, :2])
        return out


class PcaProduct:
    """Top-3 PCs, tertiled. A blind PRODUCT — the fairest rival to EO."""

    def __init__(self, seed=0):
        self.p = PCA(3, random_state=seed)
        self.cuts = None

    def fit(self, X):
        s = self.p.fit_transform(X)
        self.cuts = [np.quantile(s[:, k], [1 / 3, 2 / 3]) for k in range(3)]
        return self

    def apply(self, X):
        s = self.p.transform(X)
        return np.stack([np.searchsorted(self.cuts[k], s[:, k]) for k in range(3)], 1)


# ── scoring ─────────────────────────────────────────────────────────────────

def design(lab):
    X = np.zeros((len(lab), 9))
    r = np.arange(len(lab))
    X[r, lab[:, 0]] = 1
    X[r, 3 + lab[:, 1]] = 1
    X[r, 6 + lab[:, 2]] = 1
    return X


def centroids_of(X, lab, min_n=1):
    """Returns dict (i,j,k)->mean, and counts."""
    out, cnt = {}, {}
    for cell in itertools.product(range(3), repeat=3):
        m = (lab == np.array(cell)).all(1)
        if m.sum() >= min_n:
            out[cell] = X[m].mean(0)
            cnt[cell] = int(m.sum())
    return out, cnt


def r2(y, pred, base):
    return 1.0 - ((y - pred) ** 2).sum() / ((y - base) ** 2).sum()


def score(Xtr, ltr, Xte, lte):
    """PAST = held-out variance the 27 groups explain (cell means).
       FUTURE-a = held-out variance the additive product model explains.
       FUTURE-b = leave-one-CELL-out prediction of an unseen combination."""
    mu = Xtr.mean(0)
    cent, cnt = centroids_of(Xtr, ltr)
    filled = len(cent)

    # PAST: predict a held-out clause by its own cell's train mean
    pred_cell = np.array([cent.get(tuple(c), mu) for c in lte])
    past = r2(Xte, pred_cell, mu)

    # FUTURE-a: predict it from the additive model instead
    keys = sorted(cent)
    Y = np.array([cent[k] for k in keys])
    L = np.array(keys)
    B = np.linalg.pinv(design(L)) @ Y
    pred_add = design(lte) @ B
    fut_a = r2(Xte, pred_add, mu)

    # FUTURE-b: leave one whole cell out of the fit, predict its centroid
    sse = sst = 0.0
    if filled >= 10:
        Dm = design(L)
        for h in range(len(keys)):
            tr = [r for r in range(len(keys)) if r != h]
            if len(np.unique(L[tr][:, 0])) < 3 or len(np.unique(L[tr][:, 1])) < 3 \
               or len(np.unique(L[tr][:, 2])) < 3:
                continue
            Bh = np.linalg.pinv(Dm[tr]) @ Y[tr]
            p = Dm[h] @ Bh
            m = Y[tr].mean(0)
            sse += ((Y[h] - p) ** 2).sum()
            sst += ((Y[h] - m) ** 2).sum()
    fut_b = 1.0 - sse / sst if sst > 0 else float("nan")

    return {"cells_filled": filled,
            "past_cellmean_r2": float(past),
            "future_additive_r2": float(fut_a),
            "future_unseen_cell_r2": float(fut_b),
            "product_ratio": float(fut_a / past) if past > 0 else float("nan")}


def flat(l):
    return l[:, 0] * 9 + l[:, 1] * 3 + l[:, 2]


def aligned_ari(a, b):
    ua, ub = np.unique(a), np.unique(b)
    M = np.zeros((len(ua), len(ub)))
    for i, x in enumerate(ua):
        for j, y in enumerate(ub):
            M[i, j] = ((a == x) & (b == y)).sum()
    r, c = linear_sum_assignment(-M)
    return adjusted_rand_score(a, b), float(M[r, c].sum() / len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="the run's real embeddings.npz")
    ap.add_argument("--emb-b", default=None,
                    help="a SECOND embedding of the same clauses. Schemes are fitted in "
                         "space A and scored in both. A scheme whose cells are defined BY "
                         "space A is a product there by construction; only a scheme carried "
                         "by the text survives into space B.")
    ap.add_argument("--all", action="store_true", help="use all clauses, not just consensus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--out", default="recursive_split_results.json")
    args = ap.parse_args()

    X, lab, extra = load_run(Path(args.emb), consensus_only=not args.all)
    print(f"{len(X):,} clauses  dim={X.shape[1]}  "
          f"({'all labelled' if args.all else 'consensus only'})")
    if "language" in extra:
        u = np.unique(extra["language"])
        print(f"languages: {len(u)}  {', '.join(sorted(u)[:12])}{' ...' if len(u) > 12 else ''}")

    XB = None
    if args.emb_b:
        XB_all, labB, extraB = load_run(Path(args.emb_b), consensus_only=not args.all)
        if "ids" in extra and "ids" in extraB:
            pos = {str(v): i for i, v in enumerate(extraB["ids"])}
            order = np.array([pos[str(v)] for v in extra["ids"]])
            XB = XB_all[order]
            assert (labB[order] == lab).all(), "space B labels disagree with space A"
        else:
            XB = XB_all
        print(f"space B: {XB.shape[1]}-d, aligned on {len(XB):,} clauses")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    ncut = int(len(X) * (1 - args.test_frac))
    tr, te = perm[:ncut], perm[ncut:]
    Xtr, Xte = X[tr], X[te]
    print(f"train {len(tr):,}  test {len(te):,}\n")

    schemes = {}
    schemes["eo"] = (lab[tr], lab[te])

    tree = Tree(args.seed).fit(Xtr)
    schemes["tree(recursive kmeans)"] = (tree.apply(Xtr), tree.apply(Xte))

    pca = PcaProduct(args.seed).fit(Xtr)
    schemes["pca-tertile(blind product)"] = (pca.apply(Xtr), pca.apply(Xte))

    rl = rng.integers(0, 3, size=(len(X), 3))
    schemes["random"] = (rl[tr], rl[te])

    def table(A_tr, A_te, title):
        print(f"\n{title}")
        print(f"{'scheme':28} {'cells':>5} {'PAST':>8} {'FUTURE':>8} {'UNSEEN':>8} {'ratio':>7}")
        print(f"{'':28} {'':>5} {'cellmean':>8} {'additive':>8} {'cell':>8}")
        print("-" * 70)
        out = {}
        for name, (a, b) in schemes.items():
            sc = score(A_tr, a, A_te, b)
            out[name] = sc
            print(f"{name:28} {sc['cells_filled']:5d} {sc['past_cellmean_r2']:+8.4f} "
                  f"{sc['future_additive_r2']:+8.4f} {sc['future_unseen_cell_r2']:+8.4f} "
                  f"{sc['product_ratio']:7.3f}")
        return out

    results = {"space_a_fit": table(Xtr, Xte, "SCORED IN SPACE A (the space the blind schemes were fitted in)")}
    if XB is not None:
        results["space_b_transfer"] = table(
            XB[tr], XB[te],
            "SCORED IN SPACE B (transfer - blind cells were never fitted here)")

    # product vs tree
    subs = tree.branch_subspaces()
    print("\nPRODUCT vs TREE — depth-2 split direction, compared across branches")
    print("  (principal angles; near 0 = the same question asked everywhere = product)")
    pv = {}
    for a, b in itertools.combinations(range(3), 2):
        sv = np.linalg.svd(subs[a].T @ subs[b], compute_uv=False)
        ang = [round(float(x), 1) for x in np.degrees(np.arccos(np.clip(sv, -1, 1)))]
        pv[f"branch{a+1}_vs_branch{b+1}"] = ang
        print(f"    branch {a+1} vs branch {b+1}:  {ang} deg")

    print("\ncross-walk against EO (test split, 27 flat labels)")
    cw = {}
    for name, (_, b) in schemes.items():
        if name == "eo":
            continue
        ari, acc = aligned_ari(flat(lab[te]), flat(b))
        per = [round(adjusted_rand_score(lab[te][:, k], b[:, k]), 4) for k in range(3)]
        cw[name] = {"ari_flat27": round(ari, 4), "hungarian_acc": round(acc, 4), "per_axis_ari": per}
        print(f"    {name:28} ARI={ari:+.4f}  aligned acc={acc:.3f}  per-axis {per}")

    Path(args.out).write_text(json.dumps(
        {"emb": Path(args.emb).name, "n": len(X), "n_train": len(tr), "n_test": len(te),
         "scores": results, "product_vs_tree_principal_angles_deg": pv, "crosswalk_vs_eo": cw},
        indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
