"""
harness.py — shared scoring for the alt-structure discovery suite.

Reuses this repo's own established rubric instead of inventing a new one:
  - design()/score() — FUTURE (held-out additive-model R^2) and UNSEEN
    (leave-one-cell-out R^2) — imported directly from dimensionality.py,
    which already generalizes them to an arbitrary number of axes and
    levels-per-axis (that script's own "blind sweep" needs exactly this).
  - PAST (held-out cell-mean R^2), same family as recursive_split.py's.
  - a label-shuffle null (mean/sd/z over many reassignments): the
    "shuffled-label null" WHY-THESE-THREE.md sec. 2 shows is a FLOOR, not a
    ceiling — clearing it is necessary but "nearly free" on its own. The
    sharper test in that doc (re-assign the SAME 27 centroids to different
    grid positions) is cube-specific and doesn't generalize to a 1-axis,
    4-level scheme like Vendler aspect, so it isn't reproduced here.
    What substitutes for a ceiling is scoring every candidate — EO, the
    blind geometric rivals already in this repo, and the new
    conventional-linguistics schemes — through this SAME scorer and
    comparing them directly (WHY-THESE-THREE.md sec. 6's rule of thumb:
    "a null that EO beats tells you almost nothing; a ceiling that EO
    matches or beats is the whole result").

Every candidate in candidates/ returns (labels, n_lev) in this shape:
  labels : (n_clauses, n_axes) int array, values in [0, n_lev[a])
  n_lev  : list of per-axis level counts, len == n_axes
so run_discovery.py can score all of them identically regardless of
whether they came from EO, a blind PCA split, or a local-LLM scheme.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dimensionality import design, score as _future_unseen  # noqa: E402


def past_r2(Xtr, ltr, Xte, lte):
    """Held-out variance explained by each test clause's own train-cell mean."""
    mu = Xtr.mean(0)
    cells = {}
    for c in map(tuple, np.unique(ltr, axis=0)):
        m = (ltr == np.array(c)).all(1)
        cells[c] = Xtr[m].mean(0)
    pred = np.array([cells.get(tuple(c), mu) for c in lte])
    sse = ((Xte - pred) ** 2).sum()
    sst = ((Xte - mu) ** 2).sum()
    return float(1.0 - sse / sst) if sst > 0 else float("nan")


def score_structure(Xtr, ltr, Xte, lte, n_lev):
    """PAST/FUTURE/UNSEEN for one label scheme on one train/test split.

    ltr, lte : (n, len(n_lev)) int arrays, values in [0, n_lev[a]).
    """
    out = _future_unseen(Xtr, ltr, Xte, lte, list(n_lev))
    out["past"] = past_r2(Xtr, ltr, Xte, lte)
    out["product_ratio"] = out["future"] / out["past"] if out["past"] and out["past"] > 0 else float("nan")
    return out


def shuffled_null(Xtr, ltr, Xte, lte, n_lev, n_perm=200, seed=0, metric="unseen"):
    """Label-shuffle null for one metric. Returns observed/null_mean/null_sd/z.

    A FLOOR (see module docstring) — clearing it is necessary, not
    sufficient. Shuffles train labels only; test labels/embeddings stay
    real, so a real signal degrades toward this null rather than vanishing
    trivially.
    """
    rng = np.random.default_rng(seed)
    n_lev = list(n_lev)
    observed = score_structure(Xtr, ltr, Xte, lte, n_lev)[metric]
    vals = []
    n = len(ltr)
    for _ in range(n_perm):
        perm = rng.permutation(n)
        v = _future_unseen(Xtr, ltr[perm], Xte, lte, n_lev)[metric]
        if not np.isnan(v):
            vals.append(v)
    vals = np.array(vals)
    mean, sd = float(vals.mean()), float(vals.std())
    z = (observed - mean) / sd if sd > 0 else float("nan")
    return {"metric": metric, "observed": float(observed), "null_mean": mean,
            "null_sd": sd, "z": float(z), "n_perm": len(vals)}


def train_test_split(n, test_frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ncut = int(n * (1 - test_frac))
    return perm[:ncut], perm[ncut:]


def kappa_matrix(rater_labels: dict[str, list]):
    """Pairwise Cohen's kappa between raters on the same items.
    rater_labels: {rater_name: [label_per_item, ...]}, same order, same length.
    Items where either rater is None are dropped pairwise.
    """
    from sklearn.metrics import cohen_kappa_score
    names = list(rater_labels)
    n = len(names)
    mat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = rater_labels[names[i]], rater_labels[names[j]]
            pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            if len(pairs) < 2:
                continue
            xs, ys = zip(*pairs)
            if len(set(xs)) < 2 and len(set(ys)) < 2:
                continue
            mat[i, j] = cohen_kappa_score(xs, ys)
    return names, mat
