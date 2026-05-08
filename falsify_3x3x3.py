#!/usr/bin/env python3
"""
falsify_3x3x3.py — Falsification panel for the EO 3x3x3 claim.

The EO report (see run_*/analysis_report.txt) shows that clauses labeled by
three semantic axes (Q1 mode, Q2 domain, Q3 object) produce a monotonic
relationship between axis-difference count and embedding distance, plus
positive per-axis and 27-cell z-scores. The question this script tests is:

    Could ANY 3x3x3 partition produce the same structure?

If yes, the monotonicity is an artifact of binning, not evidence for EO.
If no — and only EO (and partitions that already use embedding geometry)
produce it — then EO's labels are tracking real semantic geometry.

Methods compared on the SAME embeddings:

  eo            — actual Q1/Q2/Q3 labels                    [the claim]
  random        — uniform random tertile labels             [floor]
  surface       — char-length × type-token × punct tertiles [geometry-blind confound]
  pca-tertile   — top-3 PCs, equal-population tertiles      [geometric ceiling]
  pca-kmeans    — top-3 PCs, KMeans(k=3) per axis           [geometric ceiling, soft]
  optimized     — search 3 unit directions maximising slope [absolute ceiling]

Per method, reports:
  - mean cosine distance by axis-diff count (0,1,2,3)
  - monotonicity slope mean[3]-mean[0], bootstrap p
  - per-axis z-score vs shuffled-label null
  - 27-cell z-score
  - pairwise axis ARI
  - corner-uniqueness: where the (top, top, top) cell sits in the
    1..27 ranking of cells by mean cosine distance to other cell
    centroids — the EO claim is that the top corner is most unique

Cross-walks EO against each non-EO method via Hungarian-aligned ARI on
flat 27-labelings and per-axis pair ARIs.

Usage:
  python falsify_3x3x3.py --run-dir run_2026-03-19_144302
  python falsify_3x3x3.py --run-dir run_2026-03-19_144302 --methods eo,random,pca-tertile
  python falsify_3x3x3.py --self-test       # run on synthetic data, no run-dir needed

Outputs to <run-dir>/falsify/:
  falsify_report.txt
  falsify_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score
    from scipy.optimize import linear_sum_assignment
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    print("Install with: pip install scikit-learn scipy numpy", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_embeddings(run_dir: Path) -> Tuple[np.ndarray, List[str]]:
    """Returns (vectors[N,D], ids[N]). Falls back gracefully if the file is
    actually a Drive-pointer text file (the repo ships placeholders)."""
    path = run_dir / "embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    if path.stat().st_size < 4096:
        head = path.read_bytes()[:200]
        if b"drive.google" in head or b"http" in head[:8]:
            raise RuntimeError(
                f"{path} is a Drive pointer, not the actual embeddings.\n"
                f"Fetch it via analyze_only.py first, e.g.:\n"
                f"  python analyze_only.py   # downloads embeddings into the run dir\n"
                f"Or download manually from the URL inside that file."
            )
    npz = np.load(path)
    vectors = npz["vectors"] if "vectors" in npz.files else npz["embeddings"]
    if "ids" in npz.files:
        ids = [str(x) for x in npz["ids"]]
    else:
        ids = [str(i) for i in range(len(vectors))]
    return vectors.astype(np.float32), ids


def load_classified(run_dir: Path) -> Dict[str, dict]:
    """Returns {id: classified_record}."""
    path = run_dir / "classified.jsonl"
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def load_raw(run_dir: Path) -> Dict[str, dict]:
    """Returns {id: raw_clause}. Used for surface-feature confounds."""
    path = run_dir / "raw_clauses.jsonl"
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


# ─────────────────────────────────────────────────────────────────────────────
# EO labels — best-available consensus, fall back to per-model
# ─────────────────────────────────────────────────────────────────────────────

Q1_VALS = ["DIFFERENTIATING", "RELATING", "GENERATING"]
Q2_VALS = ["EXISTENCE", "STRUCTURE", "SIGNIFICANCE"]
Q3_VALS = ["CONDITION", "PARTICULAR", "PATTERN"]


def eo_labels_for_ids(ids: List[str], classified: Dict[str, dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (q1, q2, q3, valid_mask) as int arrays in {0,1,2} (or -1 if missing)."""
    n = len(ids)
    q1 = np.full(n, -1, dtype=np.int32)
    q2 = np.full(n, -1, dtype=np.int32)
    q3 = np.full(n, -1, dtype=np.int32)
    for i, cid in enumerate(ids):
        rec = classified.get(cid)
        if rec is None:
            continue
        consensus = rec.get("consensus")
        cls = rec.get("classifications", {})
        # Prefer consensus, else claude, else gpt4
        chosen = None
        if isinstance(consensus, dict) and consensus.get("q1") in Q1_VALS:
            chosen = consensus
        elif "claude" in cls and cls["claude"].get("q1") in Q1_VALS:
            chosen = cls["claude"]
        elif "gpt4" in cls and cls["gpt4"].get("q1") in Q1_VALS:
            chosen = cls["gpt4"]
        if chosen is None:
            continue
        try:
            q1[i] = Q1_VALS.index(chosen.get("q1", ""))
            q2[i] = Q2_VALS.index(chosen.get("q2", ""))
            q3[i] = Q3_VALS.index(chosen.get("q3", ""))
        except ValueError:
            q1[i] = q2[i] = q3[i] = -1
    valid = (q1 >= 0) & (q2 >= 0) & (q3 >= 0)
    return q1, q2, q3, valid


# ─────────────────────────────────────────────────────────────────────────────
# Methods — each returns (q1, q2, q3) int arrays in {0,1,2}
# ─────────────────────────────────────────────────────────────────────────────

def method_random(vectors: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(vectors)
    return (
        rng.integers(0, 3, size=n).astype(np.int32),
        rng.integers(0, 3, size=n).astype(np.int32),
        rng.integers(0, 3, size=n).astype(np.int32),
    )


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _surface_features(text: str) -> Tuple[float, float, float]:
    n_chars = len(text)
    words = _WORD_RE.findall(text.lower())
    n_words = max(1, len(words))
    n_unique = len(set(words))
    n_punct = len(_PUNCT_RE.findall(text))
    char_len = float(n_chars)
    type_token = n_unique / n_words
    punct_density = n_punct / max(1, n_chars)
    return char_len, type_token, punct_density


def method_surface(ids: List[str], raw: Dict[str, dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tertile-cut three surface features. Geometry-blind, semantically blind."""
    feats = np.zeros((len(ids), 3), dtype=np.float64)
    for i, cid in enumerate(ids):
        rec = raw.get(cid, {})
        text = rec.get("clause", "") or rec.get("text", "")
        feats[i] = _surface_features(text)
    return tuple(_tertile_cut(feats[:, k]) for k in range(3))  # type: ignore[return-value]


def _tertile_cut(scores: np.ndarray) -> np.ndarray:
    """Equal-population tertile cut. Returns ints in {0,1,2} ordered by score."""
    order = np.argsort(scores, kind="stable")
    out = np.empty_like(order, dtype=np.int32)
    n = len(scores)
    edges = [0, n // 3, 2 * n // 3, n]
    for k in range(3):
        out[order[edges[k]:edges[k + 1]]] = k
    return out


def method_pca_tertile(vectors: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pca = PCA(n_components=n_components, random_state=0)
    proj = pca.fit_transform(vectors)
    return tuple(_tertile_cut(proj[:, k]) for k in range(3))  # type: ignore[return-value]


def method_pca_kmeans(vectors: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pca = PCA(n_components=n_components, random_state=0)
    proj = pca.fit_transform(vectors)
    out = []
    for k in range(3):
        km = KMeans(n_clusters=3, n_init=10, random_state=0)
        lbl = km.fit_predict(proj[:, [k]])
        # Re-order cluster ids by ascending centroid value so axis is monotone
        centers = km.cluster_centers_.flatten()
        order = np.argsort(centers)
        remap = {old: new for new, old in enumerate(order)}
        out.append(np.array([remap[v] for v in lbl], dtype=np.int32))
    return tuple(out)  # type: ignore[return-value]


def method_optimized(vectors: np.ndarray,
                     n_restarts: int = 12,
                     n_local: int = 8,
                     pair_budget: int = 1500,
                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Search 3 unit directions in embedding space whose tertile-cuts maximise
    the monotonicity slope mean_dist[3]-mean_dist[0]. This is the absolute
    upper bound for any 3x3x3 binning of these vectors.

    Strategy:
      1. seed each direction with a top PC then perturb
      2. coordinate-ascent: rotate one direction at a time toward a random
         tangent vector if it improves the objective
    Pair-sampling is shared across candidate evaluations so comparisons
    are paired (no Monte-Carlo confound).
    """
    rng = np.random.default_rng(seed)
    n, d = vectors.shape

    # Pre-normalise for cosine
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = vectors / norms

    # Pre-sample a fixed pair budget; we will *re-bucket* these same pairs as
    # candidate axes change. This keeps candidate scores comparable.
    pair_idx = rng.integers(0, n, size=(pair_budget, 2))
    pair_idx = pair_idx[pair_idx[:, 0] != pair_idx[:, 1]]
    pair_cos_dist = 1.0 - np.einsum("ij,ij->i", normed[pair_idx[:, 0]], normed[pair_idx[:, 1]])

    # Seed with top PCs
    pca = PCA(n_components=min(8, d), random_state=0)
    pca.fit(vectors[: min(2000, n)])
    seeds = pca.components_[:8]

    def _slope(dirs: np.ndarray) -> float:
        # Project, tertile-cut, then bucket by axis-diff count
        proj = vectors @ dirs.T  # [n, 3]
        cuts = np.stack([_tertile_cut(proj[:, k]) for k in range(3)], axis=1)
        a = cuts[pair_idx[:, 0]]
        b = cuts[pair_idx[:, 1]]
        diff = (a != b).sum(axis=1)
        means = []
        for k in range(4):
            mask = diff == k
            if mask.sum() < 30:
                return -np.inf
            means.append(pair_cos_dist[mask].mean())
        # Reward monotonicity AND magnitude
        if not all(means[i] <= means[i + 1] for i in range(3)):
            # Penalise non-monotone candidates so search stays in the
            # monotonic region; still return slope so ordering is informative.
            return (means[-1] - means[0]) - 1.0
        return means[-1] - means[0]

    def _orth(dirs: np.ndarray) -> np.ndarray:
        """Gram-Schmidt orthonormalise rows."""
        out = []
        for v in dirs:
            for u in out:
                v = v - (v @ u) * u
            nrm = np.linalg.norm(v)
            out.append(v / nrm if nrm > 1e-9 else v)
        return np.stack(out, axis=0)

    best_dirs = None
    best_score = -np.inf
    for r in range(n_restarts):
        # Random orthonormal triple seeded from the top PCs
        if r == 0:
            dirs = _orth(seeds[:3].copy())
        else:
            picks = rng.choice(len(seeds), size=3, replace=False)
            dirs = _orth(seeds[picks] + 0.25 * rng.standard_normal((3, d)).astype(np.float32))
        score = _slope(dirs)
        for _ in range(n_local):
            improved = False
            for k in range(3):
                # propose a perturbation in the tangent space of direction k
                tangent = rng.standard_normal(d).astype(np.float32)
                tangent -= dirs[k] * (tangent @ dirs[k])
                tangent /= max(1e-9, np.linalg.norm(tangent))
                for step in (0.4, 0.15, 0.05):
                    cand = dirs.copy()
                    cand[k] = math.cos(step) * dirs[k] + math.sin(step) * tangent
                    cand = _orth(cand)
                    cs = _slope(cand)
                    if cs > score + 1e-6:
                        dirs, score = cand, cs
                        improved = True
                        break
            if not improved:
                break
        if score > best_score:
            best_score = score
            best_dirs = dirs

    if best_dirs is None:
        best_dirs = _orth(seeds[:3].copy())
    proj = vectors @ best_dirs.T
    return tuple(_tertile_cut(proj[:, k]) for k in range(3))  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def proportionality(vectors: np.ndarray,
                    q1: np.ndarray, q2: np.ndarray, q3: np.ndarray,
                    target_per_bucket: int = 5000,
                    n_boot: int = 500,
                    seed: int = 42) -> dict:
    """Mean cosine distance by axis-diff count. Replicates app2.compute_proportionality."""
    n = len(vectors)
    normed = normalise(vectors)
    rng = np.random.default_rng(seed)
    buckets: Dict[int, List[float]] = {0: [], 1: [], 2: [], 3: []}
    attempts = 0
    max_attempts = target_per_bucket * 60
    while any(len(buckets[k]) < target_per_bucket for k in range(4)) and attempts < max_attempts:
        attempts += 1
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j:
            continue
        diff = int(q1[i] != q1[j]) + int(q2[i] != q2[j]) + int(q3[i] != q3[j])
        if len(buckets[diff]) < target_per_bucket:
            cd = 1.0 - float(normed[i] @ normed[j])
            buckets[diff].append(cd)

    means = {}
    for k in range(4):
        v = buckets[k]
        means[k] = {
            "mean_distance": statistics.mean(v) if v else None,
            "n_pairs": len(v),
            "stdev": statistics.stdev(v) if len(v) > 1 else 0.0,
        }
    valid_means = [means[k]["mean_distance"] for k in range(4) if means[k]["mean_distance"] is not None]
    monotone = all(valid_means[i] <= valid_means[i + 1] for i in range(len(valid_means) - 1))

    rng2 = np.random.default_rng(seed)
    boot_pass = 0
    for _ in range(n_boot):
        bm = []
        for k in range(4):
            v = buckets[k]
            if not v:
                continue
            samp = rng2.choice(v, size=len(v), replace=True)
            bm.append(float(samp.mean()))
        if all(bm[i] <= bm[i + 1] for i in range(len(bm) - 1)):
            boot_pass += 1
    p_fail = round(1 - boot_pass / n_boot, 4)

    return {
        "buckets": means,
        "monotone": bool(monotone),
        "monotone_bootstrap_p": p_fail,
        "slope": (valid_means[-1] - valid_means[0]) if len(valid_means) >= 2 else None,
    }


def compute_zscore(vectors: np.ndarray, labels: np.ndarray, n_shuffles: int = 200, seed: int = 42) -> Tuple[float, float]:
    """Replicates app2.compute_zscore. Within-vs-between cosine separation,
    z-scored against shuffled-label baseline."""
    unique = [int(l) for l in np.unique(labels) if l != -1]
    if len(unique) < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(seed + len(labels))
    normed = normalise(vectors)
    max_per_group = 200

    # Build label arrays once
    labels = labels.astype(np.int64)

    def separation(lbl: np.ndarray) -> float:
        within = []
        between = []
        for lb in unique:
            pos = np.where(lbl == lb)[0]
            if len(pos) < 2:
                continue
            if len(pos) > max_per_group:
                pos = rng.choice(pos, max_per_group, replace=False)
            other = np.where(lbl != lb)[0]
            if len(other) < 1:
                continue
            if len(other) > max_per_group:
                other = rng.choice(other, max_per_group, replace=False)
            vg = normed[pos]
            vo = normed[other]
            sims_in = vg @ vg.T
            iu = np.triu_indices_from(sims_in, k=1)
            within.append(sims_in[iu])
            between.append((vg @ vo.T).flatten())
        if not within or not between:
            return 0.0
        return float(np.concatenate(within).mean() - np.concatenate(between).mean())

    actual = separation(labels)
    nulls = []
    for _ in range(n_shuffles):
        perm = rng.permutation(labels)
        nulls.append(separation(perm))
    nulls = np.asarray(nulls)
    mu, sd = float(nulls.mean()), float(nulls.std() or 1e-9)
    z = (actual - mu) / sd
    return z, actual


def axis_ari(q1: np.ndarray, q2: np.ndarray, q3: np.ndarray) -> Dict[str, float]:
    return {
        "q1_vs_q2": float(adjusted_rand_score(q1, q2)),
        "q1_vs_q3": float(adjusted_rand_score(q1, q3)),
        "q2_vs_q3": float(adjusted_rand_score(q2, q3)),
    }


def cell_centroids(vectors: np.ndarray, q1: np.ndarray, q2: np.ndarray, q3: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int]], np.ndarray]:
    """Returns (centroids[K,D], lattice_keys[K], counts[K]) for cells with >=2 members."""
    cells: Dict[Tuple[int, int, int], List[int]] = {}
    for i in range(len(q1)):
        key = (int(q1[i]), int(q2[i]), int(q3[i]))
        cells.setdefault(key, []).append(i)
    keys = sorted(cells.keys())
    cents = np.stack([vectors[cells[k]].mean(axis=0) for k in keys], axis=0)
    counts = np.array([len(cells[k]) for k in keys], dtype=np.int64)
    return cents, keys, counts


def corner_uniqueness(vectors: np.ndarray, q1: np.ndarray, q2: np.ndarray, q3: np.ndarray) -> dict:
    """Rank cells by mean cosine distance to other cell centroids. Report the
    rank of the (top-tertile, top-tertile, top-tertile) cell — i.e. (2,2,2)."""
    cents, keys, counts = cell_centroids(vectors, q1, q2, q3)
    if len(keys) < 4:
        return {"rank_of_top_corner": None, "n_cells": len(keys)}
    cn = normalise(cents)
    sims = cn @ cn.T
    np.fill_diagonal(sims, np.nan)
    mean_dist = 1.0 - np.nanmean(sims, axis=1)
    order = np.argsort(-mean_dist)  # descending: most-unique first
    ranking = {keys[i]: int(np.where(order == i)[0][0]) + 1 for i in range(len(keys))}

    def closest(target: Tuple[int, int, int]) -> Tuple[int, int, int]:
        best, bd = None, 1e9
        for k in keys:
            d = sum(abs(a - b) for a, b in zip(k, target))
            if d < bd:
                bd, best = d, k
        return best  # type: ignore[return-value]

    top_corner = closest((2, 2, 2))
    bot_corner = closest((0, 0, 0))
    center = closest((1, 1, 1))
    return {
        "n_cells": len(keys),
        "top_corner_cell": top_corner,
        "top_corner_rank": ranking.get(top_corner),
        "bot_corner_cell": bot_corner,
        "bot_corner_rank": ranking.get(bot_corner),
        "center_cell": center,
        "center_rank": ranking.get(center),
        "most_unique_cell": keys[int(order[0])],
        "most_unique_dist": float(mean_dist[int(order[0])]),
        "rank_table": [
            {"cell": list(keys[i]), "rank": int(np.where(order == i)[0][0]) + 1,
             "mean_cos_dist_to_other_cells": float(mean_dist[i]),
             "n_members": int(counts[i])}
            for i in range(len(keys))
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-walks
# ─────────────────────────────────────────────────────────────────────────────

def hungarian_aligned_ari(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Returns (ari, accuracy_after_optimal_label_alignment)."""
    ua = np.unique(a)
    ub = np.unique(b)
    cost = np.zeros((len(ua), len(ub)), dtype=np.int64)
    for i, va in enumerate(ua):
        ai = a == va
        for j, vb in enumerate(ub):
            cost[i, j] = -int(np.sum(ai & (b == vb)))
    ri, cj = linear_sum_assignment(cost)
    matches = -cost[ri, cj].sum()
    acc = matches / len(a)
    return float(adjusted_rand_score(a, b)), float(acc)


def flat_label(q1: np.ndarray, q2: np.ndarray, q3: np.ndarray) -> np.ndarray:
    return q1 * 9 + q2 * 3 + q3


def best_axis_pairing(q_a: Tuple[np.ndarray, np.ndarray, np.ndarray],
                      q_b: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Dict[str, float]:
    """For each pair (a-axis, b-axis), compute ARI. Hungarian-match axes to
    maximise total ARI; return the matching and per-axis ARIs."""
    M = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            M[i, j] = adjusted_rand_score(q_a[i], q_b[j])
    ri, cj = linear_sum_assignment(-M)
    return {
        "matching": [(int(i), int(j), float(M[i, j])) for i, j in zip(ri, cj)],
        "max_total_ari": float(M[ri, cj].sum()),
        "matrix": M.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

ALL_METHODS = ["eo", "random", "surface", "pca-tertile", "pca-kmeans", "optimized"]


def run_method(name: str,
               vectors: np.ndarray,
               eo: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
               surface: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
               seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if name == "eo":
        if eo is None:
            raise RuntimeError("eo labels not loaded")
        return eo
    if name == "random":
        return method_random(vectors, seed=seed)
    if name == "surface":
        if surface is None:
            raise RuntimeError("surface features unavailable (raw_clauses.jsonl missing)")
        return surface
    if name == "pca-tertile":
        return method_pca_tertile(vectors)
    if name == "pca-kmeans":
        return method_pca_kmeans(vectors)
    if name == "optimized":
        return method_optimized(vectors, seed=seed)
    raise ValueError(f"unknown method {name}")


def evaluate(name: str, vectors: np.ndarray, q1: np.ndarray, q2: np.ndarray, q3: np.ndarray) -> dict:
    print(f"  evaluating {name} ...", flush=True)
    t0 = time.time()
    prop = proportionality(vectors, q1, q2, q3)
    z_q1, _ = compute_zscore(vectors, q1)
    z_q2, _ = compute_zscore(vectors, q2)
    z_q3, _ = compute_zscore(vectors, q3)
    z_27, _ = compute_zscore(vectors, flat_label(q1, q2, q3))
    aris = axis_ari(q1, q2, q3)
    corner = corner_uniqueness(vectors, q1, q2, q3)
    return {
        "method": name,
        "elapsed_s": round(time.time() - t0, 2),
        "proportionality": prop,
        "z_scores": {"q1": z_q1, "q2": z_q2, "q3": z_q3, "27cell": z_27},
        "axis_ari": aris,
        "corner": corner,
        "label_distribution": {
            "q1": Counter(int(x) for x in q1),
            "q2": Counter(int(x) for x in q2),
            "q3": Counter(int(x) for x in q3),
        },
    }


def format_report(results: List[dict], crosswalks: List[dict]) -> str:
    lines = []
    pad = lambda s, w: str(s).ljust(w)
    lines.append("=" * 82)
    lines.append("  EO 3x3x3 FALSIFICATION PANEL")
    lines.append("=" * 82)
    lines.append("")
    lines.append("  Tests whether ANY 3x3x3 partition of the same embeddings reproduces")
    lines.append("  the monotonic distance-by-axis-diff pattern attributed to EO. If")
    lines.append("  random or surface-confound labels also produce monotonicity, the")
    lines.append("  EO geometric signature is an artefact of binning, not evidence")
    lines.append("  for EO. If only EO and label-from-geometry methods produce it,")
    lines.append("  EO is tracking real semantic structure.")
    lines.append("")
    lines.append("  Methods:")
    lines.append("    eo            actual Q1/Q2/Q3 labels [the claim]")
    lines.append("    random        uniform random tertile labels [floor]")
    lines.append("    surface       char-len x type-token x punct tertiles [confound]")
    lines.append("    pca-tertile   top-3 PCs, tertile cut [geom ceiling]")
    lines.append("    pca-kmeans    top-3 PCs, k=3 means per axis [soft ceiling]")
    lines.append("    optimized     direct search for max-monotonicity dirs [absolute ceiling]")
    lines.append("")

    # Headline table
    lines.append("-" * 82)
    lines.append("  HEADLINE — does the partition reproduce EO's monotonic-distance pattern?")
    lines.append("-" * 82)
    lines.append("")
    hdr = f"  {pad('method',14)} {pad('slope',8)} {pad('mono',5)} {pad('z_q1',7)} {pad('z_q2',7)} {pad('z_q3',7)} {pad('z_27',7)} {pad('max_ari',8)}"
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for r in results:
        m = r["method"]
        slope = r["proportionality"]["slope"]
        mono = "yes" if r["proportionality"]["monotone"] else "NO"
        zs = r["z_scores"]
        max_ari = max(r["axis_ari"].values())
        lines.append(
            f"  {pad(m,14)} {slope:+.4f}  {pad(mono,5)} "
            f"{zs['q1']:+6.2f} {zs['q2']:+6.2f} {zs['q3']:+6.2f} {zs['27cell']:+6.2f}  {max_ari:+.4f}"
        )
    lines.append("")

    # Per-method detail
    for r in results:
        lines.append("-" * 82)
        lines.append(f"  METHOD: {r['method'].upper()}")
        lines.append("-" * 82)
        bk = r["proportionality"]["buckets"]
        lines.append("  Mean cosine distance by axis-difference count:")
        for k in range(4):
            v = bk[k]
            md = v["mean_distance"]
            md_s = f"{md:.4f}" if md is not None else "  --  "
            lines.append(f"    {k} axes diff  |  {md_s}   (n={v['n_pairs']})")
        lines.append(f"  monotone: {r['proportionality']['monotone']}    bootstrap_p_fail: {r['proportionality']['monotone_bootstrap_p']}")
        zs = r["z_scores"]
        lines.append(f"  z-scores:   q1 {zs['q1']:+.2f}    q2 {zs['q2']:+.2f}    q3 {zs['q3']:+.2f}    27cell {zs['27cell']:+.2f}")
        a = r["axis_ari"]
        lines.append(f"  axis ARI:   q1.q2 {a['q1_vs_q2']:+.4f}    q1.q3 {a['q1_vs_q3']:+.4f}    q2.q3 {a['q2_vs_q3']:+.4f}")
        c = r["corner"]
        if c["n_cells"]:
            lines.append(
                f"  corner:     n_cells={c['n_cells']}   "
                f"top-corner rank {c.get('top_corner_rank')} of {c['n_cells']}    "
                f"most-unique cell {c.get('most_unique_cell')}    "
                f"center rank {c.get('center_rank')}"
            )
        ld = r["label_distribution"]
        lines.append(f"  per-axis label counts:  q1 {dict(ld['q1'])}  q2 {dict(ld['q2'])}  q3 {dict(ld['q3'])}")
        lines.append("")

    # Cross-walks
    if crosswalks:
        lines.append("-" * 82)
        lines.append("  CROSS-WALKS — EO vs each non-EO method")
        lines.append("-" * 82)
        lines.append("")
        for cw in crosswalks:
            lines.append(f"  EO vs {cw['method']}:")
            lines.append(f"    flat-27 ARI:                       {cw['flat_ari']:+.4f}")
            lines.append(f"    flat-27 accuracy after Hungarian:  {cw['flat_acc']:.3f}")
            lines.append(f"    max axis-pair ARI sum (Hungarian): {cw['axis_pairing']['max_total_ari']:+.4f}")
            for ai, bj, val in cw["axis_pairing"]["matching"]:
                lines.append(f"      EO Q{ai+1} <-> {cw['method']} axis-{bj+1}   ARI {val:+.4f}")
            lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, help="Path to a run directory (containing embeddings.npz, classified.jsonl, raw_clauses.jsonl)")
    ap.add_argument("--methods", type=str, default="all", help=f"Comma-separated subset of {ALL_METHODS} or 'all'")
    ap.add_argument("--out-dir", type=str, default=None, help="Defaults to <run-dir>/falsify/")
    ap.add_argument("--n-shuffles", type=int, default=200)
    ap.add_argument("--self-test", action="store_true", help="Run on synthetic data to smoke-test the pipeline")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()

    if not args.run_dir:
        ap.error("--run-dir required (or use --self-test)")

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "falsify"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings from {run_dir}/embeddings.npz ...")
    vectors, ids = load_embeddings(run_dir)
    print(f"  {len(vectors)} clauses, dim={vectors.shape[1]}")

    print("Loading classifications ...")
    classified = load_classified(run_dir)
    q1, q2, q3, valid = eo_labels_for_ids(ids, classified)
    n_valid = int(valid.sum())
    print(f"  {n_valid} clauses with usable EO labels (best-available consensus->claude->gpt4)")

    raw = {}
    raw_path = run_dir / "raw_clauses.jsonl"
    if raw_path.exists():
        raw = load_raw(run_dir)

    methods = ALL_METHODS if args.methods == "all" else [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in ALL_METHODS]
    if bad:
        ap.error(f"unknown methods: {bad}; valid: {ALL_METHODS}")

    # Restrict to clauses with valid EO labels for fair comparison
    if "eo" in methods:
        idx = np.where(valid)[0]
        print(f"  restricting to {len(idx)} clauses with EO labels for cross-method parity")
    else:
        idx = np.arange(len(vectors))

    V = vectors[idx]
    eo_triplet = (q1[idx], q2[idx], q3[idx]) if "eo" in methods else None
    surface_triplet = method_surface([ids[i] for i in idx], raw) if ("surface" in methods and raw) else None
    if "surface" in methods and not raw:
        print("  warning: raw_clauses.jsonl missing; dropping 'surface' method")
        methods = [m for m in methods if m != "surface"]

    results = []
    triplets = {}
    for m in methods:
        try:
            triplets[m] = run_method(m, V, eo=eo_triplet, surface=surface_triplet)
        except Exception as e:
            print(f"  {m} failed: {e}")
            continue
        results.append(evaluate(m, V, *triplets[m]))

    # Cross-walks: EO vs each non-EO method
    crosswalks = []
    if "eo" in triplets:
        eo_q1, eo_q2, eo_q3 = triplets["eo"]
        eo_flat = flat_label(eo_q1, eo_q2, eo_q3)
        for m in methods:
            if m == "eo":
                continue
            mq1, mq2, mq3 = triplets[m]
            ari, acc = hungarian_aligned_ari(eo_flat, flat_label(mq1, mq2, mq3))
            crosswalks.append({
                "method": m,
                "flat_ari": ari,
                "flat_acc": acc,
                "axis_pairing": best_axis_pairing((eo_q1, eo_q2, eo_q3), (mq1, mq2, mq3)),
            })

    report = format_report(results, crosswalks)
    print(report)
    (out_dir / "falsify_report.txt").write_text(report)

    serialisable = {
        "n_clauses": len(V),
        "embedding_dim": int(V.shape[1]),
        "methods": [
            {**r,
             "label_distribution": {k: dict(v) for k, v in r["label_distribution"].items()}}
            for r in results
        ],
        "crosswalks": crosswalks,
    }
    (out_dir / "falsify_results.json").write_text(json.dumps(serialisable, indent=2, default=_jsonable))
    print(f"\nWrote {out_dir / 'falsify_report.txt'} and {out_dir / 'falsify_results.json'}")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not jsonable: {type(o)}")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test on synthetic data — verifies the pipeline runs end-to-end and that
# random partitions correctly produce ~zero z-scores while a structured
# 3x3x3 lattice produces the predicted monotonicity.
# ─────────────────────────────────────────────────────────────────────────────

def run_self_test() -> None:
    print("Self-test: synthesising a 3x3x3 lattice in 64-d ...")
    rng = np.random.default_rng(0)
    d = 64
    n_per_cell = 40
    # Three orthogonal axis directions
    A = rng.standard_normal((3, d))
    A, _ = np.linalg.qr(A.T)
    A = A.T  # 3 x d
    spacing = 0.7
    vectors = []
    q1l, q2l, q3l = [], [], []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                center = spacing * (a * A[0] + b * A[1] + c * A[2])
                pts = center + 0.4 * rng.standard_normal((n_per_cell, d))
                vectors.append(pts)
                q1l.extend([a] * n_per_cell)
                q2l.extend([b] * n_per_cell)
                q3l.extend([c] * n_per_cell)
    V = np.concatenate(vectors, axis=0).astype(np.float32)
    eo = (np.array(q1l, dtype=np.int32), np.array(q2l, dtype=np.int32), np.array(q3l, dtype=np.int32))

    methods = ["eo", "random", "pca-tertile", "pca-kmeans", "optimized"]
    triplets = {"eo": eo}
    results = [evaluate("eo", V, *eo)]
    for m in methods[1:]:
        triplets[m] = run_method(m, V, eo=eo, surface=None)
        results.append(evaluate(m, V, *triplets[m]))
    crosswalks = []
    for m in methods[1:]:
        ari, acc = hungarian_aligned_ari(flat_label(*eo), flat_label(*triplets[m]))
        crosswalks.append({
            "method": m,
            "flat_ari": ari,
            "flat_acc": acc,
            "axis_pairing": best_axis_pairing(eo, triplets[m]),
        })
    print(format_report(results, crosswalks))
    print("\nSelf-test finished. Expected: 'eo' and 'pca-tertile' both monotone with high z;")
    print("'random' near zero; cross-walk EO<->pca-tertile axis ARIs should be near 1.")


if __name__ == "__main__":
    main()
