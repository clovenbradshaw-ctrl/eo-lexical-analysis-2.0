#!/usr/bin/env python3
"""
4th null for Exp 1: corpus-statistic trichotomies.

Tests whether EO's face and 27-cell z-scores are specific to semantic
partitioning, or whether any coarse 3-way partition of the same clauses
produces comparable geometric separation. Builds three-valued "questions"
from features that should be orthogonal to semantics (length, punctuation,
frequency, positional, lexical), combines them into 3-question trichotomy
sets, and computes the same face (Q1xQ2, Q2xQ3, Q1xQ3) and full 27-cell
z-scores against the same embeddings.

If these non-semantic trichotomies produce z-scores in EO's range, the
signal is "models cluster on anything coarse," not "models cluster on
semantics."

Inputs:
  run_dir/
    embeddings.npz      (must contain "vectors" and "ids")
    raw_clauses.jsonl   (one JSON per line with at least {id, clause})

Output:
  run_dir/null_corpus_stats.json

Run:
  python null_corpus_stats.py --run-dir run_2026-03-19_144302 \
      --n-sets 30 --n-shuffles 200
"""

import argparse
import json
import random
import re
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# z-score (mirrors app2.compute_zscore exactly so numbers are comparable)
# ─────────────────────────────────────────────────────────────────────────────
def compute_zscore(vectors: np.ndarray, labels: np.ndarray, n_shuffles: int = 200):
    unique = [l for l in np.unique(labels) if l and l != "?"]
    if len(unique) < 2:
        return 0.0, 0.0

    max_per_group = 200
    rng = np.random.default_rng(seed=42 + len(labels))

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = vectors / norms

    if len([l for l in unique if (labels == l).sum() >= 2]) < 2:
        return 0.0, 0.0

    def separation_from_labels(lbl):
        within, between = [], []
        for label in unique:
            all_pos = np.where(lbl == label)[0]
            if len(all_pos) < 2:
                continue
            pos_s = all_pos if len(all_pos) <= max_per_group else rng.choice(all_pos, max_per_group, replace=False)
            other_pos = np.where(lbl != label)[0]
            if len(other_pos) < 1:
                continue
            other_s = other_pos if len(other_pos) <= max_per_group else rng.choice(other_pos, max_per_group, replace=False)
            vg, vo = normed[pos_s], normed[other_s]
            n_in = len(pos_s)
            if n_in < 2:
                continue
            ri_all, rj_all = np.triu_indices(n_in, k=1)
            n_unique = len(ri_all)
            n_target = min(n_unique, max_per_group * 5)
            if n_target < 1:
                continue
            sel = rng.choice(n_unique, size=n_target, replace=False)
            ri, rj = ri_all[sel], rj_all[sel]
            within.extend(np.sum(vg[ri] * vg[rj], axis=1).tolist())
            n_pairs = min(len(ri), len(pos_s), len(other_s))
            if n_pairs < 1:
                continue
            row_idx = rng.choice(len(pos_s), size=n_pairs, replace=False)
            col_idx = rng.choice(len(other_s), size=n_pairs, replace=False)
            between.extend(np.sum(vg[row_idx] * vo[col_idx], axis=1).tolist())
        if not within or not between:
            return 0.0
        return statistics.mean(within) - statistics.mean(between)

    actual = separation_from_labels(labels)

    shuffled_vals = []
    shuffled = labels.copy()
    for _ in range(n_shuffles):
        rng.shuffle(shuffled)
        shuffled_vals.append(separation_from_labels(shuffled))

    mean_s = statistics.mean(shuffled_vals)
    std_s = statistics.stdev(shuffled_vals) if len(shuffled_vals) > 1 else 1.0
    if std_s < 1e-10:
        return 0.0, actual
    return (actual - mean_s) / std_s, actual


# ─────────────────────────────────────────────────────────────────────────────
# Tercile assignment — rank-based so ties don't collapse bins
# ─────────────────────────────────────────────────────────────────────────────
def tercile_labels(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    n = len(values)
    out = np.empty(n, dtype=object)
    out[:] = "MID"
    out[ranks < n // 3] = "LOW"
    out[ranks >= (2 * n) // 3] = "HIGH"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractors — each returns one float per clause
# ─────────────────────────────────────────────────────────────────────────────
STOPWORDS = set((
    "the a an and or but if then of to in on at by for with from as is are was "
    "were be been being have has had do does did will would can could should "
    "may might must not no nor so that this these those it its they them their "
    "he she him her his hers we us our you your i me my mine which who whom whose"
).split())

_word_re = re.compile(r"[A-Za-z']+")
_vowel_group_re = re.compile(r"[aeiouyAEIOUY]+")


def _words(text):
    return _word_re.findall(text)


def _features_per_clause(text: str, corpus_freq: Counter):
    words = _words(text)
    lower = [w.lower() for w in words]
    n_words = max(len(words), 1)
    n_chars = max(len(text), 1)
    n_letters = max(sum(c.isalpha() for c in text), 1)

    syllables = sum(len(_vowel_group_re.findall(w)) or 1 for w in words)
    punct = sum(1 for c in text if not c.isalnum() and not c.isspace())
    digits = sum(c.isdigit() for c in text)
    capitals = sum(c.isupper() for c in text)
    mean_word_len = sum(len(w) for w in words) / n_words
    ttr = len(set(lower)) / n_words
    stop_ratio = sum(1 for w in lower if w in STOPWORDS) / n_words
    content = [w for w in lower if w not in STOPWORDS]
    if content and corpus_freq:
        mean_log_freq = float(np.mean([np.log1p(corpus_freq.get(w, 0)) for w in content]))
    else:
        mean_log_freq = 0.0
    comma_count = text.count(",")
    semicolon_count = text.count(";") + text.count(":")

    return {
        "length_chars": float(n_chars),
        "length_words": float(n_words),
        "length_unique_words": float(len(set(lower))),
        "length_syllables": float(syllables),
        "punct_density": punct / n_chars,
        "digit_density": digits / n_chars,
        "capital_density": capitals / n_letters,
        "mean_word_len": mean_word_len,
        "type_token_ratio": ttr,
        "stopword_ratio": stop_ratio,
        "mean_log_freq": mean_log_freq,
        "comma_count": float(comma_count),
        "semicolon_count": float(semicolon_count),
    }


def build_features(clauses_by_id: dict, ids: np.ndarray):
    """
    Returns dict of feature_name → np.ndarray[str] (tercile label per id).

    Clauses missing from clauses_by_id get feature value 0 (will fall into LOW
    tercile; those rows should be rare).
    """
    # First pass: corpus frequency over all clauses
    freq = Counter()
    for c in clauses_by_id.values():
        freq.update(w.lower() for w in _words(c))

    # Second pass: per-clause features
    feature_names = list(_features_per_clause("x", freq).keys())
    raw = {fn: np.zeros(len(ids), dtype=np.float64) for fn in feature_names}
    missing = 0
    for i, cid in enumerate(ids):
        text = clauses_by_id.get(str(cid))
        if text is None:
            missing += 1
            continue
        feats = _features_per_clause(text, freq)
        for fn, val in feats.items():
            raw[fn][i] = val

    # Position-in-corpus tercile: deterministic from id ordering
    raw["position_in_corpus"] = np.arange(len(ids), dtype=np.float64)

    labels = {fn: tercile_labels(vals) for fn, vals in raw.items()}
    return labels, missing


# ─────────────────────────────────────────────────────────────────────────────
# Trichotomy set construction
# ─────────────────────────────────────────────────────────────────────────────
# Length-like features are mutually redundant; we mark them so we don't build
# trichotomies like (length_chars, length_words, length_syllables).
FEATURE_FAMILIES = {
    "length_chars": "length",
    "length_words": "length",
    "length_unique_words": "length",
    "length_syllables": "length",
    "mean_word_len": "length",
    "punct_density": "punct",
    "comma_count": "punct",
    "semicolon_count": "punct",
    "digit_density": "glyph",
    "capital_density": "glyph",
    "type_token_ratio": "lex",
    "stopword_ratio": "lex",
    "mean_log_freq": "lex",
    "position_in_corpus": "position",
}


def build_trichotomy_sets(feature_names, n_sets, seed=42):
    """
    Build n_sets trichotomies. Prefer 3 distinct feature *families* per set so
    the Q1/Q2/Q3 axes are not trivially redundant (e.g., word count x char count
    x syllable count). Families: length, punct, glyph, lex, position.
    """
    rng = random.Random(seed)
    # All unordered triples of distinct-family features
    triples = []
    for a, b, c in combinations(feature_names, 3):
        fams = {FEATURE_FAMILIES.get(a), FEATURE_FAMILIES.get(b), FEATURE_FAMILIES.get(c)}
        if len(fams) == 3:
            triples.append((a, b, c))
    rng.shuffle(triples)
    sets = triples[:n_sets]

    # If fewer unique-family triples than requested, top up with random triples
    # (allowing same-family), since users asked for "trichotomies derived from
    # unrelated corpus statistics" — same-family ones are the null's null.
    if len(sets) < n_sets:
        extras = list(combinations(feature_names, 3))
        rng.shuffle(extras)
        for t in extras:
            if t in sets:
                continue
            sets.append(t)
            if len(sets) >= n_sets:
                break
    return sets


# ─────────────────────────────────────────────────────────────────────────────
# Face z-scores for one trichotomy
# ─────────────────────────────────────────────────────────────────────────────
def face_zscores(vectors, q1, q2, q3, n_shuffles):
    out = {}
    pairs = [("act", q1, q2), ("site", q2, q3), ("resolution", q1, q3)]
    for name, a, b in pairs:
        labels = np.array([f"{x}/{y}" for x, y in zip(a, b)])
        valid = (a != "?") & (b != "?")
        if valid.sum() < 50:
            out[name] = {"z": None, "n_groups": 0}
            continue
        z, sep = compute_zscore(vectors[valid], labels[valid], n_shuffles=n_shuffles)
        out[name] = {"z": round(float(z), 2), "separation": round(float(sep), 5),
                     "n_groups": int(len(set(labels[valid])))}
    full = np.array([f"{x}/{y}/{z}" for x, y, z in zip(q1, q2, q3)])
    valid = (q1 != "?") & (q2 != "?") & (q3 != "?")
    if valid.sum() >= 50:
        z, sep = compute_zscore(vectors[valid], full[valid], n_shuffles=n_shuffles)
        out["full_27cell"] = {"z": round(float(z), 2), "separation": round(float(sep), 5),
                              "n_groups": int(len(set(full[valid])))}
    else:
        out["full_27cell"] = {"z": None, "n_groups": 0}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
def load_clauses(path):
    by_id = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            by_id[str(obj["id"])] = obj.get("clause", "")
    return by_id


def pct(values, p):
    if not values:
        return None
    return float(np.percentile(values, p))


def summarise(key, values):
    v = [x for x in values if x is not None]
    if not v:
        return {"n": 0}
    return {
        "n": len(v),
        "min": round(min(v), 2),
        "p50": round(pct(v, 50), 2),
        "p75": round(pct(v, 75), 2),
        "p95": round(pct(v, 95), 2),
        "max": round(max(v), 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--n-sets", type=int, default=30)
    ap.add_argument("--n-shuffles", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSON (default: <run-dir>/null_corpus_stats.json)")
    args = ap.parse_args()

    run_dir = args.run_dir
    emb_path = run_dir / "embeddings.npz"
    clauses_path = run_dir / "raw_clauses.jsonl"
    if not emb_path.exists() or emb_path.stat().st_size < 1024:
        raise SystemExit(
            f"{emb_path} missing or is a drive pointer. Download the real "
            "embeddings.npz before running (see analyze_only.py)."
        )
    if not clauses_path.exists():
        raise SystemExit(f"{clauses_path} not found.")

    print(f"Loading embeddings from {emb_path} ...")
    data = np.load(emb_path, allow_pickle=True)
    vectors = data["vectors"].astype(np.float32)
    ids = data["ids"]
    print(f"  {len(vectors):,} vectors, dim {vectors.shape[1]}")

    print(f"Loading clauses from {clauses_path} ...")
    clauses_by_id = load_clauses(clauses_path)
    print(f"  {len(clauses_by_id):,} clauses loaded")

    print("Extracting corpus-statistic features ...")
    feature_labels, missing = build_features(clauses_by_id, ids)
    print(f"  {len(feature_labels)} features; {missing} ids had no matching clause text")

    sets = build_trichotomy_sets(list(feature_labels.keys()), args.n_sets, seed=args.seed)
    print(f"Running {len(sets)} trichotomies x {args.n_shuffles} shuffles per face ...")

    rows = []
    for i, (a, b, c) in enumerate(sets, 1):
        q1 = feature_labels[a]
        q2 = feature_labels[b]
        q3 = feature_labels[c]
        res = face_zscores(vectors, q1, q2, q3, args.n_shuffles)
        composite = [res[k]["z"] for k in ("act", "site", "resolution")
                     if res[k]["z"] is not None]
        row = {
            "idx": i,
            "q1": a, "q2": b, "q3": c,
            "act": res["act"]["z"],
            "site": res["site"]["z"],
            "resolution": res["resolution"]["z"],
            "full_27cell": res["full_27cell"]["z"],
            "composite_mean": round(float(np.mean(composite)), 2) if composite else None,
        }
        rows.append(row)
        print(
            f"  [{i:2d}/{len(sets)}] {a:22s} x {b:22s} x {c:22s} "
            f"-> act={row['act']}, site={row['site']}, "
            f"res={row['resolution']}, 27={row['full_27cell']}"
        )

    eo_ref = None
    results_path = run_dir / "results.json"
    if results_path.exists():
        try:
            eo = json.loads(results_path.read_text()).get("face_zscores", {})
            eo_ref = {
                "act": eo.get("operators_act", {}).get("z"),
                "site": eo.get("face_site", {}).get("z"),
                "resolution": eo.get("face_resolution", {}).get("z"),
                "full_27cell": eo.get("full_27cell", {}).get("z"),
            }
        except Exception:
            pass

    distribution = {
        face: summarise(face, [r[face] for r in rows])
        for face in ("act", "site", "resolution", "full_27cell", "composite_mean")
    }

    passes = None
    if eo_ref and all(v is not None for v in eo_ref.values()):
        def p95(face):
            return distribution[face]["p95"] if distribution[face]["n"] else None

        passes = {
            face: (eo_ref[face] is not None and p95(face) is not None
                   and eo_ref[face] > p95(face))
            for face in ("act", "site", "resolution", "full_27cell")
        }
        passes["all_four"] = all(passes.values())

    payload = {
        "run_dir": str(run_dir),
        "n_sets": len(rows),
        "n_shuffles": args.n_shuffles,
        "seed": args.seed,
        "features": sorted(feature_labels.keys()),
        "trichotomies": rows,
        "distribution": distribution,
        "eo_reference": eo_ref,
        "eo_exceeds_p95": passes,
    }
    out = args.out or (run_dir / "null_corpus_stats.json")
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")

    print("\n— Distribution of corpus-statistic trichotomy z-scores —")
    for face, s in distribution.items():
        if s["n"] == 0:
            print(f"  {face:16s} (no data)")
            continue
        print(f"  {face:16s} n={s['n']:3d}  min={s['min']:+6.2f}  p50={s['p50']:+6.2f}  "
              f"p75={s['p75']:+6.2f}  p95={s['p95']:+6.2f}  max={s['max']:+6.2f}")
    if eo_ref:
        print("\n— EO reference —")
        for k, v in eo_ref.items():
            print(f"  {k:16s} {v}")
    if passes:
        print("\n— EO above 95th percentile of corpus-stat null —")
        for k, v in passes.items():
            mark = "PASS" if v else "FAIL"
            print(f"  {k:16s} {mark}")


if __name__ == "__main__":
    main()
