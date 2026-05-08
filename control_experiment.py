"""
Control experiment: do alternative trichotomies show the same embedding-space
geometry as the EO axes?

Constraints in this sandbox:
- text-embedding-3-large (the original embedder) is unreachable.
- HuggingFace is firewalled, so multilingual neural embedders cannot be loaded.
- ConceptNet Numberbatch English (300d) is reachable via S3.
- No LLM API access for re-classification; alternative trichotomies are
  derived from deterministic English-side heuristics (POS / VADER / surface).

Scope: English clauses (n=500), of which 213 have EO consensus labels.
We embed once with numberbatch and compute the same geometry suite for
six trichotomies (q1, q2, q3 from EO; tense / sentiment / speech-act).
"""
import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import nltk
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).parent
RUN = ROOT / "run_2026-03-15_122636"
NBATCH = Path("/tmp/nbatch/numberbatch-en.txt.gz")
OUT = ROOT / "control_results"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(0xE0)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------- corpus
def load_english():
    """Return a list of dicts for every English clause with EO consensus."""
    rows = []
    with open(RUN / "classified.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get("language") != "en":
                continue
            cons = d.get("consensus")
            if not cons:
                continue
            rows.append({
                "id": d["id"],
                "text": d["clause"],
                "q1": cons["q1"],
                "q2": cons["q2"],
                "q3": cons["q3"],
            })
    log(f"english consensus clauses: {len(rows)}")
    return rows


# ---------------------------------------------------------- numberbatch loader
TOK_RX = re.compile(r"[a-z][a-z'-]*")


def tokenize(text: str) -> list[str]:
    return TOK_RX.findall(text.lower())


def load_numberbatch_for_vocab(vocab: set[str]) -> dict[str, np.ndarray]:
    """Stream the gzipped numberbatch file and keep only vectors we need."""
    vec = {}
    needed = vocab.copy()
    with gzip.open(NBATCH, "rt") as f:
        f.readline()  # header "n_words n_dim"
        for line in f:
            sp = line.find(" ")
            w = line[:sp]
            if w in needed:
                v = np.fromstring(line[sp + 1 :], sep=" ", dtype=np.float32)
                vec[w] = v
                needed.discard(w)
                if not needed:
                    break
    log(f"loaded {len(vec)} / {len(vocab)} vectors from numberbatch")
    return vec


def embed_clauses(rows, vectors, dim=300):
    """Average word vectors for each clause; rows missing all words get zero."""
    X = np.zeros((len(rows), dim), dtype=np.float32)
    miss = 0
    for i, r in enumerate(rows):
        toks = tokenize(r["text"])
        vs = [vectors[t] for t in toks if t in vectors]
        if vs:
            X[i] = np.mean(vs, axis=0)
        else:
            miss += 1
    log(f"embedded {len(rows)} clauses, {miss} had zero in-vocab tokens")
    # L2 normalize so cosine ~= dot
    X = normalize(X, axis=1)
    return X


# -------------------------------------------------------- alternative labelers
def label_tense(rows):
    """English tense via NLTK POS tags. Past / present / future."""
    out = []
    for r in rows:
        toks = nltk.word_tokenize(r["text"])
        tags = nltk.pos_tag(toks)
        past = sum(1 for _, t in tags if t in ("VBD", "VBN"))
        pres = sum(1 for _, t in tags if t in ("VBP", "VBZ", "VBG"))
        fut = 0
        text_low = r["text"].lower()
        if re.search(r"\b(will|shall|won't|gonna|going to)\b|'ll\b", text_low):
            fut += 2  # modal/periphrastic future
        if past == 0 and pres == 0 and fut == 0:
            # default: bare verb / no inflection signal -> treat as present
            out.append("present")
            continue
        m = max(past, pres, fut)
        if fut == m:
            out.append("future")
        elif past == m:
            out.append("past")
        else:
            out.append("present")
    return out


def label_sentiment(rows):
    """VADER compound score -> negative / neutral / positive."""
    sia = SentimentIntensityAnalyzer()
    out = []
    for r in rows:
        c = sia.polarity_scores(r["text"])["compound"]
        if c <= -0.05:
            out.append("negative")
        elif c >= 0.05:
            out.append("positive")
        else:
            out.append("neutral")
    return out


IMPER_RX = re.compile(
    r"^\s*(please\s+)?(do|don't|do not|let's|let us|stop|start|go|come|"
    r"give|take|put|set|make|use|run|try|see|look|listen|consider|note|"
    r"remember|remove|add|delete|read|write|tell|ask|find|check|help|"
    r"send|open|close|click|select|copy|paste|enable|disable|"
    r"build|create|update)\b",
    re.IGNORECASE,
)
EXPR_RX = re.compile(
    r"\b(thank|thanks|sorry|congrat|wow|alas|oh|hooray|damn|hell|"
    r"unfortunately|fortunately|amazing|wonderful|awful|terrible|"
    r"great|excellent|love|hate|fear|hope)\b",
    re.IGNORECASE,
)


def label_speech_act(rows):
    """Searle-style 3-way: assertive / directive / expressive.

    Rule order:
      1. trailing '?' or imperative pattern -> directive
      2. trailing '!' or expressive lexical cue -> expressive
      3. else -> assertive
    """
    out = []
    for r in rows:
        t = r["text"].strip()
        end = t[-1] if t else "."
        is_imper = bool(IMPER_RX.match(t))
        if end == "?" or is_imper:
            out.append("directive")
        elif end == "!" or EXPR_RX.search(t):
            out.append("expressive")
        else:
            out.append("assertive")
    return out


# ----------------------------------------------------------------- geometry
def fisher_ratio(X, y):
    """Trace(Sb) / Trace(Sw) -- per-axis class separability."""
    classes = np.unique(y)
    mu = X.mean(axis=0)
    sb = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    sw = np.zeros_like(sb)
    for c in classes:
        Xc = X[y == c]
        n = len(Xc)
        if n < 2:
            continue
        muc = Xc.mean(axis=0)
        d = (muc - mu).reshape(-1, 1)
        sb += n * (d @ d.T)
        sw += np.cov(Xc.T, bias=False) * (n - 1)
    return float(np.trace(sb) / max(np.trace(sw), 1e-12))


def class_centroid_zscore(X, y, n_perm=2000, seed=0):
    """Z-score of mean within-class cosine similarity vs label permutation.

    Higher z = labels carry geometric structure relative to chance.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    def within_sim(labels):
        mu = np.array([X[labels == c].mean(axis=0) for c in classes])
        mu = normalize(mu, axis=1)
        # mean cos(x_i, mu_y_i) across all i
        idx = {c: i for i, c in enumerate(classes)}
        muxs = mu[np.array([idx[c] for c in labels])]
        return float((X * muxs).sum(axis=1).mean())

    obs = within_sim(y)
    null = np.empty(n_perm)
    yp = y.copy()
    for k in range(n_perm):
        rng.shuffle(yp)
        null[k] = within_sim(yp)
    z = (obs - null.mean()) / max(null.std(ddof=1), 1e-12)
    return {
        "obs": obs,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "z": float(z),
        "p_one_sided": float((null >= obs).mean()),
    }


def lda_explained_variance(X, y):
    classes = np.unique(y)
    if len(classes) < 2:
        return None
    n_components = min(len(classes) - 1, X.shape[1])
    if n_components < 1:
        return None
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    try:
        lda.fit(X, y)
        ev = lda.explained_variance_ratio_.tolist()
    except Exception as e:
        return {"error": str(e)}
    return {
        "n_components": n_components,
        "explained_variance_ratio": ev,
        "total": float(sum(ev)),
    }


def lda_subspace(X, y):
    """Return the top-(k-1) LDA discriminant directions as a (d, k-1) matrix."""
    classes = np.unique(y)
    nc = min(len(classes) - 1, X.shape[1])
    if nc < 1:
        return None
    lda = LinearDiscriminantAnalysis(n_components=nc)
    lda.fit(X, y)
    W = lda.scalings_[:, :nc]
    # orthonormalize
    Q, _ = np.linalg.qr(W)
    return Q


def principal_angles(A, B):
    """Principal angles between subspaces spanned by columns of A and B."""
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s)).tolist()


def residualize(X, *Ws):
    """Project X onto the orthogonal complement of the span of all columns in Ws."""
    cols = [w for w in Ws if w is not None]
    if not cols:
        return X
    M = np.concatenate(cols, axis=1)
    Q, _ = np.linalg.qr(M)
    return X - X @ Q @ Q.T


def random_baseline(X, n_classes=3, trials=20, n_perm=2000, seed=0):
    """z-score for uniformly random 3-class labels -- should be ~0."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(trials):
        y = rng.integers(0, n_classes, size=len(X)).astype(str)
        out.append(class_centroid_zscore(X, y, n_perm=n_perm, seed=seed + t)["z"])
    return {"z_mean": float(np.mean(out)),
            "z_std": float(np.std(out, ddof=1)),
            "z_max": float(np.max(out)),
            "trials": trials}


# ----------------------------------------------------------------- run
def main():
    rows = load_english()
    vocab = set()
    for r in rows:
        vocab.update(tokenize(r["text"]))
    vectors = load_numberbatch_for_vocab(vocab)
    X = embed_clauses(rows, vectors)

    # EO labels
    y_q1 = np.array([r["q1"] for r in rows])
    y_q2 = np.array([r["q2"] for r in rows])
    y_q3 = np.array([r["q3"] for r in rows])

    # Alt labels
    y_tense = np.array(label_tense(rows))
    y_sent = np.array(label_sentiment(rows))
    y_act = np.array(label_speech_act(rows))

    schemes = {
        "EO_q1_mode": y_q1,
        "EO_q2_domain": y_q2,
        "EO_q3_object": y_q3,
        "ALT_tense": y_tense,
        "ALT_sentiment": y_sent,
        "ALT_speech_act": y_act,
    }

    log("class distributions:")
    for name, y in schemes.items():
        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        log(f"  {name}: {dist}")

    # ------- per-scheme geometry -------
    per_scheme = {}
    for name, y in schemes.items():
        log(f"computing geometry for {name}")
        per_scheme[name] = {
            "n_classes": int(len(np.unique(y))),
            "class_sizes": dict(zip(*[a.tolist() for a in np.unique(y, return_counts=True)])),
            "fisher_ratio": fisher_ratio(X, y),
            "centroid_zscore": class_centroid_zscore(X, y, n_perm=2000),
            "lda": lda_explained_variance(X, y),
            "silhouette": float(
                silhouette_score(X, y, metric="cosine") if len(np.unique(y)) > 1 else 0.0
            ),
        }

    # ------- pairwise principal angles between axis subspaces -------
    subspaces = {name: lda_subspace(X, y) for name, y in schemes.items()}
    pairwise = {}
    names = list(schemes.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            sa, sb = subspaces[a], subspaces[b]
            if sa is None or sb is None:
                continue
            ang = principal_angles(sa, sb)
            pairwise[f"{a}__vs__{b}"] = {
                "principal_angles_deg": ang,
                "min_angle_deg": float(min(ang)),
                "max_angle_deg": float(max(ang)),
                "cos_min": float(np.cos(np.radians(min(ang)))),
            }

    # ------- random-label baseline -------
    log("computing random 3-class baseline (20 trials)")
    baseline = random_baseline(X, n_classes=3, trials=20)

    # ------- residualization: does EO carry information beyond alt? -------
    log("residualizing X against alt subspaces and recomputing EO geometry")
    alt_subspaces = [subspaces["ALT_tense"], subspaces["ALT_sentiment"],
                     subspaces["ALT_speech_act"]]
    X_res_alt = residualize(X, *alt_subspaces)
    eo_subspaces = [subspaces["EO_q1_mode"], subspaces["EO_q2_domain"],
                    subspaces["EO_q3_object"]]
    X_res_eo = residualize(X, *eo_subspaces)

    residual = {}
    for label, X_, who in [
        ("after_remove_alt", X_res_alt, ["EO_q1_mode", "EO_q2_domain", "EO_q3_object",
                                         "ALT_tense", "ALT_sentiment", "ALT_speech_act"]),
        ("after_remove_eo", X_res_eo, ["EO_q1_mode", "EO_q2_domain", "EO_q3_object",
                                       "ALT_tense", "ALT_sentiment", "ALT_speech_act"]),
    ]:
        residual[label] = {}
        for name in who:
            y = schemes[name]
            residual[label][name] = {
                "centroid_zscore": class_centroid_zscore(X_, y, n_perm=2000),
                "fisher_ratio": fisher_ratio(X_, y),
            }

    out = {
        "embedder": "conceptnet-numberbatch-en-19.08 (300d, mean of word vectors)",
        "n_clauses": int(len(rows)),
        "vocab_in_corpus": int(len(vocab)),
        "vocab_in_numberbatch": int(len(vectors)),
        "schemes": per_scheme,
        "random_3class_baseline": baseline,
        "pairwise_principal_angles": pairwise,
        "residualization": residual,
        "notes": [
            "English consensus subset only (n=213).",
            "EO numbers cannot be compared to the published 3072d figures; "
            "the in-sandbox baseline is computed in numberbatch space.",
            "ALT axis labels are deterministic English heuristics: tense via "
            "NLTK POS tags + future modals, sentiment via VADER, speech-act "
            "via clause-final punctuation + imperative regex + expressive "
            "lexical cues.",
            "Residualization: 'after_remove_alt' projects each clause "
            "vector orthogonal to the LDA subspaces of all three alt axes, "
            "then asks whether EO axes still discriminate. 'after_remove_eo' "
            "is the symmetric check.",
        ],
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    log(f"wrote {OUT/'results.json'}")

    # also persist the labels for replication
    with open(OUT / "labels.jsonl", "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({
                "id": r["id"],
                "q1": y_q1[i], "q2": y_q2[i], "q3": y_q3[i],
                "tense": y_tense[i], "sentiment": y_sent[i], "speech_act": y_act[i],
            }) + "\n")
    np.savez_compressed(OUT / "embeddings.npz",
                        ids=np.array([r["id"] for r in rows]),
                        X=X)
    log("done")


if __name__ == "__main__":
    main()
