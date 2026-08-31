"""
local_embeddings.py — self-contained clause embeddings for this suite.

The repo's own per-run embeddings.npz files are Google Drive pointers
(docs/WHY-THESE-THREE.md: "the shipped embeddings.npz is an 83-byte Drive
pointer"), fetched from a link that may not always resolve. Rather than
depend on that, this suite computes its own embeddings with
paraphrase-multilingual-MiniLM-L12-v2 — the SAME model WHY-THESE-THREE.md
already uses to build the 27 archetype centroids and as the transfer
target ("space B") in Result 1 — so results stay comparable in spirit
while the suite stays fully reproducible offline from just this repo plus
one `pip install`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def embed(clauses: list[str]) -> np.ndarray:
    model = _get_model()
    return np.asarray(model.encode(list(clauses), show_progress_bar=False, batch_size=32))


def embed_cached(clauses: list[str], cache_path: str | Path) -> np.ndarray:
    """Disk-cached embeddings keyed by exact clause text — re-running
    discovery over the same sample shouldn't re-embed it."""
    cache_path = Path(cache_path)
    cache = {}
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        cache = dict(zip(data["clauses"].tolist(), data["vectors"]))

    missing = [c for c in clauses if c not in cache]
    if missing:
        for c, v in zip(missing, embed(missing)):
            cache[c] = v
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        all_clauses = list(cache.keys())
        np.savez(cache_path,
                  clauses=np.array(all_clauses, dtype=object),
                  vectors=np.array([cache[c] for c in all_clauses]))

    return np.array([cache[c] for c in clauses])
