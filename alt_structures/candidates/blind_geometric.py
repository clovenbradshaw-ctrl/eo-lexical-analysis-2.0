"""
Blind / mechanical rivals already established in this repo, wrapped to
harness.py's (labels, n_lev) contract. No LLM involved — these answer "how
much of EO's apparent structure comes free from geometry or surface
statistics alone."

Tree and PcaProduct are recursive_split.py's own classes, imported
directly rather than reimplemented, so results are guaranteed consistent
with the rest of the repo (same KMeans/PCA/tertile logic, same seeds).
Surface reproduces falsify_3x3x3.py's exact char-length / type-token-ratio
/ punctuation-density features (see its _surface_features), with a
fit(train)/apply(test) split added so it can run through this harness's
held-out PAST/FUTURE/UNSEEN scorer instead of falsify_3x3x3.py's
whole-corpus monotonicity test.
"""
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recursive_split import Tree, PcaProduct  # noqa: E402

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _surface_features(text: str):
    n_chars = len(text)
    words = _WORD_RE.findall(text.lower())
    n_words = max(1, len(words))
    n_unique = len(set(words))
    n_punct = len(_PUNCT_RE.findall(text))
    return float(n_chars), n_unique / n_words, n_punct / max(1, n_chars)


class Surface:
    """char-length x type-token-ratio x punctuation-density, tertiled.
    Geometry-blind AND semantically-blind — the cheapest conventional
    baseline in corpus/readability work."""

    def __init__(self, seed=0):
        self.cuts = None

    def fit(self, clauses):
        feats = np.array([_surface_features(c) for c in clauses])
        self.cuts = [np.quantile(feats[:, k], [1 / 3, 2 / 3]) for k in range(3)]
        return self

    def apply(self, clauses):
        feats = np.array([_surface_features(c) for c in clauses])
        return np.stack([np.searchsorted(self.cuts[k], feats[:, k]) for k in range(3)], 1).astype(int)


# name -> (needs_raw_text: bool, n_lev, factory(seed))
# needs_raw_text=False candidates fit/apply on embeddings X; True ones fit/apply on clause strings.
GEOMETRIC_CANDIDATES = {
    "tree(recursive kmeans)": (False, [3, 3, 3], lambda seed: Tree(seed)),
    "pca-tertile(blind product)": (False, [3, 3, 3], lambda seed: PcaProduct(seed)),
    "surface(char/ttr/punct)": (True, [3, 3, 3], lambda seed: Surface(seed)),
}
