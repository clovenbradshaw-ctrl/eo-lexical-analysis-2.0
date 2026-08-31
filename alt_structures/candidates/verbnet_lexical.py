"""
VerbNet-based lexical class candidate. Deterministic, no LLM call: looks
up the main verb's VerbNet (Levin-style) class via NLTK.

Unlike Vendler/Halliday, VerbNet has no small textbook inventory (~270
leaf classes, ~50+ top-level numbered classes) -- there is no principled
small axis to prompt an LLM for. This instead buckets to the `top_k` most
frequent top-level classes observed when FIT, plus an OTHER catch-all --
a stated simplification specific to whatever sample it's fit on, not a
universal taxonomy the way Vendler's four categories are.

Main-verb selection is a POS-tag heuristic (first non-auxiliary VB* token,
falling back to the first verb token if the clause is all auxiliaries),
not a dependency parse -- another stated simplification. Polysemous verbs
take NLTK's first-listed VerbNet class, not a sense-disambiguated one.
"""
import re
from collections import Counter

import numpy as np

_TOP_RE = re.compile(r"-(\d+)")
_AUX = {"be", "is", "are", "was", "were", "been", "being", "am",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "can", "could", "shall", "should", "may", "might", "must"}

_ready = False


def _ensure_nltk_data():
    global _ready
    if _ready:
        return
    import os
    os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
    import nltk
    for pkg, path in [
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
        ("wordnet", "corpora/wordnet"),
        ("verbnet", "corpora/verbnet"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
    _ready = True


def main_verb_topclass(clause: str):
    """Returns the top-level VerbNet class number (e.g. '26' from
    'build-26.1-1') for the clause's main verb, or None if no verb / no
    VerbNet entry for its lemma."""
    _ensure_nltk_data()
    import nltk
    from nltk.corpus import verbnet as vn
    from nltk.stem import WordNetLemmatizer

    toks = nltk.word_tokenize(clause)
    tags = nltk.pos_tag(toks)
    verbs = [w for w, t in tags if t.startswith("VB")]
    if not verbs:
        return None
    non_aux = [w for w in verbs if w.lower() not in _AUX]
    chosen = (non_aux or verbs)[0]
    lemma = WordNetLemmatizer().lemmatize(chosen.lower(), pos="v")
    classids = vn.classids(lemma)
    if not classids:
        return None
    m = _TOP_RE.search(classids[0])
    return m.group(1) if m else None


class VerbNetLexical:
    def __init__(self, seed=0, top_k=5):
        self.top_k = top_k
        self.vocab = None  # top_k class ids + trailing 'OTHER'/'NONE' bucket

    def fit(self, clauses):
        classes = [main_verb_topclass(c) for c in clauses]
        counts = Counter(c for c in classes if c is not None)
        self.vocab = [c for c, _ in counts.most_common(self.top_k)] + ["OTHER"]
        return self

    def apply(self, clauses):
        idx = {c: i for i, c in enumerate(self.vocab)}
        other = len(self.vocab) - 1
        out = [idx.get(main_verb_topclass(c), other) for c in clauses]
        return np.array(out, dtype=int).reshape(-1, 1)


# name -> (needs_raw_text, n_lev, factory(seed))
VERBNET_CANDIDATES = {
    "verbnet-lexical(top5+other)": (True, [6], lambda seed: VerbNetLexical(seed, top_k=5)),
}
