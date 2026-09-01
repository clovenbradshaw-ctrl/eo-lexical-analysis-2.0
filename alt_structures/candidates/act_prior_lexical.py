"""
ActPrior-based lexical candidate: the main verb's act, via live_priors'
own ActPrior@1 lexicon (derived-priors/act-priors/act-prior-en.json) -- a
disclosed, hand-built mapping from every VerbNet-attested verb FORM to one
of the ecosystem's nine acts (NUL/SIG/INS/SEG/CON/SYN/DEF/EVA/REC), the
same lexicon eoreader7's phasepost.js consumes for real. Far more
principled than this suite's own verbnet_lexical.py bucketing (which only
sees the top-5 most frequent VerbNet top-level classes in whatever sample
it happens to be fit on) -- and verbnet_lexical.py's full-corpus score
(-0.0004, indistinguishable from zero) is exactly why this replacement is
worth building rather than tuning the old one further.

Cross-repo dependency, by design: this candidate tests whether a lexicon
that already lives in production (live_priors + eoreader7) structures
these clauses, which needs that repo's file. Set LIVE_PRIORS_PATH if it
isn't a sibling checkout of this repo; fit() raises a clear error rather
than silently skipping if the file can't be found.

For a 'contested' verb form (872 of 4,569 in the lexicon -- multiple
candidate acts on record), this takes the FIRST listed candidate's op --
a stated simplification. act-prior-en.json's own "candidates" list is the
authority on what the alternates were and why; this doesn't adjudicate
between them the way a real disambiguation pass would.
"""
import json
import os
from pathlib import Path

import numpy as np

_ACTS = ("NUL", "SIG", "INS", "SEG", "CON", "SYN", "DEF", "EVA", "REC")


def _find_act_prior_file():
    env = os.environ.get("LIVE_PRIORS_PATH")
    candidates = []
    if env:
        candidates.append(Path(env) / "derived-priors" / "act-priors" / "act-prior-en.json")
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "live_priors" / "derived-priors" / "act-priors" / "act-prior-en.json")
    candidates.append(Path("/home/user/live_priors/derived-priors/act-priors/act-prior-en.json"))
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "act-prior-en.json not found. This candidate needs a live_priors checkout "
        "(derived-priors/act-priors/act-prior-en.json) -- set LIVE_PRIORS_PATH to its "
        "root, or clone it as a sibling of this repo. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


_lexicon = None
_lexicon_meta = None


def _load_lexicon():
    global _lexicon, _lexicon_meta
    if _lexicon is None:
        data = json.loads(_find_act_prior_file().read_text())
        _lexicon = data["forms"]
        _lexicon_meta = data.get("counts", {})
    return _lexicon


def verb_act(verb_form: str):
    """Looks up a single verb FORM in the lexicon (case-insensitive) and
    returns its act code, or None if the form isn't covered."""
    if not verb_form:
        return None
    lex = _load_lexicon()
    entry = lex.get(verb_form.lower())
    if entry is None:
        return None
    if "op" in entry:
        return entry["op"]
    candidates = entry.get("candidates") or []
    return candidates[0]["op"] if candidates else None


def main_verb_lemma(clause: str):
    """Same POS-tag main-verb heuristic as candidates.verbnet_lexical, so
    results are directly comparable to that candidate (and reuses its NLTK
    setup rather than duplicating it)."""
    from candidates.verbnet_lexical import _ensure_nltk_data, _AUX
    _ensure_nltk_data()
    import nltk
    from nltk.stem import WordNetLemmatizer

    toks = nltk.word_tokenize(clause)
    tags = nltk.pos_tag(toks)
    verbs = [w for w, t in tags if t.startswith("VB")]
    if not verbs:
        return None
    non_aux = [w for w in verbs if w.lower() not in _AUX]
    chosen = (non_aux or verbs)[0]
    return WordNetLemmatizer().lemmatize(chosen.lower(), pos="v")


class ActPriorLexical:
    """Single-axis, 10-level candidate (the 9 canonical acts + OTHER for
    verbs the lexicon doesn't cover, or clauses with no verb found)."""

    def __init__(self, seed=0):
        self.vocab = list(_ACTS) + ["OTHER"]

    def fit(self, clauses):
        _load_lexicon()  # fail fast here, not on first apply()
        return self

    def apply(self, clauses):
        idx = {a: i for i, a in enumerate(self.vocab)}
        other = len(self.vocab) - 1
        out = []
        for c in clauses:
            lemma = main_verb_lemma(c)
            act = verb_act(lemma) if lemma else None
            out.append(idx.get(act, other))
        return np.array(out, dtype=int).reshape(-1, 1)


# name -> (needs_raw_text, n_lev, factory(seed))
ACT_PRIOR_CANDIDATES = {
    "actprior-lexical(9 acts+other)": (True, [10], lambda seed: ActPriorLexical(seed)),
}
