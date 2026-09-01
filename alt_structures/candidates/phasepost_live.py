"""
phasepost_live.py -- the REAL production classifier, not a Python
reimplementation. Chains eoreader7's real `extractRelations` (subject/
verb/object capture, negation/polarity) into eoreader7's real
`phasepost.js` (the 27-cell overlay, backed by live_priors' real
ActPrior@1 lexicon) via `bridge/phasepost_live_bridge.mjs`. This module's
own job is narrow: find each clause's main verb (the same POS-tag
heuristic act_prior_lexical.py/verbnet_lexical.py already use), hand it
to Node, and shape the JS side's verdicts into harness.py's contract. No
classification decision is made in Python.

Why this exists (PR#18, this repo): "eoreader7 and the-fold aren't in
this session's GitHub access... Testing whether the cube earns its keep
inside that live pipeline needs that access granted first." This module
is that test.

Three label schemes are exposed, all built from the SAME underlying
verdicts, differing only in how an unresolved case is counted -- this
matters because the naive act_prior_lexical.py's own "take the first
candidate" simplification is exactly the policy phasepost.js's own P56
discipline refuses to take:

  - PhasepostLiveTyped   -- the honest reading. contested/gap/no-match all
    fold to OTHER. Never coin-flips.
  - PhasepostLiveFirstCandidate -- the CONTROL: contested folds to its
    FIRST listed candidate's op (mirroring act_prior_lexical.py's own
    policy, but now downstream of real subject/object/negation capture
    instead of a bare verb lookup). Answers "what does honest typing
    cost, geometrically, against the very policy PR#18 flagged."
  - PhasepostLiveCell -- 2-axis (op x grain), typed (never coin-flips),
    OTHER on either axis when unresolved. Tests whether op and grain
    compose the way EO's own three axes are tested to.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_ACTS = ("NUL", "SIG", "INS", "SEG", "CON", "SYN", "DEF", "EVA", "REC")
_GRAINS = ("Ground", "Figure", "Pattern")
HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent / "bridge" / "phasepost_live_bridge.mjs"


def _ensure_nltk_data():
    os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
    import nltk
    for pkg, path in [
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_AUX = {"be", "is", "are", "was", "were", "been", "being", "am",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "can", "could", "shall", "should", "may", "might", "must"}


def main_verb_surface(clause: str):
    """The clause's main verb, as it actually appears (not lemmatized --
    extractRelations matches the literal surface token). Same POS-tag
    heuristic as verbnet_lexical.py/act_prior_lexical.py: first
    non-auxiliary VB* token, falling back to the first verb token."""
    _ensure_nltk_data()
    import nltk
    toks = nltk.word_tokenize(clause)
    tags = nltk.pos_tag(toks)
    verbs = [w for w, t in tags if t.startswith("VB")]
    if not verbs:
        return None
    non_aux = [w for w in verbs if w.lower() not in _AUX]
    return (non_aux or verbs)[0]


_cache = {}  # keyed by (tuple(ids), dr45) -> list of result dicts, id-indexed


def run_live_pipeline(items: list[dict], dr45: bool = False, node_bin: str = "node"):
    """items: [{id, clause}, ...] (already restricted to English). Returns
    {id: verdict_dict} for every item Node produced a line for (matched or
    not) -- Node is a required step here, not optional; this raises rather
    than degrading silently if it fails, so a bridge break is never
    mistaken for "the cube found nothing"."""
    key = (tuple(sorted(i["id"] for i in items)), dr45)
    if key in _cache:
        return _cache[key]

    with_verbs = []
    for it in items:
        v = main_verb_surface(it["clause"])
        with_verbs.append({"id": it["id"], "clause": it["clause"], "verb": v or ""})

    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "in.jsonl"
        outp = Path(td) / "out.jsonl"
        with open(inp, "w") as f:
            for row in with_verbs:
                f.write(json.dumps(row) + "\n")
        env = dict(os.environ)
        if dr45:
            env["DR45"] = "1"
        proc = subprocess.run(
            [node_bin, str(BRIDGE), str(inp), str(outp)],
            capture_output=True, text=True, env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"phasepost_live_bridge.mjs failed (dr45={dr45}):\n{proc.stderr}")
        print(f"[phasepost_live] {proc.stderr.strip()}", file=sys.stderr)
        results = {}
        with open(outp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                results[r["id"]] = r

    _cache[key] = results
    return results


def coverage_report(results: dict) -> dict:
    """Honest standing breakdown -- never collapsed into a single number."""
    from collections import Counter
    c = Counter()
    for r in results.values():
        if not r.get("matched"):
            c["no_match (extractRelations found no SVO triple)"] += 1
        else:
            c[r["standing"]] += 1
    return dict(c)


def make_live_candidates(id_to_clause: dict, dr45: bool = False):
    """Runs the bridge ONCE over every id in id_to_clause and returns
    ready-to-fit candidate objects sharing that one result set, plus the
    raw results dict (for coverage_report / diagnostics).

    Candidates' fit()/apply() take a list of clause IDS, not clause text --
    a deliberate departure from ACT_PRIOR_CANDIDATES/verbnet_lexical.py's
    own text-keyed convention, found necessary rather than assumed: this
    corpus has 305 clauses whose exact text repeats (3,595 rows, 3,290
    distinct strings), so a text->result lookup would silently collide.
    IDs are what the corpus itself already guarantees unique."""
    items = [{"id": i, "clause": c} for i, c in id_to_clause.items()]
    results = run_live_pipeline(items, dr45=dr45)

    class PhasepostLiveTyped:
        """Single-axis, 10-level (9 acts + OTHER). Never coin-flips:
        contested/gap/no-match all fold to OTHER."""
        vocab = list(_ACTS) + ["OTHER"]

        def fit(self, ids):
            return self

        def apply(self, ids):
            idx = {a: i for i, a in enumerate(self.vocab)}
            other = len(self.vocab) - 1
            out = []
            for i in ids:
                r = results.get(i)
                op = r.get("op") if r and r.get("matched") else None
                out.append(idx.get(op, other))
            return np.array(out, dtype=int).reshape(-1, 1)

    class PhasepostLiveFirstCandidate:
        """Control: mirrors act_prior_lexical.py's own 'take the first
        candidate' policy -- contested folds to candidates[0], not OTHER.
        Everything downstream (subject/object capture, grain, mechanical/
        copula rules) is still the real pipeline; only THIS class chooses
        to coin-flip where phasepost.js itself refuses to."""
        vocab = list(_ACTS) + ["OTHER"]

        def fit(self, ids):
            return self

        def apply(self, ids):
            idx = {a: i for i, a in enumerate(self.vocab)}
            other = len(self.vocab) - 1
            out = []
            for i in ids:
                r = results.get(i)
                op = None
                if r and r.get("matched"):
                    op = r.get("op") or (r.get("candidates") or [None])[0]
                out.append(idx.get(op, other))
            return np.array(out, dtype=int).reshape(-1, 1)

    class PhasepostLiveCell:
        """2-axis: op (9+OTHER) x grain (3+OTHER). Typed, never coin-flips."""
        n_lev = [10, 4]

        def fit(self, ids):
            return self

        def apply(self, ids):
            op_idx = {a: i for i, a in enumerate(list(_ACTS) + ["OTHER"])}
            gr_idx = {g: i for i, g in enumerate(list(_GRAINS) + ["OTHER"])}
            out = []
            for i in ids:
                r = results.get(i)
                op = r.get("op") if r and r.get("matched") else None
                gr = r.get("grain") if r and r.get("matched") else None
                out.append([op_idx.get(op, 3), gr_idx.get(gr, 3)])
            return np.array(out, dtype=int)

    return {
        "phasepost-live-typed(9 acts+other)": (False, [10], lambda seed=0: PhasepostLiveTyped()),
        "phasepost-live-firstcand(9 acts+other)": (False, [10], lambda seed=0: PhasepostLiveFirstCandidate()),
    }, PhasepostLiveCell(), results
