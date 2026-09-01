"""
Classification schemes runnable through the local models.

'eo' replays app2.py's own Q1/Q2/Q3 rubric VERBATIM — same system prompt,
same user-prompt template, same JSON parser and label maps, imported
directly from app2.py rather than retyped — so the local models answer the
exact same question Claude and GPT-4o-mini already answered. Combined with
classify_local.py's --ids-from filter (only clauses a given rater already
labeled), this guarantees all four raters are scored on identical clause
spans, not just the same prompt text.

The other schemes are new, more conventional linguistic alternatives to
Q1/Q2/Q3, proposed in response to "what are more conventional ways to
split up text" — run through the same two local models on the same spans,
then scored by harness.py using the identical PAST/FUTURE/UNSEEN rubric
already established in recursive_split.py / factorization_test.py, so the
comparison is apples-to-apples with EO and with the existing blind
geometric rivals (PCA-tertile, KMeans tree, surface stats).
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import app2  # noqa: E402  (reuse the exact prompt/parser used for claude/gpt4)


def _strip_fences(text):
    return re.sub(r"```json|```", "", text).strip()


# ── EO — replication of the original rubric, for the 4-way rater comparison ─

EO_SCHEME = dict(
    name="eo",
    label="EO Q1xQ2xQ3 (original rubric, replicated)",
    system=app2.CLASSIFICATION_SYSTEM,
    prompt_template=app2.CLASSIFICATION_PROMPT,
    parser=app2.parse_classification,
    axes=("q1", "q2", "q3"),
    levels={"q1": ("DIFFERENTIATING", "RELATING", "GENERATING"),
            "q2": ("EXISTENCE", "STRUCTURE", "SIGNIFICANCE"),
            "q3": ("CONDITION", "ENTITY", "PATTERN")},
)


# ── Vendler aspect: states / activities / accomplishments / achievements ────
# The conventional linguistic answer to "does this clause describe a
# transformation at all" — WHY-THESE-THREE.md's "domain error" section found
# 48% of the corpus isn't one. A stative/dynamic/telic split should catch
# that directly rather than post-hoc filtering.

VENDLER_SYSTEM = """\
You are a linguist analyzing the lexical aspect (Aktionsart) of clauses,
following Vendler's four-way classification.

Answer with exactly one of the options listed.
Do not add explanation unless asked.
Return your answer as JSON: {"aspect": "..."}
"""

VENDLER_PROMPT = """\
Clause: {clause}

Classify the MAIN predication's lexical aspect (judge the underlying event
structure, not surface tense/aspect marking):

  STATE          — a static property, relation, or condition with no
                   internal phases or endpoint (know, own, resemble, love,
                   be tall, contain)
  ACTIVITY       — an ongoing dynamic process with no inherent endpoint
                   (run, push a cart, think about something)
  ACCOMPLISHMENT — a dynamic process with a natural endpoint or result
                   (build a house, walk to the store, write a letter)
  ACHIEVEMENT    — an instantaneous change of state (notice, arrive, win,
                   die, recognize)

Return JSON only: {{"aspect": "STATE|ACTIVITY|ACCOMPLISHMENT|ACHIEVEMENT"}}
"""

_VENDLER_VALID = {"STATE", "ACTIVITY", "ACCOMPLISHMENT", "ACHIEVEMENT"}


def _parse_vendler(text):
    try:
        data = json.loads(_strip_fences(text))
        v = str(data.get("aspect", "")).upper().strip()
        return {"aspect": v} if v in _VENDLER_VALID else None
    except Exception:
        return None


VENDLER_SCHEME = dict(
    name="vendler",
    label="Vendler aspectual class (state/activity/accomplishment/achievement)",
    system=VENDLER_SYSTEM,
    prompt_template=VENDLER_PROMPT,
    parser=_parse_vendler,
    axes=("aspect",),
    levels={"aspect": tuple(sorted(_VENDLER_VALID))},
)


# ── Halliday SFG transitivity: process type ──────────────────────────────────
# The closest established precedent to EO's Q1 ("how is the transformation
# structured") — an off-the-shelf, corpus-validated typology of "what kind
# of process is this clause" instead of an invented 3-way split.

HALLIDAY_SYSTEM = """\
You are a linguist coding clauses for TRANSITIVITY (process type) in the
Systemic Functional Grammar tradition (Halliday).

Answer with exactly one of the options listed.
Do not add explanation unless asked.
Return your answer as JSON: {"process": "..."}
"""

HALLIDAY_PROMPT = """\
Clause: {clause}

Classify the main clause's PROCESS TYPE:

  MATERIAL     — doing/happening: physical or abstract actions and events
                 (build, run, break, grow)
  MENTAL       — sensing: perception, cognition, affection
                 (see, know, want, feel, believe)
  RELATIONAL   — being/having: attribution or identification
                 (is, has, becomes, means, equals)
  VERBAL       — saying: communication processes
                 (say, tell, ask, report, argue)
  BEHAVIOURAL  — physiological/psychological behavior
                 (breathe, laugh, stare, cough)
  EXISTENTIAL  — existing: something exists or happens, no participant acts
                 (there is/exists, occurs, happens)

Return JSON only:
{{"process": "MATERIAL|MENTAL|RELATIONAL|VERBAL|BEHAVIOURAL|EXISTENTIAL"}}
"""

_HALLIDAY_VALID = {"MATERIAL", "MENTAL", "RELATIONAL", "VERBAL", "BEHAVIOURAL", "EXISTENTIAL"}


def _parse_halliday(text):
    try:
        data = json.loads(_strip_fences(text))
        v = str(data.get("process", "")).upper().strip()
        return {"process": v} if v in _HALLIDAY_VALID else None
    except Exception:
        return None


HALLIDAY_SCHEME = dict(
    name="halliday",
    label="Halliday SFG process type (transitivity)",
    system=HALLIDAY_SYSTEM,
    prompt_template=HALLIDAY_PROMPT,
    parser=_parse_halliday,
    axes=("process",),
    levels={"process": tuple(sorted(_HALLIDAY_VALID))},
)


# ── SRL-style argument structure (valence pattern) ───────────────────────────
# The conventional NLP answer: instead of Q1/Q2/Q3's invented 3-way split,
# decompose the clause by its predicate-argument structure the way
# PropBank/FrameNet semantic role labeling would (core roles: who did what
# to whom). A full SRL parser outputs free-form Arg0..Arg4 role spans, which
# don't reduce to a small discrete axis for this harness -- this is the
# coarse valence-pattern summary of that output (argument count + type),
# the standard first cut in argument-structure typology. It is an
# LLM-prompted PROXY for what an SRL parser's role count would give, not
# the parser itself -- stated once here rather than left implicit.

SRL_SYSTEM = """\
You are a linguist analyzing predicate-argument (semantic role) structure,
in the PropBank/FrameNet tradition.

Answer with exactly one of the options listed.
Do not add explanation unless asked.
Return your answer as JSON: {"valence": "..."}
"""

SRL_PROMPT = """\
Clause: {clause}

Classify the MAIN predicate's core argument (semantic role) pattern:

  COPULAR       — a linking/attributive predicate with no true verbal
                  argument structure (be, seem, become + complement):
                  "The sky is blue." / "She seems tired."
  INTRANSITIVE  — one core argument (an Agent, Theme, or Experiencer;
                  no direct object): "The mountain exists." / "She ran."
  TRANSITIVE    — two core arguments (e.g. Agent+Patient/Theme):
                  "She built a house." / "He knows the answer."
  DITRANSITIVE  — three or more core arguments (e.g. Agent+Recipient+
                  Theme): "She gave him a book." / "They told us the news."

Return JSON only: {{"valence": "COPULAR|INTRANSITIVE|TRANSITIVE|DITRANSITIVE"}}
"""

_SRL_VALID = {"COPULAR", "INTRANSITIVE", "TRANSITIVE", "DITRANSITIVE"}


def _parse_srl(text):
    try:
        data = json.loads(_strip_fences(text))
        v = str(data.get("valence", "")).upper().strip()
        return {"valence": v} if v in _SRL_VALID else None
    except Exception:
        return None


SRL_SCHEME = dict(
    name="srl",
    label="SRL-style argument-structure / valence pattern",
    system=SRL_SYSTEM,
    prompt_template=SRL_PROMPT,
    parser=_parse_srl,
    axes=("valence",),
    levels={"valence": tuple(sorted(_SRL_VALID))},
)


# ── PDTB-style discourse relation ────────────────────────────────────────────
# The conventional answer for how clauses RELATE to their context, as
# opposed to what a clause IS in isolation (Vendler/Halliday) or does
# internally (SRL). Real PDTB annotation labels a connective between two
# argument spans (Arg1/Arg2); applied here to single extracted clauses
# (which mostly lack a paired neighbor span in this corpus), this asks
# what relation the clause's OWN connective/structure signals to its
# context, or NONE if it carries no such signal -- a stated
# simplification of PDTB's actual annotation unit, not the full scheme.

DISCOURSE_SYSTEM = """\
You are a linguist analyzing discourse relations, in the Penn Discourse
TreeBank (PDTB) tradition -- the level-1 sense classes.

Answer with exactly one of the options listed.
Do not add explanation unless asked.
Return your answer as JSON: {"relation": "..."}
"""

DISCOURSE_PROMPT = """\
Clause: {clause}

Classify the discourse relation this clause signals to its surrounding
context, via its own connective or structure (if any):

  EXPANSION    — elaborates, restates, or adds to the surrounding context
                 (and, in addition, for example, that is)
  CONTINGENCY  — a cause/reason/condition/result relation
                 (because, so, if, therefore, as a result)
  COMPARISON   — a contrast or concession relation
                 (but, however, although, on the other hand)
  TEMPORAL     — a sequencing or synchrony relation
                 (then, before, while, after, meanwhile)
  NONE         — no discourse connective or relation signal; the clause
                 reads as self-contained

Return JSON only:
{{"relation": "EXPANSION|CONTINGENCY|COMPARISON|TEMPORAL|NONE"}}
"""

_DISCOURSE_VALID = {"EXPANSION", "CONTINGENCY", "COMPARISON", "TEMPORAL", "NONE"}


def _parse_discourse(text):
    try:
        data = json.loads(_strip_fences(text))
        v = str(data.get("relation", "")).upper().strip()
        return {"relation": v} if v in _DISCOURSE_VALID else None
    except Exception:
        return None


DISCOURSE_SCHEME = dict(
    name="discourse",
    label="PDTB-style discourse relation (level-1 sense)",
    system=DISCOURSE_SYSTEM,
    prompt_template=DISCOURSE_PROMPT,
    parser=_parse_discourse,
    axes=("relation",),
    levels={"relation": tuple(sorted(_DISCOURSE_VALID))},
)


SCHEMES = {s["name"]: s for s in
           (EO_SCHEME, VENDLER_SCHEME, HALLIDAY_SCHEME, SRL_SCHEME, DISCOURSE_SCHEME)}
