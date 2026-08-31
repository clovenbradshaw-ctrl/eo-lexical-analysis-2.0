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


SCHEMES = {s["name"]: s for s in (EO_SCHEME, VENDLER_SCHEME, HALLIDAY_SCHEME)}
