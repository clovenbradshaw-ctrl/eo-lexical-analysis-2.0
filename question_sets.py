"""question_sets.py — Registry of trichotomous classification question sets.

The EO 3x3x3 claim is that the three EO questions produce a 3x3x3 lattice
with monotonic distance-by-axis-difference structure in embedding space.
The falsifiability test in falsify_3x3x3.py controls for partition
*geometry* (random, surface, PCA, optimized) but never varies the
*questions themselves*.

This module exposes alternative "exhaustive trichotomy" question sets that
swap in the same place EO sits: same LLMs, same corpus, same embeddings,
same 3x3x3 evaluation. If a sham or rival set reproduces EO's signature,
the signature is generic to LLM-applied trichotomy; if only EO produces
it, EO's specificity claim survives.

Families:
  eo          - the actual EO questions (verbatim from app2.py)
  sham        - plausibly exhaustive but theoretically unmotivated
  rival       - theoretically motivated competitor ontologies
  adversarial - degenerate by construction (should fail; canary check)

A QuestionSet is a self-contained record: prompt text + value maps + the
canonical axis-value lists used by falsify_3x3x3.py. All sets share the
JSON output contract {"q1": "...", "q2": "...", "q3": "..."}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class QuestionSet:
    name: str
    family: str  # "eo" | "sham" | "rival" | "adversarial"
    system_prompt: str
    user_prompt_template: str  # must contain "{clause}"
    value_maps: Dict[str, Dict[str, str]]  # {"q1": {raw -> canonical}, ...}
    axis_value_lists: Dict[str, List[str]]  # {"q1": [3 canonicals], ...}


REGISTRY: Dict[str, QuestionSet] = {}


def register(qs: QuestionSet) -> QuestionSet:
    if qs.name in REGISTRY:
        raise ValueError(f"duplicate registry name: {qs.name}")
    REGISTRY[qs.name] = qs
    return qs


def get(name: str) -> QuestionSet:
    if name not in REGISTRY:
        raise KeyError(f"unknown question set: {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def names() -> List[str]:
    return sorted(REGISTRY)


def validate_all() -> None:
    """Assert every registered set is well-formed."""
    for name, qs in REGISTRY.items():
        assert "{clause}" in qs.user_prompt_template, f"{name}: prompt missing {{clause}}"
        for axis in ("q1", "q2", "q3"):
            assert axis in qs.axis_value_lists, f"{name}: missing axis_value_lists[{axis}]"
            assert axis in qs.value_maps, f"{name}: missing value_maps[{axis}]"
            canonicals = qs.axis_value_lists[axis]
            assert len(canonicals) == 3, f"{name}.{axis}: need 3 canonicals, got {len(canonicals)}"
            assert len(set(canonicals)) == 3, f"{name}.{axis}: canonicals not unique"
            mapped = set(qs.value_maps[axis].values())
            assert mapped <= set(canonicals), (
                f"{name}.{axis}: value_map produces {mapped - set(canonicals)} "
                f"outside canonicals {canonicals}"
            )
            assert set(canonicals) <= mapped, (
                f"{name}.{axis}: canonicals {set(canonicals) - mapped} unreachable from value_map"
            )


# ============================================================================
# eo - verbatim copy of CLASSIFICATION_SYSTEM / CLASSIFICATION_PROMPT and
# Q1_MAP/Q2_MAP/Q3_MAP from app2.py (the baseline; do not edit).
# ============================================================================

register(QuestionSet(
    name="eo",
    family="eo",
    system_prompt=(
        "You are a linguist analyzing clauses. For each clause you will answer three\n"
        "short questions about the kind of transformation the clause describes.\n\n"
        "Answer each question with exactly one of the options listed.\n"
        "Do not add explanation unless asked.\n"
        'Return your answer as JSON: {"q1": "...", "q2": "...", "q3": "..."}\n'
    ),
    user_prompt_template=(
        "Clause: {clause}\n\n"
        "Answer these three questions about the transformation this clause describes:\n\n"
        "Q1 - How is the transformation structured?\n"
        "  SEPARATING   - the clause is primarily about dividing, distinguishing,\n"
        "                 analyzing, or drawing something apart\n"
        "  CONNECTING   - the clause is primarily about linking, bridging, relating,\n"
        "                 or holding things together\n"
        "  PRODUCING    - the clause is primarily about making, generating, creating,\n"
        "                 or causing something to happen\n\n"
        "Q2 - What level of reality is being transformed?\n"
        "  EXISTENCE    - whether something is: presence, absence, coming into being,\n"
        "                 disappearing\n"
        "  ORGANIZATION - how things are arranged: structure, boundaries, relations,\n"
        "                 composition\n"
        "  MEANING      - what something signifies: interpretation, value, perspective,\n"
        "                 what something registers as\n\n"
        "Q3 - What kind of thing is being acted on?\n"
        "  BACKGROUND   - an ambient condition, an environment, a field or substrate,\n"
        "                 the context something happens within\n"
        "  PARTICULAR   - a specific individual thing: this named object, this event,\n"
        "                 this person\n"
        "  PATTERN      - a recurring regularity: a rule, a type, a schema, something\n"
        "                 that holds across many instances\n\n"
        'Return JSON only: {{"q1": "SEPARATING|CONNECTING|PRODUCING",\n'
        '                   "q2": "EXISTENCE|ORGANIZATION|MEANING",\n'
        '                   "q3": "BACKGROUND|PARTICULAR|PATTERN"}}\n'
    ),
    value_maps={
        "q1": {"SEPARATING": "DIFFERENTIATING", "CONNECTING": "RELATING", "PRODUCING": "GENERATING"},
        "q2": {"EXISTENCE": "EXISTENCE", "ORGANIZATION": "STRUCTURE", "MEANING": "SIGNIFICANCE"},
        "q3": {"BACKGROUND": "CONDITION", "PARTICULAR": "ENTITY", "PATTERN": "PATTERN"},
    },
    axis_value_lists={
        "q1": ["DIFFERENTIATING", "RELATING", "GENERATING"],
        "q2": ["EXISTENCE", "STRUCTURE", "SIGNIFICANCE"],
        "q3": ["CONDITION", "ENTITY", "PATTERN"],
    },
))


# ============================================================================
# SHAM TRICHOTOMIES - plausibly exhaustive, theoretically unmotivated.
# If any of these reproduce EO's signature, the signature is generic to
# "any LLM-applied trichotomy on this corpus."
# ============================================================================

_SHAM_SYSTEM = (
    "You are a linguist analyzing clauses. For each clause you will answer three\n"
    "short categorical questions about its surface and grammatical features.\n\n"
    "Answer each question with exactly one of the options listed.\n"
    "Do not add explanation unless asked.\n"
    'Return your answer as JSON: {"q1": "...", "q2": "...", "q3": "..."}\n'
)

register(QuestionSet(
    name="tense",
    family="sham",
    system_prompt=_SHAM_SYSTEM,
    user_prompt_template=(
        "Clause: {clause}\n\n"
        "Answer three questions about the clause's temporal grammar:\n\n"
        "Q1 - Temporal locus of the main predicate\n"
        "  PAST    - completed before the moment of utterance\n"
        "  PRESENT - holding at or around the moment of utterance\n"
        "  FUTURE  - projected after the moment of utterance\n"
        "  (For tenseless or generic clauses, choose PRESENT.)\n\n"
        "Q2 - Aspect\n"
        "  PERFECTIVE  - bounded, completed, viewed as a whole\n"
        "  PROGRESSIVE - in progress, ongoing, viewed from inside\n"
        "  HABITUAL    - recurring, characteristic, generic\n\n"
        "Q3 - Modality\n"
        "  ACTUAL         - asserted as the case\n"
        "  POTENTIAL      - possible, permitted, intended, or required\n"
        "  COUNTERFACTUAL - hypothetical, conditional, or negated relative to fact\n\n"
        'Return JSON only: {{"q1": "PAST|PRESENT|FUTURE",\n'
        '                   "q2": "PERFECTIVE|PROGRESSIVE|HABITUAL",\n'
        '                   "q3": "ACTUAL|POTENTIAL|COUNTERFACTUAL"}}\n'
    ),
    value_maps={
        "q1": {"PAST": "PAST", "PRESENT": "PRESENT", "FUTURE": "FUTURE"},
        "q2": {"PERFECTIVE": "PERFECTIVE", "PROGRESSIVE": "PROGRESSIVE", "HABITUAL": "HABITUAL"},
        "q3": {"ACTUAL": "ACTUAL", "POTENTIAL": "POTENTIAL", "COUNTERFACTUAL": "COUNTERFACTUAL"},
    },
    axis_value_lists={
        "q1": ["PAST", "PRESENT", "FUTURE"],
        "q2": ["PERFECTIVE", "PROGRESSIVE", "HABITUAL"],
        "q3": ["ACTUAL", "POTENTIAL", "COUNTERFACTUAL"],
    },
))

register(QuestionSet(
    name="register",
    family="sham",
    system_prompt=_SHAM_SYSTEM,
    user_prompt_template=(
        "Clause: {clause}\n\n"
        "Answer three questions about the clause's stylistic register:\n\n"
        "Q1 - Formality\n"
        "  FORMAL   - academic, legal, or technical phrasing\n"
        "  NEUTRAL  - ordinary expository or descriptive prose\n"
        "  INFORMAL - colloquial, conversational, or slang phrasing\n\n"
        "Q2 - Affect\n"
        "  POSITIVE - approving, hopeful, or celebratory tone\n"
        "  NEUTRAL  - factual, detached, or descriptive tone\n"
        "  NEGATIVE - disapproving, critical, or pessimistic tone\n\n"
        "Q3 - Specificity\n"
        "  CONCRETE - names particular people, objects, places, or events\n"
        "  MIXED    - blends concrete reference with general claims\n"
        "  ABSTRACT - states general principles, categories, or relations only\n\n"
        'Return JSON only: {{"q1": "FORMAL|NEUTRAL|INFORMAL",\n'
        '                   "q2": "POSITIVE|NEUTRAL|NEGATIVE",\n'
        '                   "q3": "CONCRETE|MIXED|ABSTRACT"}}\n'
    ),
    value_maps={
        "q1": {"FORMAL": "FORMAL", "NEUTRAL": "NEUTRAL_F", "INFORMAL": "INFORMAL"},
        "q2": {"POSITIVE": "POSITIVE", "NEUTRAL": "NEUTRAL_A", "NEGATIVE": "NEGATIVE"},
        "q3": {"CONCRETE": "CONCRETE", "MIXED": "MIXED", "ABSTRACT": "ABSTRACT"},
    },
    axis_value_lists={
        "q1": ["FORMAL", "NEUTRAL_F", "INFORMAL"],
        "q2": ["POSITIVE", "NEUTRAL_A", "NEGATIVE"],
        "q3": ["CONCRETE", "MIXED", "ABSTRACT"],
    },
))

register(QuestionSet(
    name="agency",
    family="sham",
    system_prompt=_SHAM_SYSTEM,
    user_prompt_template=(
        "Clause: {clause}\n\n"
        "Answer three questions about who/what acts and who/what is acted on:\n\n"
        "Q1 - Agent type (the entity initiating the action or state)\n"
        "  HUMAN      - a person, group, institution, or human collective\n"
        "  NONHUMAN   - an animal, natural force, object, or abstract entity\n"
        "  IMPERSONAL - no identifiable agent (existential, weather, passive)\n\n"
        "Q2 - Patient type (what is acted on or undergoes the state)\n"
        "  HUMAN      - a person, group, or human collective is affected\n"
        "  NONHUMAN   - an object, place, or abstract entity is affected\n"
        "  NONE       - no patient (intransitive or stative without affected object)\n\n"
        "Q3 - Transitivity\n"
        "  TRANSITIVE   - action transfers to a direct object\n"
        "  INTRANSITIVE - action stays with the subject; no direct object\n"
        "  STATIVE      - describes a state, attribute, or relation rather than an action\n\n"
        'Return JSON only: {{"q1": "HUMAN|NONHUMAN|IMPERSONAL",\n'
        '                   "q2": "HUMAN|NONHUMAN|NONE",\n'
        '                   "q3": "TRANSITIVE|INTRANSITIVE|STATIVE"}}\n'
    ),
    value_maps={
        "q1": {"HUMAN": "AG_HUMAN", "NONHUMAN": "AG_NONHUMAN", "IMPERSONAL": "AG_IMPERSONAL"},
        "q2": {"HUMAN": "PT_HUMAN", "NONHUMAN": "PT_NONHUMAN", "NONE": "PT_NONE"},
        "q3": {"TRANSITIVE": "TRANSITIVE", "INTRANSITIVE": "INTRANSITIVE", "STATIVE": "STATIVE"},
    },
    axis_value_lists={
        "q1": ["AG_HUMAN", "AG_NONHUMAN", "AG_IMPERSONAL"],
        "q2": ["PT_HUMAN", "PT_NONHUMAN", "PT_NONE"],
        "q3": ["TRANSITIVE", "INTRANSITIVE", "STATIVE"],
    },
))


# ============================================================================
# RIVAL ONTOLOGIES - theoretically motivated competitor frameworks.
# A "win" here that ALSO has low Hungarian-ARI with EO would falsify EO's
# specificity claim; a "win" with high ARI would mean the rival is EO in a
# different vocabulary (which would arguably strengthen EO).
# ============================================================================

_RIVAL_SYSTEM = (
    "You are a philosophically trained reader analyzing clauses. For each clause\n"
    "you will answer three questions drawn from a classical ontological framework.\n\n"
    "Answer each question with exactly one of the options listed.\n"
    "Do not add explanation unless asked.\n"
    'Return your answer as JSON: {"q1": "...", "q2": "...", "q3": "..."}\n'
)

register(QuestionSet(
    name="aristotle",
    family="rival",
    system_prompt=_RIVAL_SYSTEM,
    user_prompt_template=(
        "Clause: {clause}\n\n"
        "Answer three questions in the Aristotelian frame. Choose the option that\n"
        "best fits even if the clause does not foreground these distinctions.\n\n"
        "Q1 - Which kind of cause is most prominent?\n"
        "  MATERIAL  - the stuff or substrate out of which something is\n"
        "  FORMAL    - the pattern, shape, definition, or essence\n"
        "  EFFICIENT - the source of motion or change that brings something about\n\n"
        "Q2 - Mode of being\n"
        "  POTENTIA      - capacity, possibility, what could become\n"
        "  ENERGEIA      - activity, what is currently being exercised\n"
        "  ENTELECHEIA   - completion, what has reached its end or fulfilment\n\n"
        "Q3 - Level of substance\n"
        "  PRIME_MATTER - undifferentiated stuff, background substrate\n"
        "  PARTICULAR   - a primary substance, this individual thing\n"
        "  UNIVERSAL    - a kind, species, or general predicate\n\n"
        'Return JSON only: {{"q1": "MATERIAL|FORMAL|EFFICIENT",\n'
        '                   "q2": "POTENTIA|ENERGEIA|ENTELECHEIA",\n'
        '                   "q3": "PRIME_MATTER|PARTICULAR|UNIVERSAL"}}\n'
    ),
    value_maps={
        "q1": {"MATERIAL": "MATERIAL", "FORMAL": "FORMAL", "EFFICIENT": "EFFICIENT"},
        "q2": {"POTENTIA": "POTENTIA", "ENERGEIA": "ENERGEIA", "ENTELECHEIA": "ENTELECHEIA"},
        "q3": {"PRIME_MATTER": "PRIME_MATTER", "PARTICULAR": "AR_PARTICULAR", "UNIVERSAL": "UNIVERSAL"},
    },
    axis_value_lists={
        "q1": ["MATERIAL", "FORMAL", "EFFICIENT"],
        "q2": ["POTENTIA", "ENERGEIA", "ENTELECHEIA"],
        "q3": ["PRIME_MATTER", "AR_PARTICULAR", "UNIVERSAL"],
    },
))

register(QuestionSet(
    name="peirce",
    family="rival",
    system_prompt=_RIVAL_SYSTEM,
    user_prompt_template=(
        "Clause: {clause}\n\n"
        "Answer three questions in Peirce's phenomenological categories. Choose\n"
        "the option that best fits even if the clause does not foreground them.\n\n"
        "Q1 - Category of the subject\n"
        "  FIRSTNESS  - quality, feeling, monadic; what something is in itself\n"
        "  SECONDNESS - reaction, brute existence, dyadic; relation to another\n"
        "  THIRDNESS  - mediation, law, habit; triadic relation through a rule\n\n"
        "Q2 - Category of the predicate or relation\n"
        "  FIRSTNESS  - sheer quality or possibility\n"
        "  SECONDNESS - actual reaction, fact, contrast\n"
        "  THIRDNESS  - rule, generality, lawful pattern\n\n"
        "Q3 - Category of the object or outcome\n"
        "  FIRSTNESS  - a felt quality or possibility\n"
        "  SECONDNESS - an existing particular fact\n"
        "  THIRDNESS  - a generalization, type, or habit\n\n"
        'Return JSON only: {{"q1": "FIRSTNESS|SECONDNESS|THIRDNESS",\n'
        '                   "q2": "FIRSTNESS|SECONDNESS|THIRDNESS",\n'
        '                   "q3": "FIRSTNESS|SECONDNESS|THIRDNESS"}}\n'
    ),
    value_maps={
        "q1": {"FIRSTNESS": "FIRST_S", "SECONDNESS": "SECOND_S", "THIRDNESS": "THIRD_S"},
        "q2": {"FIRSTNESS": "FIRST_P", "SECONDNESS": "SECOND_P", "THIRDNESS": "THIRD_P"},
        "q3": {"FIRSTNESS": "FIRST_O", "SECONDNESS": "SECOND_O", "THIRDNESS": "THIRD_O"},
    },
    axis_value_lists={
        "q1": ["FIRST_S", "SECOND_S", "THIRD_S"],
        "q2": ["FIRST_P", "SECOND_P", "THIRD_P"],
        "q3": ["FIRST_O", "SECOND_O", "THIRD_O"],
    },
))

register(QuestionSet(
    name="hegel",
    family="rival",
    system_prompt=_RIVAL_SYSTEM,
    user_prompt_template=(
        "Clause: {clause}\n\n"
        "Answer three questions in a Hegelian dialectical frame. For each\n"
        "question, choose the moment the clause most clearly enacts.\n\n"
        "Q1 - Dialectical moment of the premise (what is posited)\n"
        "  THESIS     - a position is asserted, affirmed, or stated as given\n"
        "  ANTITHESIS - a counter, negation, or opposition is foregrounded\n"
        "  SYNTHESIS  - a reconciliation or higher unity is proposed\n\n"
        "Q2 - Dialectical moment of the tension (the contradiction at stake)\n"
        "  THESIS     - immediate identity, no contradiction yet surfaced\n"
        "  ANTITHESIS - contradiction or division is the central tension\n"
        "  SYNTHESIS  - the contradiction is being mediated or resolved\n\n"
        "Q3 - Dialectical moment of the resolution (the outcome)\n"
        "  THESIS     - the original position holds\n"
        "  ANTITHESIS - the opposite is established\n"
        "  SYNTHESIS  - a new, mediated determination emerges\n\n"
        'Return JSON only: {{"q1": "THESIS|ANTITHESIS|SYNTHESIS",\n'
        '                   "q2": "THESIS|ANTITHESIS|SYNTHESIS",\n'
        '                   "q3": "THESIS|ANTITHESIS|SYNTHESIS"}}\n'
    ),
    value_maps={
        "q1": {"THESIS": "THESIS_P", "ANTITHESIS": "ANTI_P", "SYNTHESIS": "SYNTH_P"},
        "q2": {"THESIS": "THESIS_T", "ANTITHESIS": "ANTI_T", "SYNTHESIS": "SYNTH_T"},
        "q3": {"THESIS": "THESIS_R", "ANTITHESIS": "ANTI_R", "SYNTHESIS": "SYNTH_R"},
    },
    axis_value_lists={
        "q1": ["THESIS_P", "ANTI_P", "SYNTH_P"],
        "q2": ["THESIS_T", "ANTI_T", "SYNTH_T"],
        "q3": ["THESIS_R", "ANTI_R", "SYNTH_R"],
    },
))


# ============================================================================
# ADVERSARIAL - three "axes" that are really sub-flavours of one EO axis,
# so by construction they are non-orthogonal. Used as a canary: if even
# this passes the falsification rule, the rule is too lax.
# ============================================================================

register(QuestionSet(
    name="eo_q1_split",
    family="adversarial",
    system_prompt=_SHAM_SYSTEM,
    user_prompt_template=(
        "Clause: {clause}\n\n"
        "The clause describes some kind of differentiation, separation, or\n"
        "boundary-making. Answer three questions about HOW that separation\n"
        "is structured. (If the clause is not about separation at all, pick\n"
        "the option that comes closest.)\n\n"
        "Q1 - Cleanness of the separation\n"
        "  CLEAN   - sharp, decisive, no remainder\n"
        "  PARTIAL - gradient or incomplete; some overlap remains\n"
        "  FUSED   - the supposed separation is barely discernible\n\n"
        "Q2 - Directionality of the separation\n"
        "  INWARD   - pulling apart from within a single thing\n"
        "  LATERAL  - separating side-by-side things from each other\n"
        "  OUTWARD  - distinguishing something from its surrounding context\n\n"
        "Q3 - Scope of the separation\n"
        "  LOCAL    - within a single small region or moment\n"
        "  REGIONAL - across a bounded subdomain\n"
        "  GLOBAL   - across the whole field under discussion\n\n"
        'Return JSON only: {{"q1": "CLEAN|PARTIAL|FUSED",\n'
        '                   "q2": "INWARD|LATERAL|OUTWARD",\n'
        '                   "q3": "LOCAL|REGIONAL|GLOBAL"}}\n'
    ),
    value_maps={
        "q1": {"CLEAN": "CLEAN", "PARTIAL": "PARTIAL", "FUSED": "FUSED"},
        "q2": {"INWARD": "INWARD", "LATERAL": "LATERAL", "OUTWARD": "OUTWARD"},
        "q3": {"LOCAL": "LOCAL", "REGIONAL": "REGIONAL", "GLOBAL": "GLOBAL"},
    },
    axis_value_lists={
        "q1": ["CLEAN", "PARTIAL", "FUSED"],
        "q2": ["INWARD", "LATERAL", "OUTWARD"],
        "q3": ["LOCAL", "REGIONAL", "GLOBAL"],
    },
))


validate_all()
