# Why these three? — an experiment to falsify or support the three axes

Status: preliminary result committed (`factorization_test.py`), decisive
experiment specified but not yet run.

---

## 1. The question, as two rival hypotheses

**H_EO** — Q1 mode, Q2 domain, Q3 object are three real, independent
dimensions of transformation. Any clause has a value on each; the same three
questions apply everywhere.

**H_ARB** — *any* three sufficiently orthogonal directions in embedding space,
quantile-binned into three levels, would produce the results the report
reports. The three axes are one arbitrary coordinate frame among many.

These are not distinguished by anything currently in `analysis_report.txt`.
That is the finding this document starts from.

## 2. What the headline results actually establish: nothing

RESULT 2 (proportionality) reports that mean cosine distance rises with the
number of differing axes — 0.8044 → 0.8694, bootstrap p = 0.0000 — and reads
this as confirming "the three axes form a real coordinate structure."

It does not. Distance growing with bin-difference count is a property of
quantile-binning, not of EO. Split any direction into tertiles: two points in
different tertiles are, on average, farther apart *along that direction*, and
squared distance decomposes over directions, so the sum inherits the ordering.
The sign of the effect is fixed before any data is collected.

`factorization_test.py --arbitrary` shows this concretely: three arbitrary
orthogonal directions over data with **no planted structure at all** produce

```
  0 axes differ   0.6945
  1 axes differ   0.8567
  2 axes differ   1.0000
  3 axes differ   1.1447      monotone spread +0.4502
```

Perfect monotonicity, p → 0, pairwise ARI ≈ 0 ("independent axes"), on noise.
The EO corpus's own spread is +0.0650. The absolute scales are not directly
comparable — anisotropy and the share of variance the directions capture both
shift them — so this is an existence proof, not a head-to-head. But it is
sufficient: monotonicity cannot discriminate H_EO from H_ARB, and neither can
per-axis z-scores against a shuffled-label null, which any labeling correlated
with position in the space will beat.

The same applies to RESULT 3. Low pairwise ARI is what independent quantile
splits give you by construction.

## 3. What does discriminate: is the cube a *product*?

H_EO says more than "three directions exist." It says the 27 cells are the
**product** of three axes — that the shift from DIFFERENTIATING to GENERATING
is the same shift whichever domain and object it is paired with. Formally the
cell centroids should be additive:

```
    c(i,j,k)  ~  mu + a_i + b_j + c_k
```

with small interaction residual. Arbitrary orthogonal directions do not confer
this on a *given* set of 27 groups: additivity is a fact about how the groups
are **arranged** on the grid, and almost all arrangements fail it.

That gives the sharp null. Hold the 27 archetype centroids fixed and re-assign
them to grid positions. Every assignment fits the same model with the same
parameter count, so the only thing that varies is the arrangement.

### Preliminary result

`factorization_test.py`, over the 27 archetype centroids
(`paraphrase-multilingual-MiniLM-L12-v2`, 384-d, built from
`run_2026-03-15_122636` exemplars):

| | additive R² |
|---|---|
| **EO's own assignment** | **0.7022** |
| 20,000 random re-assignments | mean 0.2312, sd 0.0328, **max 0.3856** |
| greedy ascent, 60 random restarts | 0.7022 — never beats EO |
| greedy ascent started at EO | 0.7022 — EO is a local maximum |

EO sits at z = **+14.4** against the re-assignment null; **0 of 20,000** random
assignments beat it, and a search over assignments cannot find a better one.
Variance is split near-evenly across the three axes (Q1 23.2%, Q2 22.3%,
Q3 24.7%, interaction 29.8%), and the main-effect subspaces are near-orthogonal
(principal angles 66°–86°).

So: *why these three* has a real answer available. Not "because three
orthogonal directions produce monotonicity" — any would — but "because among
all the ways to lay these 27 groups on a 3×3×3 grid, EO's is the arrangement
that makes them additive, and search cannot improve on it."

### The confound this result does not rule out

The 27 groups were themselves defined by EO-labeled classification. The
embedder never saw EO, so the *geometry* is independent — but the *grouping* is
not. If the classifying models answer Q1 largely off surface vocabulary, then
cells sharing a Q1 value share words, and additivity would follow from lexical
overlap rather than from a semantic dimension.

This is the one live threat, and it is what the experiment below is for.
Everything else about the product test is sound.

---

## 4. The experiment

Three arms. Arm A is the confound control and must run first; Arms B and C are
the positive tests and are independent of each other.

### Arm A — is the signal lexical? (control, blocking)

1. Fit a bag-of-words / bag-of-lemmas classifier (logistic regression, no
   embeddings) to predict Q1, Q2, Q3 from the raw clause. Report macro-F1
   against the 33% chance floor and against inter-model kappa (0.585 / 0.534 /
   0.457) as the ceiling of what the labels themselves support.
2. Extract the top-*k* discriminative tokens per axis value.
3. **Ablate them** — mask those tokens from the clause, re-embed, rebuild the
   27 centroids, re-run the product test.

**Falsifies H_EO** if additive R² collapses toward the re-assignment null after
ablation: the cube was a vocabulary artifact.
**Supports H_EO** if R² survives ablation substantially above the null: the
arrangement is carried by something other than the trigger words.

### Arm B — unsupervised recovery, and *product vs tree*

This is your "split by whatever three things create the most distinction, then
three within those." It is worth running in a specific form, because the naive
form and the sharp form give different answers.

1. **Discover, blind.** With no EO labels, find the 3-way partition of the
   embedded clauses that maximises distinction (KMeans k=3, and the top
   discriminant direction tertiled). Recurse to depth 3 → 27 leaves.
2. **Product vs tree — the load-bearing step.** At depth 2, compare the split
   found *inside branch 1* with the one found *inside branch 2* and *branch 3*.
   - If they are the **same** split (same direction up to sign, high ARI
     across branches), the corpus's own structure is a **product** — three
     questions that apply everywhere. That is EO's shape.
   - If they **differ** by branch, the structure is a **tree** — a hierarchy of
     distinctions where the second question depends on the answer to the first.
     A tree is not three dimensions, and this would falsify the coordinate
     claim regardless of how good the 27 leaves look.
3. **Map.** Hungarian-align the 27 discovered leaves against EO's 27 cells;
   report flat ARI and per-axis ARI after best axis pairing.
4. **Rank EO among rivals.** Run the discovered axes through the *same* product
   test in §3. If a blind partition achieves additive R² ≥ EO's, then EO is one
   of several equally good frames and "why these three" has no geometric answer
   — the answer would have to be askability (Arm C) instead.

`falsify_3x3x3.py` already has the machinery for steps 3–4 (`pca-tertile`,
`pca-kmeans`, `optimized`, Hungarian-aligned ARI). It needs the recursive
splitter and the branch-consistency comparison in step 2, which is the part
that actually decides product vs tree. Note that it currently cannot run: the
shipped `embeddings.npz` is an 83-byte Drive pointer.

### Arm C — transportability (the property arbitrary directions cannot have)

An arbitrary direction is a fact about one corpus and one embedder. A real
dimension should survive transfer.

1. **Across embedders.** Rebuild the 27 centroids under a second, unrelated
   model (`multilingual-e5-small` is already in this repo; add one non-sentence
   -transformer model). Re-run the product test.
2. **Across corpora.** Hold out an entire register — the arXiv quantum-physics
   slice is the obvious one — build centroids on the rest, and test additivity
   on the held-out register.
3. **Across languages.** The report claims 39 languages but this run has
   `n_languages = 1`. Re-run per-language and test whether the *same axis
   directions* transfer, not merely whether each language shows a z-score.
4. **Askability.** Report inter-model kappa alongside. A blind direction has no
   question that recovers it; that asymmetry is the strongest non-geometric
   evidence available, and the current kappas (0.46–0.59, moderate) are honest
   but not strong.

**Falsifies H_EO** if additivity is corpus- or embedder-specific.
**Supports H_EO** if the axis directions align across embedders and registers.

---

## 5. Pre-committed thresholds

Locked before Arms A–C are run.

| # | Prediction | Falsified if |
|---|---|---|
| 1 | Additive R² survives lexical ablation | R² drops below null max (≈0.39) |
| 2 | No blind partition beats EO on additive R² | any blind partition ≥ 0.7022 |
| 3 | Depth-2 splits agree across branches (product, not tree) | cross-branch ARI < 0.10 |
| 4 | Additivity transfers to a second embedder | R² < 0.50 under the second model |
| 5 | Additivity transfers to a held-out register | R² < 0.50 on the held-out slice |
| 6 | Axis independence | q1×q2 ARI ≥ 0.10 — **already at risk**: 0.0959 full corpus, 0.2711 excluding sparse cells, Spearman r = −0.277 (p ≈ 0) |

Prediction 6 is reported here as already contested by the repo's own data. The
helix directional test (asymmetry −0.0666 bits, permutation p = 0.985) also
runs against the report's stated expectation. Neither should be quietly
re-framed after the fact.

## 6. On using eoreader's own null machinery — is it circular?

Not circular, with one condition.

The NUL apparatus (`nul/index.js`'s `ground`/`difference`/`pattern`, the
redealt null arms, the licensing gate) constructs its null by destroying
structure while preserving marginals. That construction is purely statistical:
it never consults EO's vocabulary, and it would refuse an unlicensed
perturbation regardless of what the labels mean. Using it here is no more
circular than using a t-test to evaluate a theory that also happens to
mention means.

The circularity lives somewhere else: in the **comparison class**. A
shuffled-label null is a *floor*, and beating a floor is nearly free — §2 is
exactly that failure. The apparatus stops being tautological only when the
panel includes *ceilings* generated without EO: the best partition search can
find, the best a blind recursive split can find, the best a lexical classifier
can find. `falsify_3x3x3.py` already takes this posture (`optimized` is named
"absolute ceiling"). The product test above keeps it: its null is not shuffled
labels but the same 27 groups arranged every other way.

Rule of thumb for this repo: **a null that EO beats tells you almost nothing;
a ceiling that EO matches or beats is the whole result.**

## 7. Reproducing

```bash
pip install numpy
python factorization_test.py                 # product test + re-assignment null
python factorization_test.py --arbitrary     # the monotonicity artifact
```

Runs on data already committed here; needs no Drive fetch. Arms A–C do need the
real `embeddings.npz`.

---

# Results — 2026-08-24

Run on the **real** per-clause embeddings, fetched from the Drive pointers this
repo ships (`embeddings.npz` in each run dir is an 83-byte URL; the actual file
is 3072-d, 193 MB for the multilingual run):

```
curl -sSL -o embeddings.npz \
  "https://drive.usercontent.google.com/download?id=<ID>&export=download&confirm=t"
```

Primary corpus: `run_2026-03-15_122636`, 9,221 consensus-labelled clauses,
41 languages. All scores are on **held-out clauses**, averaged over 3 seeds.

## The criterion, operationalised

The standard set for this work: *the proof is its ability to make the past
understandable and the future more predictable.* Three numbers, in increasing
order of how hard they are to fake:

| | asks | who wins it cheaply |
|---|---|---|
| **PAST** | how much held-out clause variance the 27 groups explain | anything that clusters |
| **FUTURE** | predict a held-out clause's position from its three labels alone | anything with a product structure |
| **UNSEEN** | leave one whole cell out; predict it from the other 26 | **only a genuine product** |
| **ratio** | FUTURE / PAST | — |

## Result 1 — the blind recursive split

Split the embedded clauses into the three most meaningful distinctions
(KMeans k=3), split each of those three ways, and again: 27 leaves, no EO
labels anywhere. Fitted on train, applied to held-out test clauses, then
**re-scored in a second embedding space** (`paraphrase-multilingual-MiniLM-L12-v2`,
384-d) that the blind schemes were never fitted in.

| scheme | PAST | FUTURE | **UNSEEN** | ratio | | PAST_B | FUTURE_B | **UNSEEN_B** | ratio_B |
|---|---|---|---|---|---|---|---|---|---|
| **eo** | +0.010 | +0.009 | **+0.277** | 0.83 | | +0.025 | +0.022 | **+0.509** | **0.87** |
| tree (recursive KMeans) | **+0.134** | +0.039 | **−0.010** | 0.29 | | +0.072 | +0.023 | **−0.089** | 0.31 |
| pca-tertile (blind product) | +0.065 | +0.049 | +0.498 | 0.76 | | +0.061 | +0.049 | +0.535 | 0.79 |
| random | −0.004 | −0.001 | −0.290 | — | | −0.004 | −0.001 | −0.292 | — |

**The recursive tree is not three dimensions.** It wins compression outright
and cannot predict an unseen cell in either space (−0.010, −0.089; across seeds
−0.017 / +0.009 / −0.010, i.e. indistinguishable from zero — the additive model
fitted on 26 of its cells predicts the 27th no better than the training mean).
The direct diagnostic agrees: the depth-2 split found inside branch 1 is
near-orthogonal to the ones inside branches 2 and 3 (principal angles 81–90°).
The second question depends on the answer to the first. That is a hierarchy of
distinctions, not a coordinate system, and the distinction is now measured
rather than argued.

**EO is the most product-like scheme measured** (ratio 0.83 → 0.87 on transfer)
and its structure *improves* under transfer while the tree's halves.

## Result 2 — the language confound, and a correction

An earlier reading of the table above concluded that blind splits beat EO on
every criterion. That reading was wrong, and the check that overturns it is
one line:

| scheme | ARI vs language | **AMI vs language** |
|---|---|---|
| eo | +0.003 | **0.026** |
| tree | +0.464 | **0.804** |
| pca-tertile | +0.171 | **0.404** |

The recursive tree is 80% language identification. Its compression win was
rediscovering what language the clause was written in — and language alone
explains 0.193 of the variance in this space, more than any 27-group scheme
achieves. EO is essentially independent of language (AMI 0.026), which is
what an axis claiming to be about transformations rather than about surface
form should be.

So **variance-explained is the wrong yardstick here.** The dominant structure
in a multilingual text embedding is language and topic; a scheme that cuts
across both will always look weak on it. EO's ~1–2% is not obviously a defect,
and the blind schemes' advantage is substantially an artifact.

## Result 3 — the archetype cube (centroid scale)

Product test on the 27 committed archetype centroids (384-d, top-100 exemplars
per cell, heavily denoised relative to single clauses):

| | additive R² | leave-one-cell-out R² |
|---|---|---|
| **EO's own assignment** | **0.7022** | **+0.497** |
| 20,000 random re-assignments of the same 27 centroids | mean 0.231, max 0.386 | mean −0.301, max −0.094 |
| greedy search, 200 restarts | 0.7022 — never beats EO | — |

Every random arrangement predicts an unseen cell *worse than the training
mean*. EO's arrangement is the one that makes these 27 groups additive, it is
a local maximum, and search cannot improve on it (z = +14.4 / +14.5).

This is a real result about the **arrangement of EO's own cells**. It is a
different question from whether a rival scheme with different cells could do
better, and it should not be quoted as if it answered that.

## Result 4 — why three? Not earned by this corpus

Ablation on the same 9,221 clauses (3 seeds). Held-out scoring, so differing
parameter counts are handled by construction:

| configuration | cells | par | FUTURE | UNSEEN | ΔFUTURE |
|---|---|---|---|---|---|
| **EO 3×3×3 (full)** | 27 | 7 | +0.0084 | +0.2748 | — |
| without Q1 mode | 9 | 5 | +0.0081 | +0.4089 | **−0.0003** |
| without Q2 domain | 9 | 5 | +0.0069 | +0.3844 | −0.0015 |
| without Q3 object | 9 | 5 | +0.0065 | +0.1757 | −0.0019 |
| Q1: DIFFERENTIATING+GENERATING merged | 18 | 6 | +0.0085 | +0.3254 | **+0.0002** |
| Q3: CONDITION+PATTERN merged | 18 | 6 | +0.0088 | +0.4054 | **+0.0004** |
| Q2: EXISTENCE+STRUCTURE merged | 18 | 6 | +0.0084 | +0.3243 | +0.0000 |
| Q2: STRUCTURE+SIGNIFICANCE merged | 18 | 6 | +0.0051 | +0.2054 | −0.0033 |
| Q3: ENTITY+PATTERN merged | 18 | 6 | +0.0046 | +0.1647 | −0.0038 |

Three of the nine level-collapses are free or better. Merging DIFFERENTIATING
with GENERATING — the two poles of mode — costs nothing, so Q1 behaves like a
binary (*RELATING vs not*), not a trichotomy. Dropping Q1 entirely costs 3.5%
of an already-tiny FUTURE. Only two distinctions are clearly earned:
**STRUCTURE vs SIGNIFICANCE** and **ENTITY vs PATTERN**.

Blind sweep over (axes × levels) for comparison — the shape the space itself
prefers:

| | 2 levels | 3 levels | 4 levels |
|---|---|---|---|
| 2 axes | +0.026 / −0.177 | +0.035 / +0.203 | +0.041 / +0.300 |
| 3 axes | +0.037 / +0.277 | **+0.048 / +0.429** | +0.053 / +0.420 |
| 4 axes | +0.044 / +0.374 | +0.052 / +0.347 | +0.056 / +0.231 |
| 5 axes | +0.050 / **+0.440** | +0.059 / +0.227 | — |

(FUTURE / UNSEEN.) 3×3 is a local sweet spot for UNSEEN but not the maximum.
**On corpus evidence alone, 3×3×3 is not earned.**

## What these tests can and cannot establish

Every test above asks whether EO is recoverable from embedding geometry. In
Chomsky's terms (*Reflections on Language*, "On Cognitive Capacity") that is
LT(embedding-model, text) — and it is not the object EO claims. His four
stages are: set the cognitive domain D; determine how the organism O
characterises data in D **pretheoretically**; determine the cognitive
structure attained; then determine LT(O,D) relating the two.

EO's three questions are a proposal about **stage 2** — how an observer carves
a transformation before any theory. Two consequences follow, and they cut in
opposite directions:

1. **The domain may be mis-set.** D for EO is *transformations*; this corpus
   supplies *clauses*, many of which are not transformations at all. Chomsky's
   own note applies: a failure at stage 4 may reveal that "our original
   delimitation of D was faulty." The weak geometric signal is confounded with
   a domain-specification error that nothing here has separated out.

2. **Low corpus-recoverability is not by itself a refutation.** If the three
   questions are observer-side structure, underdetermination by the data is
   predicted, not surprising.

Point 2 is also the standard way a theory becomes unfalsifiable, so it must
come with its own discipline. It does: it relocates the test from the corpus
to the observer, where it is sharp.

## The next test — poverty of the stimulus, bounded

If the 27 is observer-side structure, observers must converge on it **beyond
what the stimulus supports**:

- `A` = agreement rate between two independent observers
- `S` = accuracy of the best function of the surface at predicting one
  observer's assignment
- `E` = accuracy of the best function of the embedding at the same

`A >> S, E` means the agreement comes from shared structure in the observers,
and the low geometric R² is *explained* rather than excused. `A ≈ S` means the
questions elicit vocabulary cues and the 27 has no observer-side reality
either. Current κ (0.585 / 0.534 / 0.457) is the raw material; `S` and `E`
have never been computed.

**Stated limit, up front:** Claude and GPT-4 are not independent observers, so
a high `A` is confounded by shared pretraining. The bounded version measures
the surface component and bounds the rest. Establishing the observer component
needs human annotators across languages — a study, not a script.

## Status of the pre-committed thresholds

| # | prediction | status |
|---|---|---|
| 1 | additive R² survives lexical ablation | not yet run |
| 2 | no blind partition beats EO on additive R² | **failed on raw clauses** (pca-tertile +0.498 vs EO +0.277 on UNSEEN) — but see Result 2: partly a language artifact |
| 3 | depth-2 splits agree across branches | **failed** — 81–90°, the blind structure is a tree |
| 4 | additivity transfers to a second embedder | **held** — EO improves (0.277 → 0.509) |
| 5 | additivity transfers to a held-out register | not yet run |
| 6 | axis independence | **at risk** — unchanged from prior entry |

Prediction 3 failing is informative rather than damaging: it falsifies the
*blind* structure as a coordinate system, and EO passes the same test.
Prediction 2 failing is the live problem.

## Reproducing

```bash
pip install numpy scikit-learn scipy
python factorization_test.py                     # centroid product + predictive test
python factorization_test.py --arbitrary         # the monotonicity artifact
python recursive_split.py --emb <real.npz> --emb-b <second-space.npz>
python dimensionality.py  --emb <real.npz>
```

---

# The domain error — 2026-08-24, second pass

## What the study actually asked

`app2.py`'s `CLASSIFICATION_PROMPT` opens:

> Answer these three questions about **the transformation this clause describes**.

There is no refusal option. Every clause is forced into one of the 27 cells, and
nothing anywhere asks whether the clause describes a transformation at all.
Clause selection (`extract_clauses_from_conllu`) was purely syntactic — at least
one VERB token, 8–30 tokens, declarative, deduplicated — which admits pure
statives, properties, attitude reports and directives.

In Chomsky's terms this is a **stage-1 failure**: the cognitive domain D was
never delimited, it was assumed. Every geometric result in this document was
computed over a corpus whose membership in EO's own domain was never checked.

## How large the error is

200 English clauses hand-judged against a criterion fixed in advance and applied
blind to which agreement stratum a clause came from — *does the main predication
assert a change, as opposed to a static state, property, classification,
attitude report, or directive?* (`data/transformation-judgments.json`; single
annotator, disclosed, not a validated gold standard.)

| | n | share |
|---|---|---|
| transformation | 65 | **32%** |
| borderline | 38 | 19% |
| not a transformation | 97 | **48%** |

So between a third and a half of the corpus is in EO's stated domain. The rest
was assigned cells anyway.

## Agreement is not a domain signal — the cheap filter fails

The tempting free filter: if a clause is not a transformation, the three
questions have no determinate answer, so two labellers should converge at
chance. Measured against the hand judgments, it does not hold.

| stratum (axes on which the two labellers agree) | n | mean transformation score |
|---|---|---|
| 0–1 of 3 | 20 | 0.425 |
| 2 of 3 | 20 | 0.500 |
| 3 of 3 | 20 | 0.400 |

Spearman r = **−0.015**, p = 0.91; Mann-Whitney 3/3 vs 0–1, p = 0.54. Flat.

This also **corrects an intermediate reading**. EO's geometry does rise with
labeller agreement (PAST −0.0002 → +0.0021 → +0.0068 at equal n, language AMI
flat at ~0.02 throughout), and that looked like domain membership. It is not.
It is ordinary label noise attenuating a signal.

## A filter is buildable, and the corpus still cannot use it

162 clean judgments train a logistic regression on the run's own embeddings:
**5-fold AUC 0.907, accuracy 0.833**. Applied corpus-wide it calls 28%
in-domain, against a hand-judged English rate of 32% strict — the cross-lingual
transfer is at least plausible.

| arm (equal n = 2,710, 10 seeds) | PAST | FUTURE | UNSEEN |
|---|---|---|---|
| RANDOM subsample (control) | −0.0001 ±0.0021 | +0.0005 ±0.0019 | −0.0574 ±0.0371 |
| in-domain | −0.0030 ±0.0017 | −0.0091 ±0.0016 | −0.1909 ±0.0093 |
| out-of-domain | +0.0011 ±0.0011 | +0.0024 ±0.0009 | −0.0173 ±0.0208 |

Read without the control this says filtering to transformations *hurts* EO. The
control says otherwise: **at n = 2,710 a matched random subsample also scores
nothing.** Only 2,710 consensus clauses survive the filter, and that is below
the sample size at which anything is measurable here. The in-domain arm is also
worse-conditioned (cell evenness 0.664 vs 0.771, one cell holding a single
clause), which depresses the additive fit independently of domain.

**The corpus cannot answer the domain question.** Filtering has to happen
*before* labelling, not after — a re-run of the labelling pass, not a post-hoc
repair.

## The power curve, which is a finding on its own

EO on random subsamples of the consensus set, 8 seeds:

| n | PAST | FUTURE | UNSEEN |
|---|---|---|---|
| 1,000 | −0.0236 ±0.0064 | −0.0151 ±0.0054 | −0.1786 ±0.0301 |
| 2,000 | −0.0063 ±0.0021 | −0.0036 ±0.0023 | −0.1122 ±0.0412 |
| 2,710 | −0.0001 ±0.0021 | +0.0003 ±0.0019 | −0.0609 ±0.0363 |
| 4,000 | +0.0043 ±0.0009 | +0.0043 ±0.0006 | +0.0269 ±0.0208 |
| 6,000 | +0.0076 ±0.0006 | +0.0067 ±0.0005 | +0.1500 ±0.0172 |
| **9,221 (all)** | **+0.0100 ±0.0003** | **+0.0082 ±0.0003** | **+0.2656 ±0.0104** |

Two things follow, and the second matters more than the domain question.

**The estimator crosses zero at n ≈ 2,700.** Below that, cell centroids are
noisier than the grand mean, so predicting a clause from its own cell is worse
than predicting it from nothing. Negative values at small n are overfitting of
the centroids, not absence of structure — any subset comparison at that scale is
uninterpretable.

**EO's scores are still climbing at the full corpus and have not plateaued.**
The "~1% of variance" figure quoted throughout this document is a **floor set by
corpus size**, not an estimate of the effect. Whether it saturates at 2%, 5% or
higher is unknown and is a straightforward thing to find out: label more
clauses. That is now the single cheapest way to move any of these numbers.

## Revised standing

| claim | status |
|---|---|
| monotonicity / per-axis z-scores are evidence | **refuted** — arbitrary directions reproduce them on noise |
| EO is a product, not a hierarchy | **holds** — survives transfer; the blind tree fails UNSEEN in both spaces |
| EO's arrangement of its own 27 cells is optimal | **holds** — z = +14.4, unbeaten by 20k rearrangements or by search |
| 3×3×3 specifically is earned | **not supported** — three of nine level-collapses are free or better |
| EO beats a blind product | **not supported** — pca-tertile wins, though partly on language |
| EO explains ~1% of variance | **superseded** — that is a corpus-size floor, not an estimate |
| the corpus is in EO's domain | **refuted** — 32% strict, 52% inclusive |
| a domain filter fixes it post hoc | **refuted** — too few in-domain clauses to measure anything |

## Reproducing

```bash
python domain_filter.py --emb <real.npz>          # filter, apply, compare with control
python domain_filter.py --emb <real.npz> --power  # the power curve
```

---

# The coordinates, and the instrument that finally worked — 2026-08-25

## The method, first, because it rescues everything else

Four subset analyses in this investigation died the same death: apparent
structure that turned out to be sample size. Per-tier additive R² against tier
n came back **Spearman +0.917, p = 0.0005** — the tier "gradient" was tier size
and nothing else. Cells in this corpus hold between **20 and 2,296** clauses.

`unique_exemplars.py` fixes it by selecting equally from every cell, taking the
clauses most distinctively their own: a leave-one-out margin (cosine to the
cell's centroid computed without the clause, minus the best cosine to any other
cell). Only **31.5%** of clauses score positive — most sit closer to some other
cell than their own.

The trap is that selecting the clauses which best fit the 27-cell structure and
then measuring that structure is selecting on the dependent variable. So the
null undergoes the **identical selection** — labels shuffled, margins recomputed
against the shuffled cells, top-k taken per shuffled cell.

| k/cell | n | EO additive R² | matched null | z |
|---|---|---|---|---|
| 10 | 270 | 0.3730 | 0.2349 ±0.0120 | +11.5 |
| 15 | 405 | 0.4000 | 0.2366 ±0.0087 | +18.8 |
| **20** | **540** | **0.4271** | **0.2337 ±0.0115** | **+16.8** |

At n = 540 — where a *random* subsample scores nothing at all, the estimator
crossing zero at n ≈ 2,700 — balanced most-unique selection recovers a 17-sigma
signal. **This is the only subset method in the investigation that survives its
own control**, and every subset question that died is re-runnable through it.

## The tier question, answered once the confound is gone

Balanced selection makes every tier exactly 9k clauses by construction.

| tier | additive R² | matched null | z |
|---|---|---|---|
| Q2 = EXISTENCE `[NUL SIG INS]` | 0.6214 | 0.4990 ±0.0230 | +5.3 |
| Q2 = STRUCTURE `[SEG CON SYN]` | 0.6207 | 0.5070 ±0.0288 | +4.0 |
| Q2 = SIGNIFICANCE `[DEF EVA REC]` | 0.6483 | 0.5030 ±0.0265 | +5.5 |

Spreads: Q1 0.039, Q2 0.028, Q3 0.021 — every one inside a single null sd. Q2
was 7× the others when confounded; now it is the middle one. **The tiers are
indistinguishable, and all nine are equally real** (z = +3.0 to +7.3).

The arithmetic / geometric / transcendental split *is* verifiably in the code —
`gamma ** (t - cell.t)` for presence, `-Math.log2(p)` for information, bare
`>= 2` order comparisons for the log tier. It is simply not in the lexical
geometry, and was never a claim about clause embeddings.

## The canonical coordinates fail, on the better instrument

`the-axis-triad-and-its-coordinates` gives every form a three-coordinate
address — Mode {0,1,2} arithmetic, Domain {−1,+1,√2} geometric, Object
{2,√2,2^√2} transcendental. `coordinate_geometry.py` verifies the
reconstruction against all five of the article's own worked examples before
testing anything.

**Mantel, predicted vs observed 27×27 distances: r = +0.0401, null +0.0015 ±
0.0868, z = +0.44, beaten by 327 of 500 permutations.** No relationship.

| axis | predicted | observed | max triangle angle |
|---|---|---|---|
| Mode ARITHMETIC | 1 : 1.000 : 2.000 | 1 : 0.928 : 0.875 | 67.3° |
| Domain GEOMETRIC | 1 : 0.207 : 1.207 | 1 : 1.068 : 0.919 | 67.5° |
| Object TRANSCENDENTAL | 1 : 2.135 : 1.135 | 1 : 1.062 : 0.931 | 66.6° |

The angle column is the deeper finding. **Scalar coordinates place three levels
on a line** — a degenerate triangle, 180°. Every axis measures ~67°: a
near-equilateral simplex, which is what categorical variables with no ordering
give you. *No scalar assignment of any character can describe this*, so the
step-ratio question is moot before the numbers are chosen. This confirms the
wiki's own honest line — *"Not met: the coordinate-geometry step-ratio
predictions"* — with a stronger instrument than the original test had.

Composition law fitted per level-step (LOO, top-6 PC subspace): arithmetic
+0.363, transcendental +0.369, geometric +0.289. Geometric is decisively worst;
arithmetic and transcendental are indistinguishable.

## Reachability — the claim the coordinates actually make

The coordinates were never metric claims. Gelfond–Schneider encodes
*reachability*: 2^√2 is unreachable from the algebraics by finite operation, and
the wiki reads that structurally — *"you cannot arrive at a regularity by any
finite sequence of operations."* `reachability.py` tests it without using the
coordinate values at all.

**Extrapolation.** Fit the cube on two levels, reach the third linearly.

| axis | reach top level | matched null | z | interpolate to middle | z |
|---|---|---|---|---|---|
| Mode (0 crises) | −1.0541 | −0.7011 | **−3.0** | +0.0133 | **+5.9** |
| Domain (1 crisis) | −0.8804 | −0.7133 | −1.4 | −0.0359 | +4.3 |
| Object (2 crises) | −1.2321 | −0.8166 | **−3.1** | +0.0350 | **+6.2** |

The crisis-count ordering **does not hold** — observed Domain > Mode > Object
against a predicted Mode > Domain > Object. Object is worst as predicted, but
Mode, predicted to cross *zero* crises, extrapolates worse than Domain.

What does hold is the general shape: **every extrapolation lands below its null,
every interpolation above it.** The structure supports filling in between
observations and not reaching past them.

**Hazard.** For a candidate regularity ("every clause containing word W lands in
cell C") surviving N observations, does observation N+1 refute it?

| N | survived | refuted | hazard | iid null | gap |
|---|---|---|---|---|---|
| 1 | 5,575 | 4,265 | 0.765 | 0.861 ±0.004 | −0.096 |
| 2 | 1,310 | 787 | 0.601 | 0.788 ±0.021 | −0.187 |
| 3 | 290 | 149 | 0.514 | 0.769 ±0.054 | −0.256 |
| 4 | 91 | 32 | 0.352 | 0.752 ±0.138 | −0.400 |
| 5 | 41 | 10 | 0.244 | 0.730 ±0.267 | −0.487 |

Hazard sits below the iid null at every N and the gap widens monotonically:
**inductive support genuinely accumulates.** But it never approaches zero — a
regularity holding five straight times still breaks on the sixth one time in
four — and only 41 candidates survive that far.

**The honest limit, which decides how much either test can carry.**
Gelfond–Schneider is a claim about *deductive closure*; a corpus supplies
*inductive support*. No finite sample can show that no finite sample suffices,
nor that one does. The mapping between the algebraic fact and the epistemic
claim is an analogy, and counting cannot reach it. What is measurable is whether
the system *behaves* as if regularities are reachable — and two unrelated tests
say the same thing: fill in between, never reach past.

## Standing, revised again

| claim | status |
|---|---|
| monotonicity / per-axis z-scores are evidence | **refuted** — arbitrary directions reproduce them on noise |
| EO is a product, not a hierarchy | **holds** — survives transfer; the blind tree fails UNSEEN in both spaces |
| EO's arrangement of its own cells is optimal | **holds** — z = +14.4, unbeaten by 20k rearrangements or by search |
| the product structure is real at all | **holds, strengthened** — z = +16.8 on balanced most-unique selection |
| 3×3×3 specifically is earned | **not supported** — three of nine level-collapses are free or better |
| axis levels are ordered (layers, a ladder) | **refuted** — every axis is a ~67° simplex, indistinguishable from a shuffled null |
| the canonical coordinate addresses | **refuted as metric** — Mantel z = +0.44; collinearity precondition fails outright |
| the tiers carry different maths | **verified in code, absent from the geometry** |
| the crisis-count reachability ordering | **not supported** — Mode and Domain invert |
| reachability in general | **consistent, unproven** — interpolation yes, extrapolation no, hazard falls but never to zero |

## Reproducing

```bash
python unique_exemplars.py     --emb <real.npz>          # the k sweep
python unique_exemplars.py     --emb <real.npz> --tiers  # per-tier, balanced
python coordinate_geometry.py  --emb <real.npz>          # address model
python reachability.py         --emb <real.npz>          # extrapolation + hazard
```

Real embeddings are Drive pointers in each run dir; fetch with
`curl -sSL -o embeddings.npz "https://drive.usercontent.google.com/download?id=<ID>&export=download&confirm=t"`.
