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
