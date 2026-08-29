# Control experiment — alternative trichotomies

## Question
Does the embedding-space geometry that the EO framework reports survive when
we substitute three other jointly-exhaustive trichotomies for q1/q2/q3? If
geometry collapses on the alternatives, EO is the load-bearing structure. If
geometry is preserved, EO's strongest claim — that *its specific* axes pick
out structure embedded models put there — weakens.

## Method
- **Corpus:** the English clauses from `run_2026-03-15_122636/classified.jsonl`
  that received EO consensus, n = 213.
- **Embedder:** ConceptNet Numberbatch English 19.08, 300d. Sentence vector =
  mean of in-vocabulary lowercase word vectors, L2-normalized. 1205 / 1264
  English tokens in the subset are covered.
  - **Caveat (load-bearing).** The published EO geometry uses
    `text-embedding-3-large` (3072d). That endpoint and every HF / S3 model
    mirror are firewalled in this sandbox. Numberbatch is the strongest
    embedder reachable. Numbers below are not directly comparable to the
    published `coord_geometry.json` / `subspace_geometry.json`; the EO axes
    are recomputed in numberbatch space as the in-sandbox baseline.
- **Alternative trichotomies (deterministic, English):**
  - **Tense** — past / present / future. NLTK POS tagger; future also
    triggered by `will / shall / 'll / gonna / going to`.
  - **Sentiment** — negative / neutral / positive. VADER compound score
    with cuts at ±0.05.
  - **Speech-act** — assertive / directive / expressive. Clause-final
    punctuation + imperative-verb regex + expressive lexical cues.
  - **Caveat.** These labels are not LLM consensus; they are noisier than
    the EO labels, which biases the comparison *against* alt axes. Any
    finding of "alt matches EO geometry" is therefore conservative.
- **Geometry probes** (same suite the existing pipeline reports):
  - `centroid_zscore` — observed mean cosine(x_i, centroid(class(x_i))) vs
    label permutation null (n_perm = 2000).
  - `fisher_ratio` — Trace(S_b) / Trace(S_w).
  - `lda_explained_variance` — top-(k-1) ratios from
    `LinearDiscriminantAnalysis`.
  - `principal_angles` between all pairs of axis subspaces.
  - `random_3class_baseline` — uniformly random 3-class labels, 20 trials.
  - **Residualization** — project the embedding orthogonal to the union of
    all alt LDA subspaces, then re-test EO axes. Reverse direction too.

## Class distributions (n = 213)

| scheme | classes |
|---|---|
| EO q1 (mode) | RELATING 143, DIFFERENTIATING 37, GENERATING 33 |
| EO q2 (domain) | STRUCTURE 92, SIGNIFICANCE 92, EXISTENCE 29 |
| EO q3 (object) | ENTITY 172, PATTERN 32, CONDITION 9 |
| tense | past 100, present 91, future 22 |
| sentiment | positive 94, neutral 68, negative 51 |
| speech-act | assertive 191, expressive 16, directive 6 |

Heavy imbalance in EO q3 and speech-act. Inflates Fisher ratios for the
majority class and depresses LDA stability — reported as-is.

## Per-scheme structure

| scheme | fisher | centroid z | LDA total | silhouette |
|---|---:|---:|---:|---:|
| EO_q1_mode      | 0.0204 |  7.13 | 1.000 | -0.032 |
| EO_q2_domain    | 0.0367 | 18.06 | 1.000 |  0.016 |
| EO_q3_object    | 0.0204 |  6.87 | 1.000 | -0.048 |
| ALT_tense       | 0.0406 | **19.82** | 1.000 | -0.030 |
| ALT_sentiment   | 0.0189 |  6.14 | 1.000 | -0.002 |
| ALT_speech_act  | 0.0185 |  5.45 | 1.000 |  0.022 |

Random 3-class baseline: z = 0.018 ± 0.925 (max 1.42 over 20 trials), so the
detection threshold is ≈ 2. Every scheme clears it by a wide margin.

**Tense outscores every EO axis on both Fisher ratio and centroid z.**
Sentiment and speech-act sit in the EO_q1 / EO_q3 range. The six trichotomies
form one cluster; EO is not above the alt set.

## Pairwise principal angles (sorted by min angle, low = aligned)

| pair | min° | max° |
|---|---:|---:|
| EO_q1 vs EO_q2                    | 50.9 | 81.7 |
| EO_q2 vs sentiment                | 52.3 | 85.1 |
| EO_q1 vs tense                    | 58.7 | 67.9 |
| EO_q3 vs sentiment                | 59.4 | 77.4 |
| sentiment vs speech-act           | 61.9 | 78.7 |
| tense vs sentiment                | 64.8 | 80.3 |
| tense vs speech-act               | 65.4 | 83.7 |
| EO_q2 vs tense                    | 66.4 | 89.4 |
| EO_q3 vs tense                    | 66.5 | 73.3 |
| EO_q1 vs EO_q3                    | 67.1 | 70.4 |
| EO_q1 vs sentiment                | 67.4 | 85.3 |
| EO_q2 vs EO_q3                    | 71.1 | 79.7 |
| EO_q3 vs speech-act               | 72.2 | 88.3 |
| EO_q1 vs speech-act               | 77.0 | 82.4 |
| EO_q2 vs speech-act               | 78.2 | 87.0 |

Two EO–EO pairs (q1/q2 at 50.9°, q1/q3 at 67.1°) are *more* aligned than
several EO–alt pairs. The "axis subspaces are mutually distinguishable" claim
applies to alt axes at least as much as to EO axes.

## Residualization

Project each clause vector orthogonal to the LDA subspace of the *other*
family (≤6 dims removed from a 300-dim space) and re-test.

| scheme | raw z | after remove alt | after remove EO |
|---|---:|---:|---:|
| EO_q1_mode      |  7.13 |  7.10 |  6.59 |
| EO_q2_domain    | 18.06 | 17.97 | 17.24 |
| EO_q3_object    |  6.87 |  6.80 |  6.16 |
| ALT_tense       | 19.82 | 18.34 | 19.75 |
| ALT_sentiment   |  6.14 |  5.10 |  6.11 |
| ALT_speech_act  |  5.45 |  4.61 |  5.45 |

Both directions: residualizing against one family barely dents the other.
The two label sets occupy near-orthogonal information channels, each
carrying its own structure.

## Reading

In numberbatch-en the alt trichotomies show z-scores comparable to or
larger than every EO axis, the principal-angle structure is no more
distinctive among EO axes than among alt axes, and residualization shows
EO and alt are mutually non-redundant rather than alt being a relabel of EO.

The conservative reading is: in this embedder, **EO is one of several
trichotomies that produce a clean 3-class geometry**, not the unique one.
The framework's strongest claim — that EO axes specifically pick out
structure already in the embedder — weakens. A weaker, defensible claim
remains: EO axes carry signal that is independent of these three
linguistic axes, since EO z-scores survive removing the alt subspaces.

The strong reading — "geometry collapses on alt axes" — is *not*
supported. Observed alt z-scores would have to be near the random
baseline (~2) for that, and they are between 5.4 and 19.8.

## Limits to honor in the writeup

1. The embedder is wrong. `text-embedding-3-large` may sharpen EO axes
   relative to alt axes; we cannot rule out that the result is an artifact
   of moving to a 300d static embedder. The honest move is to repeat this
   in `text-embedding-3-large` once API access is restored and report
   either confirmation or contradiction.
2. n = 213 (English consensus). The full 19,764-clause / 41-language
   geometry should be re-run with the same alt labels under a multilingual
   embedder. Heuristic alt labels do not generalize cleanly to 41
   languages without per-language POS / sentiment models.
3. Heuristic alt labels are noisier than LLM consensus; this *under*-states
   alt z-scores. Cleaner LLM-derived alt labels would only strengthen the
   "geometry preserved" reading.
4. q3 and speech-act have a dominant majority class (>80%). This boosts
   apparent within-class similarity for the majority and is part of why
   their z-scores are still positive.

## Files

- `control_experiment.py` — full pipeline, deterministic, ~12 s end-to-end.
- `control_results/results.json` — all numbers above.
- `control_results/labels.jsonl` — per-clause labels for all six schemes.
- `control_results/embeddings.npz` — the 213×300 numberbatch sentence vectors.
