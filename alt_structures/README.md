# alt_structures — discovering and scoring alternatives to Q1×Q2×Q3

`docs/WHY-THESE-THREE.md` already runs a serious falsification program
against EO's own 3×3×3 (Q1 mode × Q2 domain × Q3 object). This folder does
two more things in that spirit:

1. **Discovers alternative structures** — both blind/mechanical rivals
   already implemented elsewhere in this repo (KMeans tree, PCA-tertile,
   surface stats) and new **conventional linguistic** schemes (Vendler
   aspect, Halliday SFG transitivity) — and scores all of them through the
   *same* held-out PAST/FUTURE/UNSEEN rubric `recursive_split.py` /
   `factorization_test.py` / `dimensionality.py` already established, so
   the comparison is apples-to-apples.
2. **Redoes the classification with two more, independently-trained local
   models** (Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3 — both run
   entirely on CPU) as a 3rd and 4th rater alongside the original Claude
   and GPT-4o-mini labels already in `classified.jsonl`, on the *exact
   same clause spans*, so inter-rater agreement no longer rests on two
   raters (Claude, GPT-4) that share frontier-lab pretraining —
   `docs/WHY-THESE-THREE.md`'s own "poverty of stimulus" section flags
   this as unresolved.

## Layout

```
alt_structures/
├── harness.py              shared PAST/FUTURE/UNSEEN/shuffled-null scorer,
│                            generalized to any (n_axes, levels-per-axis)
├── sampling.py              the balanced 27-cell clause sample, shared by
│                            classification and scoring so labels line up
├── local_embeddings.py      self-contained clause embeddings (see below)
├── candidates/
│   ├── blind_geometric.py   Tree / PcaProduct (imported from
│   │                        recursive_split.py) + Surface (reproduces
│   │                        falsify_3x3x3.py's feature defs), wrapped to
│   │                        harness.py's contract
│   ├── verbnet_lexical.py   top-5-most-frequent-VerbNet-class heuristic
│   │                        (superseded in practice by act_prior_lexical.py
│   │                        below -- kept for the comparison)
│   ├── act_prior_lexical.py live_priors' own ActPrior@1 lexicon
│   │                        (cross-repo dependency, by design -- see
│   │                        "live_priors, eoreader7, and the-fold" below)
│   └── phasepost_live.py    the REAL production classifier (eoreader7's
│                            extractRelations -> phasepost.js, live_priors'
│                            real ActPrior@1) via bridge/ -- see "The live
│                            pipeline, tested for real" below
├── bridge/
│   └── phasepost_live_bridge.mjs  the one crossing: imports eoreader7 +
│                            live_priors modules unmodified, emits JSONL
├── local_models/
│   ├── registry.py          the two GGUF models
│   ├── download_models.sh   idempotent fetch (weights/*.gguf is gitignored)
│   ├── schemes.py           prompt/parser for eo (imported verbatim from
│                            app2.py) / vendler / halliday / srl / discourse
│   ├── classify_local.py    runs one (model, scheme) over a balanced
│                            sample; output matches classified.jsonl's
│                            per-rater shape
│   └── run_model_suite.sh   runs all six schemes through one model
├── run_discovery.py         scores every candidate on a balanced sample,
│                            prints/writes the comparison report + 4-way kappa
├── run_full_corpus.py       the no-LLM candidates, on the FULL 7,808
│                            -clause corpus (fast: embeddings only)
├── run_actprior_english_only.py  act_prior_lexical.py, correctly scoped
│                            to the 213 English clauses it can cover
├── run_phasepost_live.py    the REAL live pipeline, scored at n=1,354/3,595
│                            -- see "The live pipeline, tested for real" below
└── results/                 jsonl label files + *_report.json files
```

## Why a fresh, self-contained embedding

The repo's own per-run `embeddings.npz` files are Google Drive pointers
(`docs/WHY-THESE-THREE.md`: *"the shipped embeddings.npz is an 83-byte
Drive pointer"*), not always resolvable offline. `local_embeddings.py`
computes its own with `paraphrase-multilingual-MiniLM-L12-v2` — the same
model `WHY-THESE-THREE.md` already uses for the 27 archetype centroids and
as the "space B" transfer target — so this suite runs end-to-end from just
this repo plus `pip install sentence-transformers`, with no private link
dependency.

## Why these two local models

Both are 7B-class instruct models, Q4_K_M GGUF, run via `llama-cpp-python`
on CPU only:

| | lab | why |
|---|---|---|
| Qwen2.5-7B-Instruct | Alibaba | different pretraining lineage from both Claude and GPT |
| Mistral-7B-Instruct-v0.3 | Mistral AI | different lineage again |

`AVAILABLE_CLASSIFIERS` in `app2.py` shows the original raters were
`claude-sonnet-4-6` and `gpt-4o-mini` — both efficient/small-tier models,
not the largest frontier checkpoints — which makes a well-chosen 7B local
model a more comparable rival than it might first sound, though it is
still not claimed to match either on nuanced judgment. The honest limit,
stated once here rather than left implicit: **four models, all
transformer LLMs post-trained on broadly overlapping web/instruction
data, are still not four independent observers** in the strong sense
`WHY-THESE-THREE.md`'s "poverty of stimulus" section wants (that needs
human annotators across languages). What this does buy: a same-question,
same-span, same-scoring-rubric comparison across two more model families
that share **no** pretraining lineage with Claude or GPT — which is
strictly more than the 2-rater status quo, and directly produces the `A`
(agreement) side of that section's `A vs S vs E` design.

## Running it

```bash
cd alt_structures/local_models
./download_models.sh                     # fetch weights (~8.9GB total)

# check /proc/cpuinfo for avx512 before a plain `pip install`. If present,
# build with it disabled from the start (see "A CPU-instability tax" below)
# rather than risk an illegal-instruction crash partway through a run:
CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX512=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON" \
  pip install --no-cache-dir llama-cpp-python

# classify a balanced sample (N/cell x 27 cells) through both models,
# all six schemes — this is the slow step, budget real CPU time (see below)
./run_model_suite.sh qwen    4 <per-cell>
./run_model_suite.sh mistral 4 <per-cell>

cd ..
pip install sentence-transformers scikit-learn scipy nltk
python run_discovery.py --per-cell <same per-cell as above>
```

**Timing, measured on this box (4 vCPU, 15GB RAM):** ~13-16s/clause/model
running one model at a time at full thread count with AVX512 (see below
if that's not available/stable on yours — budget ~2x that). Running both
models *concurrently* at half the threads each was **not** faster —
thread contention on 4 cores pushed per-clause time to ~40s, worse than
serial — so `run_model_suite.sh` is meant to be run sequentially per
model, not backgrounded in parallel. Budget roughly `81 clauses/scheme ×
3 schemes × 2 models × ~15s ≈ 2 hours` at `per-cell=3`; scale linearly
with per-cell and scheme count.

### A CPU-instability tax, found the hard way

Mid-run at `per_cell=10`, `llama-cpp-python`'s CPU backend (`libggml
-cpu.so.0`) crashed with an illegal-instruction trap — confirmed via
`dmesg` (`trap invalid opcode ... in libggml-cpu.so.0`). `/proc/cpuinfo`
on this box advertises full AVX512 (including VNNI), and a plain `pip
install llama-cpp-python` builds with `-march=native`, which trusts that
advertisement. On at least this cloud VM, that trust wasn't warranted —
a known class of issue where advertised CPUID flags aren't reliably
executable in some virtualized environments. Rebuilding with AVX512
disabled (command above) fixed it — verified stable across dozens of
subsequent calls — at a real cost: **roughly 2x slower per clause**
(AVX512 was genuinely accelerating the quantized matmuls). If your box's
CPU is stable with AVX512, skip the CMAKE_ARGS and get the faster
numbers above; if you hit the same crash, this is the fix.

## Sampling

`sampling.balanced_sample()` draws `per_cell` clauses from each of the 27
EO consensus cells rather than a random slice — `unique_exemplars.py`'s
own finding (`docs/WHY-THESE-THREE.md`, "The coordinates, and the
instrument that finally worked") is that a **random** sample only becomes
usable past roughly n≈2,700 clauses, while a small **balanced** one (a few
hundred, spread evenly) already gives a strong, clean signal (z=+16.8 at
n=540, k=20/cell). The same reasoning should transfer to any new label
scheme scored against embedding geometry, not just EO's own — so this
suite's pilot uses the same strategy at a smaller `per_cell` to fit CPU
classification into one session.

Stated limit: this takes the *first* `per_cell` matches per cell in file
order, not `unique_exemplars.py`'s leave-one-out-margin "most
representative" selection. A follow-up run should switch to that
selection for a tighter, less order-dependent sample.

## The new candidate schemes

**Vendler aspect** (`STATE / ACTIVITY / ACCOMPLISHMENT / ACHIEVEMENT`) —
the conventional linguistic answer to whether a clause is a transformation
at all. `WHY-THESE-THREE.md`'s "domain error" section found only 32-52% of
the corpus qualifies as a transformation under EO's own stated domain; a
stative/dynamic/telic split should surface that distinction directly
rather than needing a separate post-hoc filter.

**Halliday SFG transitivity** (`MATERIAL / MENTAL / RELATIONAL / VERBAL /
BEHAVIOURAL / EXISTENTIAL`) — the closest established precedent to Q1's
"how is the transformation structured" — an off-the-shelf, corpus
-validated process-type typology instead of an invented 3-way split.

**SRL-style argument structure / valence pattern** (`COPULAR /
INTRANSITIVE / TRANSITIVE / DITRANSITIVE`) — the conventional NLP answer:
decompose the clause by predicate-argument structure the way
PropBank/FrameNet semantic role labeling would, instead of Q1/Q2/Q3's
invented 3-way split. A real SRL parser outputs free-form Arg0..Arg4 role
spans, which don't reduce to a small discrete axis for this harness — this
is an LLM-prompted proxy for the coarse valence-pattern summary of that
output (argument count + type), not the parser itself.

**PDTB-style discourse relation** (`EXPANSION / CONTINGENCY / COMPARISON /
TEMPORAL / NONE`) — the conventional answer for how a clause relates to
its context, as opposed to what it is (Vendler/Halliday) or does
internally (SRL). Real PDTB annotates a connective between two argument
spans (Arg1/Arg2); applied here to single extracted clauses, this asks
what relation the clause's own connective/structure signals to its
context — a stated simplification of PDTB's actual annotation unit.

**VerbNet lexical class** (`candidates/verbnet_lexical.py`) — the one
non-LLM addition: NLTK POS-tags the clause, lemmatizes the main verb, and
looks up its VerbNet (Levin-style) class. VerbNet has no small textbook
inventory the way Vendler's four categories do (~270 leaf classes, ~50+
top-level ones), so there's no principled fixed axis to prompt an LLM
for — this instead buckets to the 5 most frequent top-level classes
observed when fit, plus OTHER, which is specific to whatever sample it's
fit on rather than a universal taxonomy. Main-verb selection is a
POS-tag heuristic (first non-auxiliary VB* token), not a dependency
parse, and polysemous verbs take NLTK's first-listed class rather than a
sense-disambiguated one — both stated simplifications.

**Not added: AMR parsing.** A real AMR parser's output is a full semantic
graph, which doesn't reduce to a small discrete axis without engineering
disproportionate to what it would buy here (unlike SRL's argument count or
PDTB's connective, there's no similarly natural coarse summary). Left out
rather than faked.

Five of the six new schemes are single-axis (4-6 levels), vs. EO's three
3-level axes, which is exactly why `harness.py` imports its FUTURE/UNSEEN
scorer from `dimensionality.py` rather than `recursive_split.py`'s
EO-specific 3×3×3 version — it already generalizes to arbitrary axis/level
counts (`dimensionality.py`'s own "blind sweep" needs the same thing).

## Reading the results table

Columns match `WHY-THESE-THREE.md`'s own tables: **PAST** (held-out
cell-mean R²), **FUTURE** (held-out additive-model R²), **UNSEEN**
(leave-one-cell-out R² — *"only a genuine product structure can do
this"*), **ratio** (FUTURE/PAST), **UNSEEN_z** (vs. a label-shuffle null).

That null is a **floor**, not a ceiling — section 2 and 6 of
`WHY-THESE-THREE.md` are explicit that beating a shuffled-label null is
"nearly free" and not by itself evidence of real structure. The sharper
re-assignment-of-the-same-centroids test in that document is specific to a
fixed cube and doesn't generalize to a 1-axis, 4-level scheme like Vendler
aspect, so it isn't reproduced here. What substitutes for a ceiling is
scoring every candidate — EO, the blind rivals, and the new
conventional-linguistics schemes — through the identical rubric and
comparing them directly, per that document's own rule of thumb: *"a null
that EO beats tells you almost nothing; a ceiling that EO matches or beats
is the whole result."*

## Results — pilot run, per_cell=3 (81 clauses)

```
candidate                            n  cells     PAST   FUTURE   UNSEEN   UNSEEN_z
eo-consensus                        81     27  -0.6297  -0.0445  -0.1834      4.99
qwen-eo                             79     12  -0.0646  -0.2048  -0.9273      1.16
qwen-vendler                        81      4  -0.0806  -0.0806     nan       nan
qwen-halliday                       70      5  -0.0671  -0.0671     nan       nan
qwen-srl                            79      3  -0.0058  -0.0058     nan       nan
qwen-discourse                      81      4  -0.0604  -0.0604     nan       nan
mistral-eo                          78      5  -0.0318  -0.0426     nan       nan
mistral-vendler                     80      2  -0.0278  -0.0288     nan       nan
mistral-halliday                    75      5  -0.0241  -0.0241     nan       nan
mistral-srl                         80      3  -0.1229  -0.1229     nan       nan
mistral-discourse                   81      4  -0.0126  -0.0126     nan       nan
tree(recursive kmeans)              81      -  insufficient data at this n (see below)
pca-tertile(blind product)          81     23  -0.1884  -0.0199  -0.1304      nan
surface(char/ttr/punct)             81     15  -0.1885  -0.1163     nan       nan
verbnet-lexical(top5+other)         81      6  -0.0965  -0.0965     nan       nan
```

4-way rater agreement (Cohen's kappa, flat 27-cell label):

```
              claude      gpt4      qwen   mistral
claude            --     1.000     0.079     0.067
gpt4           1.000        --     0.079     0.067
qwen           0.079     0.079        --     0.003
mistral        0.067     0.067     0.003        --
```

Full machine-readable version: `results/discovery_report.json`.

### Read this table carefully — three things are NOT what they look like

**claude/gpt4 κ=1.000 is a sampling artifact, not a finding.**
`balanced_sample()` draws only from clauses with a non-null `consensus`
field, and `consensus` is only ever set when claude and gpt4 *already
agreed* (see `app2.py`'s classification pipeline). Scoring agreement on a
sample that is agreement by construction is circular. The real,
non-circular claude/gpt4 kappa is the one already in this repo
(`docs/WHY-THESE-THREE.md`: 0.585/0.534/0.457 per axis, computed over the
full labeled corpus including disagreements) — don't cite the 1.000 above
as if it were comparable to that number.

**Most PAST/FUTURE values are noise, not signal.** `unique_exemplars.py`'s
own finding (cited in `docs/WHY-THESE-THREE.md`) is that this kind of
held-out structure score needs several hundred balanced clauses before it
separates from a null — n=270 was their smallest clean signal (z=+11.5).
This pilot ran at n=81 (`per_cell=3`) specifically to fit local-CPU
classification into one session; most of the negative PAST/FUTURE values
above are exactly the small-n noise floor that document describes, not
evidence against any scheme (EO's own PAST is -0.63 here, which nobody
should read as "EO has negative structure" — it's an n=81 artifact).

**The `tree` candidate isn't broken, the sample is too small for it.**
Recursive depth-3 KMeans needs enough points per branch to run
`KMeans(k=3)` at the leaves; with ~56 training clauses spread (at depth 2)
over up to 9 branches, some branches had a single point left. This needs
a larger `per_cell` to even attempt, unrelated to whether it's a good
scheme.

### What IS real at this n: local-model agreement

Kappa doesn't need as much data as the R²-based structure tests to be
informative, and this result doesn't depend on the consensus-sampling
artifact above (qwen/mistral never touch the consensus field):

- **qwen vs mistral: κ=0.003** — indistinguishable from chance. Two
  different 7B-class model families, given the identical prompt and the
  identical clause, essentially don't converge on the same Q1/Q2/Q3
  answer.
- **qwen/mistral vs claude/gpt4: κ=0.067–0.079** — "slight" on the
  standard Landis-Koch scale, well below the original pair's own
  moderate 0.46–0.59 per-axis kappa.

Read cautiously: this could mean the three-question rubric is genuinely
hard to elicit consistently (strengthening `WHY-THESE-THREE.md`'s
"poverty of stimulus" concern), or it could mean these particular 7B CPU
models are simply less capable at this specific judgment call than
`claude-sonnet-4-6`/`gpt-4o-mini` — this data can't separate those two
explanations, and shouldn't be over-read as settling either one. What it
does deliver, cleanly: a same-question, same-span comparison across two
more model families with no shared pretraining lineage with Claude or
GPT, which is strictly more than the 2-rater status quo had.

Parse reliability was solid across all six schemes × both models
(0–2.5% JSON-parse failure for srl/discourse, 0–13.6% for eo/vendler
/halliday; `qwen-halliday` was the overall worst at 11/81).

### SRL, discourse, and VerbNet: no baseline to compare against

Unlike the `eo` replication, SRL/discourse/VerbNet are new axes with no
prior claude/gpt4 labels to compute agreement against — there's no
non-circular kappa to report for them at any n. Their PAST/FUTURE/UNSEEN
rows sit at the same n=81 noise floor as everything else above (`cells`
2-6, mostly negative PAST/FUTURE, UNSEEN unscoreable). What this pilot
does establish for them: the classification pipeline runs end-to-end on
all three (parse failure 0-2.5% for the two LLM-based ones), and
VerbNet's lexicon lookup correctly discriminates real clauses (spot
-checked separately — "she gave him a book" and "he knows the answer"
land in different classes). Whether any of them structure the embedding
space better than EO or the other rivals is exactly the question a
properly-powered run (below) would answer; this one can't.

## Results — properly-powered eo run, per_cell=10 (270 clauses)

The `eo` scheme was rerun at `per_cell=10` (270 clauses, vs. the 81-clause
pilot above) specifically because it's the one with a prior baseline to
compare against. `vendler`/`halliday`/`srl`/`discourse` below are still
the OLD 81-clause files — `run_discovery.py` correctly reports their true
(smaller) `n`, so don't read them as upgraded; only `eo-consensus`,
`qwen-eo`, `mistral-eo`, and the candidates that don't depend on
pre-existing label files (`pca-tertile`, `surface`, `verbnet-lexical`; not
`tree`, which needed still more data) got the full n=270.

```
candidate                            n  cells     PAST   FUTURE   UNSEEN   UNSEEN_z
eo-consensus                       270     27  -0.0967  +0.0155  +0.0173     14.95
qwen-eo                            264     15  -0.0279  -0.0646  -0.5082      3.74
mistral-eo                         261     11  -0.0450  -0.1122  -1.4971      0.22
pca-tertile(blind product)         270     27  -0.0410  +0.0563  +0.1360       nan
surface(char/ttr/punct)            270     16  -0.0686  -0.0425     nan        nan
verbnet-lexical(top5+other)        270      6  -0.0128  -0.0128     nan        nan
(qwen/mistral-vendler/halliday/srl/discourse: unchanged, still n=70-81)
```

kappa (unchanged raters, larger sample):

```
              claude      gpt4      qwen   mistral
claude            --     1.000     0.098     0.064
qwen           0.098                          0.006
mistral        0.064               0.006
```

### This is where the story changes

**EO's own structure flips from noise to real signal.** At n=81:
FUTURE=-0.044, UNSEEN=-0.183, z=4.99. At n=270: FUTURE=**+0.016**,
UNSEEN=**+0.017**, **z=14.95**. This is exactly the pattern
`WHY-THESE-THREE.md`'s own power curve describes (their Table: estimator
crosses zero around n≈2,700 *random* clauses; balanced sampling gets
there far faster) — reproduced independently here at a smaller scale,
which is a real validation that this suite's methodology is sound, not
just a restatement of their number.

**pca-tertile still beats EO on UNSEEN (+0.136 vs +0.017)**, replicating
`WHY-THESE-THREE.md`'s own Result 1 (there: pca-tertile +0.498 vs EO
+0.277, at their larger n) — the same qualitative finding shows up here
independently, at a fraction of their sample size. That result carried a
big caveat there (partly a language-identity confound); this pilot
doesn't have the language-AMI check needed to say whether the same
caveat applies here.

**qwen and mistral diverge, not just in agreement but in whether their
labels carry structure at all.** qwen-eo's UNSEEN is still negative
(-0.508) but clears the null meaningfully (z=3.74); mistral-eo's UNSEEN
(-1.497) does not (z=0.22, indistinguishable from shuffled labels). Both
produced valid JSON at a similar rate (97.8%/96.7%), so this isn't a
parsing artifact — mistral's classifications, even though well-formed,
don't carry a coherent geometric pattern the way qwen's (weakly) do.

**The kappa finding gets more robust, not less, with 3.3x the data.**
qwen-vs-mistral: 0.003 → 0.006. Still chance-level. This was always
going to be the more likely explanation once ruled out as small-sample
noise: two 7B-class model families, prompted identically, genuinely do
not converge on this task.

## Scaling up further

The other five schemes could get the same per_cell=10 treatment. A real
constraint surfaced doing this run, worth knowing before requesting it:
partway through the original attempt at all six schemes, `llama-cpp
-python`'s CPU backend crashed with an illegal-instruction trap (confirmed
via `dmesg`) tied to this box's AVX512 support being unreliable at
runtime despite being advertised in `/proc/cpuinfo` — a known class of
issue on some cloud VMs. Fixed by rebuilding with
`CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX512=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON"`,
verified stable, but at roughly **2x slower per clause** (AVX512 was
genuinely accelerating the quantized matmuls). Re-running the remaining
five schemes at `per_cell=10` (`5 schemes × 10 × 27 cells × 2 models ≈
2,700 clauses`) is now roughly **10-13 hours of CPU time** at the
post-fix rate, not the ~8-9h estimated before the crash was found.
Scoped down to fewer schemes (e.g. just `eo` for the kappa question, or
just the new candidates) proportionally less. Not run here to keep the
pilot inside one session.

## The full 7,808-clause corpus — the no-LLM candidates, decisively

Since `pca-tertile`, `tree`, `surface`, and `verbnet-lexical` need only
embeddings + labels already in `classified.jsonl` (no new LLM
classification), they aren't bottlenecked by CPU generation time. Run at
the corpus's actual scale (`run_full_corpus.py`, ~131s total — embedding
7,808 clauses is minutes, not hours):

```
candidate                             n  cells     PAST   FUTURE   UNSEEN  UNSEEN_z
eo-consensus                       7808     27  +0.0208  +0.0156  +0.4009     26.22
tree(recursive kmeans)             7808     27  +0.1829  +0.0940  +0.4724     33.90
pca-tertile(blind product)         7808     27  +0.1141  +0.0943  +0.6353     38.73
surface(char/ttr/punct)            7808     18  +0.0150  +0.0102     nan        nan
verbnet-lexical(top5+other)        7808      6  -0.0004  -0.0004     nan        nan
```

**pca-tertile wins outright at full scale** — UNSEEN +0.635 vs EO's own
+0.401, the highest z of any candidate (38.7). This is the clearest,
least caveated version of the "does a blind split beat EO" finding
across this whole suite.

**tree (recursive KMeans) also beats EO here** — UNSEEN +0.472 vs +0.401.
Worth flagging precisely: `WHY-THESE-THREE.md`'s own Result 1 found the
*opposite* for tree (UNSEEN ≈ -0.01, "not a product, it's a hierarchy,"
confirmed by a branch-consistency diagnostic this run doesn't reproduce).
Both runs use the identical `Tree` class from `recursive_split.py` — the
difference is the embedding space (this suite's self-generated
`paraphrase-multilingual-MiniLM-L12-v2` vs. their original 3072-d model).
**This is a real, unresolved discrepancy, not a replication** — it says
tree's product-vs-hierarchy behavior may be embedding-space-dependent,
not that their finding was wrong. Confirming which requires running their
own branch-consistency check (principal angles between depth-2 splits
inside each depth-1 branch) here, which hasn't been done.

**verbnet-lexical (the top-5-VerbNet-class heuristic) is confirmed dead
weight at full scale** (-0.0004) — not small-n noise this time, a real
null result for that specific bucketing. Motivated building something
better, below.

## live_priors, eoreader7, and the-fold: a real bridge, and its limits

This session doesn't have GitHub access to `eoreader7` or `the-fold` —
only `eo-lexical-analysis-2.0` and `live_priors` are in scope, and
`list_repos`/`add_repo` aren't available in this session to add them.
That matters here because those two repos are where the cube is actually
*used* — `the-fold` is the workbench/server that consumes readings
(`explore-server.mjs`'s `/api/priors/*`), `eoreader7` is the reading
engine whose organs produce them (`native/adapters/text/phasepost.js`
consumes the 27-cell overlay this section is about). Testing whether the
cube earns its keep *inside that live pipeline* — as opposed to testing
its geometry in isolation, which is all this suite has done — needs
access to those repos. Grant it (org admin at
`claude.ai/admin-settings/claude-tag`, or reconnect GitHub at
`claude.ai/customize/connectors`) and this can go further.

What `live_priors` alone already provides, though, is real: **`ActPrior@1`**
(`derived-priors/act-priors/act-prior-en.json`) — a disclosed, hand-built
mapping from every VerbNet-attested verb FORM (4,569 of them, from 325
VerbNet classes) to one of the ecosystem's nine acts, built by
`scripts/build-act-prior.mjs` and consumed for real by eoreader7's
`phasepost.js`. Its own header names 872/4,569 forms (19.1%) as
"contested" (multiple candidate acts, each with the VerbNet class that
produced it) rather than silently picking one.

**A small but real cross-repo finding along the way:** `live_priors`'s
own canonical grid (`goldens/reading/RULE.md`, Part II) is:

| | Existence | Structure | Interpretation |
|---|---|---|---|
| Differentiate | NUL | SEG | **DEF** |
| Relate | **SIG** | CON | **EVA** |
| Generate | INS | SYN | REC |

`eo-lexical-analysis-2.0`'s own `ACT_FACE` table (`app2.py`) has the
identical 3×3 layout but three different names at the same three
positions: **ALT** (not DEF), **SEN** (not SIG), **SUP** (not EVA). Six
of nine match exactly; three don't. Worth knowing if anyone treats the
two as interchangeable — they're the same shape, not the same table.

### Building the better VerbNet candidate — and a bug it caught immediately

`candidates/act_prior_lexical.py` looks up each clause's main verb (same
POS-tag heuristic as `verbnet_lexical.py`) against ActPrior's lexicon.
First run, full corpus: **4.5% coverage** (352/7,808) — because
`act-prior-en.json` is English-only and this corpus is 2.7% English
spread across ~39 languages, and the lookup was run against all of it
regardless of language. Not a bug in the lexicon; a bug in testing it
against 97% of a corpus it was never going to cover. Restricted to the
213 actual English consensus clauses: coverage jumps to **89.7%**
(191/213).

At that corrected, fair n=213, scored against `eo-consensus` on the exact
same subset for a clean comparison:

```
actprior-lexical    cells=9   PAST=-0.0616  FUTURE=-0.0616  UNSEEN=nan
eo-consensus (n=213) cells=21 PAST=-0.0700  FUTURE=-0.0699  UNSEEN=-0.3039  z=3.59
```

**Genuinely inconclusive, not a loss for ActPrior** — EO's own structure
is *also* close to the noise floor at this same n (z=3.59, the same
small-n regime as the 81-clause pilot earlier in this document). Unlike
the LLM-classified schemes, this one can't just be scaled up with more
CPU time: 213 is *all* the English in this corpus. Answering it properly
needs either an English-heavy corpus or extending `act-prior-en.json`'s
approach to another language ActPrior doesn't yet cover.

### A concrete limitation the RULE.md examples caught

`act_prior_lexical.py` resolves a contested verb form by taking the
first-listed candidate — documented as a stated simplification when
written. Checking it against RULE.md's own worked examples immediately
found a real miss: **"divide"** is contested (`EVA` from a "multiply"
VerbNet sense, `SYN` from a change-of-state sense, `SEG` from a
"separate" sense) and the first-candidate policy picks `EVA` — but
RULE.md's own table lists "divided" as *its* worked example for `SEG`
("cutting an arrangement"). Same story for **"marry"** (`SYN` picked,
`CON` intended). This isn't a hypothetical caveat; it's confirmed on two
of the ecosystem's own canonical examples. A context-aware disambiguation
pass (e.g. asking a local model to pick among the lexicon's *own* listed
candidates for that specific clause, rather than open-ended
classification) is the natural fix — out of scope here since it
reintroduces the LLM-classification bottleneck this candidate was built
specifically to avoid.

---

# The live pipeline, tested for real — 2026-09-01

Access to `eoreader7` and `the-fold` — named above as the exact blocker
("Grant it... and this can go further") — is now in scope for this
session. This section is that test: does the cube earn its keep *inside*
the live pipeline, not just in isolation, using the REAL, unmodified
production code, not another Python port of it?

## The bridge, and what it is honest about not being

`bridge/phasepost_live_bridge.mjs` + `candidates/phasepost_live.py` chain
eoreader7's real `extractRelations` into eoreader7's real `phasepost.js`
(the 27-cell overlay), backed by live_priors' real `ActPrior@1` lexicon —
all three imported unmodified from their sibling checkouts. Neither
eoreader7 nor live_priors needed a single line changed to answer this
question; both were already general enough.

One thing does not carry over cleanly, and it was checked rather than
assumed: `extractRelations`'s vocabulary-*discovery* step
(`discoverRelationVocab`) anchors candidate verbs on capitalised
referents that **recur across a document** (`relations.js`'s own header:
"the token immediately FOLLOWING a candidate referent surface"). This
corpus is one decontextualized clause per row — concatenating 60 real
clauses from this run and feeding them through the real
`extractSurfaces`/`discoverRelationVocab` pair (`minSurfaces: 2`, the
same recurrence floor production uses) surfaced 14 real proper-noun
surfaces (Schrodinger, Bob, Einstein, Alice, Heisenberg, …) but discovered
a "verb vocabulary" of exactly `{"'s", "state"}` — noise, not verbs. That
mechanism is real and does real work on a real document (this repo's own
sibling files record it working on Frankenstein, civic prose, Shakespeare);
it is simply not the tool for a corpus with no document to recur across,
confirmed by running it rather than inferred from the header comment
alone. So this bridge supplies each clause's own main verb directly (the
same POS-tag heuristic `act_prior_lexical.py`/`verbnet_lexical.py` already
use in this repo) as a one-word vocabulary, and lets the REAL, unmodified
`extractRelations` do everything downstream of that — subject/object
capture, NP structure, negation/polarity, clause-boundary arithmetic —
exactly as any other caller would get it.

**Configuration, stated rather than defaulted silently.** `extractRelations`
has two opt-in flags, DR4 (`nounPhraseSubjects`) and DR5
(`phrasalPredicates`). live_priors' own actual corpus-mining driver
(`scripts/eot-digest.mjs`'s `loadOrgans()`, called with no arguments at
its top-level call site) runs with **both off** — a separate goldens
-evaluation path in that repo (`measure-dr45-at-scale.mjs`,
`goldens/reading/diff-golden.mjs`) turns both on. This bridge defaults to
the same (off, off) configuration the actual mining driver runs today
(`DR45=1` reproduces the other reading) — the numbers below are both
reported, and they barely move (see below), so nothing here turns on this
choice.

## Coverage, at real scale — the P56 finding revised, not the "limitation"

Run over all 3,595 English clauses in `run_2026-03-19_144302` (not just
PR#18's own 213-clause subset — this run is drawn from a **held-out
register**, arXiv quantum-physics prose, that `WHY-THESE-THREE.md`'s own
Arm C.2 names as untested):

```
                                                            n     %
contested (P56: a candidate set, never a coin-flip)      1391  38.7%
lexical (unanimous ActPrior@1 verdict)                   1091  30.3%
no_match (extractRelations found no SVO triple)           622  17.3%
copula (RULE.md's predicate-shape rules)                   244   6.8%
gap (verb unattested, even lemma-widened)                   222   6.2%
mechanical (existential-negative subject, A4)                25   0.7%
```

**82.7% of clauses (2,973/3,595) got a real subject/verb/object match** —
from a corpus this pipeline was never tuned against, in a register
(technical academic prose) neither `relations.js` nor `ActPrior@1` were
built for. Of those, every one of the 1,391 contested verdicts carried
`op: null` — **zero** silently resolved to a single act. Asserted, not
just observed: the run script's own `assert` on this fails loudly if it
is ever untrue.

**This directly revises PR#18's own "concrete limitation," without
touching PR#18's text** (append, don't rewrite — see above). PR#18 found
`act_prior_lexical.py`'s *first-candidate* policy mis-resolves "divide"
and "marry" against RULE.md's own worked examples. Checked against the
REAL production classifier, not the Python stand-in:

```
classify(subject:"the surveyor", verb:"divided", object:"the plot into three lots")
  -> standing: "contested", op: null, candidates: [EVA, SYN, SEG]
classify(subject:"the priest", verb:"married", object:"the couple")
  -> standing: "contested", op: null, candidates: [SYN, CON]
```

`phasepost.js` never picks either — it types the ambiguity and stops,
exactly as P56 requires. The "concrete limitation… confirmed on two of
the ecosystem's own canonical examples" is real, but it describes
`act_prior_lexical.py`'s own simplification, not eoreader7's production
code, which was never at risk of it. The disambiguation pass PR#18 named
as "the natural fix" is a fix for the Python test harness, not for
production.

**DR4/DR5 barely change the coverage shape.** With both on: contested
38.4%, lexical drops to 25.2% (some of it now typed `copula-participle`,
5.0%, a bucket the (off,off) run folds into `lexical`/`copula`), gap 6.6%,
no_match unchanged at 17.3% (DR4/DR5 widen *captured spans*, not *whether*
a match is found at all). Full numbers: `results/phasepost_live_report_dr45.json`.

## Structure — the real classifier, scored the same way EO is

On the 1,354-clause claude/gpt4-consensus subset (6.4x PR#18's own
213-clause English sample — large enough that `unique_exemplars.py`'s own
power finding, echoed in `WHY-THESE-THREE.md`, puts this comfortably past
the noise floor), the SAME held-out split, embedded once and reused:

```
candidate                                   n  cells     PAST   FUTURE   UNSEEN  UNSEEN_z
eo-consensus                             1354     27  +0.0329  +0.0238  +0.0480     10.83
phasepost-live-typed (9 acts+OTHER)      1354      9  -0.0103  -0.0103     nan        nan
phasepost-live-firstcand (control)       1354      9  -0.0005  -0.0005     nan        nan
phasepost-live-cell (op x grain)         1354     21  -0.0005  -0.0552     nan        nan
actprior-lexical [PR#18 baseline]        1354      9  -0.0018  -0.0018     nan        nan
tree (recursive kmeans)                  1354     27  +0.1997  +0.1291  +0.2701     19.12
pca-tertile (blind product)              1354     27  +0.1452  +0.1441  +0.5837     30.42
```

**UNSEEN is `nan` for every single-axis/near-single-axis scheme here, not
a bug** — `dimensionality.py::score`'s leave-one-cell-out step requires
every declared level of every axis to still be observed after removing
one cell; a 9-op(+OTHER)-level axis with only 9 of 10 declared levels
ever occupied can never satisfy that for *any* held-out cell, the same
reason `verbnet-lexical`/`actprior-lexical` already show `UNSEEN nan`
earlier in this document. FUTURE is the load-bearing number here.

**The honest reading: op alone does not structure this space, on this
corpus, at this register — and neither does the coin-flip control.**
`phasepost-live-typed`'s FUTURE (-0.0103) and `phasepost-live-firstcand`'s
(-0.0005) are both indistinguishable from `actprior-lexical`'s own
-0.0018 — all three sit in the same noise band. **Typing the ambiguity
honestly, versus coin-flipping it, changes almost nothing geometrically
here** — which means PR#18's own finding (EO-consensus's real signal
"genuinely inconclusive, not a loss for ActPrior" at n=213) was the right
call, not an artifact of too little data: at 6.4x the n, op-only structure
is still flat. `phasepost-live-cell` (adding grain as a second axis) is
*worse* (FUTURE -0.0552) than op alone — grain, as currently computed
(occurrence-level, "honestly rough" by its own docstring), adds noise
here, not signal. Meanwhile **eo-consensus's own three axes (mode x
domain x object) show real, positive, non-null structure on this same
subset** (FUTURE +0.0238, UNSEEN +0.0480, z=10.83) — richer than op alone
by construction (three questions, not one), and it shows.

**pca-tertile beats everything, again, on an independent corpus.**
UNSEEN +0.584 (z=30.42) vs. eo-consensus's +0.048 — replicating PR#18's
own full-corpus finding (there: pca-tertile +0.635 vs EO +0.401) on a
corpus this suite never touched before, in a different register. This
was not the question this section set out to answer, but it is exactly
the cross-corpus transportability check `WHY-THESE-THREE.md`'s own Arm
C.2 names as unrun ("Hold out an entire register — the arXiv
quantum-physics slice is the obvious one") — landing, incidentally, on
the same side PR#18 already found.

**DR4/DR5 do not change this reading.** With both on, `phasepost-live
-typed` FUTURE is -0.0106 (vs -0.0103 off); `-cell` is -0.0423 (vs
-0.0552). Same noise band either way.

## Why the gaps are shaped the way they are, checked not guessed

Sampling `no_match` clauses directly: the dominant pattern is **passive
voice and nominalised predicates** typical of academic prose — "Three
frequent objections to this solution *are rebutted*," "the problem…
*remains* unresolved," "we propose a two-ancilla protocol that *provides*
an experimentally accessible readout" — constructions where the
POS-tagged "main verb" is a past participle in a passive frame, or sits
too far from any subject the regex's literal subject-verb adjacency can
reach. `extractRelations` is doing exactly what it says: declining to
fabricate a triple it cannot literally match, on a register (arXiv
abstracts) it was never built against — the SAME "will not fabricate,
declared as such" discipline `relations.js`'s own header states, now
observed on a corpus that stresses it harder than civic prose or
Gutenberg novels do.

Sampling `gap` clauses: the verbs ARE found — "contradicts," "generalized,"
"proven," "postulates," "hypothesize," "bunching" — they simply are not
in `ActPrior@1`'s 4,569-form VerbNet-derived lexicon. This is a REGISTER
coverage gap, the same shape as PR#18's own LANGUAGE coverage finding
(an English-only lexicon against a 97%-non-English corpus) one level
over: a general-domain lexicon (VerbNet) against a technical-academic
register whose predicates ("hypothesize," "postulate," "bunching" as a
verbed nominalisation) skew toward exactly the vocabulary a general
resource under-covers.

## What this settles, and what it doesn't

**Settled:** the live pipeline is not a paper exercise — it runs, at real
coverage (82.7%), on a real independent corpus, in a register it was
never tuned for, without a single code change to either consuming repo.
Its P56 discipline (never coin-flip a contested verb) holds at scale, not
just on hand-picked examples. PR#18's "concrete limitation" was real but
mis-attributed to the wrong layer — corrected here, not erased there.

**Not settled, and not oversold:** the cube's OP axis alone — the one
axis `phasepost.js` actually computes from real material today — does not
structure this embedding space, on this corpus. That is a materially
narrower claim than "the cube doesn't structure the space": EO's own
three-axis consensus labels (mode x domain x object, still the richer,
LLM-elicited reading of "what transformation is this") DO show real
structure on the identical subset. Whether a richer, three-axis LIVE
reading — closer in spirit to EO's own Q1/Q2/Q3 rather than op alone —
would recover eo-consensus's signal from real material is the natural
next question this leaves open, not answered here.

## Reproducing

```bash
cd alt_structures
python3 run_phasepost_live.py            # DR4/DR5 off (live_priors' own mining default)
python3 run_phasepost_live.py --dr45     # DR4/DR5 on (live_priors' own goldens config)
PHASEPOST_LIVE_LIMIT=80 python3 run_phasepost_live.py   # fast smoke run
```

Needs `EOREADER7_PATH`/`LIVE_PRIORS_PATH`/`THE_FOLD_PATH` only if those
repos aren't sibling checkouts of this one (`bridge/phasepost_live_bridge.mjs`'s
own default, matching `act_prior_lexical.py`'s existing convention).
Full reports: `results/phasepost_live_report.json`,
`results/phasepost_live_report_dr45.json`.
