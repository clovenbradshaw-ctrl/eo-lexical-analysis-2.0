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
│   └── blind_geometric.py   Tree / PcaProduct (imported from
│                            recursive_split.py) + Surface (reproduces
│                            falsify_3x3x3.py's feature defs), wrapped to
│                            harness.py's contract
├── local_models/
│   ├── registry.py          the two GGUF models
│   ├── download_models.sh   idempotent fetch (weights/*.gguf is gitignored)
│   ├── schemes.py           prompt/parser for eo (imported verbatim from
│                            app2.py) / vendler / halliday
│   ├── classify_local.py    runs one (model, scheme) over a balanced
│                            sample; output matches classified.jsonl's
│                            per-rater shape
│   └── run_model_suite.sh   runs all 3 schemes through one model
├── run_discovery.py         scores every candidate, prints/writes the
│                            comparison report + 4-way kappa
└── results/                 jsonl label files + discovery_report.json
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
question, or just the new candidates) proportionally less. Not run here
to keep the pilot inside one session.
