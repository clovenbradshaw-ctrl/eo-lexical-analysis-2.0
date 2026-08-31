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
pip install llama-cpp-python

# classify a balanced sample (N/cell x 27 cells) through both models,
# all three schemes — this is the slow step, budget real CPU time (see below)
./run_model_suite.sh qwen    4 <per-cell>
./run_model_suite.sh mistral 4 <per-cell>

cd ..
pip install sentence-transformers scikit-learn scipy
python run_discovery.py --per-cell <same per-cell as above>
```

**Timing, measured on this box (4 vCPU, 15GB RAM):** ~13-16s/clause/model
running one model at a time at full thread count. Running both models
*concurrently* at half the threads each was **not** faster — thread
contention on 4 cores pushed per-clause time to ~40s, worse than serial —
so `run_model_suite.sh` is meant to be run sequentially per model, not
backgrounded in parallel. Budget roughly `81 clauses/scheme × 3 schemes ×
2 models × ~15s ≈ 2 hours` at `per-cell=3`; scale linearly with per-cell.

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

Both are single-axis schemes (4 and 6 levels respectively, vs. EO's three
3-level axes), which is exactly why `harness.py` imports its FUTURE/UNSEEN
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

## Status

Infrastructure built and smoke-tested (synthetic-data unit checks on
`harness.py` + all three geometric candidates; a 3-clause live run through
Qwen at 100% JSON-parse success). The full classification + discovery
pass for this pilot (`per_cell=3`, 81 clauses/scheme/model) was kicked off
in the background; results and the filled-in comparison table will be
appended here once it completes. See `results/discovery_report.json` for
the machine-readable version once present.
