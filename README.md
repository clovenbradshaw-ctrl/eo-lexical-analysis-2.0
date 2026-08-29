# EO Lexical Analysis 2.0

Tests whether the Emergent Ontology three-axis structure (Mode × Domain × Object)
corresponds to real semantic dimensions in natural language. The pipeline pulls
clauses from several multilingual corpora, classifies each one along three
plain-language axes via LLM classifiers, embeds the original text with
`text-embedding-3-large`, and then measures whether the classified labels
match geometric structure in the embedding space.

## Run the experiment from scratch

```bash
# 1. Install dependencies (main.py will also offer to do this for you)
pip install -r requirements.txt

# 2. Provide API credentials
cp .env.example .env
# then edit .env and paste in your keys
#   ANTHROPIC_API_KEY  — required for the Claude classifier
#   OPENAI_API_KEY     — required (used for both GPT-4o classification and embeddings)
#   GEMINI_API_KEY     — optional third classifier

# 3. Run the pipeline
python main.py
```

`main.py` is the single entry point and runs the entire experiment end to end:

1. **Corpus** — downloads & caches Universal Dependencies, FLORES-200, arXiv
   quantum physics abstracts, Bible Wisdom, and philosophy texts in `data/`.
2. **Classify** — sends each clause to the configured classifier(s) with three
   plain questions (Q1 mode, Q2 domain, Q3 object).
3. **Embed** — embeds the original clauses with OpenAI `text-embedding-3-large`.
4. **Analyze** — per-axis z-scores, proportionality, axis-independence (ARI),
   operator/face structure, coordinate geometry, helix dependency tests.
5. **Centroids** — geometric centroids for each EO cell + exemplar reports.

All output goes to a fresh `run_<timestamp>/` directory.

### Useful flags

```bash
python main.py --help                    # full flag list
python main.py --sample 200              # quick test run on 200 clauses
python main.py --phase classify          # re-run a single phase
python main.py --resume --run-dir run_…  # resume an interrupted run
python main.py --models claude,gpt4      # restrict classifier set
python main.py --no-ud --no-flores       # disable specific sources
```

## Supporting scripts

These reuse functions from `main.py` and act on existing `run_*/` outputs:

| Script | Purpose | Needs API keys? |
| --- | --- | --- |
| `analyze_only.py` | Re-runs only Phase 4 + Phase 5 on an existing run dir. Will download the published pre-classified / pre-embedded datasets if missing. | No |
| `falsify_3x3x3.py` | Falsification panel — compares EO labels against random / surface / PCA / optimised 3×3×3 partitions on the same embeddings. | No |
| `null_corpus_stats.py` | Builds non-semantic (length, punctuation, frequency, …) trichotomies as a 4th null for face and 27-cell z-scores. | No |

## Repository layout

```
main.py                    # full pipeline entry point
analyze_only.py            # analysis-only re-run (no API keys)
falsify_3x3x3.py           # falsification panel
null_corpus_stats.py       # corpus-statistic null
requirements.txt
.env.example
docs/                      # ProofWiki cache fetcher + falsification HTML viewer
run_2026-03-15_122636/     # published run output (embeddings, labels, reports)
run_2026-03-19_144302/     # published run output
Source data                # link to Google Drive folder with bulk artefacts
```
