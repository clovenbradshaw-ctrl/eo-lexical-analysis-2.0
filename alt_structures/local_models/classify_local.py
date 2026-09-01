#!/usr/bin/env python3
"""
classify_local.py — run a classification scheme through a local CPU model.

Mirrors app2.py's classify_clause_anthropic / classify_clause_openai: same
system+user prompt contract, same JSON-in/JSON-out shape. Output rows carry
the clause `id` from the source classified.jsonl, so results line up
exactly with the "claude" / "gpt4" entries already there for the same
clause — same spans, different rater.

Usage:
  # smoke test: 3 clauses, one model, the original EO rubric
  python classify_local.py --model qwen --scheme eo \\
      --in ../../run_2026-03-15_122636/classified.jsonl \\
      --n 3 --out results/smoke_qwen_eo.jsonl

  # balanced pilot: 10 clauses per EO consensus cell (270 total)
  python classify_local.py --model mistral --scheme vendler \\
      --in ../../run_2026-03-15_122636/classified.jsonl \\
      --balanced-by consensus --per-cell 10 \\
      --out results/pilot_mistral_vendler.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from registry import MODELS, check_available  # noqa: E402
from schemes import SCHEMES  # noqa: E402


def load_llm(model_key, n_threads=4, n_ctx=1024):
    from llama_cpp import Llama
    cfg = MODELS[model_key]
    return Llama(model_path=str(cfg["path"]), n_ctx=n_ctx, n_threads=n_threads, verbose=False)


def classify_one(llm, scheme, clause, max_tokens=100, temperature=0.0):
    prompt = scheme["prompt_template"].format(clause=clause)
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": scheme["system"]},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = out["choices"][0]["message"]["content"]
    return scheme["parser"](text), text


def iter_clauses(path, ids_from=None, limit=None, balanced_by=None, per_cell=None):
    """Yields (id, clause_text, consensus_cell_or_None) from an existing
    classified.jsonl.

    balanced_by='consensus' draws `per_cell` clauses from each of the 27 EO
    consensus cells rather than a random/sequential slice — the repo's own
    unique_exemplars.py finding is that small BALANCED samples give a clean
    signal here while much larger random ones don't (estimator crosses zero
    around n=2,700 random clauses; a few hundred balanced ones already give
    a double-digit-sigma signal). Same rationale applies to any new scheme.
    """
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if ids_from and ids_from not in r.get("classifications", {}):
                continue
            rows.append(r)

    if balanced_by == "consensus" and per_cell:
        by_cell = defaultdict(list)
        for r in rows:
            c = r.get("consensus")
            if not c:
                continue
            by_cell[(c["q1"], c["q2"], c["q3"])].append(r)
        rows = []
        for cell in sorted(by_cell):
            rows.extend(by_cell[cell][:per_cell])

    if limit:
        rows = rows[:limit]

    for r in rows:
        cell = None
        if r.get("consensus"):
            c = r["consensus"]
            cell = (c["q1"], c["q2"], c["q3"])
        yield r["id"], r["clause"], cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--scheme", required=True, choices=list(SCHEMES))
    ap.add_argument("--in", dest="inp", required=True,
                     help="an existing classified.jsonl to draw clause spans from")
    ap.add_argument("--ids-from", default="claude",
                     help="only reclassify clauses this rater already labeled "
                          "(guarantees identical spans across raters); '' to disable")
    ap.add_argument("--balanced-by", default=None, choices=[None, "consensus"])
    ap.add_argument("--per-cell", type=int, default=None)
    ap.add_argument("--n", type=int, default=None, help="cap total clauses (applied after balancing)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    check_available()
    scheme = SCHEMES[args.scheme]

    print(f"loading {MODELS[args.model]['label']} ...", file=sys.stderr)
    t_load = time.time()
    llm = load_llm(args.model, n_threads=args.threads)
    print(f"loaded in {time.time() - t_load:.1f}s", file=sys.stderr)

    items = list(iter_clauses(
        args.inp,
        ids_from=(args.ids_from or None),
        limit=args.n,
        balanced_by=args.balanced_by,
        per_cell=args.per_cell,
    ))
    print(f"{len(items)} clauses  model={args.model}  scheme={args.scheme}", file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0
    t_total = 0.0
    with open(args.out, "w") as out_f:
        for i, (cid, clause, cell) in enumerate(items):
            t0 = time.time()
            parsed, raw = classify_one(llm, scheme, clause)
            dt = time.time() - t0
            t_total += dt
            n_ok += parsed is not None
            n_fail += parsed is None
            out_f.write(json.dumps({
                "id": cid,
                "clause": clause,
                "consensus_cell": cell,
                "model": args.model,
                "scheme": args.scheme,
                "parsed": parsed,
                "raw": raw,
                "seconds": round(dt, 2),
            }) + "\n")
            out_f.flush()
            avg = t_total / (i + 1)
            eta = avg * (len(items) - i - 1)
            print(f"[{i+1}/{len(items)}] {dt:5.1f}s  avg={avg:5.1f}s  "
                  f"ok={n_ok}/{i+1}  eta={eta/60:5.1f}min  id={cid}", file=sys.stderr)

    print(f"\ndone. {n_ok} parsed, {n_fail} failed "
          f"({100*n_fail/max(1,len(items)):.1f}% parse failure). "
          f"avg {t_total/max(1,len(items)):.1f}s/clause. wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
