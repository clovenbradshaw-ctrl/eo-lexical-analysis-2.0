#!/usr/bin/env python3
"""
run_discovery.py — score every candidate structure through the SAME
PAST/FUTURE/UNSEEN + shuffled-null rubric (harness.py), on the SAME
balanced clause sample, and print/write a comparison report in the spirit
of docs/WHY-THESE-THREE.md's own tables.

Candidates scored:
  eo-consensus          the original Q1xQ2xQ3, as already labeled in
                         classified.jsonl (claude+gpt4 consensus)
  {model}-eo              qwen/mistral replaying the exact same rubric on
                         the exact same spans -> 4-way rater agreement
  {model}-vendler          qwen/mistral under the Vendler aspect scheme
  {model}-halliday         qwen/mistral under the Halliday transitivity scheme
  {model}-srl              qwen/mistral under the SRL-style valence-pattern scheme
  {model}-discourse        qwen/mistral under the PDTB-style discourse-relation scheme
  tree(recursive kmeans)  blind, already in this repo (recursive_split.py)
  pca-tertile             blind product, already in this repo
  surface                 char-len x type-token x punct tertiles, already
                          in this repo (falsify_3x3x3.py), geometry-blind
  verbnet-lexical         top-5 most frequent Levin-style VerbNet classes
                          of the main verb (+OTHER), lexicon lookup, no LLM

Usage:
  python run_discovery.py --per-cell 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE / "local_models"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import harness  # noqa: E402
from sampling import balanced_sample, load_local_labels  # noqa: E402
from local_embeddings import embed_cached  # noqa: E402
from candidates.blind_geometric import GEOMETRIC_CANDIDATES  # noqa: E402
from candidates.verbnet_lexical import VERBNET_CANDIDATES  # noqa: E402

ALL_GEOMETRIC_CANDIDATES = {**GEOMETRIC_CANDIDATES, **VERBNET_CANDIDATES}
from schemes import SCHEMES  # noqa: E402

RUN_DIR = HERE.parent / "run_2026-03-15_122636"
CLASSIFIED = RUN_DIR / "classified.jsonl"


def encode(parsed_list, scheme):
    """parsed_list: list of parsed dicts or None, aligned to sample order.
    Returns (keep_mask[n], labels[n_kept, n_axes])."""
    axes, levels = scheme["axes"], scheme["levels"]
    keep, rows = [], []
    for p in parsed_list:
        if p is None:
            keep.append(False)
            continue
        try:
            idx = [levels[a].index(p[a]) for a in axes]
        except (KeyError, ValueError):
            keep.append(False)
            continue
        keep.append(True)
        rows.append(idx)
    labels = np.array(rows, dtype=int) if rows else np.zeros((0, len(axes)), int)
    return np.array(keep), labels


def eo_consensus_labels(sample):
    eo = SCHEMES["eo"]
    parsed = [{"q1": s["cell"][0], "q2": s["cell"][1], "q3": s["cell"][2]} for s in sample]
    return encode(parsed, eo)


def score_candidate(name, X, labels, n_lev, seed=0, n_perm=200):
    tr, te = harness.train_test_split(len(X), seed=seed)
    sc = harness.score_structure(X[tr], labels[tr], X[te], labels[te], n_lev)
    null = harness.shuffled_null(X[tr], labels[tr], X[te], labels[te], n_lev,
                                  n_perm=n_perm, seed=seed, metric="unseen")
    return {"name": name, "n": len(X), "n_lev": n_lev, **sc, "unseen_null": null}


def flat27(d):
    if not d:
        return None
    try:
        return f"{d['q1']}|{d['q2']}|{d['q3']}"
    except KeyError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=3,
                     help="must match the per-cell used when running classify_local.py")
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    out_path = Path(args.out) if args.out else results_dir / "discovery_report.json"

    sample = balanced_sample(CLASSIFIED, per_cell=args.per_cell)
    clauses = [s["clause"] for s in sample]
    print(f"{len(sample)} clauses ({args.per_cell}/cell x 27 cells)", file=sys.stderr)

    print(f"embedding via {__import__('local_embeddings').MODEL_NAME} (cached) ...", file=sys.stderr)
    X_full = embed_cached(clauses, results_dir / "sample_embeddings.npz")

    report = {"n_sample": len(sample), "per_cell": args.per_cell, "candidates": []}

    # ── EO consensus: the structure this whole investigation is about ──────────
    keep, labels = eo_consensus_labels(sample)
    report["candidates"].append(
        score_candidate("eo-consensus", X_full[keep], labels, [3, 3, 3], args.seed, args.n_perm))

    # ── local-LLM schemes, both models: original rubric + 2 conventional alts ──
    for model in ("qwen", "mistral"):
        for scheme_name in ("eo", "vendler", "halliday", "srl", "discourse"):
            path = results_dir / f"{model}_{scheme_name}.jsonl"
            local = load_local_labels(path)
            if not local:
                print(f"skip {model}-{scheme_name}: no results at {path}", file=sys.stderr)
                continue
            parsed_list = [local.get(s["id"]) for s in sample]
            scheme = SCHEMES[scheme_name]
            keep, labels = encode(parsed_list, scheme)
            if keep.sum() < 30:
                print(f"skip {model}-{scheme_name}: only {int(keep.sum())} parsed (need >=30)", file=sys.stderr)
                continue
            n_lev = [len(scheme["levels"][a]) for a in scheme["axes"]]
            report["candidates"].append(
                score_candidate(f"{model}-{scheme_name}", X_full[keep], labels, n_lev, args.seed, args.n_perm))

    # ── blind geometric rivals already in this repo ─────────────────────────────
    tr, te = harness.train_test_split(len(sample), seed=args.seed)
    for name, (needs_text, n_lev, factory) in ALL_GEOMETRIC_CANDIDATES.items():
        cand = factory(args.seed)
        try:
            if needs_text:
                ctr, cte = [clauses[i] for i in tr], [clauses[i] for i in te]
                cand.fit(ctr)
                ltr, lte = cand.apply(ctr), cand.apply(cte)
            else:
                cand.fit(X_full[tr])
                ltr, lte = cand.apply(X_full[tr]), cand.apply(X_full[te])
            sc = harness.score_structure(X_full[tr], ltr, X_full[te], lte, n_lev)
            report["candidates"].append({"name": name, "n": len(sample), "n_lev": n_lev, **sc})
        except ValueError as e:
            # e.g. Tree's depth-3 recursive KMeans needs more train points per
            # branch than a small pilot sample provides (~len(train)/27 avg).
            # Not a bug in the candidate -- this sample is just too small for
            # it; report that rather than aborting the whole run.
            print(f"skip {name}: {e} (sample too small for this candidate -- "
                  f"needs more train points per branch than {len(tr)} total gives it)", file=sys.stderr)
            report["candidates"].append({"name": name, "n": len(sample), "n_lev": n_lev,
                                          "error": f"insufficient data: {e}"})

    # ── 4-way rater agreement on the EO scheme ───────────────────────────────────
    raw = {}
    with open(CLASSIFIED) as f:
        for line in f:
            r = json.loads(line)
            raw[r["id"]] = r
    qwen_local = load_local_labels(results_dir / "qwen_eo.jsonl")
    mistral_local = load_local_labels(results_dir / "mistral_eo.jsonl")
    raters = {"claude": [], "gpt4": [], "qwen": [], "mistral": []}
    for s in sample:
        cls = raw[s["id"]]["classifications"]
        raters["claude"].append(flat27(cls.get("claude")))
        raters["gpt4"].append(flat27(cls.get("gpt4")))
        raters["qwen"].append(flat27(qwen_local.get(s["id"])))
        raters["mistral"].append(flat27(mistral_local.get(s["id"])))
    names, kappa = harness.kappa_matrix(raters)
    report["rater_agreement"] = {"raters": names, "kappa": kappa.tolist()}

    # ── print ────────────────────────────────────────────────────────────────
    print(f"\n{'candidate':32} {'n':>5} {'cells':>6} {'PAST':>8} {'FUTURE':>8} {'UNSEEN':>8} {'ratio':>7} {'UNSEEN_z':>9}")
    print("-" * 92)
    for c in report["candidates"]:
        if "error" in c:
            print(f"{c['name']:32} {c['n']:5d}    -- {c['error']}")
            continue
        z = c.get("unseen_null", {}).get("z", float("nan"))
        print(f"{c['name']:32} {c['n']:5d} {c['cells']:6d} {c['past']:+8.4f} {c['future']:+8.4f} "
              f"{c['unseen']:+8.4f} {c['product_ratio']:7.3f} {z:9.2f}")

    print("\nrater agreement (Cohen's kappa, flat 27-cell label, pairwise on parsed items)")
    print(" " * 10 + "".join(f"{n:>10}" for n in names))
    for i, n in enumerate(names):
        row = "".join(f"{kappa[i, j]:10.3f}" if not np.isnan(kappa[i, j]) else f"{'--':>10}"
                       for j in range(len(names)))
        print(f"{n:10}{row}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
