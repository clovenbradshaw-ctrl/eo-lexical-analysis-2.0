#!/usr/bin/env python3
"""falsify_aggregate.py — Cross-set comparison and pre-registered
falsification verdict for the question-set falsifiability test.

Reads:
  <base-run-dir>/falsify/falsify_results.json       — EO baseline (full run)
  <pilot-dir>/<set>/falsify/falsify_results.json    — per-set pilot results
  <pilot-dir>/<set>/classified.jsonl                — per-set labels (for ARI)
  <base-run-dir>/classified.jsonl                   — EO labels (for ARI)
  <base-run-dir>/embeddings.npz                     — id ordering only

Emits:
  <pilot-dir>/cross_set/comparison.json
  <pilot-dir>/cross_set/comparison_report.txt

Pre-registered falsification rule:
  EO is falsified iff some non-EO question set S, that is neither geometric
  (PCA-based) nor a relabeling of EO (Hungarian-aligned ARI vs EO <= 0.10
  on the 27-flat), satisfies ALL of:
    (a) monotonicity slope >= EO_slope - 1 * SE_EO
    (b) corner-uniqueness rank <= EO_corner_rank
    (c) class-balance check passed: every axis has
        min-tertile share >= 0.15 AND max-tertile share <= 0.55
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

import question_sets
from falsify_3x3x3 import labels_for_ids, load_classified, load_embeddings, flat_label


MIN_TERTILE = 0.15
MAX_TERTILE = 0.55
ARI_RELABEL_THRESHOLD = 0.10


def _eo_method_block(eo_results: dict) -> dict:
    for m in eo_results.get("methods", []):
        if m.get("method") == "eo":
            return m
    raise KeyError("no 'eo' method block in EO baseline falsify_results.json")


def _slope_se(block: dict) -> float:
    """Approximate SE of the slope estimate from the per-bucket stdevs.
    slope = mean[3] - mean[0]; SE ≈ sqrt(var[3]/n[3] + var[0]/n[0])."""
    prop = block.get("proportionality", {})
    buckets = prop.get("buckets", {})
    b0 = buckets.get("0") or buckets.get(0) or {}
    b3 = buckets.get("3") or buckets.get(3) or {}
    n0, s0 = b0.get("n_pairs", 0), b0.get("stdev", 0.0) or 0.0
    n3, s3 = b3.get("n_pairs", 0), b3.get("stdev", 0.0) or 0.0
    if n0 < 2 or n3 < 2:
        return float("inf")
    return math.sqrt(s0 ** 2 / n0 + s3 ** 2 / n3)


def _slope(block: dict) -> Optional[float]:
    prop = block.get("proportionality", {})
    return prop.get("slope")


def _corner_rank(block: dict) -> Optional[int]:
    corner = block.get("corner_uniqueness", {})
    return corner.get("top_corner_rank")


def _label_balance(block: dict) -> Dict[str, Dict[str, float]]:
    """Return per-axis {min_share, max_share}."""
    dist = block.get("label_distribution", {})
    out = {}
    for axis_key in ("q1", "q2", "q3"):
        counts = dist.get(axis_key, {}) or {}
        # Ignore -1 (missing)
        clean = {k: v for k, v in counts.items() if str(k) != "-1"}
        total = sum(clean.values()) or 1
        shares = [v / total for v in clean.values()]
        out[axis_key] = {
            "min_share": min(shares) if shares else 0.0,
            "max_share": max(shares) if shares else 1.0,
        }
    return out


def _balance_ok(balance: Dict[str, Dict[str, float]]) -> bool:
    for axis_data in balance.values():
        if axis_data["min_share"] < MIN_TERTILE or axis_data["max_share"] > MAX_TERTILE:
            return False
    return True


def hungarian_aligned_ari(a: np.ndarray, b: np.ndarray) -> float:
    """ARI score (already permutation-invariant; we return Hungarian for
    parity with the other module — actually adjusted_rand_score IS
    permutation invariant, so we just return it)."""
    return float(adjusted_rand_score(a, b))


def hungarian_accuracy(a: np.ndarray, b: np.ndarray) -> float:
    ua = np.unique(a)
    ub = np.unique(b)
    cost = np.zeros((len(ua), len(ub)), dtype=np.int64)
    for i, va in enumerate(ua):
        ai = a == va
        for j, vb in enumerate(ub):
            cost[i, j] = -int(np.sum(ai & (b == vb)))
    ri, cj = linear_sum_assignment(cost)
    matches = -cost[ri, cj].sum()
    return float(matches / len(a))


def axis_pair_ari_matrix(eo_triplet, s_triplet) -> List[List[float]]:
    M = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            M[i][j] = float(adjusted_rand_score(eo_triplet[i], s_triplet[j]))
    return M


def load_set_labels(run_dir: Path, set_name: str) -> Tuple[np.ndarray, List[str], dict]:
    vectors, ids = load_embeddings(run_dir)
    classified = load_classified(run_dir)
    q1, q2, q3, valid = labels_for_ids(ids, classified, set_name=set_name)
    return (q1, q2, q3, valid), ids, classified


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", required=True, help="e.g. <base>/pilot")
    ap.add_argument("--base-run-dir", required=True, help="Base run-dir holding EO baseline + embeddings")
    args = ap.parse_args()

    pilot_root = Path(args.pilot_dir)
    base = Path(args.base_run_dir)

    eo_results_path = base / "falsify" / "falsify_results.json"
    if not eo_results_path.exists():
        # Some workflows write to <run-dir>/falsify_results.json directly
        alt = base / "falsify_results.json"
        if alt.exists():
            eo_results_path = alt
        else:
            print(f"ERROR: EO baseline not found at {eo_results_path}")
            return 2
    eo_results = json.loads(eo_results_path.read_text())
    eo_block = _eo_method_block(eo_results)
    eo_slope = _slope(eo_block) or 0.0
    eo_se = _slope_se(eo_block)
    eo_corner = _corner_rank(eo_block) or 27
    eo_balance = _label_balance(eo_block)
    print(f"EO baseline: slope={eo_slope:+.4f}  SE≈{eo_se:.4f}  corner_rank={eo_corner}")

    # Load EO labels (for cross-set ARI) — restrict to ids that appear in
    # each pilot's embedding subset.
    print("Loading EO labels from base run-dir ...")
    eo_vectors, eo_ids = load_embeddings(base)
    eo_classified = load_classified(base)
    eo_q1, eo_q2, eo_q3, eo_valid = labels_for_ids(eo_ids, eo_classified, set_name="eo")
    eo_id_to_idx = {cid: i for i, cid in enumerate(eo_ids)}

    rows = []
    for set_dir in sorted(p for p in pilot_root.iterdir() if p.is_dir() and p.name != "cross_set"):
        set_name = set_dir.name
        if set_name == "eo":
            continue
        try:
            qs = question_sets.get(set_name)
        except KeyError:
            print(f"  skipping unknown set dir: {set_name}")
            continue
        results_path = set_dir / "falsify" / "falsify_results.json"
        if not results_path.exists():
            print(f"  [{set_name}] missing falsify_results.json — skipping")
            continue
        set_results = json.loads(results_path.read_text())
        block = _eo_method_block(set_results)  # "eo" label here is the set under test
        slope = _slope(block) or 0.0
        corner = _corner_rank(block) or 27
        balance = _label_balance(block)
        balance_ok = _balance_ok(balance)

        # Hungarian-aligned ARI with EO on the SAME ids (intersection of EO
        # labels and pilot labels)
        try:
            (s_q1, s_q2, s_q3, s_valid), pilot_ids, _ = load_set_labels(set_dir, set_name)
        except Exception as e:
            print(f"  [{set_name}] failed to load pilot labels: {e}")
            continue
        # Align ids
        common_idx_s = []
        common_idx_e = []
        for i, cid in enumerate(pilot_ids):
            if not s_valid[i]:
                continue
            j = eo_id_to_idx.get(cid)
            if j is None or not eo_valid[j]:
                continue
            common_idx_s.append(i)
            common_idx_e.append(j)
        if not common_idx_s:
            print(f"  [{set_name}] no overlapping valid ids with EO — skipping ARI")
            ari_eo = float("nan")
            acc_eo = float("nan")
            pair_matrix = None
        else:
            cis = np.array(common_idx_s)
            cie = np.array(common_idx_e)
            s_flat = flat_label(s_q1[cis], s_q2[cis], s_q3[cis])
            e_flat = flat_label(eo_q1[cie], eo_q2[cie], eo_q3[cie])
            ari_eo = hungarian_aligned_ari(e_flat, s_flat)
            acc_eo = hungarian_accuracy(e_flat, s_flat)
            pair_matrix = axis_pair_ari_matrix((eo_q1[cie], eo_q2[cie], eo_q3[cie]),
                                               (s_q1[cis], s_q2[cis], s_q3[cis]))

        # Pre-registered verdict per set
        cond_slope = slope >= (eo_slope - eo_se)
        cond_corner = corner <= eo_corner
        cond_balance = balance_ok
        cond_not_relabeling = (not math.isnan(ari_eo)) and (ari_eo <= ARI_RELABEL_THRESHOLD)
        falsifies = (qs.family != "eo"
                     and cond_slope and cond_corner and cond_balance and cond_not_relabeling)

        rows.append({
            "set_name": set_name,
            "family": qs.family,
            "slope": slope,
            "slope_vs_eo": slope - eo_slope,
            "corner_rank": corner,
            "balance": balance,
            "balance_ok": balance_ok,
            "ari_vs_eo_flat27": ari_eo,
            "hungarian_acc_vs_eo": acc_eo,
            "axis_pair_ari_matrix": pair_matrix,
            "cond_slope_ge_eo_minus_1se": cond_slope,
            "cond_corner_le_eo": cond_corner,
            "cond_balance_ok": cond_balance,
            "cond_not_relabeling": cond_not_relabeling,
            "FALSIFIES_EO": falsifies,
        })

    cross_dir = pilot_root / "cross_set"
    cross_dir.mkdir(exist_ok=True)
    out_json = {
        "base_run_dir": str(base),
        "pilot_dir": str(pilot_root),
        "eo_baseline": {
            "slope": eo_slope,
            "slope_se": eo_se,
            "corner_rank": eo_corner,
            "balance": eo_balance,
        },
        "thresholds": {
            "min_tertile": MIN_TERTILE,
            "max_tertile": MAX_TERTILE,
            "ari_relabel": ARI_RELABEL_THRESHOLD,
        },
        "sets": rows,
        "any_set_falsifies": any(r["FALSIFIES_EO"] for r in rows),
    }
    (cross_dir / "comparison.json").write_text(json.dumps(out_json, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else (int(o) if isinstance(o, np.integer) else o.tolist() if isinstance(o, np.ndarray) else str(o))))

    # Text report
    lines = []
    lines.append("=" * 78)
    lines.append("CROSS-SET FALSIFICATION COMPARISON")
    lines.append("=" * 78)
    lines.append(f"Base run-dir: {base}")
    lines.append(f"Pilot dir:    {pilot_root}")
    lines.append("")
    lines.append(f"EO baseline:  slope = {eo_slope:+.4f}    1 SE ≈ {eo_se:.4f}")
    lines.append(f"              corner-rank = {eo_corner}    balance min/max per axis:")
    for axis, b in eo_balance.items():
        lines.append(f"                  {axis}: min={b['min_share']:.2f}  max={b['max_share']:.2f}")
    lines.append("")
    lines.append(f"Thresholds:   slope >= EO - 1 SE        corner <= EO")
    lines.append(f"              balance: min tertile >= {MIN_TERTILE}, max <= {MAX_TERTILE}")
    lines.append(f"              not-relabeling: Hungarian-ARI(EO,S) on 27-flat <= {ARI_RELABEL_THRESHOLD}")
    lines.append("")
    header = f"{'set':<14} {'family':<11} {'slope':>+8} {'Δslope':>+8} {'corner':>6} {'bal':>3} {'ARI(EO)':>+8} {'verdict':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        bal = "ok" if r["balance_ok"] else "BAD"
        ari = r["ari_vs_eo_flat27"]
        ari_s = f"{ari:+.3f}" if not math.isnan(ari) else "  n/a"
        verdict = "FALSIFIES" if r["FALSIFIES_EO"] else "—"
        lines.append(f"{r['set_name']:<14} {r['family']:<11} "
                     f"{r['slope']:>+8.4f} {r['slope_vs_eo']:>+8.4f} "
                     f"{r['corner_rank']:>6} {bal:>3} {ari_s:>8} {verdict:>10}")
    lines.append("")
    if out_json["any_set_falsifies"]:
        lines.append(">>> EO IS FALSIFIED by at least one non-EO, non-relabeling set <<<")
    else:
        lines.append(">>> EO SURVIVES: no non-EO, non-relabeling set passes all conditions. <<<")
    lines.append("")

    report = "\n".join(lines)
    (cross_dir / "comparison_report.txt").write_text(report)
    print(report)
    print(f"\nWrote {cross_dir / 'comparison.json'} and {cross_dir / 'comparison_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
