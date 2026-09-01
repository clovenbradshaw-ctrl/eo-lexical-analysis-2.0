"""
sampling.py — the balanced clause sample shared by classify_local.py runs
and run_discovery.py's scoring, so local-model labels line up 1:1 with
what gets scored.

balanced_sample() draws `per_cell` clauses from each of the 27 EO
consensus cells rather than a random or sequential slice of the corpus.
That mirrors unique_exemplars.py's own finding (docs/WHY-THESE-THREE.md,
"The coordinates, and the instrument that finally worked"): a random
sample only gives a usable signal past roughly n=2,700 clauses, while a
small BALANCED one (a few hundred, spread evenly across cells) already
gives a strong, clean signal. The same reasoning should apply to any new
label scheme scored against embedding geometry, not just EO's own.

Caveat, stated once here rather than silently: this takes the first
`per_cell` matches per cell in file order, not unique_exemplars.py's
leave-one-out-margin "most representative" selection. A production run
should switch to that; this suite trades some of that rigor for a sample
small enough to reclassify with two 7B CPU models in a single session.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def balanced_sample(classified_path, per_cell, ids_from="claude"):
    """Returns a list of dicts: {id, clause, language, source, cell}
    (cell = (q1, q2, q3) EO consensus tuple), `per_cell` per of the 27
    cells, drawn only from clauses `ids_from` already rated (so every
    rater — claude/gpt4/local models — was shown the same span)."""
    by_cell = defaultdict(list)
    with open(classified_path) as f:
        for line in f:
            r = json.loads(line)
            if ids_from and ids_from not in r.get("classifications", {}):
                continue
            c = r.get("consensus")
            if not c:
                continue
            cell = (c["q1"], c["q2"], c["q3"])
            by_cell[cell].append({
                "id": r["id"], "clause": r["clause"],
                "language": r.get("language"), "source": r.get("source"),
                "cell": cell,
            })

    out = []
    for cell in sorted(by_cell):
        out.extend(by_cell[cell][:per_cell])
    return out


def load_local_labels(path):
    """Reads a classify_local.py output jsonl into {id: parsed_dict_or_None}."""
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r["parsed"]
    return out
