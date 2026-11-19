#!/usr/bin/env python3
"""Compare the single-researcher arm with the society arm.

Usage: python society/compare_arms.py path/to/arm-single path/to/arm-society

Reports, per arm:
  runs used        training runs from results.tsv plus every audit seed run
  promoted         hypotheses that survived the multi-seed audit
  champion         mean val_bpb of the final champion over the audit seeds
  gain             champion improvement over the starting baseline (seed means)
  subsystems       distinct subsystems tested, and the normalised entropy of
                   tests across subsystems (1.0 = spread evenly, 0 = all in one)
  confirms by half confirmations in the first and second half of the run, which
                   shows whether the arm kept finding things or flattened out
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from board import split_contracts, field  # noqa: E402

CANON = ["attention", "activation", "mlp", "residual", "normalisation", "positional",
         "embedding", "training-loop", "optimiser", "output"]


def canon(sub):
    s = sub.lower().replace("normalization", "normalisation")
    for c in CANON:
        if c.rstrip("s") in s or c in s:
            return c
    return "other"


def arm_summary(root: Path):
    hyp = split_contracts((root / "hypotheses.md").read_text()) if (root / "hypotheses.md").exists() else []
    tested = [(n, st, canon(field(b, "Subsystem"))) for n, _, st, b in hyp
              if st not in ("PROPOSED", "APPROVED", "REJECTED")]
    results = [l.split("\t") for l in (root / "results.tsv").read_text().splitlines()[1:]] \
        if (root / "results.tsv").exists() else []
    exp_runs = sum(1 for r in results if r and r[0] != "baseline")
    audits = [json.loads(l) for l in (root / "audits.jsonl").read_text().splitlines() if l.strip()] \
        if (root / "audits.jsonl").exists() else []
    audit_runs = sum(1 for a in audits if a["kind"] == "run")
    verdicts = [a for a in audits if a["kind"] == "verdict"]
    promoted = [v for v in verdicts if v["decision"] == "PROMOTE"]

    def mean(xs):
        xs = [x for x in xs if x == x]
        return sum(xs) / len(xs) if xs else float("nan")

    first = verdicts[0]["champion_vals"] if verdicts and "champion_vals" in verdicts[0] else []
    if promoted:
        last = promoted[-1]["candidate_vals"]
    else:
        last = first
    counts = Counter(s for _, _, s in tested)
    k = len(counts)
    total = sum(counts.values())
    ent = (-sum(c / total * math.log(c / total) for c in counts.values()) / math.log(k)) if k > 1 else 0.0
    confirms = [n for n, st, _ in tested if st in ("CONFIRMED", "PROMOTED", "AUDIT_FAILED", "AUDIT_CONFLICT")]
    ids = sorted(n for n, _, _ in tested)
    half = ids[len(ids) // 2] if ids else 0
    return {
        "runs used": exp_runs + audit_runs,
        "experiments": exp_runs,
        "audit runs": audit_runs,
        "confirmed (single run)": len(confirms),
        "promoted (multi-seed)": len(promoted),
        "start baseline": round(mean(first), 6) if first else None,
        "final champion": round(mean(last), 6) if last else None,
        "gain": round(mean(first) - mean(last), 6) if first and last else None,
        "subsystems tested": k,
        "subsystem entropy": round(ent, 3),
        "by subsystem": dict(counts),
        "confirms 1st half / 2nd half": (sum(1 for n in confirms if n < half), sum(1 for n in confirms if n >= half)),
    }


if __name__ == "__main__":
    arms = sys.argv[1:]
    if not arms:
        sys.exit(__doc__)
    sums = {Path(a).name: arm_summary(Path(a)) for a in arms}
    keys = list(next(iter(sums.values())).keys())
    w = max(len(k) for k in keys)
    print(" " * w + "  " + "  ".join(f"{n:>28}" for n in sums))
    for k in keys:
        print(f"{k:<{w}}  " + "  ".join(f"{str(s[k]):>28}" for s in sums.values()))
