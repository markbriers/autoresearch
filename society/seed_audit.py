#!/usr/bin/env python3
"""Multi-seed audit: the gate between CONFIRMED and PROMOTED.

The paper found that checks which execute code catch what text review misses
(running referee objections as scripts overturned 49% of headline conclusions).
In v9 a single run with delta < -0.003 counted as CONFIRMED, against a seed
s.d. of about 0.002, so roughly a one-sigma gate. Later hypotheses then stacked
on top of results that may have been noise.

This script applies a CONFIRMED experiment commit on top of the current
champion, runs champion and candidate on several seeds, and promotes only if
the mean improvement clears the threshold and is at least two standard errors
from zero. Promotion advances the `champion` branch, so the next cycle's
hypotheses build on a gain that has been replicated. Applying onto the current
champion also makes every promotion an explicit stacking test.

Usage:
  python society/seed_audit.py --hypothesis 7 --commit <sha>
  python society/seed_audit.py --baseline-only        # measure seed noise

Every training run is logged to audits.jsonl and counts against the arm's
run budget.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = os.environ.get("AR_PYTHON", str(ROOT / ".venv" / "bin" / "python"))
WT = ROOT / "worktrees"
CACHE = ROOT / "audit_cache.json"
LOG = ROOT / "audits.jsonl"


def git(*args, cwd=ROOT, check=True):
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True).stdout.strip()


def run_train(tree: Path, seed: int) -> float:
    env = dict(os.environ, AR_SEED=str(seed))
    t0 = time.time()
    p = subprocess.run([PY, "train.py"], cwd=tree, env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr
    (tree / f"audit_seed{seed}.log").write_text(out)
    m = re.search(r"^val_bpb:\s+([0-9.]+)", out, re.M)
    val = float(m.group(1)) if m else float("nan")
    with LOG.open("a") as f:
        f.write(json.dumps({"kind": "run", "tree": tree.name, "seed": seed, "val_bpb": val,
                            "seconds": round(time.time() - t0, 1),
                            "commit": git("rev-parse", "HEAD", cwd=tree)}) + "\n")
    return val


def stats(xs):
    xs = [x for x in xs if not math.isnan(x)]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    return mu, sd, n


def champion_scores(seeds, cache):
    champ = git("rev-parse", "champion")
    tree = WT / "champion"
    if tree.exists():
        git("worktree", "remove", "--force", str(tree))
    git("worktree", "add", "--detach", str(tree), champ)
    vals = []
    for s in seeds:
        key = f"{champ}:{s}"
        if key not in cache:
            cache[key] = run_train(tree, s)
            CACHE.write_text(json.dumps(cache, indent=1))
        vals.append(cache[key])
    git("worktree", "remove", "--force", str(tree))
    return champ, vals


def set_status(n, status):
    hyp = ROOT / "hypotheses.md"
    if not hyp.exists():
        return
    text = hyp.read_text()
    text = re.sub(rf"^(## Hypothesis {n}: .*? \| Status: )\w+", rf"\g<1>{status}", text, flags=re.M)
    hyp.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis", type=int)
    ap.add_argument("--commit")
    ap.add_argument("--seeds", default=os.environ.get("AR_AUDIT_SEEDS", "1,2,3"))
    ap.add_argument("--threshold", type=float, default=float(os.environ.get("AR_THRESHOLD", "0.003")))
    ap.add_argument("--baseline-only", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    WT.mkdir(exist_ok=True)

    champ, base = champion_scores(seeds, cache)
    bmu, bsd, bn = stats(base)
    print(f"champion {champ[:7]}: mean {bmu:.6f} sd {bsd:.6f} over {bn} seeds {base}")
    if a.baseline_only:
        return

    # Apply the candidate's change on top of the current champion.
    parent = git("rev-parse", f"{a.commit}^")
    tree = WT / f"audit-H{a.hypothesis}"
    if tree.exists():
        git("worktree", "remove", "--force", str(tree))
    git("worktree", "add", "--detach", str(tree), champ)
    diff = git("diff", parent, a.commit, "--", "train.py")
    ap_res = subprocess.run(["git", "apply", "--3way", "-"], cwd=tree, input=diff + "\n", text=True, capture_output=True)
    if ap_res.returncode != 0:
        verdict = {"kind": "verdict", "hypothesis": a.hypothesis, "commit": a.commit, "champion": champ,
                   "decision": "CONFLICT", "detail": ap_res.stderr[-500:]}
        LOG.open("a").write(json.dumps(verdict) + "\n")
        set_status(a.hypothesis, "AUDIT_CONFLICT")
        git("worktree", "remove", "--force", str(tree))
        print("CONFLICT: change does not apply cleanly to the current champion")
        return
    git("-c", "user.name=autoresearch", "-c", "user.email=autoresearch@localhost",
        "commit", "-am", f"H{a.hypothesis} on champion {champ[:7]}", cwd=tree)
    cand = [run_train(tree, s) for s in seeds]
    cmu, csd, cn = stats(cand)

    delta = cmu - bmu
    se = math.sqrt((bsd ** 2) / max(bn, 1) + (csd ** 2) / max(cn, 1)) if bn > 1 and cn > 1 else float("nan")
    z = delta / se if se and not math.isnan(se) and se > 0 else float("nan")
    promote = cn == len(seeds) and delta < -a.threshold and (not math.isnan(z) and z < -2.0)
    decision = "PROMOTE" if promote else "HOLD"
    new_head = git("rev-parse", "HEAD", cwd=tree)
    verdict = {"kind": "verdict", "hypothesis": a.hypothesis, "commit": a.commit, "champion": champ,
               "candidate_on_champion": new_head, "champion_vals": base, "candidate_vals": cand,
               "delta_mean": round(delta, 6), "se": round(se, 6) if not math.isnan(se) else None,
               "z": round(z, 2) if not math.isnan(z) else None, "decision": decision}
    LOG.open("a").write(json.dumps(verdict) + "\n")
    git("worktree", "remove", "--force", str(tree))
    if promote:
        git("branch", "-f", "champion", new_head)
        # The candidate's seed runs are the new champion's seed runs: reuse them.
        for s_, v in zip(seeds, cand):
            cache[f"{new_head}:{s_}"] = v
        CACHE.write_text(json.dumps(cache, indent=1))
        set_status(a.hypothesis, "PROMOTED")
    else:
        set_status(a.hypothesis, "AUDIT_FAILED")
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    sys.exit(main())
