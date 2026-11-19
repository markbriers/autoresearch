#!/usr/bin/env python3
"""Shared board for the society protocol.

The paper "A Society of AI Scientists" (alphaXiv 2611.society-ai-scientists,
Section 4.5) found that agents who could see each other's scores crowded around
the current leader (88% of later trials), while agents who saw only *what* had
been tried, not how well it scored, kept searching broadly (30%).

This script enforces that split mechanically rather than by prompt:

  merge   Move PROPOSED contracts from labs/<lab>/proposals.md into the shared
          hypotheses.md, numbering them and tagging each with its lab.
  render  Write board.md (every lab's coordinates: lab, subsystem, intervention,
          tested or not, and nothing else) and labs/<lab>/own.md (that lab's own
          contracts with their verdicts, still without numbers).

In the "single" arm the lone researcher reads hypotheses.md and evaluations.md
directly, as in v9, so it sees every score. In the "society" arm labs read only
board.md and their own own.md.
"""
import argparse
import re
from pathlib import Path

HYP_HEADER = re.compile(r"^## Hypothesis (\d+|N): (.*?) \| Status: (\w+)\s*$", re.M)
FIELD = lambda name: re.compile(r"^\*\*" + re.escape(name) + r":\*\*\s*(.*)$", re.M)
NUMBER = re.compile(r"[-+]?\d*\.\d+|\b\d+\b")


def split_contracts(text):
    """Return list of (number, title, status, body) for each contract in text."""
    out = []
    matches = list(HYP_HEADER.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((int(m.group(1)) if m.group(1).isdigit() else 0, m.group(2).strip(), m.group(3), text[m.start():end]))
    return out


def field(body, name):
    m = FIELD(name).search(body)
    return m.group(1).strip() if m else ""


def strip_numbers(s):
    """Remove anything that looks like a score. Coordinates only."""
    return NUMBER.sub("#", s) if re.search(r"bpb|delta|%", s, re.I) else s


def merge(root: Path):
    hyp = root / "hypotheses.md"
    text = hyp.read_text() if hyp.exists() else "# Hypotheses\n"
    existing = split_contracts(text)
    next_n = max([n for n, *_ in existing], default=0) + 1
    added = []
    for prop in sorted((root / "labs").glob("*/proposals.md")):
        lab = prop.parent.name
        ptext = prop.read_text()
        contracts = split_contracts(ptext)
        if not contracts:
            continue
        for _, title, status, body in contracts:
            body = HYP_HEADER.sub(f"## Hypothesis {next_n}: {title} | Status: PROPOSED", body, count=1)
            if not FIELD("Lab").search(body):
                body = body.replace("### Sprint Contract", f"### Sprint Contract\n\n**Lab:** {lab}", 1)
            text = text.rstrip() + "\n\n---\n\n" + body.strip() + "\n"
            added.append((next_n, lab, title))
            next_n += 1
        prop.write_text("")  # proposals consumed
    hyp.write_text(text)
    for n, lab, title in added:
        print(f"merged H{n} from {lab}: {title}")
    if not added:
        print("no proposals to merge")


def render(root: Path):
    hyp = root / "hypotheses.md"
    contracts = split_contracts(hyp.read_text()) if hyp.exists() else []
    labs_dir = root / "labs"
    labs = sorted(p.name for p in labs_dir.iterdir() if p.is_dir()) if labs_dir.exists() else []

    rows = []
    per_lab = {lab: [] for lab in labs}
    for n, title, status, body in contracts:
        lab = field(body, "Lab") or "unassigned"
        subsystem = field(body, "Subsystem")
        intervention = strip_numbers(field(body, "Intervention"))
        tested = {"PROPOSED": "proposed", "APPROVED": "queued", "REJECTED": "not run"}.get(status, "tested")
        rows.append(f"| H{n} | {lab} | {subsystem} | {intervention} | {tested} |")
        if lab in per_lab:
            per_lab[lab].append((n, title, status, subsystem, intervention))

    board = [
        "# Shared board",
        "",
        "What every lab has tried. Outcomes and scores are deliberately left out: "
        "use this to avoid duplicating work, not to decide where the winners are.",
        "",
        "| ID | Lab | Subsystem | Intervention | State |",
        "|----|-----|-----------|--------------|-------|",
        *rows,
        "",
    ]
    (root / "board.md").write_text("\n".join(board))

    for lab, items in per_lab.items():
        lines = [f"# {lab}: your own contracts and verdicts", "",
                 "Verdicts from the Evaluator for your lab only. PROMOTED means the change "
                 "survived the multi-seed audit and is now part of the shared train.py.", ""]
        for n, title, status, subsystem, intervention in items:
            lines.append(f"- H{n} [{status}] {title} ({subsystem}): {intervention}")
        (labs_dir / lab / "own.md").write_text("\n".join(lines) + "\n")
    print(f"board.md: {len(rows)} entries; own.md for {len(per_lab)} labs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["merge", "render"])
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    {"merge": merge, "render": render}[a.cmd](Path(a.root))
