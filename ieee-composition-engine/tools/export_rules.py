"""
export_rules.py - write the IEEE rulebook as a readable table.

    python tools/export_rules.py

Reads src/rules/ieee_rules.py and writes docs/IEEE_RULES.md: every rule with
its ID, value, status, source, the file that applies it and the compliance
check that tests it. Run it again after changing the rulebook (a test fails
if the document is out of date).
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.rules import ieee_rules as R          # noqa: E402

OUT = os.path.join(ROOT, "docs", "IEEE_RULES.md")
STATUS = {R.ENFORCED: "Applied automatically", R.CHECKED: "Checked and reported",
          R.AUTHOR: "Author's judgement"}


def cell(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def build() -> str:
    counts = R.summary()
    lines = [
        "# IEEE Rulebook",
        "",
        "Every IEEE formatting rule the IEEE Manuscript Composition Engine follows.",
        "This file is generated from `src/rules/ieee_rules.py` by `python tools/export_rules.py`;",
        "edit the rulebook, not this file.",
        "",
        f"**{len(R.RULES)} rules:** {counts[R.ENFORCED]} applied automatically, "
        f"{counts[R.CHECKED]} checked and reported, {counts[R.AUTHOR]} left to the author's "
        "judgement.",
        "",
        "| Status | Meaning |",
        "|---|---|",
        "| Applied automatically | The engine changes the manuscript to follow the rule. |",
        "| Checked and reported | The engine measures the rule and reports it, but a program "
        "cannot fix it safely. |",
        "| Author's judgement | Written down for completeness; it needs a human decision. |",
        "",
        "## Sources",
        "",
    ]
    for key, (name, url) in R.SOURCES.items():
        lines.append(f"- **{key}**: [{name}]({url})")
    group = None
    for r in R.RULES:
        if r.group != group:
            group = r.group
            lines += ["", f"## {group}", "",
                      "| ID | Rule | Value | Status | Source | Applied in | Score check |",
                      "|---|---|---|---|---|---|---|"]
        source = r.source + (f": {r.section}" if r.section else "")
        lines.append(f"| {r.id} | {cell(r.rule)} | {cell(r.value)} | {STATUS[r.status]} | "
                     f"{cell(source)} | {cell(r.where and '`' + r.where + '`')} | "
                     f"{cell(r.check and '`' + r.check + '`')} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(build())
    counts = R.summary()
    print(f"Wrote {os.path.relpath(OUT, ROOT)}: {len(R.RULES)} rules "
          f"({counts[R.ENFORCED]} applied, {counts[R.CHECKED]} checked, "
          f"{counts[R.AUTHOR]} author's judgement)")


if __name__ == "__main__":
    main()
