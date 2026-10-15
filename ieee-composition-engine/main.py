"""
main.py - the command you run.

Runs the stages built so far:
    stage 1  ingest    - read the file into lines with layout clues
    stage 2  structure - label every line, then rebuild the paper
    stage 3  map       - match the author's sections onto the IEEE skeleton

Usage:
    python main.py data/input/sample_paper.pdf
    python main.py data/input/sample_paper.pdf --lines 30     (also show labelled lines)
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

from src.ingest import load_document
from src.structure import map_to_ieee, structure_document

WIDTH = 78


def banner(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Printing stage 1
# ---------------------------------------------------------------------------

def print_stage1(doc) -> None:
    banner("STAGE 1 - INGEST")
    print(f"  File            : {doc.source_path}")
    print(f"  Format          : {doc.source_format}")
    print(f"  Pages           : {doc.page_count}")
    print(f"  Lines extracted : {len(doc.blocks)}")
    print(f"  Body font size  : {doc.body_font_size} pt")


# ---------------------------------------------------------------------------
# Printing stage 2
# ---------------------------------------------------------------------------

def print_labelled_lines(doc, limit: int) -> None:
    if limit <= 0:
        return
    print()
    print(f"  Each line with the label the classifier gave it (first {min(limit, len(doc.blocks))}):")
    print()
    print(f"  {'#':>3}  {'size':>5}  {'label':<16}  text")
    print(f"  {'-'*3}  {'-'*5}  {'-'*16}  {'-'*45}")
    for i, b in enumerate(doc.blocks[:limit], start=1):
        print(f"  {i:>3}  {b.font_size:>5}  {b.block_type.value:<16}  {short(b.text, 45)}")


def print_structure(paper) -> None:
    print()
    print("  Label counts:")
    for label, count in paper.label_counts.items():
        print(f"    {label:<18} {count:>3}")

    banner("STAGE 2 - THE PAPER, REBUILT")

    def field(name: str, value: str) -> None:
        wrapped = textwrap.wrap(value or "(none found)", WIDTH - 18)
        print(f"  {name:<14}: {wrapped[0]}")
        for line in wrapped[1:]:
            print(f"  {'':<14}  {line}")

    field("Title", paper.title)
    field("Authors", " | ".join(paper.authors))
    field("Affiliations", " | ".join(paper.affiliations))
    field("Abstract", short(paper.abstract, 220))
    print(f"  {'':<14}  ({len(paper.abstract.split())} words)")
    field("Keywords", " | ".join(paper.keywords))

    print()
    print("  Sections:")
    for s in paper.sections:
        indent = "    " * s.level
        name = f"{s.number} {s.heading}".strip() or "(text before first heading)"
        counts = []
        paragraphs = len(s.paragraphs)
        captions = sum(1 for e in s.elements if e.kind == "caption")
        equations = sum(1 for e in s.elements if e.kind == "equation")
        if paragraphs:
            counts.append(f"{paragraphs} para")
        if captions:
            counts.append(f"{captions} caption")
        if equations:
            counts.append(f"{equations} equation")
        detail = ", ".join(counts) or "empty"
        print(f"  {indent}{short(name, 40):<40}  [{detail}, {s.word_count} words]")

    print()
    print(f"  References    : {len(paper.references)} found")
    for i, ref in enumerate(paper.references, start=1):
        print(f"    {i:>2}. {short(ref, 66)}")



# ---------------------------------------------------------------------------
# Printing stage 3
# ---------------------------------------------------------------------------

def print_stage3(paper) -> None:
    banner("STAGE 3 - MAPPED TO THE IEEE SKELETON")
    print()
    print(f"  {'Author wrote':<30}    {'IEEE heading':<28}  {'sure?':>5}")
    print(f"  {'-'*30}    {'-'*28}  {'-'*5}")
    for s in paper.sections:
        indent = "  " * (s.level - 1)
        original = indent + (f"{s.number} {s.heading}".strip() or "(untitled text)")
        ieee = indent + f"{s.ieee_number}. {s.ieee_heading}".lstrip(". ")
        if s.level == 1:
            ieee = indent + (f"{s.ieee_number}. " if s.ieee_number else "") + s.ieee_heading.upper()
            sure = f"{int(s.mapping_confidence * 100)}%"
        else:
            ieee = indent + f"{s.ieee_number}. {s.ieee_heading}"
            sure = ""
        print(f"  {short(original, 30):<30} -> {short(ieee, 28):<28}  {sure:>5}")

    print()
    print("  Why each section was mapped where it was:")
    for s in paper.sections:
        if s.level == 1:
            name = s.heading or "(untitled text)"
            print(f"    {short(name, 24):<24} {s.ieee_slot:<14} because {s.mapping_reason}")

    print()
    if paper.missing_sections:
        print("  Missing IEEE sections : " + ", ".join(paper.missing_sections))
    else:
        print("  Missing IEEE sections : none - every required section is present")

    if paper.warnings:
        print()
        print("  Warnings:")
        for w in paper.warnings:
            print(f"    ! {w}")


# ---------------------------------------------------------------------------

def main() -> int:
    # Windows terminals default to an old text encoding that cannot print
    # characters like "—" or "≤", which real papers are full of. This line
    # stops Python from crashing on them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="IEEE Manuscript Composition Engine")
    parser.add_argument("input_file", help="path to a .pdf or .docx file")
    parser.add_argument("--lines", type=int, default=0,
                        help="how many labelled lines from stage 2 to print (default: none)")
    parser.add_argument("--outdir", default="data/output",
                        help="where to write the JSON results")
    args = parser.parse_args()

    try:
        doc = load_document(args.input_file)
    except (FileNotFoundError, ValueError) as error:
        print(f"\n  ERROR: {error}\n")
        return 1

    print_stage1(doc)

    paper = structure_document(doc)          # stage 2: labels + rebuild
    print_labelled_lines(doc, args.lines)
    print_structure(paper)

    paper = map_to_ieee(paper)               # stage 3: IEEE skeleton
    print_stage3(paper)

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input_file))[0]
    lines_path = os.path.join(args.outdir, f"{stem}_stage2_labelled_lines.json")
    paper_path = os.path.join(args.outdir, f"{stem}_stage3_ieee_paper.json")
    doc.to_json(lines_path)
    paper.to_json(paper_path)

    print()
    print("  Saved:")
    print(f"    {lines_path}   (every line with its label)")
    print(f"    {paper_path}   (the paper, mapped to IEEE)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
