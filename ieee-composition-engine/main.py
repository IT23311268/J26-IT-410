"""
main.py - the command you run.

Runs the stages built so far:
    stage 1  ingest    - read the file into lines with layout clues
    stage 2  structure - label every line, then rebuild the paper
    stage 3  map       - match the author's sections onto the IEEE skeleton
    stage 5  compose   - IEEE references and citations, wording rules, index terms,
                         abstract (Gemini, checked), optional polish
    stage 4  render    - write the two-column IEEE Word document
(Stage 5 runs before stage 4: the text is finished first, then printed.)

Input can be a paper (.pdf / .docx) OR a verified content package (.json) -
the format the other agents in the group pipeline hand over.

Usage:
    python main.py data/input/sample_paper.pdf
    python main.py data/input/sample_paper.pdf --offline      (no Gemini, rules only)
    python main.py data/input/sample_paper.pdf --polish       (Gemini rewording, checked)
    python main.py data/input/sample_verified_content.json    (content from other agents)
    python main.py data/input/sample_paper.pdf --lines 30     (also show labelled lines)
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

from src.compose import compose, make_llm
from src.ingest import PackageError, load_document, read_package
from src.render import render_docx
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
# Printing stage 5
# ---------------------------------------------------------------------------

def print_stage5(paper, llm) -> None:
    banner("STAGE 5 - IEEE COMPOSITION")
    print(f"  AI model   : {llm.name}")
    if not llm.available:
        print(f"  Why        : {getattr(llm, 'reason', '')}")
    print()
    for line in paper.compose_report:
        for i, part in enumerate(textwrap.wrap(line, WIDTH - 8)):
            print(("    - " if i == 0 else "      ") + part)
    if getattr(llm, "calls_made", 0) or getattr(llm, "cache_hits", 0):
        print()
        print(f"  Gemini requests: {llm.calls_made} new, {llm.cache_hits} reused from cache")
    if paper.ai_log:
        print()
        print("  Written or changed by AI / automatic extraction (for transparency):")
        for entry in paper.ai_log:
            for i, part in enumerate(textwrap.wrap(entry, WIDTH - 8)):
                print(("    * " if i == 0 else "      ") + part)
    if paper.references:
        print()
        print("  First references in IEEE style:")
        for i, ref in enumerate(paper.references[:3], start=1):
            for j, part in enumerate(textwrap.wrap(ref, WIDTH - 10)):
                print((f"    [{i}] " if j == 0 else "        ") + part)


# ---------------------------------------------------------------------------

def main() -> int:
    # Windows terminals default to an old text encoding that cannot print
    # characters like "—" or "≤", which real papers are full of. This line
    # stops Python from crashing on them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="IEEE Manuscript Composition Engine")
    parser.add_argument("input_file", help="a .pdf / .docx paper, or a .json content package")
    parser.add_argument("--lines", type=int, default=0,
                        help="how many labelled lines from stage 2 to print (default: none)")
    parser.add_argument("--outdir", default="data/output",
                        help="where to write the results")
    parser.add_argument("--offline", action="store_true",
                        help="do not use Gemini; rules only")
    parser.add_argument("--polish", action="store_true",
                        help="let Gemini reword paragraphs (every change is checked)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input_file))[0]
    lines_path = os.path.join(args.outdir, f"{stem}_stage2_labelled_lines.json")
    paper_path = os.path.join(args.outdir, f"{stem}_ieee_paper.json")
    saved = []

    if args.input_file.lower().endswith(".json"):
        # Route B: verified content handed over by the other agents (TAF task 1)
        if not os.path.exists(args.input_file):
            print(f"\n  ERROR: No such file: {args.input_file}\n")
            return 1
        try:
            paper = read_package(args.input_file)
        except PackageError as error:
            print(f"\n  ERROR: {error}\n")
            return 1
        banner("STAGES 1-2 - VERIFIED CONTENT PACKAGE RECEIVED")
        print(f"  File     : {args.input_file}")
        print(f"  Title    : {short(paper.title, 60)}")
        print(f"  Sections : {len(paper.sections)}   "
              f"References: {len(paper.references)}")
    else:
        # Route A: read the paper itself (stand-in for Member 1's ingestion)
        try:
            doc = load_document(args.input_file)
        except (FileNotFoundError, ValueError) as error:
            print(f"\n  ERROR: {error}\n")
            return 1
        print_stage1(doc)
        paper = structure_document(doc)          # stage 2: labels + rebuild
        print_labelled_lines(doc, args.lines)
        print_structure(paper)
        doc.to_json(lines_path)
        saved.append((lines_path, "every line with its label"))

    paper = map_to_ieee(paper)                   # stage 3: IEEE skeleton
    print_stage3(paper)

    llm = make_llm(offline=args.offline)         # stage 5: composition
    paper = compose(paper, llm, polish=args.polish)
    print_stage5(paper, llm)

    paper.to_json(paper_path)
    saved.append((paper_path, "the finished paper, as data"))

    # stage 4: the IEEE Word document
    docx_path = os.path.join(args.outdir, f"{stem}_IEEE.docx")
    try:
        render_notes = render_docx(paper, docx_path)
    except PermissionError:
        print(f"\n  ERROR: cannot write {docx_path}")
        print("  Is it open in Word? Close it and run the command again.\n")
        return 1

    banner("STAGE 4 - IEEE WORD DOCUMENT")
    print(f"  Written to : {docx_path}")
    print("  Layout     : US Letter, two columns, Times New Roman, IEEE margins")
    if render_notes:
        print()
        print("  Notes:")
        for note in render_notes:
            for i, line in enumerate(textwrap.wrap(note, WIDTH - 8)):
                print(("    - " if i == 0 else "      ") + line)

    print()
    print("  Saved:")
    for path, what in saved:
        print(f"    {path}   ({what})")
    print(f"    {docx_path}   (OPEN THIS ONE IN WORD)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
