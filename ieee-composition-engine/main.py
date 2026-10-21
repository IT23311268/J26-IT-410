"""
main.py - the command you run.

Runs the stages built so far:
    stage 1  ingest    - read the file into lines with layout clues
    stage 2  structure - label every line, then rebuild the paper
    stage 3  map       - match the author's sections onto the IEEE skeleton
    stage 5  compose   - IEEE references and citations, wording rules, index terms,
                         abstract (Gemini, checked), optional polish
    stage 6  visuals   - place figures/tables after their first mention, crop
                         them (and broken equations) from the PDF as images
    stage 4  render    - write the two-column IEEE Word document
    score    evaluate  - IEEE compliance score, before vs. after
(Stages 5 and 6 run before stage 4: the content is finished first, then printed.)

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

from src.pipeline import PackageError, run_pipeline
from src.render import ieee_style as S

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
# Printing stage 6 and the compliance score
# ---------------------------------------------------------------------------

def print_stage6(paper, exported) -> None:
    banner("STAGE 6 - FIGURES, TABLES AND EQUATIONS")
    found = paper.visual_stats or {}
    if found.get("captions"):
        print(f"  Visual areas found on the pages : {found['found']} of {found['captions']} captions")
    images = ", ".join(f"{v} {k}" for k, v in exported.items() if v)
    print(f"  Images cropped from the source  : {images or 'none'}")

    report = paper.placement_report
    if not report:
        print("  Contextual placement            : no figures or tables to place")
        return
    print()
    print(f"  {'Visual':<12} {'Placed':<44} {'distance':>12}")
    print(f"  {'-'*12} {'-'*44} {'-'*12}")
    for r in report:
        before = "-" if r["distance_before"] is None else r["distance_before"]
        after = "-" if r["distance_after"] is None else r["distance_after"]
        print(f"  {r['label']:<12} {short(r['method'], 44):<44} {str(before):>5} -> {str(after):<4}")
    print("  (distance = paragraphs between a visual and its first mention;")
    print("   0 = directly after it, negative = placed BEFORE it)")

    summary = paper.placement_summary
    b, a = summary.get("before", {}), summary.get("after", {})
    if b.get("mean_distance") is not None:
        print()
        print(f"  {'':<34} {'before':>8} {'after':>8}")
        print(f"  {'Mean distance to first mention':<34} {b['mean_distance']:>8} {a['mean_distance']:>8}")
        print(f"  {'Placed after first mention':<34} {str(b['after_first_mention']) + '%':>8} "
              f"{str(a['after_first_mention']) + '%':>8}")
        print(f"  {'Directly after first mention':<34} {str(b['adjacent']) + '%':>8} "
              f"{str(a['adjacent']) + '%':>8}")


def print_compliance(report) -> None:
    banner("IEEE COMPLIANCE SCORE")
    print()
    print(f"  {'Check':<44} {'rules':<16} {'before':>7} {'after':>7}")
    print(f"  {'-'*44} {'-'*16} {'-'*7} {'-'*7}")
    group = None
    for c in report.checks:
        if c.group != group:
            group = c.group
            print(f"  {group}")
        fmt = lambda v: "n/a" if v is None else f"{round(v * 100)}%"
        rules = short(",".join(c.rules), 16)
        name = c.name if c.scored else c.name + " *"
        print(f"    {short(name, 42):<42} {rules:<16} {fmt(c.before):>7} {fmt(c.after):>7}")
    print(f"  {'-'*44} {'-'*16} {'-'*7} {'-'*7}")
    before = "n/a" if report.before_score is None else f"{report.before_score}%"
    after = "n/a" if report.after_score is None else f"{report.after_score}%"
    print(f"  {'OVERALL':<61} {before:>7} {after:>7}")
    print("  * reported only, not in the score. Rule IDs: see src/rules/ieee_rules.py")
    weak = [c for c in report.checks if c.after is not None and c.after < 1]
    if weak:
        print()
        print("  Not yet fully compliant:")
        for c in weak:
            print(f"    - {c.name} ({', '.join(c.rules)}): {c.after_reason}")


def print_rubric(r) -> None:
    from src.rubric import rubric_score, summary
    banner("IEEE RUBRIC - EVERY RULE, AND WHERE IT WAS APPLIED")
    counts = summary(r.rubric)
    ticked, applicable, pct = rubric_score(r.rubric)
    print(f"  Rules that apply to this paper : {applicable} of {len(r.rubric)}")
    print(f"    applied by the engine        : {counts['applied']}")
    print(f"    already followed             : {counts['met']}")
    print(f"    partly / not followed        : {counts['partly'] + counts['not_met']}")
    print(f"    for a person to check        : {counts['review']}")
    print(f"  Suggested rubric score         : {pct}%  ({ticked}/{applicable} ticked)")
    print()
    print("  Examples of where rules were applied:")
    shown = 0
    for item in r.rubric:
        placed = [e for e in item.evidence if e.get("anchor") != "document"]
        if item.state == "applied" and placed and shown < 8:
            e = placed[0]
            change = f"'{short(e['before'], 28)}' -> '{short(e['after'], 28)}'" if e["before"] \
                else short(e["after"], 50)
            print(f"    {item.id:<7} {short(e['where'], 30):<30} {change}")
            shown += 1
    print()
    print("  Tick the rubric on the web page (streamlit run app.py), or open")
    print(f"  {os.path.basename(r.annotated_path)} in Word to see every rule in the margin.")


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
    parser.add_argument("--paper", default="letter", choices=["letter", "a4"],
                        help="paper size from the IEEE rulebook (rule LAY-01): letter or a4")
    args = parser.parse_args()

    def show(stage: str, r) -> None:
        """Called by the pipeline after each stage - prints that stage."""
        if stage == "ingest" and r.is_package:
            banner("STAGES 1-2 - VERIFIED CONTENT PACKAGE RECEIVED")
            print(f"  File     : {r.input_path}")
            print(f"  Title    : {short(r.paper.title, 60)}")
            print(f"  Sections : {len(r.paper.sections)}   "
                  f"References: {len(r.paper.references)}")
        elif stage == "ingest":
            print_stage1(r.source_doc)
        elif stage == "structure" and not r.is_package:
            print_labelled_lines(r.source_doc, args.lines)
            print_structure(r.paper)
        elif stage == "map":
            print_stage3(r.paper)
        elif stage == "compose":
            print_stage5(r.paper, r.llm)
        elif stage == "visuals":
            print_stage6(r.paper, r.exported)
        elif stage == "render":
            banner("STAGE 4 - IEEE WORD DOCUMENT")
            print(f"  Written to : {r.docx_path}")
            print(f"  Layout     : {S.PAPER.name}, two columns, {S.FONT}, IEEE margins "
                  "(values from src/rules/ieee_rules.py)")
            if r.render_notes:
                print()
                print("  Notes:")
                for note in r.render_notes:
                    for i, line in enumerate(textwrap.wrap(note, WIDTH - 8)):
                        print(("    - " if i == 0 else "      ") + line)
        elif stage == "score":
            print_compliance(r.compliance)
            print_rubric(r)

    # All the stages live in src/pipeline.py - the web page (app.py) uses the same code.
    try:
        result = run_pipeline(args.input_file, outdir=args.outdir, offline=args.offline,
                              polish=args.polish, on_stage=show, paper_size=args.paper)
    except (FileNotFoundError, ValueError, PackageError) as error:
        print(f"\n  ERROR: {error}\n")
        return 1
    except PermissionError as error:
        print(f"\n  ERROR: cannot write {error.filename or 'the Word file'}")
        print("  Is it open in Word? Close it and run the command again.\n")
        return 1

    print()
    print("  Saved:")
    for path, what in result.saved:
        print(f"    {path}   ({what})")
    print(f"    {result.docx_path}   (OPEN THIS ONE IN WORD)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
