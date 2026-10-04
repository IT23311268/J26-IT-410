"""Run Box 3 over a folder of papers and report what looks wrong.

Why this exists
---------------
Every threshold in `ingestion/layout.py` was measured on one or two real
pages. That is the honest way to pick a number, but it is not evidence
that the number holds across a corpus — and "it worked on the paper I
tested it on" is exactly the answer a panel will not accept.

So this script does not ask whether the output is *right*; nothing here
knows the correct answer. It looks for the specific shapes that every
bug found so far produced, and prints the pages carrying them. A page
flagged here is a page worth opening in the /layout overlay. A clean
run across fifteen papers is the evidence that the rules generalise.

Usage
-----
    python scripts/check_corpus.py data/raw
    python scripts/check_corpus.py data/raw --verbose
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.layout import containment_ratio, extract_regions  # noqa: E402
from schema.ingestion_schema_v1 import LayoutRegion, RegionType  # noqa: E402

# A chart or a table that was not absorbed scatters into many very short
# BODY regions — "Telugu", "25.0%", "0-shot". That signature is what
# both the axis-label bug and the table-cell bug looked like from the
# outside, so it is the first thing to look for.
#
# Short alone is not enough: a page can legitimately carry short lines
# (a list, a footnote, a page number). What marks the scatter is short
# text *crowded around an artifact*, so the proximity is part of the
# test. Without it the check needed a count so high it missed the very
# page it was written for, which carried six stray labels rather than
# the seventy the bug was first noticed on.
SHORT_TEXT_CHARS = 25
NEAR_ARTIFACT = 30.0
MANY_SHORT_BODY = 4

# A table and a figure are different artifacts and go to Member 4
# separately. Overlapping boxes mean one detector claimed the other's
# territory — the bug where a table's rule chain ran into a chart.
ARTIFACT_OVERLAP = 0.5

# A region covering most of the page is either a scan (one figure, which
# is correct and expected) or something merged that should not have.
GIANT_REGION = 0.6


def _page_rect(regions: list[LayoutRegion]) -> tuple[float, float, float, float]:
    """The page's extent, inferred from the regions on it. Good enough
    for the area ratios below without re-opening the PDF."""
    return (
        min(r.bbox.x0 for r in regions),
        min(r.bbox.y0 for r in regions),
        max(r.bbox.x1 for r in regions),
        max(r.bbox.y1 for r in regions),
    )


def _area(box) -> float:
    return max(box.x1 - box.x0, 0.0) * max(box.y1 - box.y0, 0.0)


def _gap(a, b) -> float:
    """Distance between two region boxes in points; zero if they touch."""
    dx = max(b.x0 - a.x1, a.x0 - b.x1, 0.0)
    dy = max(b.y0 - a.y1, a.y0 - b.y1, 0.0)
    return max(dx, dy)


def check_page(page_no: int, regions: list[LayoutRegion]) -> list[str]:
    """Return the names of the problems visible on this one page."""
    problems: list[str] = []
    kinds = Counter(r.region_type for r in regions)

    if not any(r.text.strip() for r in regions):
        # Correct behaviour, not a bug: a scan has no text layer, so
        # geometry has nothing to read and Box 8 has to take over.
        problems.append("no-text-layer (scan? needs Box 8)")

    artifacts = [
        r for r in regions if r.region_type in (RegionType.FIGURE, RegionType.TABLE)
    ]

    stray = [
        r
        for r in regions
        if r.region_type == RegionType.BODY
        and len(r.text.strip()) <= SHORT_TEXT_CHARS
        and any(_gap(r.bbox, a.bbox) <= NEAR_ARTIFACT for a in artifacts)
    ]
    if len(stray) >= MANY_SHORT_BODY:
        problems.append(
            f"{len(stray)} tiny BODY regions around an artifact "
            "(chart labels or table cells not absorbed)"
        )

    for i, a in enumerate(artifacts):
        for b in artifacts[i + 1 :]:
            if a.region_type == b.region_type:
                continue
            box_a = (a.bbox.x0, a.bbox.y0, a.bbox.x1, a.bbox.y1)
            box_b = (b.bbox.x0, b.bbox.y0, b.bbox.x1, b.bbox.y1)
            if max(
                containment_ratio(box_a, box_b), containment_ratio(box_b, box_a)
            ) >= ARTIFACT_OVERLAP:
                problems.append("a TABLE and a FIGURE overlap (one swallowed the other)")
                break
        else:
            continue
        break

    if regions:
        px0, py0, px1, py1 = _page_rect(regions)
        page_area = max((px1 - px0) * (py1 - py0), 1.0)
        for r in regions:
            if r.region_type in (RegionType.FIGURE, RegionType.TABLE):
                continue
            if _area(r.bbox) / page_area >= GIANT_REGION:
                problems.append(f"a {r.region_type.value.upper()} region covers the page")
                break

    if kinds[RegionType.CAPTION] and not (
        kinds[RegionType.FIGURE] or kinds[RegionType.TABLE]
    ):
        problems.append("a caption with nothing to caption")

    return problems


def check_paper(pdf: Path, verbose: bool) -> tuple[Counter, int]:
    """Print one paper's report. Returns (region counts, problem count)."""
    try:
        regions = extract_regions(pdf, pdf.stem)
    except Exception as exc:  # a corrupt or encrypted PDF should not stop the run
        print(f"\n{pdf.name}\n  !! could not read: {exc}")
        return Counter(), 1

    by_page: dict[int, list[LayoutRegion]] = defaultdict(list)
    for region in regions:
        by_page[region.bbox.page].append(region)

    kinds = Counter(r.region_type.value for r in regions)
    flagged: list[tuple[int, list[str]]] = []
    for page_no in sorted(by_page):
        problems = check_page(page_no, by_page[page_no])
        if problems:
            flagged.append((page_no, problems))

    status = "ok" if not flagged else f"{len(flagged)} page(s) flagged"
    print(f"\n{pdf.name}")
    print(f"  {len(by_page)} pages, {len(regions)} regions — {status}")
    if verbose or flagged:
        counts = "  ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
        print(f"  {counts}")
    for page_no, problems in flagged:
        for problem in problems:
            print(f"    page {page_no}: {problem}")

    # No headings at all means Box 4 has nothing to build a section tree
    # from, which is worth knowing before Box 4 is written.
    if not kinds.get("heading"):
        print("    (no headings on any page — Box 4 would find no sections)")

    return kinds, len(flagged)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="a folder of PDFs")
    parser.add_argument("--verbose", action="store_true", help="counts for every paper")
    args = parser.parse_args()

    pdfs = sorted(p for p in args.folder.rglob("*.pdf"))
    if not pdfs:
        print(f"no PDFs under {args.folder}")
        return 1

    totals: Counter = Counter()
    flagged_papers = 0
    for pdf in pdfs:
        kinds, flagged = check_paper(pdf, args.verbose)
        totals.update(kinds)
        if flagged:
            flagged_papers += 1

    print("\n" + "=" * 58)
    print(f"{len(pdfs)} papers, {flagged_papers} with something to look at")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(totals.items())))
    print("\nOpen a flagged page in the overlay to see it:")
    print("  http://127.0.0.1:8000/paper/{paper_id}/page/{n}/layout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
