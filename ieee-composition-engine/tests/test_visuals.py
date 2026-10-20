"""
Tests for Step 6: visual extraction, contextual placement, compliance score.

Run with:   pytest -q
"""

from __future__ import annotations

import copy
import os
import sys

import docx
import pymupdf
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compose import NoLLM, compose                                  # noqa: E402
from src.evaluate import evaluate                                       # noqa: E402
from src.ingest import load_document                                    # noqa: E402
from src.models import Element, Section, StructuredPaper                # noqa: E402
from src.placement import mentions, place_visuals                       # noqa: E402
from src.render import render_docx                                      # noqa: E402
from src.structure import map_to_ieee, structure_document               # noqa: E402
from src.visuals import export_visuals                                  # noqa: E402

SAMPLE_PDF = os.path.join("data", "input", "sample_paper.pdf")


# ---------------------------------------------------------------------------
# Finding mentions in the text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("as shown in Fig. 3", [("figure", 3)]),
    ("see Figure 2 and Fig. 5", [("figure", 2), ("figure", 5)]),
    ("Figs. 3 and 4 compare", [("figure", 3), ("figure", 4)]),
    ("Figs. 3–5 show", [("figure", 3), ("figure", 4), ("figure", 5)]),
    ("results in Table II and Table 4", [("table", 2), ("table", 4)]),
    ("Tables II and III", [("table", 2), ("table", 3)]),
    ("Fig. 1a shows", [("figure", 1)]),
    ("following Algorithm 1", [("algorithm", 1)]),
    ("no references here", []),
])
def test_mentions(text, expected):
    assert mentions(text) == expected


# ---------------------------------------------------------------------------
# Contextual placement on a tiny hand-made paper
# ---------------------------------------------------------------------------

def _paper(*items) -> StructuredPaper:
    paper = StructuredPaper(source_path="x")
    section = Section(heading="Results", number="1", level=1)
    for kind, text in items:
        section.elements.append(Element(kind=kind, text=text))
    paper.sections.append(section)
    return paper


def test_figure_floated_before_its_mention_is_moved_after_it():
    paper = _paper(("caption", "Figure 1. The pipeline."),
                   ("paragraph", "Some introduction text that does not mention it."),
                   ("paragraph", "The pipeline is shown in Fig. 1 below."),
                   ("paragraph", "More text."))
    report = place_visuals(paper)
    order = [e.text[:12] for e in paper.sections[0].elements]
    assert order == ["Some introdu", "The pipeline", "Figure 1. Th", "More text."]
    assert report[0]["distance_before"] < 0 and report[0]["distance_after"] == 0
    assert paper.placement_summary["after"]["after_first_mention"] == 100


def test_visual_is_not_inserted_between_a_paragraph_and_its_equation():
    paper = _paper(("caption", "Figure 1. Loss curve."),
                   ("paragraph", "As Fig. 1 shows, the loss is:"),
                   ("equation", "L = a + b (1)"),
                   ("paragraph", "where a is the Dice term."),
                   ("paragraph", "Next paragraph."))
    place_visuals(paper)
    kinds = [e.kind for e in paper.sections[0].elements]
    assert kinds == ["paragraph", "equation", "paragraph", "caption", "paragraph"]


def test_unmentioned_table_uses_semantic_match_or_stays():
    paper = _paper(("paragraph", "We tuned the learning rate and batch size carefully."),
                   ("paragraph", "Unrelated discussion of hospitals and ethics."),
                   ("caption", "Table 1: Learning rate and batch size settings."))
    report = place_visuals(paper)
    assert report[0]["method"].startswith("semantic match")
    assert paper.sections[0].elements[1].kind == "caption"     # after the tuning paragraph


def test_no_confident_match_keeps_the_visual_in_place():
    paper = _paper(("paragraph", "Completely different words entirely."),
                   ("caption", "Figure 9: Quantum chromodynamics lattice."))
    report = place_visuals(paper)
    assert report[0]["method"].startswith("kept in place")


# ---------------------------------------------------------------------------
# The sample paper end to end: extraction, placement, score
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    if not os.path.exists(SAMPLE_PDF):
        pytest.skip("run tools/make_sample_paper.py first")
    out = tmp_path_factory.mktemp("out")
    doc = load_document(SAMPLE_PDF)
    paper = map_to_ieee(structure_document(doc))
    source = copy.deepcopy(paper)
    compose(paper, NoLLM())
    place_visuals(paper)
    exported = export_visuals(paper, SAMPLE_PDF, str(out / "assets"))
    docx_path = str(out / "paper.docx")
    render_docx(paper, docx_path)
    report = evaluate(doc, source, paper, docx_path)
    return paper, exported, docx_path, report


def test_figure_and_table_regions_are_found(pipeline):
    paper, exported, _, _ = pipeline
    assert paper.visual_stats["found"] == 2
    assert exported["figures"] == 1 and exported["tables"] == 1
    figure = next(e for s in paper.sections for e in s.elements if e.text.startswith("Figure 1"))
    width = figure.region[2] - figure.region[0]
    assert width > 300                       # the whole block diagram, not one box


def test_cropped_images_exist_and_are_in_the_docx(pipeline):
    paper, _, docx_path, _ = pipeline
    images = [e.image for s in paper.sections for e in s.elements if e.image]
    assert images and all(os.path.exists(i) for i in images)
    assert len(docx.Document(docx_path).inline_shapes) == len(images)


def test_sample_figure_moved_after_its_mention(pipeline):
    paper, _, _, _ = pipeline
    fig = next(r for r in paper.placement_report if r["label"] == "Fig. 1")
    assert fig["distance_before"] < 0                         # floated above its mention
    assert fig["distance_after"] == 0
    architecture = next(s for s in paper.sections if s.heading == "Network Architecture")
    kinds = [e.kind for e in architecture.elements]
    assert kinds == ["paragraph", "equation", "paragraph", "caption"]


def test_compliance_score_improves(pipeline):
    _, _, _, report = pipeline
    assert report.before_score < 30
    assert report.after_score > 90
    by_key = {c.key: c for c in report.checks}
    assert by_key["citation_style"].before == 0 and by_key["citation_style"].after == 1
    assert by_key["placement"].after == 1
    assert by_key["abstract_length"].after == 0            # 83 words - honestly not fixed


def test_every_check_explains_itself(pipeline):
    _, _, _, report = pipeline
    assert all(c.after_reason for c in report.checks)
    assert len(report.checks) == 24                        # 23 scored + 1 advisory
    assert all(c.rules for c in report.checks)             # every check names its rulebook rules
    assert [c.key for c in report.checks if not c.scored] == ["acronyms"]


def test_text_inside_a_figure_is_removed_from_the_body(pipeline):
    paper, _, _, _ = pipeline
    body = " ".join(e.text for s in paper.sections for e in s.elements if e.kind == "paragraph")
    assert "Encoder (5 stages)" not in body


def test_side_by_side_figures_each_get_their_own_area(tmp_path):
    """Three charts in one row, three captions: no chart may be claimed twice."""
    path = str(tmp_path / "row.pdf")
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(pymupdf.Rect(72, 60, 540, 120),
                        "Body text before the figures runs across the whole width of "
                        "the page so that it is clearly ordinary prose for the engine.",
                        fontsize=10, fontname="tiro")
    for i, x in enumerate((72, 236, 400)):
        page.draw_rect(pymupdf.Rect(x, 140, x + 140, 240), color=(0, 0, 1), width=1)
        page.draw_line((x + 10, 230), (x + 130, 150), color=(1, 0, 0), width=1)
        page.insert_textbox(pymupdf.Rect(x, 250, x + 140, 280), f"Figure {i + 1}: Chart {i + 1}.",
                            fontsize=9, fontname="tiro")
    page.insert_textbox(pymupdf.Rect(72, 300, 540, 360),
                        "Body text after the figures also runs across the whole width of the "
                        "page and mentions Fig. 1, Fig. 2 and Fig. 3 for completeness here.",
                        fontsize=10, fontname="tiro")
    doc.save(path)
    paper = structure_document(load_document(path))
    regions = [e.region for s in paper.sections for e in s.elements if e.kind == "caption"]
    assert len(regions) == 3 and all(regions)
    for a, b in zip(regions, regions[1:]):
        assert a[2] <= b[0] + 1                 # no overlap between neighbours
