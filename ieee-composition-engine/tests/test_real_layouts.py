"""
Tests for layouts found in real papers (first found in arXiv 2610.20597v1,
an ICLR-style paper). Each test rebuilds one tricky pattern in miniature.

Run with:   pytest -q
"""

from __future__ import annotations

import os
import sys

import pymupdf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import load_document                                   # noqa: E402
from src.models import BlockType, ParsedDocument, TextBlock            # noqa: E402
from src.structure import map_to_ieee, structure_document              # noqa: E402
from src.structure.assembler import _split_authors                     # noqa: E402
from src.structure.classifier import classify, is_math_fragment        # noqa: E402


def _block(text, size=10.0, max_size=None, bold=False, y=0.0, page=1):
    return TextBlock(text=text, page=page, font_size=size, font_name="Times",
                     bold=bold, italic=False, x0=72, y0=y, x1=500, y1=y + 10,
                     max_font_size=max_size or size)


def _iclr_paper() -> ParsedDocument:
    """A miniature ICLR-style paper, line by line."""
    lines = [
        _block("CO-EVOLVING ACTING AND FORECASTING", 13.8, 17.2),
        _block("FOR LANGUAGE AGENTS", 13.8, 17.2),
        _block("Haofei Yu1 ∗", bold=True), _block("Yixing Chen2 ∗", bold=True),
        _block("1University of Illinois Urbana-Champaign"),
        _block("3Meta AI"),
        _block("ABSTRACT", 9.6, 12.0),
        _block("Training language agents is expensive and slow to do well today."),
        _block("We propose a new method that learns from every single step."),
        _block("1 INTRODUCTION", 9.6, 12.0),                     # small caps, not bold
        _block("Large language models enable autonomous agents to act."),
        _block("COPE outperforms observation-based auxiliary", 10.1, bold=True),
        _block("losses on aggregate metrics. On the other hand, it is fast."),
        _block("Success-conditioned action chunk. Fix a task and a policy.", bold=True),
        _block("h P"), _block("= EPr+"), _block("t CE"),          # scraps of a formula
        _block("5 COPE: CO-EVOLVING ACTING AND FORECASTING", 9.6, 12.0),
        _block("Our method trains a single network to act and forecast."),
        _block("5.1 SHARING PARAMETERS", 8.0, 10.0),
        _block("Both heads share one backbone network for efficiency."),
        _block("10 CONCLUSION", 9.6, 12.0),
        _block("We presented a method. Future work will examine more tasks."),
        _block("ETHICS STATEMENT", 9.6, 12.0),
        _block("This work uses only public benchmark environments."),
        _block("REFERENCES", 9.6, 12.0),
        _block("Victor Barres, Honghua Dong, and Karthik Narasimhan. A paper. 2025."),
        _block("A ARTIFACT DETAILS", 9.6, 12.0),
        _block("A.1 MODEL LICENSE", 8.0, 10.0),
        _block("All models are used under their licences."),
    ]
    for i, b in enumerate(lines):
        b.y0, b.y1 = 40 + i * 14, 50 + i * 14
    doc = ParsedDocument(source_path="iclr", source_format="pdf", page_count=1, blocks=lines)
    doc.measure_body_font()
    return doc


def test_iclr_structure():
    paper = map_to_ieee(structure_document(_iclr_paper()))

    assert paper.title == "CO-EVOLVING ACTING AND FORECASTING FOR LANGUAGE AGENTS"
    assert paper.authors == ["Haofei Yu", "Yixing Chen"]
    assert "3Meta AI" in paper.affiliations
    assert paper.abstract.startswith("Training language agents")
    assert "Large language models" not in paper.abstract           # abstract stops at 1 INTRODUCTION

    outline = [(s.number, s.level, s.ieee_heading) for s in paper.sections]
    assert outline == [
        ("1", 1, "Introduction"),
        ("5", 1, "Methodology"),                 # heading repeats the paper's title
        ("5.1", 2, "Sharing Parameters"),
        ("10", 1, "Conclusion"),
        ("", 1, "Ethics Statement"),             # unnumbered statement, NOT under Conclusion
        ("A", 1, "Appendix A: Artifact Details"),
        ("A.1", 2, "Model License"),
    ]
    assert paper.sections[4].ieee_number == ""
    assert len(paper.references) == 1                               # appendix not swallowed


def test_bold_run_in_labels_are_body_text():
    doc = _iclr_paper()
    classify(doc)
    label = {b.text: b.block_type for b in doc.blocks}
    assert label["COPE outperforms observation-based auxiliary"] is BlockType.BODY
    assert label["Success-conditioned action chunk. Fix a task and a policy."] is BlockType.BODY


def test_formula_scraps_become_one_placeholder_region():
    paper = structure_document(_iclr_paper())
    intro = paper.sections[0]
    kinds = [e.kind for e in intro.elements]
    assert kinds.count("math_region") == 1
    assert not any(e.text == "h P" for e in intro.elements if e.kind == "paragraph")


def test_math_fragment_detection():
    assert is_math_fragment(_block("q(at:t+K | ht) ≜ Pr+(at:t+K | ht),"))
    assert is_math_fragment(_block("h P"))
    assert not is_math_fragment(_block("GRPO"))                     # a table cell
    assert not is_math_fragment(_block("Collect a group of trajectories"))


def test_author_lines_with_markers():
    assert _split_authors(["Haofei Yu1 ∗", "Yixing Chen2 ∗"]) == ["Haofei Yu", "Yixing Chen"]
    assert _split_authors(["K. M. Perera, S. Fernando and A. Jayasuriya"]) == \
        ["K. M. Perera", "S. Fernando", "A. Jayasuriya"]


def test_split_section_number_is_reattached(tmp_path):
    """'1' and 'INTRODUCTION' stored as separate pieces on the same line."""
    path = str(tmp_path / "split.pdf")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 200), "1", fontsize=12, fontname="tiro")
    page.insert_text((95, 200), "INTRODUCTION", fontsize=10, fontname="tiro")
    page.insert_text((72, 230), "Body text of the introduction goes here.", fontsize=10,
                     fontname="tiro")
    doc.save(path)
    texts = [b.text for b in load_document(path).blocks]
    assert texts[0] == "1 INTRODUCTION"
    assert "1" not in texts


def test_small_typewriter_text_is_code_not_junk():
    """Appendix prompt listings are printed small in a typewriter font. They
    are real content and must never be thrown away as figure lettering."""
    doc = ParsedDocument(source_path="x", source_format="pdf", page_count=1, blocks=[
        _block("Body text of the paper that sets the normal font size here.", y=40),
        _block("Body text of the paper that sets the normal font size again.", y=54),
        TextBlock(text='"description": "Login to your account.",', page=1, font_size=7.0,
                  font_name="NimbusMonL-Regu", bold=False, italic=False,
                  x0=72, y0=70, x1=300, y1=78, max_font_size=7.0),
        _block("73.5", size=5.0, y=90),                         # chart lettering
    ])
    doc.measure_body_font()
    classify(doc)
    assert doc.blocks[2].block_type is BlockType.CODE
    assert doc.blocks[3].block_type is BlockType.JUNK
