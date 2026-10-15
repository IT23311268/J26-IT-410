"""
Tests for stage 1.

Run them with:   pytest -q

These are not decoration. When you change the reader on day 3 to also pull out
images, these tests tell you in one second whether you broke the text path.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import load_document                     # noqa: E402
from src.models import BlockType, ParsedDocument, TextBlock   # noqa: E402

SAMPLE_PDF = os.path.join("data", "input", "sample_paper.pdf")


def test_rejects_unknown_file_type(tmp_path):
    bad = tmp_path / "notes.rtf"
    bad.write_text("hello")
    with pytest.raises(ValueError):
        load_document(str(bad))


def test_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        load_document("data/input/does_not_exist.pdf")


@pytest.mark.skipif(not os.path.exists(SAMPLE_PDF),
                    reason="run tools/make_sample_paper.py first")
def test_reads_the_sample_paper():
    doc = load_document(SAMPLE_PDF)

    assert doc.source_format == "pdf"
    assert doc.page_count >= 1
    assert len(doc.blocks) > 20

    # The body size is the most common size, so it must be smaller than the
    # title. If this ever fails, the font measurement is broken.
    assert doc.body_font_size > 0
    largest = max(b.font_size for b in doc.blocks)
    assert largest > doc.body_font_size

    # Stage 1 must not guess labels - that is stage 2's job.
    assert all(b.block_type is BlockType.UNKNOWN for b in doc.blocks)


def test_upper_case_detection():
    block = TextBlock(text="ABSTRACT", page=1, font_size=11, font_name="x",
                      bold=True, italic=False, x0=0, y0=0, x1=0, y1=0)
    assert block.is_upper
    assert block.word_count == 1

    sentence = TextBlock(text="Lung cancer remains", page=1, font_size=10,
                         font_name="x", bold=False, italic=False,
                         x0=0, y0=0, x1=0, y1=0)
    assert not sentence.is_upper
    assert sentence.word_count == 3


def test_body_font_is_measured_by_characters():
    """Many short big-font lines must not outvote the real body text."""
    doc = ParsedDocument(source_path="x", source_format="pdf", page_count=1)
    for _ in range(5):
        doc.blocks.append(TextBlock(text="Heading", page=1, font_size=16,
                                    font_name="x", bold=True, italic=False,
                                    x0=0, y0=0, x1=0, y1=0))
    doc.blocks.append(TextBlock(text="x" * 400, page=1, font_size=10,
                                font_name="x", bold=False, italic=False,
                                x0=0, y0=0, x1=0, y1=0))
    assert doc.measure_body_font() == 10.0


# ---------------------------------------------------------------------------
# Problems found on a real arXiv paper (2610.20597v1)
# ---------------------------------------------------------------------------

import pymupdf  # noqa: E402


def _tight_pdf(path):
    """A page with the three problems the real arXiv paper had."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    # 1. arXiv prints its identifier sideways down the left margin
    page.insert_text((30, 600), "arXiv:2610.20597v1 [cs.AI] 14 Oct 2026",
                     fontsize=20, fontname="tiro", rotate=90)
    # 2. LaTeX-style tight justification: words 1pt apart, no space characters
    x = 72
    for word in ["Action", "forecasting", "fits", "the", "multi-step"]:
        page.insert_text((x, 100), word, fontsize=10, fontname="tiro")
        x += pymupdf.get_text_length(word, "tiro", 10) + 1.0
    # 3. an invisible control character inside the text
    page.insert_text((72, 130), "bad\x01char here", fontsize=10, fontname="tiro")
    doc.save(path)


def test_real_paper_problems_are_fixed(tmp_path):
    path = str(tmp_path / "tight.pdf")
    _tight_pdf(path)
    texts = [b.text for b in load_document(path).blocks]

    assert not any("arXiv:" in t for t in texts)                 # sideways text skipped
    assert "Action forecasting fits the multi-step" in texts     # spaces restored
    assert "badchar here" in texts                               # control char removed


def test_letters_inside_a_word_are_not_split(tmp_path):
    path = str(tmp_path / "normal.pdf")
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "Convolutional networks", fontsize=10, fontname="tiro")
    doc.save(path)
    assert load_document(path).blocks[0].text == "Convolutional networks"
