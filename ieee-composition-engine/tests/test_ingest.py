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
