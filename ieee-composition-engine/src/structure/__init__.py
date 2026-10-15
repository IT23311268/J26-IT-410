"""
The structure package - stages 2 and 3 of the pipeline.

Stage 2 (this week, day 2):
    classify()  - label every line (title, heading, body, caption, ...)
    assemble()  - glue labelled lines into paragraphs, sections, references

Stage 3 (day 3) will add the IEEE section mapper here too.
"""

from __future__ import annotations

from ..models import ParsedDocument, StructuredPaper
from .assembler import assemble
from .classifier import classify, label_counts


def structure_document(doc: ParsedDocument) -> StructuredPaper:
    """Run all of stage 2: classify the lines, then rebuild the paper."""
    doc, notes = classify(doc)
    return assemble(doc, label_counts(doc), notes)


__all__ = ["structure_document", "classify", "assemble", "label_counts"]
