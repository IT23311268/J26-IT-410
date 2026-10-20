"""
The structure package - stages 2 and 3 of the pipeline.

Stage 2:
    classify()     - label every line (title, heading, body, caption, ...)
    assemble()     - glue labelled lines into paragraphs, sections, references
Stage 3:
    map_to_ieee()  - match the author's sections onto the IEEE skeleton
"""

from __future__ import annotations

from ..models import ParsedDocument, StructuredPaper
from .assembler import assemble
from .classifier import classify, label_counts
from .ieee_mapper import map_to_ieee


def structure_document(doc: ParsedDocument) -> StructuredPaper:
    """Run stage 2: classify the lines, find figure areas, then rebuild the paper."""
    from ..visuals import detect_regions          # imported here to avoid a circular import

    doc, notes = classify(doc)
    visual_stats = detect_regions(doc)            # Step 6: figure/table areas (PDF only)
    paper = assemble(doc, label_counts(doc), notes)
    paper.visual_stats = visual_stats
    return paper


__all__ = ["structure_document", "classify", "assemble", "label_counts", "map_to_ieee"]
