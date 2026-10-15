"""
The ingest package - stage 1 of the pipeline.

Everything outside this package calls load_document() and gets back a
ParsedDocument. Whether the source was a PDF or a Word file is this package's
problem, not anybody else's. That is what makes the rest of the pipeline easy
to test: it only ever sees one shape of data.
"""

from __future__ import annotations

import os

from ..models import ParsedDocument
from .docx_reader import read_docx
from .pdf_reader import read_pdf

SUPPORTED_FORMATS = (".pdf", ".docx")


def load_document(path: str) -> ParsedDocument:
    """Read any supported document and return it as a ParsedDocument."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}")

    extension = os.path.splitext(path)[1].lower()

    if extension == ".pdf":
        return read_pdf(path)
    if extension == ".docx":
        return read_docx(path)

    raise ValueError(
        f"Unsupported file type '{extension}'. "
        f"This engine accepts: {', '.join(SUPPORTED_FORMATS)}"
    )


__all__ = ["load_document", "read_pdf", "read_docx", "SUPPORTED_FORMATS"]
