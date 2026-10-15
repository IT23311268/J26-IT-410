"""
The render package - stage 4 of the pipeline.

render_docx() writes a StructuredPaper as a two-column IEEE Word document.
All IEEE measurements live in ieee_style.py.
"""

from .docx_renderer import render_docx

__all__ = ["render_docx"]
