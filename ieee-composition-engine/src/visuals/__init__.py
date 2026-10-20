"""
The visuals package - Step 6, part 1: figures, tables, algorithms and
equations as images.

  detect_regions()  find the page area each visual occupies   (regions.py)
  export_visuals()  crop those areas out of the PDF as images  (export.py)
"""

from .export import export_visuals
from .regions import detect_regions

__all__ = ["detect_regions", "export_visuals"]
