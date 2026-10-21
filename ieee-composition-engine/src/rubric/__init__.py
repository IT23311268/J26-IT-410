"""
The rubric package - rubric-based IEEE evaluation.

  evidence.py   a receipt for every rule the engine applies: rule ID, exact
                place, before and after
  rubric.py     one rubric line per rulebook rule: does it apply to this
                paper, what did the engine do, where - and a score from the
                rules a reviewer ticks
  export.py     the filled rubric as an Excel file
"""

from .evidence import anchor_of, record, where_of
from .rubric import RubricItem, build_rubric, rubric_score, save_rubric, summary

__all__ = ["anchor_of", "record", "where_of", "RubricItem", "build_rubric", "rubric_score",
           "save_rubric", "summary"]
