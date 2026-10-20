"""
The placement package - Step 6, part 2: contextual placement of figures,
tables and algorithms (the novelty of this component).

  place_visuals()  move each visual to just after its first mention, or the
                   most similar paragraph, and measure the improvement.
"""

from .contextual import mentions, place_visuals, summarise

__all__ = ["place_visuals", "mentions", "summarise"]
