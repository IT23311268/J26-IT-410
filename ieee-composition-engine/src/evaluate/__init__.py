"""
The evaluate package - Step 6, part 3: the IEEE Format Compliance Score.

  evaluate()  run 17 IEEE checks on the source and on the output and
              report a before/after percentage, with a reason for each.
"""

from .compliance import ComplianceReport, evaluate

__all__ = ["evaluate", "ComplianceReport"]
