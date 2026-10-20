"""
The rules package - the IEEE rulebook.

    from src.rules import ieee_rules as R
    R.SIZE_PT["title"]        -> 24
    R.rule("LAY-02")          -> the margins rule, with its value and source

Everything the engine knows about IEEE style is in ieee_rules.py.
"""

from . import ieee_rules
from .ieee_rules import RULES, Rule, rule, rules_for_check, summary

__all__ = ["ieee_rules", "RULES", "Rule", "rule", "rules_for_check", "summary"]
