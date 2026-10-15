"""
Tests for stage 3 (the IEEE mapper).

Run with:   pytest -q
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import load_document                                   # noqa: E402
from src.models import Element, Section, StructuredPaper               # noqa: E402
from src.structure import map_to_ieee, structure_document              # noqa: E402
from src.structure.ieee_mapper import score_section, to_letter, to_roman  # noqa: E402

SAMPLE_PDF = os.path.join("data", "input", "sample_paper.pdf")


def _paper(*sections) -> StructuredPaper:
    """Build a paper from (heading, level, text) tuples."""
    paper = StructuredPaper(source_path="test")
    for heading, level, text in sections:
        s = Section(heading=heading, number="", level=level)
        if text:
            s.elements.append(Element(kind="paragraph", text=text))
        paper.sections.append(s)
    return paper


def _best(heading: str, text: str = "", first=False, last=False) -> str:
    scores = score_section(heading, text, first, last)
    return max(scores.items(), key=lambda kv: kv[1][0])[0]


# ---------------------------------------------------------------------------
# Numbering helpers
# ---------------------------------------------------------------------------

def test_roman_numerals():
    assert [to_roman(n) for n in (1, 2, 3, 4, 5, 6, 9, 10)] == \
        ["I", "II", "III", "IV", "V", "VI", "IX", "X"]


def test_letters():
    assert [to_letter(n) for n in (1, 2, 3)] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Heading recognition - the common names authors use
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("heading, slot", [
    ("Introduction", "introduction"),
    ("Motivation", "introduction"),
    ("Literature Review", "related_work"),
    ("Related Works", "related_work"),
    ("Prior Work", "related_work"),
    ("Materials and Methods", "methodology"),
    ("Proposed System", "methodology"),
    ("Our Approach", "methodology"),
    ("Experimental Setup", "experiments"),
    ("Results and Discussion", "results"),
    ("Evaluation", "results"),
    ("Discussion", "discussion"),
    ("Limitations", "discussion"),
    ("Conclusions and Future Work", "conclusion"),
    ("Concluding Remarks", "conclusion"),
    ("Acknowledgements", "acknowledgment"),
])
def test_heading_names(heading, slot):
    assert _best(heading) == slot


def test_background_goes_to_introduction_when_first():
    assert _best("Background", first=True) == "introduction"


def test_background_goes_to_related_work_when_full_of_citations():
    text = "Smith et al. (2019) proposed a method. Jones et al. (2020) extended it."
    assert _best("Background", text) == "related_work"


# ---------------------------------------------------------------------------
# Whole-paper behaviour
# ---------------------------------------------------------------------------

def test_sample_paper_is_mapped():
    if not os.path.exists(SAMPLE_PDF):
        pytest.skip("run tools/make_sample_paper.py first")
    paper = map_to_ieee(structure_document(load_document(SAMPLE_PDF)))
    got = [(s.ieee_number, s.ieee_heading) for s in paper.sections]
    assert got == [
        ("I", "Introduction"),
        ("II", "Related Work"),
        ("III", "Methodology"),
        ("A", "Dataset"),
        ("B", "Network Architecture"),
        ("IV", "Results"),
        ("V", "Conclusion"),
    ]
    assert paper.missing_sections == []
    assert all(s.mapping_reason for s in paper.sections if s.level == 1)


def test_subsection_letters_restart_under_each_section():
    paper = map_to_ieee(_paper(
        ("Introduction", 1, "In this paper we propose a thing."),
        ("Method", 1, ""),
        ("Data", 2, ""), ("Model", 2, ""),
        ("Results", 1, ""),
        ("Accuracy", 2, ""),
        ("Conclusion", 1, ""),
    ))
    assert [s.ieee_number for s in paper.sections] == ["I", "II", "A", "B", "III", "A", "IV"]


def test_missing_sections_are_reported():
    paper = map_to_ieee(_paper(
        ("Introduction", 1, "In this paper we study X."),
        ("Results", 1, "Accuracy was 91 percent."),
    ))
    assert paper.missing_sections == ["Methodology", "Conclusion"]
    assert any("Methodology" in w for w in paper.warnings)


def test_unrecognised_heading_keeps_its_own_name():
    paper = map_to_ieee(_paper(
        ("Introduction", 1, ""),
        ("Ethical Considerations Under Sri Lankan Law", 1, ""),
        ("Method", 1, ""), ("Results", 1, ""), ("Conclusion", 1, ""),
    ))
    odd = paper.sections[1]
    assert odd.ieee_slot == "other"
    assert odd.ieee_heading == "Ethical Considerations Under Sri Lankan Law"
    assert odd.ieee_number == "II"                    # still numbered


def test_acknowledgment_is_not_numbered_and_does_not_steal_conclusion():
    paper = map_to_ieee(_paper(
        ("Introduction", 1, ""), ("Method", 1, ""), ("Results", 1, ""),
        ("Summary", 1, "We presented a method. Future work will examine more data."),
        ("Acknowledgements", 1, "We thank the funding agency for the grant."),
    ))
    assert paper.sections[3].ieee_slot == "conclusion"
    ack = paper.sections[4]
    assert ack.ieee_heading == "Acknowledgment" and ack.ieee_number == ""


def test_two_sections_in_same_slot_keep_distinct_names():
    paper = map_to_ieee(_paper(
        ("Introduction", 1, ""), ("Method", 1, ""),
        ("Results", 1, ""), ("Further Results", 1, ""),
        ("Conclusion", 1, ""),
    ))
    names = [s.ieee_heading for s in paper.sections]
    assert names == ["Introduction", "Methodology", "Results", "Further Results", "Conclusion"]


def test_out_of_order_sections_are_kept_but_flagged():
    paper = map_to_ieee(_paper(
        ("Introduction", 1, ""), ("Results", 1, ""),
        ("Method", 1, ""), ("Conclusion", 1, ""),
    ))
    assert [s.ieee_heading for s in paper.sections][1] == "Results"   # not reordered
    assert any("order" in w for w in paper.warnings)


def test_mapping_survives_a_save_and_reload(tmp_path):
    paper = map_to_ieee(_paper(("Introduction", 1, "Hello."), ("Conclusion", 1, "Bye.")))
    path = tmp_path / "paper.json"
    paper.to_json(str(path))
    again = StructuredPaper.from_json(str(path))
    assert again.sections[1].ieee_heading == "Conclusion"
    assert again.missing_sections == paper.missing_sections
