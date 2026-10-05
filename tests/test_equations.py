"""Tests for equation detection — Box 3.

The fixtures here are built from the real font data of arXiv:2005.11401
(the RAG paper): body text set in NimbusRomNo9L, mathematics in CMMI,
CMSY and CMEX. Those names are not invented; they came out of
`scripts/inspect_fonts.py` run on that paper.
"""

import pytest

from ingestion.layout import (
    MATH_RATIO_FOR_EQUATION,
    block_math_ratio,
    classify_block,
    is_math_font,
    merge_equation_fragments,
)
from schema.ingestion_schema_v1 import RegionType


def _block(spans: list[tuple[str, str]], bbox=(0, 0, 100, 20)) -> dict:
    """A PyMuPDF-shaped text block from (font, text) pairs."""
    return {
        "type": 0,
        "bbox": bbox,
        "lines": [{"spans": [{"font": font, "text": text, "size": 10.0} for font, text in spans]}],
    }


# --------------------------------------------------------------------------
# which faces count as mathematics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("font", ["CMMI10", "CMMI7", "CMSY10", "CMSY7", "CMEX10"])
def test_computer_modern_math_faces_are_math(font):
    assert is_math_font(font)


@pytest.mark.parametrize("font", ["MSAM10", "MSBM10", "STIXTwoMath", "XITSMath", "LMMath10"])
def test_other_common_math_faces_are_math(font):
    assert is_math_font(font)


@pytest.mark.parametrize(
    "font",
    [
        "NimbusRomNo9L-Regu",
        "NimbusRomNo9L-ReguItal",
        "NimbusRomNo9L-Medi",
        "Times-Roman",
        "Helvetica",
        "",
    ],
)
def test_text_faces_are_not_math(font):
    assert not is_math_font(font)


@pytest.mark.parametrize("font", ["CMR10", "CMR7", "CMBX10"])
def test_plain_computer_modern_is_not_treated_as_math(font):
    """CMR and CMBX appear inside equations, but they are also the body
    face of any paper typeset wholly in Computer Modern. Matching them
    would label every paragraph of an older paper as an equation."""
    assert not is_math_font(font)


# --------------------------------------------------------------------------
# how much of a block is mathematics
# --------------------------------------------------------------------------


def test_prose_has_a_math_ratio_of_zero():
    block = _block([("NimbusRomNo9L-Regu", "We jointly train the retriever and generator")])
    assert block_math_ratio(block) == 0.0


def test_a_display_equation_is_almost_all_math():
    block = _block([("CMEX10", "∑"), ("CMMI10", "p"), ("CMSY10", "∈"), ("CMMI7", "z")])
    assert block_math_ratio(block) == 1.0


def test_inline_math_in_a_paragraph_stays_low():
    """The real case that sets the threshold: a body paragraph from the
    RAG paper carried 170 ordinary characters and 7 in CMMI."""
    block = _block(
        [
            ("NimbusRomNo9L-Regu", "x" * 170),
            ("CMMI10", "p"),
            ("CMMI10", "θ"),
            ("CMR10", "()"),
            ("CMMI7", "ηz"),
            ("CMSY10", "|"),
        ]
    )
    ratio = block_math_ratio(block)
    assert ratio < 0.1
    assert ratio < MATH_RATIO_FOR_EQUATION


def test_an_empty_block_is_not_math():
    assert block_math_ratio({"type": 0, "bbox": (0, 0, 1, 1), "lines": []}) == 0.0


# --------------------------------------------------------------------------
# the upright parts of a formula
# --------------------------------------------------------------------------


def test_a_short_formula_counts_its_upright_parts():
    """Found on the RAG paper: "d(z) = BERT_d(z)" came back as BODY.

    Its variables are strong maths (CMMI) but its operator name, digits
    and parentheses are CMR and CMBX. Those belong to the formula, and
    in a block this short they are counted.
    """
    block = _block(
        [
            ("CMBX10", "d"),
            ("CMR10", "() = BERT"),
            ("CMMI7", "d"),
            ("CMR10", "("),
            ("CMMI10", "z"),
            ("CMR10", ")"),
        ]
    )
    assert block_math_ratio(block) == 1.0


def test_a_long_block_does_not_promote_its_upright_parts():
    """The failure this guards against: in a paper typeset wholly in
    Computer Modern, CMR is the *body* face. Promoting it in a long
    block would turn every paragraph into an equation."""
    block = _block([("CMR10", "x" * 400), ("CMMI10", "p"), ("CMMI10", "y")])

    ratio = block_math_ratio(block)
    assert ratio < 0.05
    assert classify_block("prose", 10.0, 10.0, is_image=False, math_ratio=ratio) == RegionType.BODY


def test_promotion_needs_some_unambiguous_maths():
    """A short block of plain Computer Modern prose, with no real maths
    in it, is not an equation."""
    block = _block([("CMR10", "Section 3 describes the method.")])
    assert block_math_ratio(block) == 0.0


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


def test_a_high_math_ratio_is_an_equation():
    assert (
        classify_block("p(y|x)", 10.0, 10.0, is_image=False, math_ratio=0.95)
        == RegionType.EQUATION
    )


def test_a_paragraph_with_inline_math_stays_body():
    assert (
        classify_block("We jointly train...", 10.0, 10.0, is_image=False, math_ratio=0.04)
        == RegionType.BODY
    )


def test_a_caption_containing_maths_is_still_a_caption():
    """"Figure 2: the distribution p(y|x)" must not become an equation."""
    assert (
        classify_block("Figure 2: the distribution", 10.0, 10.0, is_image=False, math_ratio=0.8)
        == RegionType.CAPTION
    )


def test_classification_without_a_math_ratio_is_unchanged():
    """The argument defaults, so existing callers behave as before."""
    assert classify_block("ordinary prose", 10.0, 10.0, is_image=False) == RegionType.BODY


# --------------------------------------------------------------------------
# putting a shattered equation back together
# --------------------------------------------------------------------------


def _math_fragment(x0, y0, x1, y1, text="x") -> dict:
    return _block([("CMMI10", text)], bbox=(x0, y0, x1, y1))


def test_fragments_of_one_equation_merge_into_a_single_block():
    """The RAG paper's first equation arrived as ten blocks."""
    fragments = [_math_fragment(100 + i * 12, 400, 110 + i * 12, 415) for i in range(10)]
    merged = merge_equation_fragments(fragments)

    assert len(merged) == 1
    x0, y0, x1, y1 = merged[0]["bbox"]
    assert x0 == 100
    assert x1 == 218  # 100 + 9*12 + 10


def test_two_separate_equations_stay_separate():
    first = [_math_fragment(100, 400, 150, 415), _math_fragment(155, 400, 200, 415)]
    second = [_math_fragment(100, 600, 150, 615), _math_fragment(155, 600, 200, 615)]

    merged = merge_equation_fragments(first + second)
    assert len(merged) == 2


def test_fragments_in_different_columns_do_not_merge():
    """A gutter is far wider than the merge gap, so the two sides of a
    two-column page cannot reach each other."""
    left = _math_fragment(60, 400, 260, 415)
    right = _math_fragment(330, 400, 530, 415)

    merged = merge_equation_fragments([left, right])
    assert len(merged) == 2


def test_prose_blocks_pass_through_untouched():
    prose = _block([("NimbusRomNo9L-Regu", "ordinary text")], bbox=(0, 0, 300, 20))
    equations = [_math_fragment(100, 400, 150, 415), _math_fragment(155, 400, 200, 415)]

    merged = merge_equation_fragments([prose] + equations)

    assert prose in merged
    assert len(merged) == 2  # the prose, plus one merged equation


def test_a_lone_equation_block_is_left_alone():
    one = [_math_fragment(100, 400, 200, 415)]
    assert merge_equation_fragments(one) == one


def test_merging_preserves_the_text_of_every_fragment():
    fragments = [
        _math_fragment(100, 400, 110, 415, "a"),
        _math_fragment(112, 400, 122, 415, "b"),
        _math_fragment(124, 400, 134, 415, "c"),
    ]
    merged = merge_equation_fragments(fragments)

    from ingestion.layout import _block_text

    text = _block_text(merged[0])
    assert "a" in text and "b" in text and "c" in text
