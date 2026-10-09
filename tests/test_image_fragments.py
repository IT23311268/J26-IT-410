"""Tests for putting a clipped raster image back together — Box 3.

Found by running the detector over a real corpus rather than over the
synthetic pages in these tests. Page 22 of arXiv:2610.12376 is one
page-wide tokenisation figure whose background is a bitmap; the PDF
draws that bitmap once and reveals it through 26 clipping paths, so
PyMuPDF reported 26 image blocks.

Box 3 returned **29 figures for that page**, Box 6 would have written
29 crops of one picture, and Member 3 would have received 29 artifacts
where the paper has one. Across the whole paper 33 figures came back
where there are 8, and two thirds of them were duplicates or boxes
nested inside other boxes.

None of the synthetic PDFs in this suite reproduce it: every one of
them draws its figures as plain vector paths, which is why the bug
survived 300 passing tests. That is the argument for the corpus check
in one sentence.

`transform` is the PDF's own record of where an image was placed, so
pieces that share one are the same placement by definition. There is
no threshold here to tune.
"""

import pytest

from ingestion.layout import merge_image_fragments

# The placement all 26 pieces on that page shared.
PLACEMENT = (420.9, 0.0, -0.0, 554.3, 87.2, 133.7)
OTHER = (120.0, 0.0, -0.0, 90.0, 60.0, 400.0)


def image(bbox, transform=PLACEMENT, number=0):
    return {"type": 1, "bbox": bbox, "transform": transform, "number": number}


def text(bbox, body="prose"):
    return {
        "type": 0,
        "bbox": bbox,
        "lines": [{"spans": [{"text": body, "font": "Times", "size": 9.0}]}],
    }


def vector(bbox):
    """What `vector_figure_blocks` returns: type 1, but no transform."""
    return {"type": 1, "bbox": bbox, "lines": []}


# --------------------------------------------------------------------------
# the merge
# --------------------------------------------------------------------------


def test_pieces_of_one_placement_become_one_block():
    out = merge_image_fragments(
        [image((100, 100, 120, 110)), image((200, 300, 260, 340))]
    )
    assert len(out) == 1


def test_the_box_covers_every_piece():
    out = merge_image_fragments(
        [
            image((221.5, 193.0, 240.0, 210.0)),
            image((400.0, 300.0, 482.2, 420.0)),
            image((300.0, 500.0, 350.0, 614.9)),
        ]
    )
    assert out[0]["bbox"] == (221.5, 193.0, 482.2, 614.9)


def test_two_placements_stay_two_figures():
    """Two real figures on one page must not be glued together."""
    out = merge_image_fragments(
        [
            image((100, 100, 200, 200), PLACEMENT),
            image((100, 400, 200, 500), OTHER),
        ]
    )
    assert len(out) == 2


def test_a_placement_differing_in_the_last_decimal_is_the_same_one():
    """Pieces of one placement can disagree a hair in the transform.
    A tenth of a point is far below anything that separates two real
    placements."""
    jittered = (420.94, 0.0, -0.0, 554.335, 87.17, 133.72)
    out = merge_image_fragments(
        [image((100, 100, 120, 110), PLACEMENT), image((300, 300, 320, 310), jittered)]
    )
    assert len(out) == 1


def test_a_lone_image_is_left_exactly_as_it_was():
    block = image((100, 100, 200, 200))
    out = merge_image_fragments([block])
    assert out[0]["bbox"] == (100, 100, 200, 200)
    assert out[0]["transform"] == PLACEMENT


def test_the_fragments_are_not_mutated_in_place():
    """`page.get_text()` hands back PyMuPDF's own dicts. Growing one in
    place would corrupt a second read of the same page."""
    first = image((100, 100, 120, 110))
    merge_image_fragments([first, image((300, 300, 320, 310))])
    assert first["bbox"] == (100, 100, 120, 110)


# --------------------------------------------------------------------------
# everything else passes through
# --------------------------------------------------------------------------


def test_a_vector_figure_has_no_transform_and_is_untouched():
    """`vector_figure_blocks` emits type-1 blocks with no transform.
    Keying on a missing transform would merge every chart on the page
    into one."""
    out = merge_image_fragments([vector((50, 50, 150, 150)), vector((50, 400, 150, 500))])
    assert len(out) == 2


def test_text_blocks_are_untouched():
    blocks = [text((72, 72, 540, 120)), text((72, 200, 540, 260))]
    assert merge_image_fragments(blocks) == blocks


def test_reading_order_is_preserved():
    """Blocks arrive in the order PyMuPDF found them and later stages
    depend on that, so the merged image keeps the position of its
    first piece."""
    blocks = [
        text((72, 50, 540, 70), "above"),
        image((100, 100, 120, 110)),
        text((72, 300, 540, 320), "below"),
        image((300, 350, 320, 360)),
    ]
    out = merge_image_fragments(blocks)
    assert [b["type"] for b in out] == [0, 1, 0]
    assert out[0]["lines"][0]["spans"][0]["text"] == "above"
    assert out[2]["lines"][0]["spans"][0]["text"] == "below"


def test_an_empty_page_is_fine():
    assert merge_image_fragments([]) == []


# --------------------------------------------------------------------------
# the shape of the real page
# --------------------------------------------------------------------------


def test_the_real_page_collapses_to_one_figure():
    """26 pieces, one placement — the page that started this."""
    pieces = [
        image((221.5 + 9 * i, 193.0 + 16 * i, 240.0 + 9 * i, 210.0 + 16 * i))
        for i in range(26)
    ]
    out = merge_image_fragments(pieces)
    assert len(out) == 1
    assert out[0]["bbox"][0] == pytest.approx(221.5)
    assert out[0]["bbox"][3] == pytest.approx(210.0 + 16 * 25)
