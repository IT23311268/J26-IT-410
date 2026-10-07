"""Tests for artifact extraction — Box 6.

Box 6 is the first stage whose output a human judges by looking at it,
which makes it the easiest one to be quietly wrong about: a crop that
is 10 points short still looks like a figure. So the tests measure the
crop against the page instead of trusting the picture.

Two things have to hold at once, and they pull against each other:

  the crop must contain the whole artifact, including the tick labels
  and axis titles that a chart prints *outside* its plot frame — the
  bug Box 3's `grow_figures_to_their_labels` was written for, which
  only becomes visible here;

  the crop must not contain the caption underneath it. A caption sits
  as little as 1.9 points below the figure on a real page, so padding
  that ignores its neighbours slices the caption in half and puts the
  fragment in Member 3's graph.
"""

from pathlib import Path

import pytest
from PIL import Image

from ingestion.artifacts import (
    CROP_PADDING,
    MIN_ARTIFACT_SIDE,
    extract_artifacts,
    pair_captions,
    padded_box,
)
from ingestion.layout import extract_regions
from ingestion.rasterise import rasterise_pdf
from schema.ingestion_schema_v1 import (
    ArtifactType,
    BoundingBox,
    LayoutRegion,
    PageImage,
    RegionType,
)

from .test_charts import PLOT, _build_table_and_chart_pdf

#: US letter, the page size the chart fixture is drawn on. Needed to
#: flip reportlab's bottom-left origin into PyMuPDF's top-left one.
PAGE_HEIGHT = 792.0


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def page(tmp_path: Path):
    """The table-and-chart page, rasterised, through Boxes 3 and 6."""
    pdf = _build_table_and_chart_pdf(tmp_path / "chart.pdf")
    pages = rasterise_pdf(pdf, paper_id="p", out_root=tmp_path, dpi=200)
    regions = extract_regions(pdf, "p")
    artifacts = extract_artifacts(regions, pages, "p", tmp_path)
    return regions, pages, artifacts, tmp_path


def _bbox(page: int, x0: float, y0: float, x1: float, y1: float) -> BoundingBox:
    return BoundingBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1)


def _region(
    region_id: str,
    kind: RegionType,
    box: BoundingBox,
    text: str = "",
    order: int = 0,
) -> LayoutRegion:
    return LayoutRegion(
        region_id=region_id,
        paper_id="p",
        region_type=kind,
        bbox=box,
        text=text,
        order_index=order,
    )


# --------------------------------------------------------------------------
# padded_box — the crop margin
# --------------------------------------------------------------------------


def test_padding_is_full_with_nothing_nearby():
    box = _bbox(0, 100, 100, 200, 200)
    assert padded_box(box, [], 612, 792) == (
        100 - CROP_PADDING,
        100 - CROP_PADDING,
        200 + CROP_PADDING,
        200 + CROP_PADDING,
    )


def test_padding_stops_halfway_to_a_close_neighbour():
    """A caption 2 points below gets 1 point of the gap, not 6."""
    box = _bbox(0, 100, 100, 200, 200)
    caption = _bbox(0, 100, 202, 200, 215)
    _, _, _, y1 = padded_box(box, [caption], 612, 792)
    assert y1 == pytest.approx(201.0)


def test_a_close_neighbour_does_not_shrink_the_other_edges():
    """Only the crowded edge is cut back. A figure with a caption under
    it still gets full padding above and to the sides."""
    box = _bbox(0, 100, 100, 200, 200)
    caption = _bbox(0, 100, 202, 200, 215)
    x0, y0, x1, _ = padded_box(box, [caption], 612, 792)
    assert (x0, y0, x1) == (94.0, 94.0, 206.0)


def test_a_neighbour_that_does_not_line_up_is_not_in_the_way():
    """A block in the next column is beside the figure but shares none
    of its height, so it cannot be clipping anything."""
    box = _bbox(0, 100, 100, 200, 200)
    other_column = _bbox(0, 300, 400, 400, 500)
    assert padded_box(box, [other_column], 612, 792) == (94.0, 94.0, 206.0, 206.0)


def test_padding_never_runs_off_the_page():
    box = _bbox(0, 2, 2, 610, 790)
    assert padded_box(box, [], 612, 792) == (0.0, 0.0, 612.0, 792.0)


def test_a_touching_neighbour_gives_no_padding():
    """Zero gap, zero room — the crop stops on the edge rather than
    eating into whatever is pressed against it."""
    box = _bbox(0, 100, 100, 200, 200)
    touching = _bbox(0, 100, 200, 200, 230)
    _, _, _, y1 = padded_box(box, [touching], 612, 792)
    assert y1 == pytest.approx(200.0)


# --------------------------------------------------------------------------
# pair_captions
# --------------------------------------------------------------------------


def test_each_artifact_gets_the_caption_under_it():
    regions = [
        _region("a1", RegionType.FIGURE, _bbox(0, 100, 100, 300, 200)),
        _region("c1", RegionType.CAPTION, _bbox(0, 100, 205, 300, 220), "Figure 1: one"),
        _region("a2", RegionType.FIGURE, _bbox(0, 100, 400, 300, 500)),
        _region("c2", RegionType.CAPTION, _bbox(0, 100, 505, 300, 520), "Figure 2: two"),
    ]
    assert pair_captions(regions) == {"a1": "Figure 1: one", "a2": "Figure 2: two"}


def test_one_caption_is_not_shared_by_two_artifacts():
    """Nearest first: the caption goes to the figure it is actually
    under, and the other figure gets nothing rather than a wrong
    label."""
    regions = [
        _region("near", RegionType.FIGURE, _bbox(0, 100, 100, 300, 200)),
        _region("far", RegionType.FIGURE, _bbox(0, 100, 300, 300, 380)),
        _region("c", RegionType.CAPTION, _bbox(0, 100, 203, 300, 218), "Figure 1: one"),
    ]
    paired = pair_captions(regions)
    assert paired == {"near": "Figure 1: one"}


def test_a_caption_in_the_next_column_is_not_matched():
    regions = [
        _region("a", RegionType.FIGURE, _bbox(0, 60, 100, 280, 200)),
        _region("c", RegionType.CAPTION, _bbox(0, 320, 205, 540, 220), "Figure 9: other"),
    ]
    assert pair_captions(regions) == {}


def test_tables_and_equations_get_captions_too():
    regions = [
        _region("t", RegionType.TABLE, _bbox(0, 100, 100, 300, 200)),
        _region("c", RegionType.CAPTION, _bbox(0, 100, 205, 300, 220), "Table 2: results"),
    ]
    assert pair_captions(regions) == {"t": "Table 2: results"}


# --------------------------------------------------------------------------
# extract_artifacts — on the real page
# --------------------------------------------------------------------------


def test_the_table_and_the_chart_both_come_out(page):
    _, _, artifacts, _ = page
    kinds = {a.artifact_type for a in artifacts}
    assert ArtifactType.TABLE in kinds
    assert ArtifactType.FIGURE in kinds


def test_each_artifact_carries_its_own_caption(page):
    _, _, artifacts, _ = page
    captions = {a.artifact_type: a.caption for a in artifacts}
    assert captions[ArtifactType.TABLE].startswith("Table 4")
    assert captions[ArtifactType.FIGURE].startswith("Figure 7")


def test_every_artifact_has_a_file_on_disk(page):
    _, _, artifacts, root = page
    assert artifacts
    for a in artifacts:
        assert a.image_path is not None
        assert (root / a.image_path).exists()


def test_files_are_named_by_type_and_numbered(page):
    _, _, artifacts, _ = page
    names = {Path(a.image_path).name for a in artifacts}
    assert "fig_01.png" in names
    assert "tbl_01.png" in names


def test_ids_are_stable_and_in_reading_order(page):
    _, _, artifacts, _ = page
    assert [a.artifact_id for a in artifacts] == [
        f"p::artifact::{i}" for i in range(len(artifacts))
    ]


def test_binding_to_chunks_is_left_for_box_7(page):
    _, _, artifacts, _ = page
    assert all(a.linked_chunk_ids == [] for a in artifacts)


# --------------------------------------------------------------------------
# the two claims that pull against each other
# --------------------------------------------------------------------------


def test_the_crop_reaches_past_the_plot_frame(page):
    """The chart's ticks and axis titles are printed outside its frame.
    A crop that stops at the frame loses them — which is what the
    pre-Box-3-fix output did.

    `PLOT` is in reportlab's coordinates, which put the origin at the
    bottom-left with y increasing upwards. PyMuPDF — and therefore
    every bbox in the schema — puts it at the top-left with y
    increasing downwards. Comparing the two directly passes or fails
    for the wrong reason, so the frame is flipped first.
    """
    _, _, artifacts, _ = page
    figure = next(a for a in artifacts if a.artifact_type == ArtifactType.FIGURE)
    px0, py0, _, py1 = PLOT
    frame_top, frame_bottom = PAGE_HEIGHT - py1, PAGE_HEIGHT - py0

    assert figure.bbox.x0 < px0, "y tick labels sit left of the frame"
    assert figure.bbox.y0 < frame_top, "the chart title sits above the frame"
    assert figure.bbox.y1 > frame_bottom, "the x tick labels sit below the frame"


def test_the_crop_stops_short_of_the_caption(page):
    """1.9 points of clearance on a real page. Padding that ignored it
    would put half a line of caption text in the picture."""
    regions, _, artifacts, _ = page
    figure = next(a for a in artifacts if a.artifact_type == ArtifactType.FIGURE)
    caption = min(
        (r for r in regions if r.region_type == RegionType.CAPTION),
        key=lambda r: abs(r.bbox.y0 - figure.bbox.y1),
    )
    neighbours = [r.bbox for r in regions if r.region_id != figure.artifact_id]
    _, _, _, y1 = padded_box(figure.bbox, neighbours, 612, 792)
    assert y1 <= caption.bbox.y0, "the crop runs into the caption"


def test_the_crop_is_not_blank(page):
    """A crop placed with the scale factor inverted lands on white
    space and still saves happily. Check there is ink in it."""
    _, _, artifacts, root = page
    for a in artifacts:
        image = Image.open(root / a.image_path).convert("L")
        darkest, _ = image.getextrema()
        assert darkest < 128, f"{a.image_path} is blank"


# --------------------------------------------------------------------------
# the awkward runs
# --------------------------------------------------------------------------


def test_artifacts_survive_a_run_with_no_page_images(tmp_path: Path):
    """`rasterise=false` costs the picture, not the record."""
    pdf = _build_table_and_chart_pdf(tmp_path / "chart.pdf")
    regions = extract_regions(pdf, "p")
    artifacts = extract_artifacts(regions, [], "p", tmp_path)
    assert artifacts
    assert all(a.image_path is None for a in artifacts)
    assert all(a.caption for a in artifacts)


def test_a_missing_page_file_does_not_crash(tmp_path: Path):
    """The record says the image is there and it is not — a half
    deleted data/processed/. Degrade, don't 500."""
    pdf = _build_table_and_chart_pdf(tmp_path / "chart.pdf")
    regions = extract_regions(pdf, "p")
    ghost = PageImage(
        page_index=0,
        image_path="p/pages/page_0000.png",
        width_px=1700,
        height_px=2200,
        dpi=200,
    )
    artifacts = extract_artifacts(regions, [ghost], "p", tmp_path)
    assert artifacts
    assert all(a.image_path is None for a in artifacts)


def test_slivers_are_not_artifacts(tmp_path: Path):
    """A stray rule or bullet is not worth a PNG of its own."""
    sliver = _region("s", RegionType.FIGURE, _bbox(0, 100, 100, 100 + MIN_ARTIFACT_SIDE - 1, 110))
    assert extract_artifacts([sliver], [], "p", tmp_path) == []


def test_no_artifacts_writes_no_directory(tmp_path: Path):
    """A paper of pure prose should not leave an empty artifacts/
    folder behind."""
    body = _region("b", RegionType.BODY, _bbox(0, 72, 72, 540, 300), "prose")
    assert extract_artifacts([body], [], "p", tmp_path) == []
    assert not (tmp_path / "p" / "artifacts").exists()
