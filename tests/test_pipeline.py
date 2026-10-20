"""Tests for the two pages a panel actually sees — the front door and
the pipeline walkthrough.

These are demo surfaces, not part of the Members 2/3/4 contract, so
what is worth testing is narrow: that every stage is on the page, that
the numbers on it are the real ones rather than hard-coded, and that a
half-complete run degrades to a sentence instead of a stack trace. A
page that 500s in front of a panel is worse than a page that admits a
stage did not run.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.home import PaperCard, render_home
from api.main import app
from api.pipeline import STAGES, render_pipeline
from ingestion.artifacts import extract_artifacts
from ingestion.baseline_extractor import extract_baseline
from ingestion.chunking import chunk_regions
from ingestion.layout import extract_regions
from ingestion.rasterise import rasterise_pdf
from ingestion.sections import assign_sections
from schema.ingestion_schema_v1 import ExtractionMethod

from .test_charts import _build_table_and_chart_pdf


@pytest.fixture
def ingested(tmp_path: Path):
    """One paper all the way through Boxes 1–6, plus its baseline."""
    pdf = _build_table_and_chart_pdf(tmp_path / "paper.pdf")
    result = extract_baseline(pdf)
    pid = result.paper.paper_id
    result.pages = rasterise_pdf(pdf, paper_id=pid, out_root=tmp_path, dpi=150)
    result.regions = assign_sections(extract_regions(pdf, pid))
    result.chunks = chunk_regions(result.regions, pid)
    result.artifacts = extract_artifacts(result.regions, result.pages, pid, tmp_path)
    result.paper.extraction_method = ExtractionMethod.LAYOUT_AWARE_V1
    return result, extract_baseline(pdf)


# --------------------------------------------------------------------------
# the walkthrough
# --------------------------------------------------------------------------


def test_every_stage_is_on_the_page(ingested):
    """The page is the component diagram. A missing box means the
    panel is back to being told which stage they are looking at."""
    result, baseline = ingested
    html = render_pipeline(result, baseline)
    for key, _, label in STAGES:
        assert f"id='{key}'" in html, f"stage {key} is missing"
        assert label in html


def test_it_reports_the_real_counts(ingested):
    result, baseline = ingested
    html = render_pipeline(result, baseline, source_bytes=4096)
    assert f">{len(result.chunks)}<" in html
    assert str(result.paper.page_count) in html
    assert result.schema_version in html


def test_each_artifact_gets_a_tile(ingested):
    result, baseline = ingested
    html = render_pipeline(result, baseline)
    assert result.artifacts
    for i in range(len(result.artifacts)):
        assert f"/artifact/{i}'" in html


def test_captions_reach_the_page(ingested):
    result, baseline = ingested
    html = render_pipeline(result, baseline)
    assert "Table 4" in html
    assert "Figure 7" in html


def test_every_section_in_the_paper_is_named(ingested):
    result, baseline = ingested
    html = render_pipeline(result, baseline)
    for section in {r.section.value for r in result.regions}:
        assert section.replace("_", " ") in html


def test_text_from_the_paper_is_escaped(ingested):
    """Chunk text and captions go straight into the page."""
    result, baseline = ingested
    result.chunks[0] = result.chunks[0].model_copy(
        update={"text": "<script>alert('x')</script> & more"}
    )
    html = render_pipeline(result, baseline)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


# --------------------------------------------------------------------------
# the half-complete runs
# --------------------------------------------------------------------------


def test_a_run_with_no_page_images_still_renders(ingested):
    result, baseline = ingested
    result.pages = []
    html = render_pipeline(result, baseline)
    assert "rasterise=false" in html
    assert "id='s6'" in html


def test_a_run_that_stopped_at_box_2_still_renders(ingested):
    result, baseline = ingested
    result.regions = []
    result.chunks = []
    result.artifacts = []
    html = render_pipeline(result, baseline)
    assert "detect_layout=true" in html
    assert "No chunks." in html


def test_a_paper_with_no_figures_says_so(ingested):
    result, baseline = ingested
    result.artifacts = []
    assert "No figures, tables or equations" in render_pipeline(result, baseline)


def test_a_missing_baseline_costs_one_section_not_the_page(ingested):
    result, _ = ingested
    html = render_pipeline(result, None)
    assert "baseline cannot be re-run" in html
    assert "id='s1'" in html


def test_the_comparison_is_in_shares_not_counts(ingested):
    """The two paths cut at different sizes, so raw counts would make
    the pipeline look better or worse purely by chunk size."""
    result, baseline = ingested
    html = render_pipeline(result, baseline)
    assert "shares of each run, not counts" in html
    assert "%</span>" in html


def test_both_sides_of_each_measure_are_drawn(ingested):
    """Two measures, a baseline bar and an ours bar in each."""
    result, baseline = ingested
    html = render_pipeline(result, baseline)
    assert html.count("class='measure'") == 2
    assert html.count("i class='baseline'") == 2
    assert html.count("i class='ours'") == 2


def test_a_bar_is_never_identified_by_colour_alone(ingested):
    """One man in twelve cannot separate two hues reliably. Every bar
    carries its own row label and its own printed percentage."""
    result, baseline = ingested
    html = render_pipeline(result, baseline)
    assert html.count("Baseline</span>") == 2
    assert html.count("This pipeline</span>") == 2


# --------------------------------------------------------------------------
# the front door
# --------------------------------------------------------------------------


def test_the_home_page_lists_every_paper():
    html = render_home([PaperCard("alpha-0001"), PaperCard("beta-0002")])
    assert "alpha-0001" in html and "beta-0002" in html
    assert "/pipeline/alpha-0001" in html


def test_a_row_leads_with_the_title_not_the_id():
    """The id is a hash with a filename on the front. It is the right
    identifier for Members 2/3/4 and no help at all to someone choosing
    which paper to open."""
    html = render_home([PaperCard("x-00ff", "Attention Is All You Need", 15, 80, 9)])
    # The id appears in the href first, so compare against where it is
    # actually shown to a reader rather than its first occurrence.
    assert html.index("Attention Is All You Need") < html.index("class='id mono'")
    assert "15 pages" in html and "80 chunks" in html
    assert "9 figures and tables" in html


def test_a_row_with_nothing_loaded_still_appears():
    """A damaged record must be visible, not silently missing."""
    html = render_home([PaperCard("broken-0001")])
    assert "broken-0001" in html
    assert "/pipeline/broken-0001" in html


def test_one_artifact_is_not_pluralised():
    assert "1 figure or table" in render_home([PaperCard("x", "T", 1, 1, 1)])


def test_an_id_with_a_space_is_encoded():
    assert "/pipeline/Linked%20Lists" in render_home([PaperCard("Linked Lists")])


def test_an_empty_shelf_invites_an_upload():
    html = render_home([])
    assert "Nothing ingested yet" in html


def test_the_api_reference_is_still_reachable():
    """Swagger is the contract for Members 2/3/4. Moving it off the
    front page must not hide it."""
    assert "/docs" in render_home([])


# --------------------------------------------------------------------------
# the routes
# --------------------------------------------------------------------------


def test_the_root_serves_the_front_door():
    r = TestClient(app).get("/")
    assert r.status_code == 200
    assert "Drop a paper here" in r.text


def test_an_unknown_paper_404s():
    assert TestClient(app).get("/pipeline/no-such-paper").status_code == 404


# --------------------------------------------------------------------------
# the headline, and the two chunks facing each other
# --------------------------------------------------------------------------


def test_the_result_is_at_the_top_before_any_scrolling(ingested):
    """A panel reads the top of a page and then listens. Whatever is up
    there is what they take away, so it has to be the result and not
    the filename."""
    result, baseline = ingested
    html = render_pipeline(result, baseline)
    head = html[: html.index("id='s1'")]
    assert "regions found" in head
    assert "chunks carry their section" in head
    assert "figures and tables lifted out" in head


def test_the_headline_carries_the_comparison(ingested):
    result, baseline = ingested
    head = render_pipeline(result, baseline)[: render_pipeline(result, baseline).index("id='s1'")]
    assert "start mid-word" in head
    assert "without this pipeline" in head


def test_the_headline_survives_a_missing_baseline(ingested):
    """No stored PDF means no comparison, but the counts still stand."""
    result, _ = ingested
    html = render_pipeline(result, None)
    head = html[: html.index("id='s1'")]
    assert "regions found" in head
    assert "without this pipeline" not in head


@pytest.fixture
def two_column(tmp_path: Path):
    """A paper the baseline actually breaks a word in.

    The chart fixture above is a single page whose baseline happens to
    cut cleanly, so it cannot show the thing being measured. This one
    is the two-column generator at a small chunk size, which is where
    `lumn paragraphs…` comes from.
    """
    import subprocess
    import sys

    pdf = tmp_path / "cols.pdf"
    subprocess.run(
        [sys.executable, "scripts/generate_two_column_pdf.py", str(pdf)],
        check=True,
        capture_output=True,
    )
    baseline = extract_baseline(pdf, chunk_chars=300, chunk_overlap=0)
    layout = extract_baseline(pdf)
    regions = assign_sections(extract_regions(pdf, layout.paper.paper_id))
    layout.chunks = chunk_regions(regions, layout.paper.paper_id)
    layout.regions = regions
    return layout, baseline


def test_the_broken_opening_is_marked_not_just_counted(two_column):
    """50% against 0% is the measure. This is what the 50% looks like."""
    result, baseline = two_column
    html = render_pipeline(result, baseline)
    assert "mark class='cut'" in html
    assert "Opens inside a word" in html


def test_the_clean_opening_is_marked_too(two_column):
    result, baseline = two_column
    html = render_pipeline(result, baseline)
    assert "mark class='keep'" in html
    assert "Opens at the heading" in html


def test_a_baseline_that_never_breaks_a_word_shows_no_pair(ingested):
    """A one-paragraph paper can cut cleanly by luck. Inventing a
    broken chunk to illustrate the point would be a lie."""
    result, baseline = ingested
    baseline.chunks = [
        c.model_copy(update={"text": "Clean opening. " + c.text}) for c in baseline.chunks
    ]
    html = render_pipeline(result, baseline)
    assert "Opens inside a word" not in html
    assert "id='ev'" in html


def test_the_reading_order_claim_is_above_the_overlay(ingested):
    """A caption under an image is read after the image, if at all."""
    result, baseline = ingested
    html = render_pipeline(result, baseline)
    assert "class='claim'" in html
    assert html.index("reading order</strong>") < html.index("/layout'")


# --------------------------------------------------------------------------
# the app rail
# --------------------------------------------------------------------------


def test_both_pages_carry_the_same_rail(ingested):
    """Two pages that navigate differently read as two prototypes."""
    result, baseline = ingested
    for html in (render_home([]), render_pipeline(result, baseline)):
        assert "class='apprail'" in html


def test_the_home_rail_leaves_out_what_needs_a_paper():
    """Four of the six destinations are views of one paper. Before a
    paper is chosen they have nothing to point at, and a control that
    is never enabled is a dead end with a tooltip."""
    html = render_home([])
    assert ">Papers</span>" in html
    assert ">API</span>" in html
    for absent in (">Pipeline</span>", ">Pages</span>", ">Compare</span>", ">JSON</span>"):
        assert absent not in html


def test_the_pipeline_rail_reaches_every_view_of_this_paper(ingested):
    result, baseline = ingested
    html = render_pipeline(result, baseline)
    pid = result.paper.paper_id
    for href in (f"/pipeline/{pid}", f"/gallery/{pid}", f"/compare/{pid}", f"/paper/{pid}"):
        assert f"href='{href}'" in html


def test_the_rail_marks_where_you_are(ingested):
    result, baseline = ingested
    assert render_pipeline(result, baseline).count("aria-current='page'") == 1
    assert render_home([]).count("aria-current='page'") == 1


def test_a_rail_entry_is_never_an_icon_on_its_own(ingested):
    """A panel meeting the service for the first time would have to
    hover six shapes to find out what they are. Pinned as "no anchor
    ends at its icon" rather than by counting a particular icon size,
    so resizing them is not a test failure."""
    result, baseline = ingested
    for html in (render_pipeline(result, baseline), render_home([])):
        assert "</svg></a>" not in html


def test_the_stages_are_nested_inside_the_rail(ingested):
    """Two rails down the left of one screen is a navigation scheme
    explaining itself twice."""
    result, baseline = ingested
    html = render_pipeline(result, baseline)
    # Find the rail's OWN closing tag, not the first one on the page.
    # The page may carry a second <nav> above the rail — a topbar, say —
    # and a bare index("</nav>") then returns a slice that ends before
    # the rail even opens, which is an empty string that fails for a
    # reason that has nothing to do with the rail.
    start = html.index("<nav class='apprail'")
    nav = html[start:html.index("</nav>", start)]
    for _, _, label in STAGES:
        assert label in nav


def test_the_six_stages_are_called_one_thing_everywhere(ingested):
    """The landing page used to write its own titles, so a panel read
    "The PDF arrives" on one page and "PDF in" on the next and had to
    work out that they were the same six stages."""
    result, baseline = ingested
    home = render_home([])
    walkthrough = render_pipeline(result, baseline)
    for key, _, label in STAGES:
        if key == "ev":
            continue
        assert label in home, f"{label} is missing from the landing page"
        assert label in walkthrough


def test_each_step_is_numbered_as_a_step():
    """A bare 3 in the corner of a card is a count as easily as it is
    a position, and this is an order."""
    home = render_home([])
    for n in range(1, 7):
        assert f">Step {n}</span>" in home


def test_an_image_opens_over_the_page_not_in_a_bare_tab(ingested):
    """A raw bitmap in a tab of its own has a dead back button,
    because nothing preceded it."""
    result, baseline = ingested
    html = render_pipeline(result, baseline)
    assert "class='zoom'" in html
    assert "target='_blank'" not in html


def test_a_paper_id_with_a_space_is_encoded_in_the_rail(ingested):
    result, baseline = ingested
    result.paper.paper_id = "Linked Lists"
    assert "/gallery/Linked%20Lists" in render_pipeline(result, baseline)


def test_the_masthead_does_not_repeat_the_rail(ingested):
    """Those four links used to sit under the title as well."""
    result, baseline = ingested
    assert "class='elsewhere'" not in render_pipeline(result, baseline)


def test_no_page_has_a_dark_theme(ingested):
    """The theme must not be the demo laptop's to choose.

    A projector's lamp adds light to every pixel, so a dark background
    arrives on the screen as grey and the contrast the page was
    measured for is gone. The supervisor saw this happen. Light is the
    only theme, and `color-scheme: light` says so to the browser as
    well, which is what keeps the scrollbars and the file picker from
    coming back dark on a laptop set to dark.
    """
    result, baseline = ingested
    for name, html in (
        ("home", render_home([])),
        ("pipeline", render_pipeline(result, baseline)),
    ):
        assert "prefers-color-scheme" not in html, f"{name} still follows the OS"
        assert "color-scheme: light;" in html, f"{name} does not pin the theme"


def test_the_planned_chip_is_placed_on_the_grid_not_floated():
    """It used to say `float: right`, which a grid item ignores.

    The chip took a row of its own and stretched the full width of the
    card, so "Planned · PP2" read as an empty input box sitting in the
    middle of the stage. A card is a grid, so the chip is placed in a
    cell like everything else in it.
    """
    css = render_home([])
    chip = css.split(".step .todo {")[1].split("}")[0]
    assert "float" not in chip
    assert "grid-row: 1" in chip and "justify-self: end" in chip


def test_the_planned_step_is_not_faded_out_of_readability():
    """Marked as not built, not made hard to read.

    At 0.72 its body text measured 2.76:1 against the card under a
    projector — a contrast failure, and in practice an invisible card
    on the one stage a panel is most likely to ask about. The greyed
    icon and the chip already carry the meaning.
    """
    assert "opacity: 0.72" not in render_home([])
