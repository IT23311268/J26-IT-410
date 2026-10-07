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
