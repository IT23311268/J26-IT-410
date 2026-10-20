"""
Tests for Step 7 - the shared pipeline and the Streamlit web page.

Run with:   pytest -q

The web page is driven by Streamlit's own test runner (AppTest): it "clicks"
the button without opening a browser. Gemini is switched off, and results are
written to a temporary folder, so the tests never touch data/output.
"""

from __future__ import annotations

import io
import os
import sys
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.compose.llm import NoLLM                               # noqa: E402
from src.evaluate.compliance import Check, ComplianceReport     # noqa: E402
from src.pipeline import STAGES, run_pipeline                   # noqa: E402
from src.ui import views as V                                   # noqa: E402

SAMPLE_PDF = os.path.join(ROOT, "data", "input", "sample_paper.pdf")
PACKAGE = os.path.join(ROOT, "data", "input", "sample_verified_content.json")


# ---------------------------------------------------------------------------
# The pipeline both front ends share
# ---------------------------------------------------------------------------

def test_pipeline_reports_every_stage_in_order(tmp_path):
    if not os.path.exists(SAMPLE_PDF):
        pytest.skip("run tools/make_sample_paper.py first")
    seen = []
    result = run_pipeline(SAMPLE_PDF, outdir=str(tmp_path), llm=NoLLM("test"),
                          on_stage=lambda stage, r: seen.append(stage))
    assert seen == [key for key, _ in STAGES]
    assert os.path.exists(result.docx_path)
    assert result.compliance.after_score > result.compliance.before_score
    # every stage has a one-line fact for the web page
    assert all(V.stage_fact(stage, result) for stage in seen)


def test_pipeline_accepts_a_content_package(tmp_path):
    seen = []
    result = run_pipeline(PACKAGE, outdir=str(tmp_path), llm=NoLLM("test"),
                          on_stage=lambda stage, r: seen.append(stage))
    assert result.is_package and result.source_doc is None
    assert seen == [key for key, _ in STAGES]
    assert "content package" in V.stage_fact("ingest", result)


def test_pipeline_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_pipeline(str(tmp_path / "nope.pdf"), outdir=str(tmp_path), llm=NoLLM("test"))


# ---------------------------------------------------------------------------
# The HTML pieces
# ---------------------------------------------------------------------------

def test_stage_list_shows_done_running_and_waiting():
    markup = V.stage_list({"ingest": "3 pages"}, running="structure")
    assert markup.count('class="stage stage-done"') == 1
    assert markup.count('class="stage stage-running"') == 1
    assert markup.count('class="stage stage-waiting"') == len(STAGES) - 2
    assert "3 pages" in markup
    assert V.next_stage("ingest") == "structure" and V.next_stage("score") is None


def test_score_track_with_and_without_a_before_score():
    both = V.score_track(8.8, 94.1)
    assert "8.8%" in both and "94.1%" in both and "dot-before" in both
    package = V.score_track(None, 93.3)
    assert "dot-before" not in package and "93.3%" in package


def test_text_is_escaped():
    assert V.esc("<b>&") == "&lt;b&gt;&amp;"


def test_compliance_rows_mark_weak_checks():
    report = ComplianceReport(checks=[
        Check("a", "Abstract 150-250 words", "Front matter", 0.0, 0.0, "83 words", "83 words"),
        Check("b", "Two-column body", "Layout", None, 1.0, "", "two columns"),
    ], before_score=0.0, after_score=50.0)
    markup = V.compliance_rows(report)
    assert markup.count('class="weak"') == 1
    assert "n/a" in markup and "Overall" in markup


def test_zip_folder(tmp_path):
    assert V.zip_folder(str(tmp_path)) is None              # nothing to zip
    (tmp_path / "figure_1.png").write_bytes(b"png")
    names = zipfile.ZipFile(io.BytesIO(V.zip_folder(str(tmp_path)))).namelist()
    assert names == ["figure_1.png"]


# ---------------------------------------------------------------------------
# The web page itself
# ---------------------------------------------------------------------------

def test_web_page_composes_the_sample_paper(tmp_path, monkeypatch):
    if not os.path.exists(SAMPLE_PDF):
        pytest.skip("run tools/make_sample_paper.py first")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("GEMINI_API_KEY", "")                  # rules only
    monkeypatch.setenv("IEEE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.chdir(ROOT)

    app = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=180).run()
    assert not app.exception
    paper_choice = next(r for r in app.radio if r.label == "Paper")
    assert paper_choice.value == "sample_paper.pdf"           # sample picked by default
    assert next(r for r in app.radio if r.label == "Paper size").value == "letter"

    app.button[0].click().run()
    assert not app.exception and not app.error
    page = " ".join(m.value for m in app.markdown)
    assert "Automated Detection of Pulmonary Nodules" in page
    assert "The IEEE manuscript" in page                      # the score track
    assert os.path.exists(tmp_path / "sample_paper_IEEE.docx")
    assert [t.label for t in app.tabs][:2] == ["Compliance checks", "Section mapping"]
    assert app.tabs[-1].label.startswith("IEEE rulebook")
    assert "LAY-02" in page                                   # rule IDs shown on the checks
