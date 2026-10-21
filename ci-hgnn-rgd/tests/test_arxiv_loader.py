import json

from stage02_helpers import arxiv_record

from ci_hgnn_rgd.data.inspection import inspect_text
from ci_hgnn_rgd.data.sources.arxiv import adapt
from ci_hgnn_rgd.data.streaming import events


def test_arxiv_raw_id_normalized_id_version_and_dates_are_separate(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(arxiv_record()))
    paper, issues = adapt(next(events(path, "arxiv-metadata")), synthetic=True)
    assert not issues
    assert paper.arxiv_id_raw == "2001.00001v2"
    assert paper.arxiv_id_normalized == "2001.00001"
    assert paper.arxiv_version == "v2"
    assert paper.doi_raw == "https://doi.org/10.1234/SYNTHETIC"
    assert paper.doi_normalized == "10.1234/synthetic"
    assert paper.publication_year == 2020
    assert str(paper.publication_date) == "2020-01-01"


def test_missing_optional_fields_are_null_not_invented(tmp_path):
    path = tmp_path / "snapshot.jsonl"
    raw = arxiv_record()
    raw.pop("doi")
    raw.pop("update_date")
    path.write_text(json.dumps(raw))
    paper, issues = adapt(next(events(path, "arxiv-metadata")), synthetic=True)
    assert not issues
    assert paper.doi_raw is None and paper.doi_normalized is None
    assert paper.update_date is None


def test_malformed_rows_are_retained_as_invalid_locators(tmp_path):
    path = tmp_path / "snapshot.jsonl"
    path.write_text(
        json.dumps(arxiv_record())
        + "\n{broken\n"
        + json.dumps(arxiv_record(id="2001.00002"))
        + "\n"
    )
    result = inspect_text("arxiv-metadata", path, tmp_path / "out", synthetic=True)
    assert result["quality"]["raw_record_count"] == 3
    assert result["quality"]["malformed_records"] == 1
    assert result["quality"]["valid_record_count"] == 2
    assert result["scope"]["total_records"] == 3
    assert sum(1 for _ in open(result["staging_path"], encoding="utf-8")) == 3


def test_missing_required_metadata_is_reported_without_dropping(tmp_path):
    path = tmp_path / "snapshot.jsonl"
    path.write_text(json.dumps(arxiv_record(title="", abstract=None, categories=None, versions=[])))
    result = inspect_text("arxiv-metadata", path, tmp_path / "out", synthetic=True)
    assert result["quality"]["invalid_record_count"] == 1
    assert result["quality"]["missing_counts"]["title"] == 1
    assert result["quality"]["missing_counts"]["date"] == 1
    assert result["quality"]["year_distribution"] == {}


def test_prefix_and_reservoir_scopes_do_not_extrapolate(tmp_path):
    path = tmp_path / "snapshot.jsonl"
    path.write_text("\n".join(json.dumps(arxiv_record(id=f"2001.{n:05d}")) for n in range(50)))
    prefix = inspect_text("arxiv-metadata", path, tmp_path / "out", max_records=10, synthetic=True)
    assert prefix["scope"]["total_records"] is None
    first = inspect_text(
        "arxiv-metadata", path, tmp_path / "out", sample_size=5, seed=7, synthetic=True
    )
    second = inspect_text(
        "arxiv-metadata", path, tmp_path / "out", sample_size=5, seed=7, synthetic=True
    )
    assert first["scope"]["total_records"] == 50
    assert first["quality"]["raw_record_count"] == 5

    def ids(result):
        return [
            json.loads(line)["record"]["source_record_id"]
            for line in open(result["staging_path"], encoding="utf-8")
        ]

    assert ids(first) == ids(second)
