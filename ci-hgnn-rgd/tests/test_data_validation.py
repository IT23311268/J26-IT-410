import json
from pathlib import Path

import pytest
from stage02_helpers import arxiv_record

from ci_hgnn_rgd.data.reporting import generate_reports
from ci_hgnn_rgd.data.service import DataService


def test_synthetic_inspection_cannot_establish_real_readiness(tmp_path, config):
    source = tmp_path / "snapshot.jsonl"
    source.write_text(json.dumps(arxiv_record()))
    service = DataService(config, data_dir=tmp_path / "raw", output_dir=tmp_path / "reports")
    report = service.inspect("arxiv", path=source, synthetic=True)
    assert report["quality"]["raw_record_count"] == 1
    result = generate_reports(service)
    assert result["readiness"]["arxiv-metadata"]["readiness"] == "NOT_VALIDATED"
    schema = json.loads((tmp_path / "reports/dataset_schema_report.json").read_text())
    assert "id" in schema["datasets"]["arxiv-metadata"]["observed"]["fields"]
    assert schema["datasets"]["mag-year-h2gb"]["observed"] is None


def test_inspection_registers_real_executed_metadata_and_keeps_file(tmp_path, config):
    source = tmp_path / "snapshot.jsonl"
    raw = json.dumps(arxiv_record())
    source.write_text(raw)
    service = DataService(config, data_dir=tmp_path / "raw", output_dir=tmp_path / "reports")
    service.acquire("arxiv", path=source, offline=True)
    assert service.registry.states["arxiv-metadata"].status == "AVAILABLE"
    service.inspect("arxiv", synthetic=True)
    state = service.registry.states["arxiv-metadata"]
    assert state.status == "VALIDATED" and state.verified_at and state.inspection_path
    assert state.raw_record_count == 1 and state.valid_record_count == 1
    assert source.read_text() == raw


@pytest.mark.parametrize("action", ["replace_source", "failed_inspection"])
def test_prior_inspection_is_not_reused_for_unvalidated_current_source(tmp_path, config, action):
    source = tmp_path / "original.jsonl"
    source.write_text(json.dumps(arxiv_record()))
    service = DataService(config, output_dir=tmp_path / "reports")
    previous = service.inspect("arxiv", path=source, synthetic=True)
    if action == "replace_source":
        replacement = tmp_path / "replacement.jsonl"
        replacement.write_text("{}")
        service.acquire("arxiv", path=replacement, offline=True)
    else:
        with pytest.raises(OSError):
            service.inspect("arxiv", path=tmp_path / "missing.jsonl", synthetic=True)
    result = generate_reports(service)
    assert result["readiness"]["arxiv-metadata"]["readiness"] == "NOT_VALIDATED"
    for name in ("dataset_schema_report", "data_quality_report"):
        report = json.loads((tmp_path / "reports" / (name + ".json")).read_text())
        assert report["datasets"]["arxiv-metadata"]["observed"] is None
    assert Path(previous["inspection_path"]).exists()  # Retain the historical audit artifact.
