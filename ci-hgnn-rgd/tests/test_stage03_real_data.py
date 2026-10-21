"""Opt-in genuine local integration; never substitute synthetic records for missing sources."""

import os
from pathlib import Path

import pytest

from ci_hgnn_rgd.data.integration import run_stage03
from ci_hgnn_rgd.data.integration_manifest import iter_table
from ci_hgnn_rgd.data.stage03_config import Stage03Config


@pytest.mark.integration
@pytest.mark.parametrize(
    "dataset,variable,version_variable,format",
    [
        ("scicite", "CI_HGNN_SCICITE_PATH", None, "raw"),
        ("arxiv-metadata", "CI_HGNN_ARXIV_PATH", None, "raw"),
        ("ogbn-arxiv", "CI_HGNN_OGB_STAGE03_TABLES", "CI_HGNN_OGB_VERSION", "graph_tables"),
        ("mag-year-h2gb", "CI_HGNN_MAG_STAGE03_TABLES", "CI_HGNN_MAG_VERSION", "graph_tables"),
    ],
)
def test_bounded_real_stage03_integration(
    tmp_path, config, dataset, variable, version_variable, format
):
    path = os.environ.get(variable)
    if not path or not Path(path).exists():
        pytest.skip(f"{variable} not configured with genuine local records")
    version = os.environ.get(version_variable) if version_variable else None
    if version_variable and not version:
        pytest.skip(f"{version_variable} required for source-local graph scope")
    settings = Stage03Config.model_validate(
        {
            "integration": {
                "inputs": [
                    {"dataset": dataset, "path": path, "version": version, "format": format}
                ],
                "discover_stage02": False,
                "max_records": 1000,
                "output_dir": tmp_path / "runs",
                "report_dir": tmp_path / "reports",
                "storage_format": "jsonl",
            }
        }
    )
    result = run_stage03(config, settings)
    run = Path(result["run_dir"])
    assert result["sources"][dataset]["status"] == "PROCESSED"
    assert 0 < result["table_counts"]["cleaned_records"] <= 1000
    assert all(
        not provenance["synthetic"]
        for row in iter_table(run, "provenance")
        for provenance in row["provenance"]
    )
    if dataset == "scicite":
        contexts = list(iter_table(run, "citation_contexts"))
        assert contexts and all(
            row["normalized_intent_label"] in {"Background", "Method", "Result"} for row in contexts
        )
        assert all(row["citing_paper_id"] is None for row in contexts)
        assert result["temporal"]["unknown_availability"] > 0
