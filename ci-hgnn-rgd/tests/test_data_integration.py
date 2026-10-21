"""Real-file integration tests: never replace missing data with synthetic fixtures."""

import os
from pathlib import Path

import pytest

from ci_hgnn_rgd.data.inspection import inspect_text
from ci_hgnn_rgd.data.sources.mag_year import MAGYearAdapter
from ci_hgnn_rgd.data.sources.ogbn_arxiv import OGBNArxivAdapter


def local_source(variable):
    value = os.environ.get(variable)
    if not value or not Path(value).exists():
        pytest.skip(f"{variable} is not configured with genuine local dataset files")
    return Path(value)


@pytest.mark.integration
def test_real_scicite_principal_splits(tmp_path):
    result = inspect_text("scicite", local_source("CI_HGNN_SCICITE_PATH"), tmp_path)
    assert result["scope"]["input_exhausted"]
    assert result["quality"]["raw_record_count"] > 0
    assert not result["synthetic"]
    assert set(result["quality"]["label_distribution_by_split"]) == {"train", "dev", "test"}


@pytest.mark.integration
def test_real_arxiv_bounded_schema(tmp_path):
    result = inspect_text(
        "arxiv-metadata", local_source("CI_HGNN_ARXIV_PATH"), tmp_path, max_records=1000
    )
    assert result["quality"]["raw_record_count"] > 0
    assert "id" in result["observed_schema"]["fields"]


@pytest.mark.integration
@pytest.mark.parametrize(
    "variable,adapter",
    [("CI_HGNN_OGB_ROOT", OGBNArxivAdapter), ("CI_HGNN_MAG_ROOT", MAGYearAdapter)],
)
def test_real_official_graph_cache(variable, adapter):
    root = local_source(variable)
    if os.environ.get("CI_HGNN_TRUST_GRAPH_CACHE") != "1":
        pytest.skip(
            "graph cache trust not explicitly configured; unsafe deserialization is disabled"
        )
    report = adapter().inspect(adapter().load(root, trusted_cache=True), max_edges=10000)
    assert not report["issues"]
