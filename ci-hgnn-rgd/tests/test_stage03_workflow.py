"""Synthetic end-to-end storage, CLI and provenance tests; no scholarly ground truth."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import PROJECT
from stage02_helpers import scicite_record

from ci_hgnn_rgd.data.integration import run_stage03
from ci_hgnn_rgd.data.integration_manifest import (
    RunPublication,
    TableStore,
    iter_table,
    resolve_run,
    trace_record,
)
from ci_hgnn_rgd.data.stage03_config import (
    InputSource,
    IntegrationSettings,
    Stage03Config,
    load_stage03_config,
)
from ci_hgnn_rgd.data.stage03_reporting import REPORTS, validate_run


def settings(tmp_path, source, **changes):
    values = {
        "inputs": (InputSource(dataset="scicite", path=source, synthetic=True),),
        "discover_stage02": False,
        "output_dir": tmp_path / "processed",
        "report_dir": tmp_path / "reports",
        "storage_format": "jsonl",
        "batch_size": 2,
    }
    values.update(changes)
    return Stage03Config(integration=IntegrationSettings(**values))


def source_file(tmp_path, records=None):
    path = tmp_path / "train.jsonl"
    records = records or [scicite_record()]
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    return path


def test_offline_partial_context_integration_preserves_rows_and_trace(tmp_path, config):
    first = scicite_record()
    second = {**first, "string": first["string"] + " Another distinct context."}
    source = source_file(tmp_path, [first, second, first])
    raw = source.read_bytes()
    with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
        result = run_stage03(config, settings(tmp_path, source))
    run = Path(result["run_dir"])
    assert source.read_bytes() == raw
    contexts = list(iter_table(run, "citation_contexts"))
    assert len(contexts) == 3 and len({row["context_id"] for row in contexts}) == 3
    assert all(row["source_split"] == "train" for row in contexts)
    assert all(row["citing_paper_id"] is None for row in contexts)
    assert list(iter_table(run, "papers")) == []
    assert len(list(iter_table(run, "citations"))) == 3
    assert result["temporal"]["excluded"] == 3
    manifest = json.loads((run / "integration_manifest.json").read_text())
    assert manifest["cleaning"]["duplicate_records"] == 1
    assert manifest["scientific_stages_executed"] == []
    traced = trace_record(run, contexts[0]["context_id"])
    assert {item["table"] for item in traced["records"]} >= {
        "citation_contexts",
        "cleaned_records",
        "provenance",
        "temporal_eligibility",
    }
    for report in REPORTS:
        document = json.loads((run / (report + ".json")).read_text())
        assert document["observation_kind"] == "synthetic_software_fixture"
        assert document["real_sources"] == []
    assert validate_run(run)["valid"]


def test_failed_publication_does_not_replace_previous_run(tmp_path):
    previous = RunPublication(tmp_path, {"synthetic": True})
    final = previous.publish({"synthetic": True})
    second = RunPublication(tmp_path, {"synthetic": True})
    with patch("ci_hgnn_rgd.data.integration_manifest.os.rename", side_effect=OSError("failure")):
        with pytest.raises(OSError):
            second.publish({"synthetic": True})
    assert resolve_run(tmp_path) == final
    assert second.pending.is_dir()


def test_repeat_run_context_ids_stable_and_prior_outputs_preserved(tmp_path, config):
    source = source_file(tmp_path)
    options = settings(tmp_path, source)
    first = Path(run_stage03(config, options)["run_dir"])
    second = Path(run_stage03(config, options)["run_dir"])
    assert first != second and first.is_dir() and second.is_dir()
    assert [row["context_id"] for row in iter_table(first, "citation_contexts")] == [
        row["context_id"] for row in iter_table(second, "citation_contexts")
    ]
    assert resolve_run(options.integration.output_dir) == second


def test_quarantine_retained_and_failed_validation_honest(tmp_path, config):
    source = source_file(tmp_path, [scicite_record(label="fabricated-label")])
    run = Path(run_stage03(config, settings(tmp_path, source))["run_dir"])
    quarantined = list(iter_table(run, "quarantine"))
    assert len(quarantined) == 1
    assert quarantined[0]["raw_values"]["label"] == "fabricated-label"
    assert not list(iter_table(run, "citation_contexts"))
    assert not validate_run(run)["valid"]


def test_checksum_validation_catches_changed_output(tmp_path, config):
    run = Path(run_stage03(config, settings(tmp_path, source_file(tmp_path)))["run_dir"])
    (run / "citation_contexts.jsonl").write_text("tampered\n")
    report = validate_run(run)
    assert not report["valid"] and "citation_contexts.jsonl" in report["checksum_mismatches"]


def test_trace_missing_record_fails(tmp_path, config):
    run = Path(run_stage03(config, settings(tmp_path, source_file(tmp_path)))["run_dir"])
    with pytest.raises(ValueError, match="record not present"):
        trace_record(run, "fabricated-paper")


def test_canonical_trace_expands_to_merge_decisions_and_raw_sources(tmp_path):
    store = TableStore(tmp_path, storage_format="jsonl")
    store.write(
        "papers",
        {
            "canonical_paper_id": "synthetic:canonical",
            "source_record_ids": [
                {"record_key": "synthetic:left"},
                {"record_key": "synthetic:right"},
            ],
        },
    )
    store.write(
        "match_decisions",
        {
            "left_record_key": "synthetic:left",
            "right_record_key": "synthetic:right",
            "decision": "EXACT_VERIFIED",
        },
    )
    store.write("provenance", {"record_key": "synthetic:left", "raw_values": {"title": "α = β"}})
    store.close()
    traced = trace_record(tmp_path, "synthetic:canonical")
    assert {row["table"] for row in traced["records"]} == {
        "papers",
        "match_decisions",
        "provenance",
    }
    with pytest.raises(ValueError):
        trace_record(tmp_path, "synthetic:canonical", limit=0)


def test_partitioned_parquet_has_stable_schema_and_complete_nested_payload(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    store = TableStore(tmp_path, batch_size=2, storage_format="parquet")
    rows = [
        {
            "canonical_paper_id": f"synthetic:{i}",
            "title": "α + β = E²",
            "publication_year": 2024,
            "provenance": [{"synthetic": True}],
            "optional": None if i == 0 else {"x": [1, 2]},
        }
        for i in range(5)
    ]
    for row in rows:
        store.write("papers", row)
    store.close()
    parts = sorted((tmp_path / "papers.parquet").glob("*.parquet"))
    assert len(parts) == 3
    table = pq.read_table(tmp_path / "papers.parquet")
    assert table.schema.field("publication_year").type == pa.int64()
    assert [json.loads(value) for value in table["record_json"].to_pylist()] == rows


def test_clean_phase_does_not_claim_integration(tmp_path, config):
    result = run_stage03(
        config, settings(tmp_path, source_file(tmp_path)), action="clean", dataset="scicite"
    )
    run = Path(result["run_dir"])
    assert len(list(iter_table(run, "cleaned_records"))) == 1
    assert not list(iter_table(run, "citation_contexts"))
    manifest = json.loads((run / "integration_manifest.json").read_text())
    assert manifest["identity"]["status"] == "NOT_EXECUTED"


def test_dry_run_creates_no_outputs(tmp_path, config):
    options = settings(tmp_path, source_file(tmp_path))
    result = run_stage03(config, options, dry_run=True)
    assert result["dry_run"]
    assert not options.integration.output_dir.exists()
    assert not options.integration.report_dir.exists()


@pytest.mark.parametrize(
    "field,value", [("batch_size", 0), ("memory_budget_mb", 1), ("max_records", -1)]
)
def test_invalid_resource_settings_rejected(field, value):
    with pytest.raises(ValueError):
        IntegrationSettings(**{field: value})


def test_separate_configuration_defaults_preserve_base_project_contract(config):
    assert "stage03" not in config.model_dump()
    stage03 = load_stage03_config(None, PROJECT)
    assert stage03.temporal_integrity.policy == "STRICT_POINT_IN_TIME"
    assert stage03.integration.offline
    assert stage03.integration.output_dir == PROJECT / "data/processed/stage03"


def run_cli(tmp_path, *arguments):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "ci_hgnn_rgd",
            "data",
            *arguments,
            "--output-dir",
            str(tmp_path / "processed"),
            "--report-dir",
            str(tmp_path / "reports"),
            "--storage-format",
            "jsonl",
            "--offline",
        ],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=40,
        check=False,
    )


@pytest.mark.parametrize(
    "action",
    [
        "clean",
        "normalize-identifiers",
        "generate-match-candidates",
        "resolve-entities",
        "deduplicate",
        "integrate",
        "stage03-run",
    ],
)
def test_processing_cli_commands_return_real_outcomes(tmp_path, action):
    source = source_file(tmp_path)
    completed = run_cli(
        tmp_path, action, "--dataset", "scicite", "--path", str(source), "--synthetic"
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["action"] == action


def test_audit_cli_and_informative_failures(tmp_path):
    assert run_cli(tmp_path, "integration-report").returncode == 1
    assert run_cli(tmp_path, "clean", "--dataset", "unknown-source").returncode == 1
    source = source_file(tmp_path)
    created = run_cli(
        tmp_path, "stage03-run", "--dataset", "scicite", "--path", str(source), "--synthetic"
    )
    assert created.returncode == 0, created.stderr
    run = Path(json.loads(created.stdout)["run_dir"])
    for action in ("audit-leakage", "validate-integrated", "integration-report"):
        assert run_cli(tmp_path, action).returncode == 0
    context = next(iter_table(run, "citation_contexts"))
    trace = run_cli(tmp_path, "trace-record", "--id", context["context_id"])
    assert trace.returncode == 0, trace.stderr
    assert run_cli(tmp_path, "trace-record", "--id", "missing").returncode == 1
    assert run_cli(tmp_path, "stage03-run", "--max-records", "0").returncode == 1
