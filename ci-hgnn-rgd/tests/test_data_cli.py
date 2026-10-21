import json
import subprocess
import sys

from conftest import PROJECT
from stage02_helpers import scicite_record


def cli(tmp_path, *args):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "ci_hgnn_rgd",
            "data",
            *args,
            "--offline",
            "--config",
            str(PROJECT / "configs/base.yaml"),
            "--data-dir",
            str(tmp_path / "raw"),
            "--output-dir",
            str(tmp_path / "reports"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_data_list_and_local_inspection_exit_codes(tmp_path):
    result = cli(tmp_path, "list")
    assert result.returncode == 0 and len(json.loads(result.stdout)["datasets"]) == 6
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(scicite_record()))
    result = cli(tmp_path, "inspect", "--dataset", "scicite", "--path", str(path), "--synthetic")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["synthetic"]
    failure = cli(tmp_path, "acquire", "--dataset", "arxiv")
    assert failure.returncode == 1 and "offline" in failure.stderr


def test_validation_returns_nonzero_on_data_failures(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(scicite_record(label="unsupported")))
    result = cli(tmp_path, "validate", "--dataset", "scicite", "--path", str(path), "--synthetic")
    assert result.returncode == 1
    assert json.loads(result.stdout)["quality"]["invalid_record_count"] == 1


def test_optional_graph_block_does_not_prevent_other_commands(tmp_path):
    result = cli(tmp_path, "inspect", "--dataset", "mag_year")
    assert result.returncode == 1
    result = cli(tmp_path, "readiness")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["readiness"]["mag-year-h2gb"]["readiness"] == "BLOCKED"


def test_invalid_dataset_and_invalid_limits_return_failure(tmp_path):
    assert cli(tmp_path, "inspect", "--dataset", "invented-source").returncode == 1
    assert cli(tmp_path, "inspect", "--dataset", "arxiv", "--max-records", "0").returncode == 2


def test_reports_and_audit_commands_create_all_required_files(tmp_path):
    for command in ("audit-identifiers", "report", "readiness"):
        result = cli(tmp_path, command)
        assert result.returncode == 0, result.stderr
    for name in (
        "dataset_schema_report",
        "data_quality_report",
        "identifier_compatibility",
        "stage02_readiness",
    ):
        for suffix in (".json", ".md"):
            assert (tmp_path / "reports" / (name + suffix)).exists()
    document = json.loads((tmp_path / "reports/identifier_compatibility.json").read_text())
    assert document["pairs"] == []
    assert document["scientific_stages_executed"] == []
