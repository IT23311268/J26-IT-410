"""CLI surface tests; no command here trains or downloads a model."""

import json
import subprocess
import sys

from conftest import PROJECT


def run_cli(*arguments):
    return subprocess.run(
        [sys.executable, "-m", "ci_hgnn_rgd", *arguments],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


def test_stage04_commands_are_registered_without_import_time_model_downloads():
    result = run_cli("citation-intent", "--help")
    assert result.returncode == 0, result.stderr
    for command in (
        "inspect-data",
        "audit-leakage",
        "train-baselines",
        "train",
        "evaluate",
        "calibrate",
        "predict",
        "predict-file",
        "export-graph-evidence",
        "report",
    ):
        assert command in result.stdout


def test_stage04_cli_returns_nonzero_for_unknown_published_run():
    result = run_cli(
        "citation-intent",
        "predict",
        "--run-id",
        "run-does-not-exist",
        "--text",
        "synthetic input only",
        "--offline",
    )
    assert result.returncode == 1
    assert result.stdout == ""
    log = json.loads(result.stderr)
    assert log["level"] == "ERROR"
    assert "not found" in log["message"]
