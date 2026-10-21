import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import PROJECT


def run_cli(*args, cwd=PROJECT):
    return subprocess.run(
        [sys.executable, "-m", "ci_hgnn_rgd", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_help_exposes_required_commands():
    result = run_cli("--help")
    assert result.returncode == 0
    assert all(
        command in result.stdout for command in ("check-env", "validate-config", "show-pipeline")
    )


def test_installed_console_entry_point():
    binary = Path(sys.executable).parent / (
        "ci-hgnn-rgd.exe" if sys.platform == "win32" else "ci-hgnn-rgd"
    )
    result = subprocess.run(
        [str(binary), "--help"], text=True, capture_output=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr


def test_environment_command_reports_cpu_without_requiring_ml():
    result = run_cli("check-env", "--device", "cpu")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["ready"] and report["python_supported"]
    assert report["device"] == "cpu"


@pytest.mark.parametrize("cwd", [PROJECT, PROJECT.parent])
def test_default_config_command_works_from_project_or_repository_root(cwd):
    result = run_cli("validate-config", cwd=cwd)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "valid"
    assert not report["datasets_acquired"]
    assert report["scientific_stages_executed"] == []
    assert json.loads(result.stderr)["level"] == "INFO"


def test_explicit_config_works_outside_repository_and_writes_nothing(tmp_path):
    result = run_cli(
        "validate-config", "--config", str(PROJECT / "configs/base.yaml"), cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_missing_config_fails_without_writing(tmp_path):
    result = run_cli("validate-config", "--config", "missing.yaml", cwd=tmp_path)
    assert result.returncode == 1
    assert result.stdout == ""
    assert json.loads(result.stderr)["level"] == "ERROR"
    assert "missing.yaml" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_no_default_config_outside_repository_has_actionable_error(tmp_path):
    result = run_cli("validate-config", cwd=tmp_path)
    assert result.returncode == 1
    assert "--config" in result.stderr


def test_unknown_dataset_fails_config_validation(tmp_path, config):
    import yaml

    document = config.model_dump(mode="json")
    document["data"]["datasets"][0]["dataset_id"] = "invented-source"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document))
    result = run_cli("validate-config", "--config", str(path), "--project-root", str(PROJECT))
    assert result.returncode == 1
    assert "invented-source" in result.stderr


def test_show_pipeline_has_twelve_stages_and_only_authorized_stages_implemented():
    result = run_cli("show-pipeline")
    assert result.returncode == 0, result.stderr
    stages = json.loads(result.stdout)["stages"]
    assert [stage["number"] for stage in stages] == list(range(1, 13))
    assert [stage["number"] for stage in stages if stage["implemented"]] == [1, 2, 3, 4]


def test_module_imports_neither_download_nor_import_heavy_backends(tmp_path):
    code = """
import socket, sys, random
def forbidden(*args, **kwargs):
    raise AssertionError("network access during import")
socket.socket.connect = forbidden
state = random.getstate()
import ci_hgnn_rgd, pkgutil, importlib
for module in pkgutil.walk_packages(ci_hgnn_rgd.__path__, ci_hgnn_rgd.__name__ + "."):
    importlib.import_module(module.name)
assert state == random.getstate(), "import changed random seed"
assert not ({"torch", "torch_geometric", "transformers", "numpy", "pandas"} & sys.modules.keys())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
