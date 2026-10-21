import shutil
from pathlib import Path

import pytest
from conftest import PROJECT

from ci_hgnn_rgd.config import ConfigurationError, load_config


def copy_config(tmp_path):
    shutil.copytree(PROJECT / "configs", tmp_path / "configs")
    return tmp_path / "configs/base.yaml"


def test_project_relative_paths_are_independent_of_working_directory(tmp_path, monkeypatch):
    path = copy_config(tmp_path)
    monkeypatch.chdir(PROJECT.parent)
    config = load_config(path)
    assert config.paths.raw == tmp_path / "data/raw"
    assert config.paths.reports == tmp_path / "artifacts/reports"
    assert not config.paths.raw.exists()
    assert tuple(item.value for item in config.citation_intent.classes) == (
        "Background",
        "Method",
        "Result",
    )


@pytest.mark.parametrize(
    "old,new,message",
    [
        ("seed: 42", "seed: true", "run.seed"),
        ("seed: 42", "seed: -1", "run.seed"),
        ("seed: 42", "seed: 4294967296", "run.seed"),
        ("seed: 42", "seed: 3.5", "run.seed"),
        ("log_level: INFO", "log_level: invalid", "run.log_level"),
        ("23:59:59Z", "23:59:59", "run.cutoff"),
        ("device: auto", "device: gpu42", "run.device"),
        ("raw: data/raw", "raw: ../outside", "project root"),
        ("seed: 42", "seed: 42\n  typographical_error: true", "typographical_error"),
    ],
)
def test_invalid_base_configuration(tmp_path, old, new, message):
    path = copy_config(tmp_path)
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(ConfigurationError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    "content",
    [
        "run: {seed: 1, seed: 2}",
        "run: !!python/object/apply:os.system ['echo unsafe']",
        "x: &x [1]\ny: *x",
        "[not, a, mapping]",
        "run: [unterminated",
        "",
    ],
)
def test_unsafe_or_ambiguous_yaml_is_rejected(tmp_path, content):
    path = tmp_path / "base.yaml"
    path.write_text(content)
    with pytest.raises(ConfigurationError):
        load_config(path)


def test_missing_file_has_context(tmp_path):
    with pytest.raises(ConfigurationError, match="missing.yaml"):
        load_config(tmp_path / "missing.yaml")


def test_include_cannot_escape_config_directory(tmp_path):
    path = tmp_path / "base.yaml"
    path.write_text("includes: ['../outside.yaml']")
    with pytest.raises(ConfigurationError, match="same directory"):
        load_config(path)


def test_section_duplicates_are_rejected(tmp_path):
    path = copy_config(tmp_path)
    path.write_text(path.read_text() + "data: {}\n")
    with pytest.raises(ConfigurationError, match="duplicate sections"):
        load_config(path)


def test_scicite_cannot_be_expanded_by_config(tmp_path):
    path = copy_config(tmp_path)
    intent = path.parent / "citation_intent.yaml"
    intent.write_text(intent.read_text().replace("Result]", "Result, Investigates]"))
    with pytest.raises(ConfigurationError, match="citation_intent.classes"):
        load_config(path)


def test_future_stage_cannot_be_enabled(tmp_path):
    path = copy_config(tmp_path)
    gnn = path.parent / "gnn.yaml"
    gnn.write_text(gnn.read_text().replace("enabled: false", "enabled: true"))
    with pytest.raises(ConfigurationError, match="gnn.enabled"):
        load_config(path)


@pytest.mark.parametrize(
    "boundaries",
    [
        'train_end: "2024-01-01T00:00:00Z"\n  validation_end: null\n  test_end: null',
        (
            'train_end: "2024-01-01T00:00:00Z"\n'
            '  validation_end: "2023-01-01T00:00:00Z"\n  test_end: "2025-01-01T00:00:00Z"'
        ),
        (
            'train_end: "2023-01-01T00:00:00Z"\n'
            '  validation_end: "2024-01-01T00:00:00Z"\n  test_end: "2026-01-01T00:00:00Z"'
        ),
    ],
)
def test_temporal_split_config_must_be_complete_ordered_and_within_cutoff(tmp_path, boundaries):
    path = copy_config(tmp_path)
    (path.parent / "evaluation.yaml").write_text("evaluation:\n  " + boundaries + "\n")
    with pytest.raises(ConfigurationError):
        load_config(path)


def test_error_omits_unrecognized_secret_values(tmp_path):
    path = copy_config(tmp_path)
    path.write_text(path.read_text() + "api_key: do-not-echo-this-test-value\n")
    with pytest.raises(ConfigurationError) as caught:
        load_config(path)
    assert "api_key" in str(caught.value)
    assert "do-not-echo-this-test-value" not in str(caught.value)


def test_explicit_project_root(tmp_path):
    path = copy_config(tmp_path)
    root = tmp_path / "explicit"
    config = load_config(path, project_root=root)
    assert config.paths.raw == root / "data/raw"
    assert not root.exists()


def test_no_absolute_paths_are_hardcoded(config):
    assert all(isinstance(value, Path) for _, value in config.paths)
    assert all(value.is_relative_to(PROJECT) for _, value in config.paths)
