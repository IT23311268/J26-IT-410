"""Isolation guarantees independent of browser rendering."""

import ast
import hashlib
import json

from conftest import DASHBOARD


def test_preexisting_code_and_consumed_artifacts_unchanged():
    baseline = DASHBOARD / "reports/isolation-before.json"
    if not baseline.exists():
        import pytest

        pytest.skip("Local pre-work hash baseline is not distributed with source")
    values = json.loads(baseline.read_text(encoding="utf-8"))
    repo = DASHBOARD.parent.parent
    changed = []
    for name, expected in values.items():
        path = repo / name
        if not path.is_file():
            changed.append(name)
            continue
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                changed.append(name)
    assert not changed, changed


def test_no_pipeline_imports_or_model_deserialization():
    for source in (DASHBOARD / "backend/app").rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for entry in ast.walk(tree):
            if isinstance(entry, ast.Import):
                assert all(not n.name.startswith("ci_hgnn_rgd") for n in entry.names)
            if isinstance(entry, ast.ImportFrom):
                assert not (entry.module or "").startswith("ci_hgnn_rgd")
        text = source.read_text(encoding="utf-8")
        for forbidden in (
            "torch.load",
            "pickle.load",
            "joblib.load",
            "write_text(",
            "write_bytes(",
            "mkdir(",
        ):
            assert forbidden not in text


def test_isolated_dependency_manifests():
    assert (DASHBOARD / "frontend/package.json").is_file()
    assert (DASHBOARD / "backend/requirements.txt").is_file()
    assert (DASHBOARD / "demo_data/dataset.json").is_file()
    fixture = json.loads((DASHBOARD / "demo_data/dataset.json").read_text(encoding="utf-8"))
    assert all(
        item["synthetic"] for name in ("nodes", "edges", "intents") for item in fixture[name]
    )
