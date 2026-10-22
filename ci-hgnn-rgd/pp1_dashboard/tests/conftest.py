"""Viewer tests; demo fixtures never establish scientific findings."""

from pathlib import Path

import pytest
from app.main import create_app
from fastapi.testclient import TestClient

DASHBOARD = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def client():
    return TestClient(create_app())


@pytest.fixture(scope="session")
def real(client):
    if not (DASHBOARD.parent / "data/processed/stage06/latest.json").exists():
        pytest.skip("Published genuine graph not installed on this host")
    response = client.get("/api/overview")
    assert response.status_code == 200
    assert response.json()["status"] == "available", response.text
    return client


@pytest.fixture
def missing(tmp_path):
    return TestClient(create_app(tmp_path))
