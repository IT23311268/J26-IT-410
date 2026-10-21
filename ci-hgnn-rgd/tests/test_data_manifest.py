import json
from unittest.mock import patch

import pytest

from ci_hgnn_rgd.data.manifests import DatasetState, DatasetStatus, InspectionRegistry, atomic_json


def state():
    return DatasetState(
        dataset_id="synthetic",
        source_url="https://example.org/synthetic",
        source_organization="Synthetic test source",
        purpose="Software test only",
        access_method="local fixture",
        licence_notes="Synthetic bytes",
        citation_requirements="Not scholarly evidence",
    )


def test_atomic_json_and_failed_replace_leave_prior_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_json(path, {"version": 1})
    with patch(
        "ci_hgnn_rgd.data.manifests.os.replace", side_effect=OSError("synthetic write failure")
    ):
        with pytest.raises(OSError):
            atomic_json(path, {"version": 2})
    assert json.loads(path.read_text()) == {"version": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_registry_persists_actual_status_actions(tmp_path):
    path = tmp_path / "registry.json"
    registry = InspectionRegistry(path)
    first = registry.put(state(), action="catalogued")
    values = first.model_dump()
    values.update(status=DatasetStatus.AVAILABLE, local_path=str(tmp_path))
    registry.put(DatasetState.model_validate(values), action="local_registered")
    recovered = InspectionRegistry(path).states["synthetic"]
    assert recovered.status == DatasetStatus.AVAILABLE
    assert [step["action"] for step in recovered.history] == ["catalogued", "local_registered"]


def test_cannot_mark_validated_merely_for_having_a_loader(tmp_path):
    values = state().model_dump()
    values["status"] = "VALIDATED"
    with pytest.raises(ValueError, match="executed inspection"):
        InspectionRegistry(tmp_path / "registry.json").put(
            DatasetState.model_validate(values), action="loader_exists"
        )
