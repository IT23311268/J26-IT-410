"""Atomic, auditable inspection state. No successful status is inferred from a loader."""

import json
import os
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from pydantic import Field, model_validator

from ..schemas import Count, FrozenModel, NonEmpty


def now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".tmp-",
            suffix=".json",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()  # Only the temporary file created by this function.


class DatasetStatus(StrEnum):
    NOT_CONFIGURED = "NOT_CONFIGURED"
    AVAILABLE = "AVAILABLE"
    DOWNLOADED = "DOWNLOADED"
    VALIDATED = "VALIDATED"
    PARTIALLY_VALIDATED = "PARTIALLY_VALIDATED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class DatasetState(FrozenModel):
    dataset_id: NonEmpty
    status: DatasetStatus = DatasetStatus.NOT_CONFIGURED
    version_id: str | None = None
    source_url: str
    source_organization: str
    purpose: str
    access_method: str
    licence_notes: str
    citation_requirements: str
    local_path: str | None = None
    original_format: str | None = None
    downloaded_at: str | None = None
    verified_at: str | None = None
    checksum: str | None = None
    checksum_trust: str = "not_available"
    raw_record_count: Count | None = None
    valid_record_count: Count | None = None
    invalid_record_count: Count | None = None
    detected_schema: dict = Field(default_factory=dict)
    quality: dict = Field(default_factory=dict)
    identifiers: dict = Field(default_factory=dict)
    labels: dict = Field(default_factory=dict)
    node_types: tuple[str, ...] = ()
    edge_types: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    provenance: tuple[dict, ...] = ()
    inspection_path: str | None = None
    staging_path: str | None = None
    index_path: str | None = None
    history: tuple[dict, ...] = ()

    @model_validator(mode="after")
    def consistent_counts(self):
        counts = (self.raw_record_count, self.valid_record_count, self.invalid_record_count)
        if all(value is not None for value in counts) and counts[0] != counts[1] + counts[2]:
            raise ValueError("raw record count must equal valid plus invalid records")
        return self


class InspectionRegistry:
    """Persistent counterpart to the Stage 01 version catalogue; single-writer CLI."""

    def __init__(self, path: Path):
        self.path = path
        self.states = {}
        if path.exists():
            document = json.loads(path.read_text(encoding="utf-8"))
            self.states = {
                key: DatasetState.model_validate(value)
                for key, value in document["datasets"].items()
            }

    def put(self, state: DatasetState, *, action: str) -> DatasetState:
        if state.status in {DatasetStatus.VALIDATED, DatasetStatus.PARTIALLY_VALIDATED}:
            if not state.verified_at or not state.inspection_path:
                raise ValueError("validated status requires an executed inspection and timestamp")
        if state.status == DatasetStatus.DOWNLOADED and not (
            state.local_path and state.downloaded_at
        ):
            raise ValueError("downloaded status requires a local file and acquisition timestamp")
        previous = self.states.get(state.dataset_id)
        history = list(previous.history) if previous else []
        history.append({"at": now(), "action": action, "status": state.status.value})
        values = state.model_dump()
        values["history"] = history
        state = DatasetState.model_validate(values)
        proposed = {**self.states, state.dataset_id: state}
        atomic_json(
            self.path,
            {
                "schema_version": "1.0",
                "datasets": {
                    key: value.model_dump(mode="json") for key, value in sorted(proposed.items())
                },
            },
        )
        self.states = proposed
        return state
