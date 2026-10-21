"""Explicit stage boundaries and in-memory foundation manifests, without scientific execution."""

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from . import CONTRACT_VERSION, __version__
from .config import ProjectConfig
from .data.validation import validate_corpus
from .schemas import Corpus


@dataclass(frozen=True)
class StageDefinition:
    number: int
    name: str
    implemented: bool = False


# Proposed workflow, not a claim that any later stage has been executed or validated.
STAGES = (
    StageDefinition(1, "Project foundation and architecture", True),
    StageDefinition(2, "Dataset acquisition and source schema audits", True),
    StageDefinition(3, "Entity resolution, provenance and temporal splits", True),
    StageDefinition(4, "Citation-intent classification", True),
    StageDefinition(5, "Topic identification"),
    StageDefinition(6, "Independent direct-investigation evidence"),
    StageDefinition(7, "Heterogeneous graph construction"),
    StageDefinition(8, "Heterogeneous graph representation learning"),
    StageDefinition(9, "Baselines and ablation protocols"),
    StageDefinition(10, "Citation-asymmetry hypothesis scoring"),
    StageDefinition(11, "Independent temporal and expert evaluation"),
    StageDefinition(12, "Evidence explanations and reporting"),
)


def require_stage(number: int) -> StageDefinition:
    if type(number) is not int or not 1 <= number <= len(STAGES):
        raise ValueError("stage number must be an integer from 1 to 12")
    stage = STAGES[number - 1]
    if not stage.implemented:
        raise NotImplementedError(f"Stage {number:02d}: {stage.name} is not implemented")
    return stage


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def prepare_run(
    config: ProjectConfig, corpus: Corpus | None = None, *, allow_synthetic: bool = False
) -> dict:
    """Validate inputs and return an audit manifest; never write or execute a later stage."""
    corpus = Corpus() if corpus is None else corpus
    validate_corpus(corpus, config.run.cutoff, allow_synthetic=allow_synthetic)
    ordered = {}
    for field, id_field in (
        ("papers", "paper_id"),
        ("authors", "author_id"),
        ("topics", "topic_id"),
        ("citations", "edge_id"),
        ("contexts", "context_id"),
        ("paper_topic_evidence", "evidence_id"),
    ):
        ordered[field] = sorted(getattr(corpus, field), key=lambda item: getattr(item, id_field))
    return {
        "run_id": str(uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "component_version": __version__,
        "contract_version": CONTRACT_VERSION,
        "python_version": platform.python_version(),
        "stage": 1,
        "status": "foundation_validated",
        "synthetic_inputs_allowed": allow_synthetic,
        "config": config.model_dump(mode="json"),
        "config_sha256": fingerprint(config),
        "corpus_sha256": fingerprint(ordered),
        "counts": {name: len(records) for name, records in ordered.items()},
        "scientific_stages_executed": [],
        "stages": [asdict(stage) for stage in STAGES],
    }
