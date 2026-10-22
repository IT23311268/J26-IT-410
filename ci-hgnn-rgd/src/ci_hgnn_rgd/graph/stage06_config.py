"""Additive Stage 06 configuration; the frozen foundation graph.yaml is unchanged."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StrictInt, model_validator

from ..config import _read_yaml
from ..data.temporal_integrity import TemporalPolicy, parse_cutoff
from ..schemas import FrozenModel

Positive = Annotated[StrictInt, Field(gt=0)]


class GraphInputs(FrozenModel):
    stage03_root: Path = Path("data/processed/stage03-topics")
    stage03_run: Path | None = None
    stage05_root: Path = Path("data/processed/stage05")
    topic_run: Path | None = None
    topic_export: Path | None = None
    embedding_run: Path | None = None
    intent_export: Path | None = None
    intent_model_root: Path = Path("artifacts/models/citation_intent")


class GraphFeatures(FrozenModel):
    paper_embedding: Literal["specter2", "scibert", "none"] = "specter2"
    missing_embedding_strategy: Literal["masked_zero", "error"] = "masked_zero"
    include_publication_year: bool = True
    max_tensor_bytes: Positive = 268435456


class GraphTemporal(FrozenModel):
    policy: TemporalPolicy | None = None
    cutoff: str | None = None
    snapshot_id: str | None = None
    model_availability_policy: Literal["require_produced_by_cutoff"] = "require_produced_by_cutoff"
    split_cutoffs: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def chronological_windows(self):
        if set(self.split_cutoffs) - {"train", "dev", "test"}:
            raise ValueError("split_cutoffs accepts train/dev/test only")
        dates = [
            parse_cutoff(self.split_cutoffs[k])
            for k in ("train", "dev", "test")
            if k in self.split_cutoffs
        ]
        if dates != sorted(set(dates)):
            raise ValueError("split cutoffs must be strictly chronological")
        if self.cutoff:
            parse_cutoff(self.cutoff)
        return self


class Stage06Config(FrozenModel):
    schema_version: Literal["stage06-v1"] = "stage06-v1"
    inputs: GraphInputs = Field(default_factory=GraphInputs)
    features: GraphFeatures = Field(default_factory=GraphFeatures)
    temporal: GraphTemporal = Field(default_factory=GraphTemporal)
    output_dir: Path = Path("data/processed/stage06")
    report_dir: Path = Path("artifacts/reports/stage06")
    include_authors: bool = True
    include_topics: bool = True
    include_research_fields: bool = False
    citation_intent_representation: Literal["edge_attributes", "context_nodes", "both"] = (
        "edge_attributes"
    )
    aggregate_method: Literal["mean_probability"] = "mean_probability"
    add_reverse_message_passing_edges: bool = True
    export_predicted_relations_separately: bool = True
    allow_rule_candidate_edges: bool = False
    masked_relations: tuple[str, ...] = ()
    max_papers: Positive = 10000
    max_edges: Positive = 1000000
    batch_size: Positive = 256
    sqlite_cache_mib: Positive = 16
    max_record_bytes: Positive = 2097152
    field: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    offline: Literal[True] = True

    @model_validator(mode="after")
    def ranges(self):
        if self.start_year and self.end_year and self.start_year > self.end_year:
            raise ValueError("start_year must precede end_year")
        return self


def load_graph_config(path: Path, *, project_root: Path) -> Stage06Config:
    root = project_root.resolve()
    data = _read_yaml(path)
    for name in data.pop("includes", []):
        if Path(name).name != name:
            raise ValueError("graph includes must be sibling filenames")
        included = _read_yaml(path.parent / name)
        if "includes" in included or data.keys() & included.keys():
            raise ValueError("duplicate or recursive graph configuration include")
        data.update(included)
    config = Stage06Config.model_validate(data)
    values = config.model_dump(mode="python")

    def inside(value):
        resolved = (root / value).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("Stage 06 paths must remain inside the project root")
        return resolved

    for key, value in values["inputs"].items():
        if value is not None:
            values["inputs"][key] = inside(value)
    for key in ("output_dir", "report_dir"):
        values[key] = inside(values[key])
    return Stage06Config.model_validate(values)
