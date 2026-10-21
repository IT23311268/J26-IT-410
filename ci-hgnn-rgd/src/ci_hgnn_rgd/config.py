"""Typed safe YAML configuration; paths are relative to the project, never the caller."""

from datetime import UTC
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    AwareDatetime,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from .schemas import SHA256, FrozenModel, NonEmpty, SciCiteClass


class ConfigurationError(ValueError):
    """Configuration failure with file/field context and without input values."""


class UniqueSafeLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigurationError("YAML mapping keys must be strings")
            if key in result:
                raise ConfigurationError(f"duplicate YAML key: {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _read_yaml(path: Path) -> dict:
    try:
        if path.stat().st_size > 1_048_576:
            raise ConfigurationError("configuration exceeds the 1 MiB limit")
        content = path.read_text(encoding="utf-8-sig")
        if any(isinstance(event, yaml.AliasEvent) for event in yaml.parse(content)):
            raise ConfigurationError("YAML aliases are not supported")
        document = yaml.load(content, Loader=UniqueSafeLoader)
        if not isinstance(document, dict):
            raise ConfigurationError("YAML document must be a mapping")
        return document
    except (OSError, UnicodeError, yaml.YAMLError, ConfigurationError) as error:
        mark = getattr(error, "problem_mark", None)
        location = f", line {mark.line + 1}" if mark else ""
        detail = str(error) if isinstance(error, ConfigurationError) else type(error).__name__
        raise ConfigurationError(f"{path}{location}: {detail}") from None


class ProjectSettings(FrozenModel):
    name: Literal["ci-hgnn-rgd"] = "ci-hgnn-rgd"
    contract_version: Literal["1.0"] = "1.0"


class RunSettings(FrozenModel):
    seed: Annotated[StrictInt, Field(ge=0, le=2**32 - 1)]
    cutoff: AwareDatetime
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    deterministic: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("cutoff")
    @classmethod
    def utc_cutoff(cls, value):
        return value.astimezone(UTC)


class ProjectPaths(FrozenModel):
    raw: Path
    interim: Path
    processed: Path
    models: Path
    graphs: Path
    reports: Path


class DatasetSelection(FrozenModel):
    dataset_id: NonEmpty
    version_id: NonEmpty | None = None
    sha256: SHA256 | None = None


class DataSettings(FrozenModel):
    datasets: tuple[DatasetSelection, ...]
    allow_downloads: Literal[False] = False
    require_verified_identifier_mapping: Literal[True] = True
    max_download_bytes: Annotated[StrictInt, Field(gt=0)] = 64 * 1024**2
    max_line_bytes: Annotated[StrictInt, Field(gt=0)] = 4 * 1024**2
    short_text_threshold: Annotated[StrictInt, Field(ge=0)] = 20
    long_text_threshold: Annotated[StrictInt, Field(gt=0)] = 10000

    @model_validator(mode="after")
    def unique_datasets(self):
        ids = [item.dataset_id for item in self.datasets]
        if len(ids) != len(set(ids)):
            raise ValueError("dataset selections must be unique")
        return self


class CitationIntentSettings(FrozenModel):
    enabled: Literal[False] = False
    taxonomy: Literal["scicite"] = "scicite"
    classes: tuple[SciCiteClass, ...]
    model_id: NonEmpty | None = None
    model_revision: NonEmpty | None = None

    @field_validator("classes")
    @classmethod
    def only_scicite(cls, values):
        if len(values) != 3 or set(values) != set(SciCiteClass):
            raise ValueError("SciCite must have exactly Background, Method, and Result")
        return values


class TopicSettings(FrozenModel):
    enabled: Literal[False] = False
    algorithm: Literal["unselected"] = "unselected"
    fit_on: Literal["train_only"] = "train_only"


class GraphSettings(FrozenModel):
    enabled: Literal[False] = False
    schema_version: Literal["1.0"] = "1.0"
    include_optional_nodes: bool = False
    missing_intent: Literal["preserve_unknown"] = "preserve_unknown"


class GNNSettings(FrozenModel):
    enabled: Literal[False] = False
    architecture: Literal["unselected"] = "unselected"
    fit_on: Literal["train_only"] = "train_only"


class GapScoringSettings(FrozenModel):
    enabled: Literal[False] = False
    formula_version: NonEmpty | None = None
    interpretation: Literal["unvalidated_hypothesis"] = "unvalidated_hypothesis"
    investigation_evidence: Literal["independent_of_citation_frequency"] = (
        "independent_of_citation_frequency"
    )


class EvaluationSettings(FrozenModel):
    enabled: Literal[False] = False
    split: Literal["temporal"] = "temporal"
    train_end: AwareDatetime | None = None
    validation_end: AwareDatetime | None = None
    test_end: AwareDatetime | None = None
    ground_truth_path: Path | None = None

    @model_validator(mode="after")
    def ordered_splits(self):
        boundaries = (self.train_end, self.validation_end, self.test_end)
        if any(value is not None for value in boundaries):
            if any(value is None for value in boundaries):
                raise ValueError("provide all three temporal split boundaries or none")
            if not self.train_end < self.validation_end < self.test_end:
                raise ValueError("split boundaries must satisfy train < validation < test")
        return self


class ProjectConfig(FrozenModel):
    project: ProjectSettings
    run: RunSettings
    paths: ProjectPaths
    data: DataSettings
    citation_intent: CitationIntentSettings
    topic_model: TopicSettings
    graph: GraphSettings
    gnn: GNNSettings
    gap_scoring: GapScoringSettings
    evaluation: EvaluationSettings

    @model_validator(mode="after")
    def no_future_test_window(self):
        if self.evaluation.test_end and self.evaluation.test_end > self.run.cutoff:
            raise ValueError("evaluation test_end cannot follow the run cutoff")
        return self


def _inside_root(value: Path, root: Path) -> Path:
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise ConfigurationError("configured paths must remain inside the project root")
    return resolved


def load_config(path: str | Path, *, project_root: str | Path | None = None) -> ProjectConfig:
    """Load base and same-directory sections without creating directories or fetching data."""
    path = Path(path).resolve()
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else (path.parent.parent if path.parent.name == "configs" else path.parent)
    )
    document = _read_yaml(path)
    includes = document.pop("includes", [])
    if not isinstance(includes, list) or not all(isinstance(item, str) for item in includes):
        raise ConfigurationError(f"{path}: includes must be a list of YAML filenames")
    for include in includes:
        if Path(include).name != include or Path(include).suffix not in {".yaml", ".yml"}:
            raise ConfigurationError(f"{path}: includes must name files in the same directory")
        include_path = (path.parent / include).resolve()
        if not include_path.is_relative_to(path.parent):
            raise ConfigurationError(
                f"{path}: include resolves outside the configuration directory"
            )
        section = _read_yaml(include_path)
        if "includes" in section or document.keys() & section.keys():
            raise ConfigurationError(
                f"{include_path}: recursive or duplicate sections are forbidden"
            )
        document.update(section)
    try:
        config = ProjectConfig.model_validate(document)
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(map(str, issue['loc']))}: {issue['msg']}"
            for issue in error.errors(include_input=False, include_url=False)
        )
        raise ConfigurationError(f"{path}: {details}") from None
    resolved_paths = {name: _inside_root(value, root) for name, value in config.paths}
    evaluation = config.evaluation.model_dump()
    if evaluation["ground_truth_path"] is not None:
        evaluation["ground_truth_path"] = _inside_root(evaluation["ground_truth_path"], root)
    resolved = config.model_dump()
    resolved["paths"] = resolved_paths
    resolved["evaluation"] = evaluation
    return ProjectConfig.model_validate(resolved)
