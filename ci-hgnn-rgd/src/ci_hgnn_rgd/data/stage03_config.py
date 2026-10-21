"""Stage 03 settings, separate from the unchanged Stage 01/02 configuration contract."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StrictInt, model_validator

from ..config import ConfigurationError, _read_yaml
from ..schemas import FrozenModel

Positive = Annotated[StrictInt, Field(gt=0)]


class InputSource(FrozenModel):
    dataset: Literal[
        "scicite", "arxiv-metadata", "ogbn-arxiv", "mag-year-h2gb", "openalex", "semantic-scholar"
    ]
    path: Path | None = None
    version: str | None = None
    format: Literal["raw", "stage02", "graph_tables"] = "raw"
    synthetic: bool = False


class CleaningSettings(FrozenModel):
    unicode_form: Literal["NFC"] = "NFC"
    preserve_raw: Literal[True] = True
    quarantine_invalid: Literal[True] = True
    repair_suspect_encoding: Literal[False] = False


class ResolutionSettings(FrozenModel):
    max_candidates: Positive = 50
    max_cluster_size: Positive = 1000
    probable_title_threshold: Annotated[float, Field(ge=0, le=1)] = 0.94
    material_title_conflict_threshold: Annotated[float, Field(ge=0, le=1)] = 0.35
    year_tolerance: Annotated[StrictInt, Field(ge=0)] = 1
    candidate_methods: tuple[Literal["identifier", "title_year"], ...] = (
        "identifier",
        "title_year",
    )
    automatic_merge: Literal["EXACT_VERIFIED_ONLY"] = "EXACT_VERIFIED_ONLY"
    author_matching: Literal["source_local_only"] = "source_local_only"
    duplicate_handling: Literal["retain_all_with_decisions"] = "retain_all_with_decisions"
    conflict_handling: Literal["retain_separate"] = "retain_separate"


class TemporalSettings(FrozenModel):
    policy: Literal[
        "STRICT_POINT_IN_TIME", "PUBLICATION_DATE_PROXY", "EXPLORATORY_NO_TEMPORAL_CLAIM"
    ] = "STRICT_POINT_IN_TIME"
    cutoff: AwareDatetime | None = None
    split_policy: Literal["preserve_official"] = "preserve_official"
    near_duplicate_threshold: Annotated[float, Field(ge=0, le=1)] = 0.94
    near_candidates_per_context: Positive = 20
    max_near_comparisons: Positive = 100000


class IntegrationSettings(FrozenModel):
    inputs: tuple[InputSource, ...] = ()
    discover_stage02: bool = True
    output_dir: Path = Path("data/processed/stage03")
    report_dir: Path = Path("artifacts/reports")
    batch_size: Positive = 1000
    memory_budget_mb: Annotated[StrictInt, Field(ge=16)] = 128
    max_records: Positive | None = None
    seed: Annotated[StrictInt, Field(ge=0, le=2**32 - 1)] = 42
    storage_format: Literal["auto", "jsonl", "parquet"] = "auto"
    offline: Literal[True] = True
    enrichment: Literal["local_metadata_only"] = "local_metadata_only"
    overwrite_runs: Literal[False] = False

    @model_validator(mode="after")
    def unique_sources(self):
        datasets = [item.dataset for item in self.inputs]
        if len(datasets) != len(set(datasets)):
            raise ValueError("each input dataset must be configured once")
        return self


class Stage03Config(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    cleaning: CleaningSettings = Field(default_factory=CleaningSettings)
    entity_resolution: ResolutionSettings = Field(default_factory=ResolutionSettings)
    temporal_integrity: TemporalSettings = Field(default_factory=TemporalSettings)
    integration: IntegrationSettings = Field(default_factory=IntegrationSettings)


def load_stage03_config(path: Path | None, project_root: Path) -> Stage03Config:
    """Reuse duplicate-key/alias-safe YAML loading; includes stay in the same directory."""
    path = Path(path or project_root / "configs/integration.yaml").resolve()
    document = _read_yaml(path)
    includes = document.pop("includes", [])
    if not isinstance(includes, list) or not all(isinstance(item, str) for item in includes):
        raise ConfigurationError("Stage 03 includes must be a list of YAML filenames")
    for name in includes:
        if Path(name).name != name or Path(name).suffix not in {".yaml", ".yml"}:
            raise ConfigurationError("Stage 03 includes must stay in the configuration directory")
        included = (path.parent / name).resolve()
        if not included.is_relative_to(path.parent):
            raise ConfigurationError("Stage 03 include escapes configuration directory")
        section = _read_yaml(included)
        if "includes" in section or document.keys() & section.keys():
            raise ConfigurationError("recursive or duplicate Stage 03 sections are forbidden")
        document.update(section)
    config = Stage03Config.model_validate(document)
    values = config.model_dump()
    for field in ("output_dir", "report_dir"):
        value = (project_root / getattr(config.integration, field)).resolve()
        if not value.is_relative_to(project_root):
            raise ConfigurationError("Stage 03 configured output paths must stay in project root")
        values["integration"][field] = value
    for source in values["integration"]["inputs"]:
        if source["path"] is not None:
            source["path"] = (project_root / source["path"]).resolve()
    return Stage03Config.model_validate(values)
