"""Typed Stage 04 configuration kept separate from the frozen Stage 01 base YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StrictInt, model_validator

from ..config import ConfigurationError, _read_yaml
from ..schemas import FrozenModel, NonEmpty

PositiveInt = Annotated[StrictInt, Field(gt=0)]
ProbabilityValue = Annotated[float, Field(ge=0, le=1)]

SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"
SCIBERT_REVISION = "952d056ace307abd2cda7cee822a1d608deb20ed"


class IntentDatasetSettings(FrozenModel):
    name: Literal["scicite"] = "scicite"
    source: Literal["stage03"] = "stage03"
    stage03_output_dir: Path = Path("data/processed/stage03")
    stage03_run: Path | None = None
    citation_context_table: Literal["citation_contexts"] = "citation_contexts"
    preserve_official_splits: Literal[True] = True
    exclude_quarantined: Literal[True] = True
    allow_synthetic: Literal[False] = False


class IntentLabelSettings(FrozenModel):
    background: Literal[0] = 0
    method: Literal[1] = 1
    result: Literal[2] = 2

    def mapping(self) -> dict[str, int]:
        return {"background": self.background, "method": self.method, "result": self.result}


class IntentModelSettings(FrozenModel):
    name: Literal[SCIBERT_MODEL] = SCIBERT_MODEL
    revision: NonEmpty = SCIBERT_REVISION
    num_labels: Literal[3] = 3
    max_length: Annotated[StrictInt, Field(ge=16, le=512)] = 256
    trust_remote_code: Literal[False] = False


class IntentTrainingSettings(FrozenModel):
    mode: Literal["quick", "full"] = "full"
    epochs: PositiveInt = 4
    learning_rate: Annotated[float, Field(gt=0, le=1)] = 2e-5
    train_batch_size: PositiveInt = 16
    eval_batch_size: PositiveInt = 32
    gradient_accumulation_steps: PositiveInt = 1
    weight_decay: Annotated[float, Field(ge=0, le=1)] = 0.01
    warmup_ratio: ProbabilityValue = 0.1
    max_grad_norm: Annotated[float, Field(gt=0)] = 1.0
    seed: Annotated[StrictInt, Field(ge=0, le=2**32 - 1)] = 42
    multiple_seeds: tuple[Annotated[StrictInt, Field(ge=0, le=2**32 - 1)], ...] = ()
    deterministic: bool = True
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    precision: Literal["auto", "fp32", "fp16", "bf16"] = "auto"
    early_stopping_metric: Literal["macro_f1"] = "macro_f1"
    early_stopping_patience: Annotated[StrictInt, Field(ge=0)] = 2
    min_delta: Annotated[float, Field(ge=0)] = 0.0
    class_weighting: Literal["none", "inverse", "smoothed"] = "none"
    class_weight_smoothing: Annotated[float, Field(gt=0)] = 1.0
    weighted_sampling: bool = False
    num_workers: Annotated[StrictInt, Field(ge=0, le=8)] = 0
    quick_train_limit: PositiveInt = 768
    min_available_ram_gib: Annotated[float, Field(ge=0)] = 3.0
    min_available_gpu_gib: Annotated[float, Field(ge=0)] = 2.0

    @model_validator(mode="after")
    def mutually_exclusive_imbalance_controls(self):
        if self.weighted_sampling and self.class_weighting != "none":
            raise ValueError("weighted sampling and class weighting cannot be enabled together")
        if len(set(self.multiple_seeds)) != len(self.multiple_seeds):
            raise ValueError("multiple_seeds must not contain duplicates")
        return self


class IntentBaselineSettings(FrozenModel):
    models: tuple[Literal["majority", "tfidf_logistic_regression", "tfidf_linear_svm"], ...] = (
        "majority",
        "tfidf_logistic_regression",
        "tfidf_linear_svm",
    )
    max_features: PositiveInt = 30000
    ngram_min: Annotated[StrictInt, Field(ge=1, le=3)] = 1
    ngram_max: Annotated[StrictInt, Field(ge=1, le=3)] = 2
    min_document_frequency: Annotated[StrictInt, Field(ge=1)] = 1
    max_document_frequency: ProbabilityValue = 1.0
    max_iter: PositiveInt = 1000
    candidate_c: tuple[Annotated[float, Field(gt=0)], ...] = (0.5, 1.0, 2.0)
    class_weighting_candidates: tuple[Literal["none", "inverse", "smoothed"], ...] = (
        "none",
        "inverse",
    )
    class_weight_smoothing: Annotated[float, Field(gt=0)] = 1.0

    @model_validator(mode="after")
    def valid_ngram_range_and_candidates(self):
        if self.ngram_min > self.ngram_max:
            raise ValueError("ngram_min cannot exceed ngram_max")
        if not self.candidate_c:
            raise ValueError("at least one candidate C is required")
        if not self.class_weighting_candidates:
            raise ValueError("at least one class weighting strategy is required")
        return self


class IntentSearchCandidate(FrozenModel):
    """Explicit overrides for one trial; every trial reuses the same trainer."""

    learning_rate: Annotated[float, Field(gt=0, le=1)] | None = None
    train_batch_size: PositiveInt | None = None
    max_length: Annotated[StrictInt, Field(ge=16, le=512)] | None = None
    weight_decay: Annotated[float, Field(ge=0, le=1)] | None = None
    epochs: PositiveInt | None = None
    class_weighting: Literal["none", "inverse", "smoothed"] | None = None


class IntentSearchSettings(FrozenModel):
    enabled: bool = False
    max_runs: Annotated[StrictInt, Field(ge=1, le=24)] = 6
    candidates: tuple[IntentSearchCandidate, ...] = (
        IntentSearchCandidate(class_weighting="none"),
        IntentSearchCandidate(class_weighting="inverse"),
    )

    @model_validator(mode="after")
    def bounded_unique_candidates(self):
        if not self.candidates or len(self.candidates) > self.max_runs:
            raise ValueError("search requires between one and max_runs candidates")
        if len({item.model_dump_json() for item in self.candidates}) != len(self.candidates):
            raise ValueError("search candidates must not contain duplicates")
        return self


class IntentEvaluationSettings(FrozenModel):
    primary_metric: Literal["macro_f1"] = "macro_f1"
    preserve_test_holdout: Literal[True] = True
    leakage_audit: Literal[True] = True
    near_duplicate_threshold: Annotated[float, Field(gt=0, le=1)] = 0.94
    near_candidates_per_context: PositiveInt = 20
    max_near_comparisons: PositiveInt = 100000
    zero_division: Literal[0] = 0
    bootstrap_resamples: Annotated[StrictInt, Field(ge=0, le=10000)] = 0
    bootstrap_confidence: Annotated[float, Field(gt=0, lt=1)] = 0.95
    sensitivity_exclude_cross_split_text_overlap: bool = False


class IntentCalibrationSettings(FrozenModel):
    enabled: bool = False
    method: Literal["temperature_scaling"] = "temperature_scaling"
    split: Literal["dev"] = "dev"
    low_confidence_threshold: ProbabilityValue = 0.5
    ece_bins: Annotated[StrictInt, Field(ge=2, le=100)] = 15


class IntentInferenceSettings(FrozenModel):
    batch_size: PositiveInt = 32
    include_probabilities: Literal[True] = True
    include_provenance: Literal[True] = True
    include_text_excerpt: bool = False
    text_excerpt_characters: PositiveInt = 240


class IntentOutputSettings(FrozenModel):
    model_root: Path = Path("artifacts/models/citation_intent")
    report_root: Path = Path("artifacts/reports/stage04")
    work_root: Path = Path("data/processed/stage04")


class Stage04Config(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset: IntentDatasetSettings = Field(default_factory=IntentDatasetSettings)
    labels: IntentLabelSettings = Field(default_factory=IntentLabelSettings)
    model: IntentModelSettings = Field(default_factory=IntentModelSettings)
    training: IntentTrainingSettings = Field(default_factory=IntentTrainingSettings)
    search: IntentSearchSettings = Field(default_factory=IntentSearchSettings)
    baselines: IntentBaselineSettings = Field(default_factory=IntentBaselineSettings)
    evaluation: IntentEvaluationSettings = Field(default_factory=IntentEvaluationSettings)
    calibration: IntentCalibrationSettings = Field(default_factory=IntentCalibrationSettings)
    inference: IntentInferenceSettings = Field(default_factory=IntentInferenceSettings)
    outputs: IntentOutputSettings = Field(default_factory=IntentOutputSettings)
    offline: bool = False

    @model_validator(mode="after")
    def fixed_taxonomy(self):
        if self.labels.mapping() != {"background": 0, "method": 1, "result": 2}:
            raise ValueError("Stage 04 requires the exact SciCite label mapping")
        if self.search.enabled:
            runs = len(self.search.candidates) * (len(self.training.multiple_seeds) or 1)
            if runs > self.search.max_runs:
                raise ValueError("candidate and seed combinations exceed search.max_runs")
            for candidate in self.search.candidates:
                values = self.training.model_dump()
                values.update(candidate.model_dump(exclude_none=True, exclude={"max_length"}))
                IntentTrainingSettings.model_validate(values)
        return self


def _read_stage04_document(path: Path) -> dict:
    document = _read_yaml(path)
    includes = document.pop("includes", [])
    if not isinstance(includes, list) or not all(isinstance(item, str) for item in includes):
        raise ConfigurationError("Stage 04 includes must be a list of YAML filenames")
    for name in includes:
        if Path(name).name != name or Path(name).suffix not in {".yaml", ".yml"}:
            raise ConfigurationError("Stage 04 includes must stay in the configuration directory")
        included = (path.parent / name).resolve()
        if not included.is_relative_to(path.parent):
            raise ConfigurationError("Stage 04 include escapes configuration directory")
        section = _read_yaml(included)
        if "includes" in section or document.keys() & section.keys():
            raise ConfigurationError("recursive or duplicate Stage 04 sections are forbidden")
        document.update(section)
    return document


def _inside_project(value: Path, root: Path, *, field: str) -> Path:
    resolved = (root / value).resolve() if not value.is_absolute() else value.resolve()
    if not resolved.is_relative_to(root):
        raise ConfigurationError(f"Stage 04 {field} must remain inside the project root")
    return resolved


def load_stage04_config(path: Path | None, project_root: Path) -> Stage04Config:
    """Load an additive Stage 04 config without weakening the base YAML contract."""
    root = Path(project_root).resolve()
    config_path = Path(path or root / "configs/citation_intent_stage04.yaml").resolve()
    document = _read_stage04_document(config_path)
    try:
        config = Stage04Config.model_validate(document)
    except Exception as error:
        details = getattr(error, "errors", lambda **_: [])(include_input=False, include_url=False)
        if details:
            rendered = "; ".join(
                f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in details
            )
            raise ConfigurationError(f"{config_path}: {rendered}") from None
        raise ConfigurationError(f"{config_path}: {error}") from None
    values = config.model_dump()
    for field in ("stage03_output_dir", "stage03_run"):
        value = values["dataset"][field]
        if value is not None:
            values["dataset"][field] = _inside_project(Path(value), root, field=field)
    for field in ("model_root", "report_root", "work_root"):
        values["outputs"][field] = _inside_project(
            Path(values["outputs"][field]), root, field=field
        )
    return Stage04Config.model_validate(values)
