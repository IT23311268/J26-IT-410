"""Additive typed Stage 05 config; the original Stage 01 YAML remains compatible."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StrictInt, model_validator

from ...config import ConfigurationError, _read_yaml
from ...schemas import FrozenModel

Positive = Annotated[StrictInt, Field(gt=0)]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class DataSettings(FrozenModel):
    stage03_root: Path = Path("data/processed/stage03")
    stage03_run: Path | None = None
    max_records: Positive = 2000
    sampling: Literal["hash"] = "hash"
    allow_title_only: bool = True
    allow_abstract_only: bool = False
    max_text_characters: Positive = 100000
    allowed_languages: tuple[str, ...] = ()


class EmbeddingSettings(FrozenModel):
    backend: Literal["specter2", "scibert"] = "specter2"
    specter2_model: Literal["allenai/specter2_base"] = "allenai/specter2_base"
    specter2_revision: Revision = "3447645e1def9117997203454fa4495937bfbd83"
    adapter: Literal["allenai/specter2"] = "allenai/specter2"
    adapter_revision: Revision = "2081559630a80fc5851d8f798a05ba81e9468089"
    scibert_model: Literal["allenai/scibert_scivocab_uncased"] = "allenai/scibert_scivocab_uncased"
    scibert_revision: Revision = "952d056ace307abd2cda7cee822a1d608deb20ed"
    batch_size: Positive = 8
    max_length: Annotated[StrictInt, Field(ge=16, le=512)] = 512
    normalize: bool = True
    dimension: Positive = 768
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    quantization: Literal["none", "int8_dynamic"] = "none"
    cpu_threads: Positive | None = None
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    cache_enabled: bool = True
    min_available_ram_gib: Annotated[float, Field(ge=0)] = 1.5
    # Strict historical experiments must document the pretrained encoder's own
    # training/availability boundary, independently of the input-corpus cutoff.
    verified_model_available_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def quantized_cpu_only(self):
        if self.quantization != "none" and (
            self.device not in {"auto", "cpu"} or self.precision != "fp32"
        ):
            raise ValueError("dynamic int8 encoding requires CPU and fp32 activation precision")
        return self


class ModelSettings(FrozenModel):
    algorithm: Literal["bertopic", "nmf", "lda"] = "bertopic"
    min_topic_size: Positive = 20
    top_n_words: Annotated[StrictInt, Field(ge=2, le=50)] = 10
    n_topics: Annotated[StrictInt, Field(ge=2, le=100)] = 8
    max_features: Positive = 20000
    min_df: Positive = 2
    ngram_max: Annotated[StrictInt, Field(ge=1, le=2)] = 2
    keyword_stop_words: Literal["english"] | None = "english"
    umap_dimensions: Positive = 5
    umap_neighbors: Positive = 15
    umap_min_dist: Annotated[float, Field(ge=0, le=1)] = 0.0
    umap_metric: Literal["cosine", "euclidean"] = "cosine"
    hdbscan_min_samples: Positive = 5
    representative_papers: Positive = 3
    centroid_rejection_quantile: Annotated[float, Field(ge=0, le=0.5)] = 0.05
    max_iter: Positive = 100
    stability_seeds: tuple[int, ...] = (42, 73)


class TemporalSettings(FrozenModel):
    inherit_stage03_policy: bool = True
    policy: (
        Literal["STRICT_POINT_IN_TIME", "PUBLICATION_DATE_PROXY", "EXPLORATORY_NO_TEMPORAL_CLAIM"]
        | None
    ) = None
    cutoff: AwareDatetime | None = None
    freeze_historical_topic_models: Literal[True] = True


class InvestigationSettings(FrozenModel):
    guideline_version: str = "stage05-v1"
    min_train_annotations: Positive = 30
    require_independent_annotators: Literal[True] = True
    annotation_sample_size: Positive = 100
    epochs: Positive = 3
    learning_rate: Annotated[float, Field(gt=0)] = 2e-5
    class_weighting: Literal["none", "inverse", "smoothed"] = "inverse"
    batch_size: Positive = 4
    max_length: Annotated[StrictInt, Field(ge=16, le=512)] = 256


class Stage05Config(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset: DataSettings = Field(default_factory=DataSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    topic_model: ModelSettings = Field(default_factory=ModelSettings)
    temporal: TemporalSettings = Field(default_factory=TemporalSettings)
    investigation: InvestigationSettings = Field(default_factory=InvestigationSettings)
    seed: Annotated[StrictInt, Field(ge=0, le=2**32 - 1)] = 42
    offline: bool = True
    output_root: Path = Path("data/processed/stage05")
    model_root: Path = Path("artifacts/models/topic_discovery")
    investigation_model_root: Path = Path("artifacts/models/investigation_detection")
    report_root: Path = Path("artifacts/reports/stage05")
    cache_root: Path = Path("data/interim/stage05_embeddings")

    @model_validator(mode="after")
    def historical_cutoff_required(self):
        if not self.temporal.inherit_stage03_policy and self.temporal.policy is None:
            raise ValueError("choose an explicit temporal policy when inheritance is disabled")
        if self.temporal.policy in {"STRICT_POINT_IN_TIME", "PUBLICATION_DATE_PROXY"}:
            if self.temporal.cutoff is None and not self.temporal.inherit_stage03_policy:
                raise ValueError("historical topic modelling requires a cutoff")
        return self


def load_config(path: Path | None, root: Path) -> Stage05Config:
    root = Path(root).resolve()
    path = Path(path or root / "configs/topic_model_stage05.yaml").resolve()
    document = _read_yaml(path)
    includes = document.pop("includes", [])
    if not isinstance(includes, list):
        raise ConfigurationError("Stage 05 includes must be a list")
    for name in includes:
        if not isinstance(name, str) or Path(name).name != name:
            raise ConfigurationError("Stage 05 includes must name files in the same directory")
        target = (path.parent / name).resolve()
        if not target.is_relative_to(path.parent):
            raise ConfigurationError("Stage 05 include escapes its directory")
        extra = _read_yaml(target)
        if "includes" in extra or document.keys() & extra.keys():
            raise ConfigurationError("duplicate or recursive Stage 05 sections")
        document.update(extra)
    config = Stage05Config.model_validate(document)
    values = config.model_dump(mode="python")
    for field in (
        "output_root",
        "model_root",
        "investigation_model_root",
        "report_root",
        "cache_root",
    ):
        target = (root / values[field]).resolve()
        if not target.is_relative_to(root):
            raise ConfigurationError("Stage 05 outputs must remain inside the project")
        values[field] = target
    for field in ("stage03_root", "stage03_run"):
        if values["dataset"][field] is not None:
            values["dataset"][field] = (root / values["dataset"][field]).resolve()
    return Stage05Config.model_validate(values)
