"""Small public boundary contracts for the standalone viewer."""

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict

Mode = Literal["real", "demo"]
IntentLabel = Literal["background", "method", "result"]
DEMO_NOTICE = "Synthetic Demo Data — Not Research Findings"


T = TypeVar("T")


class Envelope(BaseModel, Generic[T]):
    mode: Mode
    status: Literal["available", "unavailable"]
    notice: str | None = None
    graph_version: str | None = None
    data: T | None = None


class Record(BaseModel):
    model_config = ConfigDict(extra="allow")


class Node(Record):
    id: str
    type: Literal["paper", "author", "topic", "research_field"]
    label: str
    year: int | None
    abstract: str | None
    keywords: list[str]
    provenance: list[dict[str, Any]]
    metadata: dict[str, Any]
    synthetic: bool


class Edge(Record):
    id: str
    source: str
    target: str
    relation: str
    provenance: list[dict[str, Any]]
    synthetic: bool


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int


class Distribution(Record):
    label: str
    count: int


class Intent(Record):
    id: str
    text: str | None
    label: IntentLabel
    confidence: float | None
    source: str
    split: str | None
    citing_paper_id: str | None
    cited_paper_id: str | None
    model_id: str
    model_version: str
    provenance: list[dict[str, Any]]
    synthetic: bool
    original_label: str | None


class Intents(Page[Intent]):
    distribution: list[Distribution]
    model_id: str | None
    model_version: str | None
    calibration: str | None


class Topic(Record):
    id: str
    label: str
    keywords: list[str]
    paper_count: int
    coverage: float
    model_version: str
    review_status: str
    synthetic: bool


class TopicDetail(BaseModel):
    topic: Topic
    papers: list[Node]
    evidence: list[dict[str, Any]]
    provenance: list[dict[str, Any]]
    membership_note: str


class GraphStats(BaseModel):
    node_counts: dict[str, int]
    edge_counts: dict[str, int]
    components: int | None
    limitations: list[str]


class Neighborhood(BaseModel):
    nodes: list[Node]
    edges: list[Edge]
    seed: str | None
    truncated: bool
    limit: int
    depth: int


class NodeDetail(BaseModel):
    node: Node
    relations: list[Edge]


class Stage(Record):
    number: int
    name: str
    status: Literal["Verified Data", "Validation Pending", "Not Yet Implemented"]
    detail: str
    evidence: list[str]


class Progress(BaseModel):
    stages: list[Stage]
    engineering_note: str
    scientific_note: str


class Dataset(Record):
    name: str
    records: int | None
    status: str
    detail: str
    source: str | None


class Datasets(BaseModel):
    items: list[Dataset]


class Overview(BaseModel):
    counts: dict[str, int]
    publication_years: list[Distribution]
    intent_distribution: list[Distribution]
    topic_coverage: dict[str, int | float]
    graph: dict[str, int | None]
    datasets: list[Dataset]
    stages: list[Stage]
    provenance: list[str]
    limitations: list[str]
    intent_model: str | None
    created_at: str | None


class Metric(BaseModel):
    model: str
    split: str
    metric: str
    value: float
    samples: int | None
    source: str


class Validation(BaseModel):
    checks: list[dict[str, str]]
    metrics: list[Metric]
    plans: list[dict[str, str]]
    missing_evidence: list[str]
    temporal_policy: str | None
