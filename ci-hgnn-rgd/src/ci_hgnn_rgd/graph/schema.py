"""Backend-independent graph schema. Intent is an optional citation-edge attribute."""

from dataclasses import dataclass
from enum import StrEnum


class NodeType(StrEnum):
    PAPER = "paper"
    AUTHOR = "author"
    TOPIC = "topic"
    INSTITUTION = "institution"
    RESEARCH_AREA = "research_area"


class RelationType(StrEnum):
    CITES = "cites"
    AUTHORED_BY = "authored_by"
    MENTIONS_TOPIC = "mentions_topic"
    INVESTIGATES_TOPIC = "investigates_topic"
    AFFILIATED_WITH = "affiliated_with"
    IN_RESEARCH_AREA = "in_research_area"


@dataclass(frozen=True)
class RelationDefinition:
    source: NodeType
    relation: RelationType
    target: NodeType
    provenance_required: bool = True


@dataclass(frozen=True)
class GraphSchema:
    version: str
    node_types: tuple[NodeType, ...]
    relations: tuple[RelationDefinition, ...]
    citation_intent_optional: bool = True


def get_graph_schema(include_optional: bool = False) -> GraphSchema:
    nodes = (NodeType.PAPER, NodeType.AUTHOR, NodeType.TOPIC)
    relations = (
        RelationDefinition(NodeType.PAPER, RelationType.CITES, NodeType.PAPER),
        RelationDefinition(NodeType.PAPER, RelationType.AUTHORED_BY, NodeType.AUTHOR),
        RelationDefinition(NodeType.PAPER, RelationType.MENTIONS_TOPIC, NodeType.TOPIC),
        RelationDefinition(NodeType.PAPER, RelationType.INVESTIGATES_TOPIC, NodeType.TOPIC),
    )
    if include_optional:
        nodes += (NodeType.INSTITUTION, NodeType.RESEARCH_AREA)
        relations += (
            RelationDefinition(NodeType.AUTHOR, RelationType.AFFILIATED_WITH, NodeType.INSTITUTION),
            RelationDefinition(
                NodeType.TOPIC, RelationType.IN_RESEARCH_AREA, NodeType.RESEARCH_AREA
            ),
        )
    return GraphSchema("1.0", nodes, relations)
