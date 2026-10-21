"""Streaming Stage 03 adapters over verified local records, never network downloads.

Text sources reuse Stage 02 streaming and source schemas. Graph inputs are explicit
JSONL node/edge exports, not unsafe serialized graph objects. Internal indices remain
scoped by dataset, release, and node type; unknown releases are not silently invented.
"""

import itertools
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .cleaning import clean_record, json_safe
from .normalization import normalize_identifier
from .provenance import file_sha256, reference
from .sources import arxiv, scicite
from .streaming import events

_ALIASES = {
    "arxiv": "arxiv-metadata",
    "cornell-arxiv": "arxiv-metadata",
    "ogbn_arxiv": "ogbn-arxiv",
    "mag-year": "mag-year-h2gb",
    "mag_year": "mag-year-h2gb",
    "h2gb": "mag-year-h2gb",
    "semantic_scholar": "semantic-scholar",
}
_SOURCE_URLS = {
    "scicite": scicite.SOURCE_URL,
    "arxiv-metadata": arxiv.SOURCE_URL,
    "openalex": "https://docs.openalex.org/api-entities/works/work-object",
    "semantic-scholar": "https://api.semanticscholar.org/api-docs/graph",
    "ogbn-arxiv": "https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv",
    "mag-year-h2gb": "https://github.com/junhongmit/H2GB",
}


def _identifier(namespace, value, *, dataset, version, verified=False, node_type=None):
    if value is None:
        return None
    return normalize_identifier(
        namespace,
        value,
        dataset=dataset,
        version=version,
        node_type=node_type,
        verified=verified,
    )


def _source_base(dataset, event, version, synthetic, acquired_at=None):
    provenance = reference(
        dataset,
        _SOURCE_URLS[dataset],
        version,
        event,
        synthetic=synthetic,
        acquired_at=acquired_at,
    ).model_dump(mode="json")
    return {
        "source_dataset": dataset,
        "dataset_version": version,
        "raw_values": json_safe(event.record) if event.record is not None else {},
        "source_split": event.split,
        "provenance": [provenance],
        "issues": [event.error] if event.error else [],
        "quarantine": event.error is not None,
    }


def _from_stage02(record, base, dataset, version):
    """Stage 02 canonical record -> loss-aware Stage 03 fields, API unchanged."""
    result = {**base, "kind": record.get("kind", "paper")}
    result["source_record_id"] = record.get("source_record_id")
    result["source_split"] = record.get("source_split", base.get("source_split"))
    source_provenance = record.get("source_provenance", record.get("identifier_provenance"))
    if source_provenance:
        references = [*base["provenance"]]
        if not any(
            item.get("locator") == source_provenance.get("locator")
            and item.get("record_sha256") == source_provenance.get("record_sha256")
            for item in references
        ):
            references.insert(0, source_provenance)
        result["provenance"] = references
    if result["kind"] == "citation_context":
        result.update(
            original_context=record.get("citation_text"),
            original_intent_label=record.get("original_intent_label"),
            normalized_intent_label=record.get("normalized_intent_label"),
        )
        for source, target in (
            ("citing_paper_identifier", "citing_identifier"),
            ("cited_paper_identifier", "cited_identifier"),
        ):
            value = record.get(source)
            result[target] = (
                _identifier(value["namespace"], value["value"], dataset=dataset, version=version)
                if value
                else None
            )
    elif result["kind"] == "paper":
        for field in (
            "title",
            "abstract",
            "authors",
            "publication_year",
            "publication_date",
            "update_date",
            "versions",
            "arxiv_version",
        ):
            if field in record:
                result[field] = record[field]
        result["categories"] = record.get("subject_categories", [])
        result["identifiers"] = [
            _identifier(
                item["namespace"], item["value"], dataset=dataset, version=version, verified=True
            )
            for item in record.get("external_ids", [])
        ]
        # Raw arXiv versions remain distinguishable, even though work matching uses base ID.
        if record.get("arxiv_id_raw"):
            result["identifiers"] = [
                item for item in result["identifiers"] if item["namespace"] != "arxiv"
            ] + [
                _identifier(
                    "arxiv", record["arxiv_id_raw"], dataset=dataset, version=version, verified=True
                )
            ]
    elif result["kind"] == "citation_edge":
        result.update(_graph_record(dataset, record, version))
    return result


def _metadata_record(dataset, raw, version):
    """Adapt observed OpenAlex/S2 fields without enrichment or inferred identifiers."""
    result: dict[str, Any] = {"kind": "paper", "identifiers": [], "authors": []}
    mapping = (
        {"doi": "doi", "openalex": "openalex", "mag": "mag"}
        if dataset == "openalex"
        else {"DOI": "doi", "ArXiv": "arxiv", "MAG": "mag"}
    )
    external = raw.get("ids", {}) if dataset == "openalex" else raw.get("externalIds", {})
    if not isinstance(external, dict):
        result.setdefault("issues", []).append("unexpected_type:external_ids")
        external = {}
    for field, namespace in mapping.items():
        if external.get(field) is not None:
            result["identifiers"].append(
                _identifier(
                    namespace, external[field], dataset=dataset, version=version, verified=True
                )
            )
    source_id = raw.get("id") if dataset == "openalex" else raw.get("paperId")
    result["source_record_id"] = source_id if isinstance(source_id, str) else None
    if source_id is not None:
        result["identifiers"].append(
            _identifier(
                "openalex" if dataset == "openalex" else "semantic_scholar",
                source_id,
                dataset=dataset,
                version=version,
                verified=True,
            )
        )
    if dataset == "openalex" and raw.get("doi") and not external.get("doi"):
        result["identifiers"].append(
            _identifier("doi", raw["doi"], dataset=dataset, version=version, verified=True)
        )
    result["title"] = raw.get("title", raw.get("display_name"))
    # OpenAlex inverted abstracts are retained raw, not silently reconstructed as original text.
    result["abstract"] = raw.get("abstract")
    result["publication_year"] = raw.get("publication_year", raw.get("year"))
    result["publication_date"] = raw.get("publication_date", raw.get("publicationDate"))
    result["venue"] = raw.get("venue")
    result["categories"] = []
    author_values = raw.get("authorships", []) if dataset == "openalex" else raw.get("authors", [])
    if not isinstance(author_values, list):
        result.setdefault("issues", []).append("unexpected_type:authors")
        author_values = []
    for authorship in author_values:
        if not isinstance(authorship, dict):
            result.setdefault("issues", []).append("unexpected_type:author")
            continue
        author = authorship.get("author", {}) if dataset == "openalex" else authorship
        if isinstance(author, dict):
            name = author.get("display_name", author.get("name"))
            if isinstance(name, str):
                result["authors"].append(name)
    result["author_source_values"] = json_safe(author_values)
    result["external_ids"] = json_safe(external)
    # Source update and retrieval times are evidence, never publication-time guarantees.
    for source, target in (
        ("created_date", "source_created_date"),
        ("updated_date", "source_updated_date"),
        ("available_at", "available_at"),
        ("snapshot_date", "dataset_snapshot_date"),
    ):
        if source in raw:
            result[target] = raw[source]
    if "updated_date" in raw:
        result["update_date"] = raw["updated_date"]
    return result


def _graph_record(dataset, raw, version):
    namespace = "ogbn-arxiv-node-index" if dataset == "ogbn-arxiv" else "h2gb-mag-year-node-index"
    kind = raw.get("kind", "node" if "node_id" in raw else "citation_edge")
    declared_type = {"paper": "paper", "author": "author", "source_field": "field"}.get(kind)
    node_type = raw.get(
        "node_type", declared_type or ("paper" if dataset == "ogbn-arxiv" else None)
    )
    result = {"node_type": node_type, "identifiers": [], "issues": []}
    if kind in {"node", "paper", "author", "source_field"}:
        index = raw.get(
            "node_id",
            raw.get(
                "source_record_id",
                raw.get("index", raw.get("source_field_id", raw.get("source_author_id"))),
            ),
        )
        identifier = _identifier(
            namespace, index, dataset=dataset, version=version, node_type=node_type
        )
        result.update(
            kind={
                "paper": "paper",
                "author": "author",
                "field": "source_field",
                "field_of_study": "source_field",
            }.get(node_type, "graph_node"),
            source_record_id=index,
            identifiers=[identifier] if identifier else [],
            node_identifier=identifier,
        )
        for field in (
            "title",
            "abstract",
            "authors",
            "categories",
            "publication_year",
            "publication_date",
            "display_name",
            "field_name",
            "field_taxonomy",
        ):
            if field in raw:
                result[field] = raw[field]
        if result["kind"] == "author" and "display_name" not in result and "name" in raw:
            result["display_name"] = raw["name"]
        if result["kind"] == "source_field":
            result["source_field_id"] = index
    else:
        source_type = raw.get("source_node_type", node_type)
        target_type = raw.get("target_node_type", node_type)
        source = raw.get("source_node_id", raw.get("source", raw.get("original_source_id")))
        target = raw.get("target_node_id", raw.get("target", raw.get("original_target_id")))
        relation = raw.get("edge_type", raw.get("relation", "cites"))
        result.update(
            kind="citation_edge" if relation in {"cites", "citation"} else "relation",
            edge_type=relation,
            original_source_id=source,
            original_target_id=target,
            source_node_type=source_type,
            target_node_type=target_type,
            identifier_namespace=namespace,
            source_identifier=_identifier(
                namespace, source, dataset=dataset, version=version, node_type=source_type
            ),
            target_identifier=_identifier(
                namespace, target, dataset=dataset, version=version, node_type=target_type
            ),
            citation_intent_status="UNOBSERVED",
        )
        if relation == "investigates_topic":
            result["issues"].append("investigation_relationship_not_validated_stage03")
            result["quarantine"] = True
        if raw.get("citation_intent") is not None:
            result["issues"].append("graph_intent_not_inferred_or_promoted")
        for field in ("publication_year", "available_at", "citation_observed_at"):
            if field in raw:
                result[field] = raw[field]
    result["source_split"] = raw.get("source_split", raw.get("split"))
    result["split_method"] = "preserved_source_assignment_unverified_officialness"
    if version is None:
        result["issues"].append("unknown_graph_dataset_version")
    return result


def iter_source_records(
    dataset: str,
    path: str | Path,
    *,
    version: str | None = None,
    max_records: int | None = None,
    synthetic: bool = False,
    max_line_bytes: int = 4 * 1024**2,
    format: str = "raw",
) -> Iterator[dict[str, Any]]:
    """Read local genuine or explicitly synthetic records with bounded per-line memory.

    ``format='stage02'`` explicitly reads prior canonical JSONL envelopes. Graph sources
    accept only observed JSONL table exports; this adapter never deserializes pickle/PT.
    Malformed rows become recoverable quarantines referencing unchanged source bytes.
    """
    dataset = _ALIASES.get(dataset, dataset)
    if dataset not in _SOURCE_URLS:
        raise ValueError(f"unsupported Stage 03 source: {dataset}")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"local Stage 03 source not found: {path}")
    if format not in {"raw", "stage02", "graph_tables"}:
        raise ValueError("source format must be raw, stage02, or graph_tables")
    if format == "graph_tables" and dataset not in {"ogbn-arxiv", "mag-year-h2gb"}:
        raise ValueError("graph_tables format requires an observed graph dataset")
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive")
    if (
        dataset in {"ogbn-arxiv", "mag-year-h2gb"}
        and path.is_file()
        and path.suffix not in {".json", ".jsonl"}
    ):
        raise ValueError("graph integration requires explicit safe JSONL node/edge exports")
    acquired_at = None
    sidecar = path.with_name(path.name + ".source.json")
    if path.is_file() and sidecar.exists():
        acquisition = json.loads(sidecar.read_text(encoding="utf-8"))
        if acquisition.get("sha256") != file_sha256(path):
            raise ValueError("source changed since acquisition; checksum mismatch")
        acquired_at = acquisition.get("downloaded_at")
    iterator = events(path, dataset, max_line_bytes=max_line_bytes)
    if max_records is not None:
        iterator = itertools.islice(iterator, max_records)
    for event in iterator:
        base = _source_base(dataset, event, version, synthetic, acquired_at)
        if event.record is None:
            yield clean_record({**base, "kind": "invalid", "quarantine": True})
            continue
        try:
            if format == "stage02":
                raw = event.record.get("record")
                base["issues"].extend(event.record.get("issues", []))
                base["issues"].append("original_raw_fields_in_source_locator_not_envelope")
                if not isinstance(raw, dict):
                    raise ValueError("Stage 02 envelope has no canonical source record")
                result = _from_stage02(raw, base, dataset, version)
            elif dataset in {"scicite", "arxiv-metadata"}:
                adapter = scicite if dataset == "scicite" else arxiv
                staged, issues = adapter.adapt(
                    event, version=version, synthetic=synthetic, acquired_at=acquired_at
                )
                result = _from_stage02(staged.model_dump(mode="json"), base, dataset, version)
                result["issues"].extend(issues)
                # Dates retain original precision; arXiv v1 date is distinct from update date.
                if dataset == "arxiv-metadata":
                    raw_versions = event.record.get("versions", [])
                    result["version_dates"] = (
                        [
                            {"version": item.get("version"), "created": item.get("created")}
                            for item in raw_versions
                            if isinstance(item, dict)
                        ]
                        if isinstance(raw_versions, list)
                        else []
                    )
            else:
                source = (
                    _metadata_record(dataset, event.record, version)
                    if dataset in {"openalex", "semantic-scholar"}
                    else _graph_record(dataset, event.record, version)
                )
                result = {**base, **source}
                result["issues"] = base["issues"] + source.get("issues", [])
                result["quarantine"] = base["quarantine"] or source.get("quarantine", False)
            yield clean_record(result)
        except (ValueError, TypeError, OverflowError, KeyError) as error:
            base["issues"].append(f"source_adaptation_failed:{type(error).__name__}")
            yield clean_record({**base, "kind": "invalid", "quarantine": True})
