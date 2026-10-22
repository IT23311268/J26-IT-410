"""Construct scholarly facts conservatively from the actual earlier-stage contracts."""

import json
import logging
import sqlite3

from ..data.graph_integrity import validate_edge
from ..nlp.intent_contracts import CitationIntentEvidence
from ..nlp.topics.contracts import PaperTopicEvidence, PreparedPaper, TopicAssignment, TopicRecord
from ..pipeline import fingerprint
from .inputs import rows, table
from .store import REVERSES
from .temporal import assessment, split_visibility

LOGGER = logging.getLogger(__name__)


def provenance(row):
    result = row.get("provenance") or row.get("source_provenance") or row.get("edge_provenance")
    if (
        not isinstance(result, list | tuple)
        or not result
        or any(not isinstance(item, dict) or not item for item in result)
    ):
        raise ValueError("source record has no provenance")
    return list(result)


def source_scope(row):
    ref = (row.get("source_record_ids") or [{}])[0]
    prov = provenance(row)[0]
    return fingerprint(
        [
            row.get("source_dataset") or ref.get("source_dataset") or prov.get("source_dataset"),
            row.get("dataset_version", ref.get("dataset_version", prov.get("dataset_version"))),
        ]
    )


def paper_key(row):
    if row.get("canonical_paper_id"):
        if not isinstance(row["canonical_paper_id"], str):
            raise ValueError("canonical paper identity must be an explicitly namespaced string")
        return str(row["canonical_paper_id"])
    ref = (row.get("source_record_ids") or [{}])[0]
    dataset = row.get("source_dataset") or ref.get("source_dataset")
    version = row.get("dataset_version", ref.get("dataset_version"))
    source_id = row.get("source_record_id", ref.get("source_record_id"))
    source_key = row.get("record_key") or ref.get("record_key")
    if not dataset or (source_id is None and source_key is None):
        raise ValueError("paper has neither canonical nor scoped source-local identity")
    if isinstance(source_id, bool):
        raise ValueError("a boolean is not a source-local paper identifier")
    return "source-local:" + fingerprint([dataset, version, source_id, source_key])


def _iter(bundle, config, name):
    yield from rows(
        table(bundle.stage03, name),
        batch_size=config.batch_size,
        max_record_bytes=config.max_record_bytes,
    )


def paper_nodes(store, bundle, config):
    for name in ("papers", "source_local_papers"):
        for row_number, row in enumerate(_iter(bundle, config, name)):
            provenance(row)
            key = paper_key(row)
            # Keep source text and references; raw source remains checksum-addressed upstream.
            payload = {
                k: v for k, v in row.items() if k not in {"raw_values", "field_alternatives"}
            }
            existing = store.aux("paper_inputs", key)
            if existing and name == "papers":
                if existing != payload:
                    raise ValueError("canonical paper identity has conflicting source records")
                store.evidence(
                    "duplicate_paper_observations",
                    str(row_number),
                    {"stable_id": key, "original": payload},
                )
            elif not existing:
                store.auxiliary("paper_inputs", key, payload)
    included = 0
    for key, payload in store.db.execute(
        "SELECT key,payload FROM auxiliary WHERE kind='paper_inputs' ORDER BY key"
    ):
        row = json.loads(payload)
        year = row.get("publication_year")
        if year is not None and (type(year) is not int or not 1000 <= year <= 9999):
            raise ValueError("invalid paper publication year")
        if (
            (config.field and config.field not in (row.get("categories") or []))
            or (config.start_year and (year is None or year < config.start_year))
            or (config.end_year and (year is None or year > config.end_year))
        ):
            store.exclude("paper", key, "field_or_year_filter", row)
            continue
        temporal = assessment(row, bundle)
        if row.get("quarantine") or not temporal["eligible"]:
            store.exclude("paper", key, "quarantined_or_temporally_ineligible", row)
            continue
        if included >= config.max_papers:
            store.exclude("paper", key, "max_papers_deterministic_sorted_id_limit", row)
            continue
        refs = row.get("source_record_ids") or []
        namespace = (
            "canonical" if row.get("canonical_paper_id") else "source-local:" + source_scope(row)
        )
        store.node(
            "paper",
            key,
            {
                **row,
                "identifier_namespace": namespace,
                "source_version": row.get("dataset_version"),
                "provenance": provenance(row),
                "temporal_eligibility": temporal,
                "source_record_ids": refs,
                "text_available": bool(row.get("title") or row.get("abstract")),
                "split_visibility": split_visibility(row, bundle, config),
            },
        )
        store.alias("paper", namespace, row.get("canonical_paper_id", key), key)
        store.alias("paper", source_scope(row), row.get("record_key", key), key)
        included += 1
        if included % config.batch_size == 0:
            LOGGER.info("Stage 06 indexed %s eligible papers", included)
    store.db.commit()


def author_nodes_and_edges(store, bundle, config):
    if not config.include_authors:
        return
    for row in _iter(bundle, config, "authors"):
        provenance(row)
        identifier = row.get("canonical_author_id")
        if not identifier:
            raise ValueError("Stage 03 author lacks its stable source-local/canonical ID")
        store.auxiliary("author_inputs", source_scope(row) + ":" + identifier, row)
        if row.get("identity_status") in {"EXACT_VERIFIED", "VERIFIED", "verified"}:
            store.auxiliary("verified_author_inputs", identifier, row)
    for offset, row in enumerate(_iter(bundle, config, "authorships")):
        source, target = row.get("source_id"), row.get("target_id")
        if not store.get_node("paper", source):
            store.exclude("authorship", str(offset), "paper_outside_snapshot_or_filter", row)
            continue
        author = store.aux("author_inputs", source_scope(row) + ":" + str(target))
        if author is None and row.get("identifier_namespace") == "canonical":
            author = store.aux("verified_author_inputs", str(target))
        if author is None:
            raise ValueError("authorship refers to a missing or differently scoped author")
        temporal = assessment(row, bundle, kind="edge")
        if not temporal["eligible"]:
            store.exclude("authorship", str(offset), "unknown_or_future_relation_availability", row)
            continue
        status = author.get("identity_status", "SOURCE_LOCAL_UNVERIFIED")
        verified = status in {"EXACT_VERIFIED", "VERIFIED", "verified"}
        key = (
            target
            if verified
            else "source-author:"
            + fingerprint([source_scope(author), target, author.get("record_key")])
        )
        value = {
            **author,
            "original_author_id": target,
            "identity_status": status,
            "identifier_namespace": "canonical" if verified else "source-local",
            "provenance": provenance(author),
            "temporal_eligibility": temporal,
        }
        existing = store.get_node("author", key)
        if existing is None:
            store.node("author", key, value)
        elif any(
            existing.get(field) != content
            for field, content in author.items()
            if field not in {"temporal_eligibility", "identifier_namespace"}
        ):
            raise ValueError("shared author identity has conflicting source metadata")
        # The same author can support several independently dated authorships.
        # Each relationship retains its own availability/provenance; node counts
        # and split visibility are recomputed from those eligible relationships.
        store.edge(
            "authored_by",
            source,
            key,
            {
                "original": row,
                "provenance": provenance(row),
                "temporal_eligibility": temporal,
                "author_position": row.get("author_position"),
                "split_visibility": split_visibility(row, bundle, config, kind="edge"),
            },
        )


def citation_edges(store, bundle, config):
    for offset, row in enumerate(_iter(bundle, config, "citations")):
        source = row.get("source_paper_id")
        target = row.get("target_paper_id")
        source = row.get("source_id") if source is None else source
        target = row.get("target_id") if target is None else target
        identifier = row.get("edge_id") or "citation:" + fingerprint(
            [row.get("record_key"), row, offset]
        )
        if source is None or target is None:
            store.exclude("citation", identifier, "unresolved_paper_endpoints", row)
            continue
        checked = validate_edge({**row, "source_id": source, "target_id": target})
        if not checked["valid"]:
            raise ValueError(
                "invalid citation source relationship: " + ",".join(checked["reasons"])
            )
        namespace = row.get("identifier_namespace", "canonical")
        for value in (source, target):
            node = store.get_node("paper", value)
            if node and node["identifier_namespace"] != namespace:
                raise ValueError(
                    "citation endpoint namespace differs from its verified paper mapping"
                )
        if not store.get_node("paper", source) or not store.get_node("paper", target):
            store.exclude("citation", identifier, "endpoint_outside_snapshot_or_filter", row)
            continue
        temporal = assessment(row, bundle, kind="citation_edge")
        if not temporal["eligible"]:
            store.exclude("citation", identifier, "unknown_or_future_citation_availability", row)
            continue
        store.edge(
            "cites",
            source,
            target,
            {
                "original": row,
                "provenance": provenance(row),
                "temporal_eligibility": temporal,
                "citation_context_ids": row.get("citation_context_ids") or [],
                "missing_intent": True,
                "context_count": 0,
                "intent_probabilities": [0.0, 0.0, 0.0],
                "aggregation_calibration_status": "not_validated",
                "citation_count": 1,
                "source_version": row.get("dataset_version"),
                "split_visibility": split_visibility(row, bundle, config, kind="citation_edge"),
            },
            identifier=identifier,
        )


def topic_nodes_and_edges(store, bundle, config):
    if not config.include_topics or bundle.topic_run is None:
        return
    produced = bundle.topic_manifest.get("created_at")
    if not produced:
        raise ValueError("topic model has no production time")
    for row in rows(table(bundle.topic_export, "topic_catalog")):
        topic = TopicRecord.model_validate(row)
        if (
            not topic.model_version.startswith(bundle.topic_manifest["run_id"] + ":")
            or topic.temporal_snapshot_id
            != bundle.topic_manifest["temporal_snapshot"]["snapshot_id"]
        ):
            raise ValueError("topic catalog does not belong to the selected fitted model/snapshot")
        key = "topic:" + fingerprint(
            [topic.model_version, topic.temporal_snapshot_id, topic.topic_id]
        )
        model_time = {
            "feature_available_at": produced,
            "training_data_cutoff": bundle.topic_manifest.get("temporal_snapshot", {}).get(
                "cutoff"
            ),
        }
        temporal = assessment(model_time, bundle, kind="feature")
        store.auxiliary(
            "topic_identity",
            topic.topic_id,
            {
                "key": key,
                "model_version": topic.model_version,
                "snapshot_id": topic.temporal_snapshot_id,
                "eligible": temporal["eligible"],
            },
        )
        if not temporal["eligible"]:
            store.exclude(
                "topic", topic.topic_id, "future_or_unknown_topic_model_availability", row
            )
            continue
        store.node(
            "topic",
            key,
            {
                **row,
                "identifier_namespace": "fitted_topic_model_snapshot",
                "provenance": provenance(row),
                "temporal_eligibility": temporal,
                "review_status": "provisional",
                "outlier_status": False,
                "original_topic_size": topic.paper_count,
                "split_visibility": split_visibility(model_time, bundle, config, kind="feature"),
            },
        )
    for row in rows(table(bundle.topic_export, "paper_topic_assignments")):
        assignment = TopicAssignment.model_validate(row)
        store.evidence("topic_assignments", fingerprint(row), row)
        if assignment.outlier_status:
            continue
        topic = store.aux("topic_identity", assignment.topic_id)
        if (
            not topic
            or topic["model_version"] != assignment.topic_model_version
            or topic["snapshot_id"] != assignment.temporal_snapshot_id
        ):
            raise ValueError("assignment topic ID, model version or snapshot is incompatible")
        if not topic["eligible"] or not store.get_node("paper", assignment.paper_key):
            store.exclude("topic_assignment", assignment.paper_key, "endpoint_not_eligible", row)
            continue
        paper = store.get_node("paper", assignment.paper_key)
        if assignment.canonical_paper_id and assignment.canonical_paper_id != paper.get(
            "canonical_paper_id"
        ):
            raise ValueError("topic assignment canonical paper identity mismatch")
        temporal = assessment(paper, bundle, kind="feature", produced_at=produced)
        if not temporal["eligible"]:
            store.exclude(
                "topic_assignment", assignment.paper_key, "future_or_unknown_assignment", row
            )
            continue
        store.edge(
            "associated_with_topic",
            assignment.paper_key,
            topic["key"],
            {
                "original": row,
                "provenance": provenance(row),
                "temporal_eligibility": temporal,
                "assignment_score": assignment.assignment_score,
                "score_semantics": assignment.score_semantics,
                "evidence_tier": "MODEL_PREDICTED",
                "split_visibility": split_visibility(
                    paper, bundle, config, kind="feature", produced_at=produced
                ),
            },
        )


def investigation_edges(store, bundle, config):
    if bundle.topic_export is None:
        return
    for row in rows(table(bundle.topic_export, "paper_topic_evidence")):
        evidence = PaperTopicEvidence.model_validate(row)
        topic = store.aux("topic_identity", evidence.topic_id)
        if (
            not topic
            or topic["model_version"] != evidence.topic_model_version
            or topic["snapshot_id"] != evidence.temporal_snapshot_id
        ):
            raise ValueError("investigation evidence topic namespace/model mismatch")
        tier = {
            "human": "HUMAN_VERIFIED",
            "model": "MODEL_PREDICTED",
            "rule_candidate": "RULE_CANDIDATE",
            "ranking_candidate": "RULE_CANDIDATE",
            "unknown": "UNKNOWN",
        }[evidence.evidence_source]
        store.evidence("paper_topic_evidence", fingerprint(row), {**row, "evidence_tier": tier})
        paper = store.get_node("paper", evidence.paper_key)
        if not topic["eligible"] or not paper:
            continue
        if evidence.canonical_paper_id and evidence.canonical_paper_id != paper.get(
            "canonical_paper_id"
        ):
            raise ValueError("investigation evidence paper identity mismatch")
        if evidence.evidence_text_span is not None:
            field = "title" if evidence.text_field == "original_title" else "abstract"
            text = paper.get(field) or ""
            if text[evidence.evidence_start_offset : evidence.evidence_end_offset] != (
                evidence.evidence_text_span
            ):
                raise ValueError("investigation span does not match its source paper")
        label = evidence.evidence_label
        candidate = (
            tier == "RULE_CANDIDATE"
            and config.allow_rule_candidate_edges
            and evidence.candidate_label in {"INVESTIGATES_TOPIC", "MENTIONS_TOPIC"}
        )
        if label == "UNKNOWN" and not candidate:
            continue
        if tier == "HUMAN_VERIFIED":
            if evidence.review_status != "human_verified" or not evidence.evidence_text_span:
                raise ValueError(
                    "verified investigation requires reviewed, grounded human evidence"
                )
            prefix = ""
        elif tier == "MODEL_PREDICTED":
            if evidence.review_status != "model_predicted":
                raise ValueError("model evidence must explicitly retain predicted review status")
            if not config.export_predicted_relations_separately:
                continue
            prefix = "predicted_"
        elif candidate:
            prefix, label = "candidate_", evidence.candidate_label
        else:
            continue
        produced = bundle.export_manifest.get("created_at")
        temporal = assessment(paper, bundle, kind="feature", produced_at=produced)
        if not temporal["eligible"]:
            store.exclude("investigation", evidence.paper_key, "future_or_unknown_evidence", row)
            continue
        relation = prefix + (
            "investigates_topic" if label == "INVESTIGATES_TOPIC" else "mentions_topic"
        )
        store.edge(
            relation,
            evidence.paper_key,
            topic["key"],
            {
                "original": row,
                "evidence_tier": tier,
                "provenance": provenance(row),
                "temporal_eligibility": temporal,
                "assignment_score": evidence.evidence_score,
                "score_semantics": evidence.score_semantics,
                "split_visibility": split_visibility(
                    paper, bundle, config, kind="feature", produced_at=produced
                ),
            },
        )


def citation_intent(store, bundle, config):
    if bundle.intent_export is None:
        return
    input_count = 0
    for row in rows(bundle.intent_export):
        input_count += 1
        prediction = CitationIntentEvidence.model_validate(row)
        if prediction.model_version != bundle.intent_manifest["run_id"]:
            raise ValueError("citation-intent model version differs from its export manifest")
        identifier = prediction.citation_context_id
        if store.db.execute(
            "SELECT 1 FROM evidence WHERE kind='citation_context_predictions' AND id=?",
            (identifier,),
        ).fetchone():
            raise ValueError("duplicate citation-context prediction ID within one model export")
        store.evidence("citation_context_predictions", identifier, row)
        source, target = prediction.citing_paper_id, prediction.cited_paper_id
        reason = None
        if prediction.prediction_status != "completed":
            reason = "skipped_empty_text"
        elif (
            not source
            or not target
            or prediction.identity_resolution_status
            not in {"EXACT_VERIFIED", "VERIFIED", "resolved_verified"}
        ):
            reason = "unverified_paper_pair"
        pair = (
            store.db.execute(
                "SELECT id FROM edges WHERE rel='cites' AND s=? AND t=?", (source, target)
            ).fetchall()
            if reason is None
            else []
        )
        if reason is None and not pair:
            reason = "no_verified_citation_relationship_in_snapshot"
        temporal = assessment(
            row, bundle, kind="feature", produced_at=prediction.prediction_timestamp.isoformat()
        )
        if reason is None and not temporal["eligible"]:
            reason = "future_or_unknown_prediction_availability"
        if reason:
            store.evidence(
                "unlinked_citation_contexts", identifier, {**row, "unlink_reason": reason}
            )
            continue
        store.evidence(
            "citation_intent_links",
            identifier,
            {
                **row,
                "citation_edge_ids": [item[0] for item in pair],
                "temporal_eligibility": temporal,
                "split_visibility": split_visibility(
                    row,
                    bundle,
                    config,
                    kind="feature",
                    produced_at=prediction.prediction_timestamp.isoformat(),
                ),
            },
        )
        store.db.executemany(
            "INSERT INTO intent_contributions VALUES(?,?,?,?,?)",
            [
                (
                    edge_id,
                    identifier,
                    prediction.prob_background,
                    prediction.prob_method,
                    prediction.prob_result,
                )
                for (edge_id,) in pair
            ],
        )
        if config.citation_intent_representation in {"context_nodes", "both"}:
            key = "context:" + fingerprint([identifier, prediction.model_version])
            store.node(
                "citation_context",
                key,
                {
                    **row,
                    "identifier_namespace": "stage04_context",
                    "temporal_eligibility": temporal,
                    "provenance": provenance(row),
                    "split_visibility": split_visibility(
                        row,
                        bundle,
                        config,
                        kind="feature",
                        produced_at=prediction.prediction_timestamp.isoformat(),
                    ),
                },
            )
            for relation, paper in (
                ("originates_from", source),
                ("refers_to", target),
                ("expresses_intent_about", target),
            ):
                store.edge(
                    relation,
                    key,
                    paper,
                    {
                        "provenance": provenance(row),
                        "temporal_eligibility": temporal,
                        "context_id": identifier,
                    },
                )
    if input_count != bundle.intent_manifest.get("records"):
        raise ValueError("citation-intent export record count differs from its manifest")
    # Each context contributes once to a paper pair, including repeated citation observations.
    for edge_id, count, p0, p1, p2 in store.db.execute(
        "SELECT edge_id,COUNT(*),AVG(p0),AVG(p1),AVG(p2) FROM intent_contributions GROUP BY edge_id"
    ):
        edge = json.loads(
            store.db.execute("SELECT payload FROM edges WHERE id=?", (edge_id,)).fetchone()[0]
        )
        edge.update(
            context_count=count,
            linked_context_table="citation_intent_links",
            missing_intent=False,
            intent_probabilities=[p0, p1, p2],
        )
        edge["split_intent_probabilities"], edge["split_context_counts"] = {}, {}
        for split in config.temporal.split_cutoffs:
            n, s0, s1, s2 = store.db.execute(
                "SELECT COUNT(*),AVG(c.p0),AVG(c.p1),AVG(c.p2) FROM intent_contributions c "
                "JOIN evidence e ON e.kind='citation_intent_links' AND e.id=c.context_id "
                "WHERE c.edge_id=? AND json_extract(e.payload,?)=1",
                (edge_id, "$.split_visibility." + split),
            ).fetchone()
            edge["split_intent_probabilities"][split] = [s0, s1, s2] if n else [0.0, 0.0, 0.0]
            edge["split_context_counts"][split] = n
        if config.citation_intent_representation == "context_nodes":
            edge.update(missing_intent=True, intent_probabilities=[0.0, 0.0, 0.0])
        store.db.execute("UPDATE edges SET payload=? WHERE id=?", (json.dumps(edge), edge_id))


def research_fields(store, bundle, config):
    if not config.include_research_fields:
        return
    for row in _iter(bundle, config, "source_fields"):
        paper = row.get("canonical_paper_id")
        if not store.get_node("paper", paper):
            continue
        original_id = row.get("source_field_id")
        if original_id is None or not row.get("field_name"):
            raise ValueError("source field lacks verified identifier/name")
        key = "field:" + fingerprint([source_scope(row), row.get("field_taxonomy"), original_id])
        temporal = assessment(store.get_node("paper", paper), bundle, kind="feature")
        if not temporal["eligible"]:
            store.exclude("research_field", key, "unknown_or_future_field_availability", row)
            continue
        node = {
            "source_field_id": original_id,
            "field_name": row["field_name"],
            "field_taxonomy": row.get("field_taxonomy"),
            "provenance": provenance(row),
            "identifier_namespace": source_scope(row),
            "temporal_eligibility": temporal,
        }
        # One taxonomy node may have multiple source observations; preserve each on its edge.
        if not store.get_node("research_field", key):
            store.node("research_field", key, node)
        store.edge(
            "classified_in",
            paper,
            key,
            {"provenance": provenance(row), "original": row, "temporal_eligibility": temporal},
        )


def reverse_relations(store, config):
    if not config.add_reverse_message_passing_edges:
        return
    for identifier, sk, relation, tk, source, target, payload in store.db.execute(
        "SELECT id,sk,rel,tk,s,t,payload FROM edges WHERE derived=0 ORDER BY id"
    ):
        reverse = REVERSES.get(relation, "rev_" + relation)
        value = {
            **json.loads(payload),
            "derived_from_edge_id": identifier,
            "interpretation": "message_passing_only_not_an_independent_scholarly_fact",
        }
        store.edge(
            reverse,
            target,
            source,
            value,
            identifier="reverse:" + identifier,
            derived=True,
            node_types=(tk, sk),
        )


def embedding_index(store, bundle, config):
    if bundle.embedding_run is None:
        return
    root = bundle.embedding_run
    for row in rows(root / "papers.jsonl"):
        paper = PreparedPaper.model_validate(row)
        node = store.get_node("paper", paper.paper_key)
        if node:
            expected = fingerprint(
                [node.get("title") or "", node.get("abstract"), "NFC_whitespace_v1"]
            )
            if paper.text_sha256 != expected:
                raise ValueError("embedding prepared paper text differs from Stage 03 text")
            store.auxiliary("prepared_text", paper.paper_key, row)
    for row in rows(root / "paper_embedding_index.jsonl"):
        number = row.get("row")
        if type(number) is not int or number < 0:
            raise ValueError("embedding row index is invalid or duplicated")
        key = row["paper_key"]
        prepared = store.aux("prepared_text", key)
        if store.get_node("paper", key):
            if not prepared or prepared["text_sha256"] != row.get("text_sha256"):
                raise ValueError("embedding index text hash mismatch")
            if row.get("canonical_paper_id") != prepared.get("canonical_paper_id"):
                raise ValueError("embedding canonical paper identity mismatch")
        try:
            store.db.execute(
                "INSERT INTO features VALUES(?,?,?,?)",
                (key, number, row.get("text_sha256"), json.dumps(row)),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError("duplicate paper identity or row in embedding index") from error
    count, maximum = store.db.execute("SELECT COUNT(*),MAX(row_number) FROM features").fetchone()
    if count and maximum != count - 1:
        raise ValueError("embedding row indices must form a complete matrix mapping")
    store.auxiliary("embedding_rows", "count", {"count": count})
