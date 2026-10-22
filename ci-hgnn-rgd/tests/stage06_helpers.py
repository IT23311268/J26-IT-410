"""Explicitly synthetic earlier-stage artifacts; no real graph results are claimed."""

import json
import math

import numpy as np

from ci_hgnn_rgd.data.manifests import atomic_json
from ci_hgnn_rgd.data.provenance import file_sha256
from ci_hgnn_rgd.nlp.topics.config import Stage05Config
from ci_hgnn_rgd.nlp.topics.data import prepare_paper

POLICY = "EXPLORATORY_NO_TEMPORAL_CLAIM"
SNAPSHOT = "snapshot:fixture"
MODEL = "run-fixture:bertopic"


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v) + "\n" for v in values), encoding="utf-8")


def refresh(root, name):
    path = root / name
    manifest = json.loads(path.read_text())
    key = "output_checksums" if name == "integration_manifest.json" else "artifact_checksums"
    manifest[key] = {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in root.rglob("*")
        if p.is_file() and p != path
    }
    atomic_json(path, manifest)


def fixture_config(tmp_path, *, mutate=None, produced="2021-01-01T00:00:00+00:00"):
    from ci_hgnn_rgd.graph.stage06_config import Stage06Config

    s3 = tmp_path / "stage03/run-fixture"
    s5 = tmp_path / "stage05/run-fixture"
    s4 = tmp_path / "intent/run-intent"
    for path in (s3, s5, s4 / "prediction_exports"):
        path.mkdir(parents=True)
    prov = [
        {
            "source_dataset": "synthetic",
            "source_record_id": "fixture",
            "available_at": "2020-01-01",
            "synthetic": True,
        }
    ]
    temporal = {
        "evidence": {
            "publication_date": "2020-01-01",
            "publication_year": 2020,
            "available_at": "2020-01-01",
        }
    }
    papers = [
        {
            "canonical_paper_id": f"p:{i}",
            "record_key": f"paper:{i}",
            "source_dataset": "synthetic",
            "dataset_version": "fixture-v1",
            "source_record_id": str(i),
            "title": f"Paper {i}",
            "abstract": "We use alpha methods to investigate the question.",
            "publication_year": 2020,
            "publication_date": "2020-01-01",
            "date_precision": "day",
            "provenance": prov,
            "temporal_eligibility": temporal,
            "source_split": "train" if i == 1 else "dev",
            "categories": ["field-A"],
        }
        for i in (1, 2, 3)
    ]
    authors = [
        {
            "canonical_author_id": f"a:{i}",
            "record_key": f"a-source:{i}",
            "source_dataset": "synthetic",
            "dataset_version": "fixture-v1",
            "display_name": "Same Name",
            "identity_status": "SOURCE_LOCAL_NAME_ONLY",
            "provenance": prov,
        }
        for i in (1, 2)
    ]
    authorships = [
        {
            "source_id": f"p:{i}",
            "target_id": f"a:{i}",
            "record_key": f"authorship:{i}",
            "identifier_namespace": "canonical",
            "source_dataset": "synthetic",
            "dataset_version": "fixture-v1",
            "author_position": i - 1,
            "provenance": prov,
            "temporal_eligibility": temporal,
        }
        for i in (1, 2)
    ]
    citations = [
        {
            "edge_id": f"c:{i}",
            "source_paper_id": "p:1",
            "target_paper_id": target,
            "identifier_namespace": "canonical",
            "provenance": prov,
            "source_dataset": "synthetic",
            "dataset_version": "fixture-v1",
            "citation_context_ids": ["ctx:1", "ctx:2"] if target == "p:2" else [],
            "temporal_eligibility": temporal,
        }
        for i, target in enumerate(["p:2", "p:2", "p:1"])
    ]
    fields = [
        {
            "canonical_paper_id": "p:1",
            "source_field_id": "field:1",
            "field_name": "field-A",
            "source_dataset": "synthetic",
            "dataset_version": "fixture-v1",
            "field_taxonomy": "synthetic_categories",
            "provenance": prov,
        }
    ]
    topics = [
        {
            "topic_id": f"{SNAPSHOT}:{MODEL}:topic:{i}",
            "local_topic_id": i,
            "model_version": MODEL,
            "temporal_snapshot_id": SNAPSHOT,
            "keywords": ["alpha"],
            "provisional_label": "fixture topic",
            "paper_count": 1,
            "corpus_coverage": 1 / 3,
            "representation_method": "synthetic",
            "provenance": prov,
        }
        for i in (0, 1)
    ]
    assignments = [
        {
            "paper_key": f"p:{i}",
            "canonical_paper_id": f"p:{i}",
            "topic_id": topics[i - 1]["topic_id"] if i < 3 else None,
            "topic_model_version": MODEL,
            "assignment_method": "synthetic",
            "assignment_score": 0.8 if i < 3 else None,
            "score_semantics": "uncalibrated",
            "outlier_status": i == 3,
            "source_provenance": prov,
            "temporal_snapshot_id": SNAPSHOT,
        }
        for i in (1, 2, 3)
    ]
    evidence = []
    for i, source, review, label in [
        (1, "human", "human_verified", "INVESTIGATES_TOPIC"),
        (2, "model", "model_predicted", "INVESTIGATES_TOPIC"),
        (3, "unknown", "unreviewed", "UNKNOWN"),
    ]:
        evidence.append(
            {
                "paper_key": f"p:{i}",
                "canonical_paper_id": f"p:{i}",
                "topic_id": topics[min(i - 1, 1)]["topic_id"],
                "evidence_label": label,
                "evidence_source": source,
                "review_status": review,
                "evidence_method": "synthetic",
                "evidence_score": 0.9,
                "score_semantics": "synthetic_score_not_calibrated",
                "evidence_text_span": "We use",
                "evidence_start_offset": 0,
                "evidence_end_offset": 6,
                "text_field": "original_abstract",
                "annotator_or_model_id": "fixture",
                "model_version": "fixture-evidence",
                "topic_model_version": MODEL,
                "source_provenance": prov,
                "temporal_snapshot_id": SNAPSHOT,
                "uncertainty_status": "synthetic_fixture",
            }
        )
    intents = []
    for i, probs in [(1, [0.2, 0.6, 0.2]), (2, [0.4, 0.4, 0.2]), (3, [0.1, 0.8, 0.1])]:
        intents.append(
            {
                "citation_context_id": f"ctx:{i}",
                "source_dataset": "synthetic",
                "source_record_id": str(i),
                "citing_paper_id": "p:1" if i < 3 else None,
                "cited_paper_id": "p:2" if i < 3 else None,
                "source_split": "train",
                "predicted_intent": ("background", "method", "result")[
                    max(range(3), key=probs.__getitem__)
                ],
                "prob_background": probs[0],
                "prob_method": probs[1],
                "prob_result": probs[2],
                "confidence": max(probs),
                "predictive_entropy": -sum(p * math.log(p) for p in probs),
                "calibration_status": "uncalibrated",
                "model_id": "synthetic-model",
                "model_version": "run-intent",
                "identity_resolution_status": "EXACT_VERIFIED" if i < 3 else "UNRESOLVED",
                "prediction_timestamp": produced,
                "provenance": prov,
            }
        )
    values = {
        "papers": papers,
        "authors": authors,
        "authorships": authorships,
        "citations": citations,
        "source_fields": fields,
        "topics": topics,
        "assignments": assignments,
        "evidence": evidence,
        "intents": intents,
    }
    if mutate:
        mutate(values)
    for name in ("papers", "authors", "authorships", "citations", "source_fields"):
        write_jsonl(s3 / (name + ".jsonl"), values[name])
    write_jsonl(s3 / "source_local_papers.jsonl", values["papers"])
    atomic_json(
        s3 / "integration_manifest.json",
        {
            "stage": 3,
            "run_id": "run-fixture",
            "status": "COMPLETE",
            "synthetic_sources": ["synthetic"],
            "table_counts": {
                name: len(values[name])
                for name in ("papers", "authors", "authorships", "citations")
            },
            "temporal": {"policy": POLICY, "cutoff": "2025-12-31"},
        },
    )
    refresh(s3, "integration_manifest.json")
    atomic_json(s3.parent / "latest.json", {"run_dir": str(s3)})
    for name, key in [
        ("topic_catalog", "topics"),
        ("paper_topic_assignments", "assignments"),
        ("paper_topic_evidence", "evidence"),
    ]:
        write_jsonl(s5 / (name + ".jsonl"), values[key])
    prepared = [
        prepare_paper(row, Stage05Config(), policy=POLICY, cutoff=None).model_dump(mode="json")
        for row in values["papers"]
    ]
    write_jsonl(s5 / "papers.jsonl", prepared)
    write_jsonl(
        s5 / "paper_embedding_index.jsonl",
        [
            {
                "paper_key": row["paper_key"],
                "canonical_paper_id": row["canonical_paper_id"],
                "text_sha256": row["text_sha256"],
                "row": i,
            }
            for i, row in enumerate(prepared)
        ],
    )
    np.save(
        s5 / "paper_embeddings.npy",
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        allow_pickle=False,
    )
    atomic_json(
        s5 / "embedding_model_manifest.json",
        {
            "encoder": {
                "backend": "specter2",
                "model_id": "synthetic",
                "model_revision": "fixture-v1",
                "dimension": 2,
            }
        },
    )
    atomic_json(s5 / "training_corpus_manifest.json", {"stage03_run_id": "run-fixture"})
    atomic_json(s5 / "temporal_snapshot.json", {"snapshot_id": SNAPSHOT, "policy": POLICY})
    atomic_json(
        s5 / "stage05_manifest.json",
        {
            "stage": 5,
            "status": "COMPLETE",
            "run_id": "run-fixture",
            "created_at": produced,
            "temporal_snapshot": {"snapshot_id": SNAPSHOT, "cutoff": "2020-12-31"},
            "corpus_sha256": "synthetic",
            "execution_status": "TOPIC_MODELS_FITTED",
            "primary_algorithm": "bertopic",
        },
    )
    refresh(s5, "stage05_manifest.json")
    intent_path = s4 / "prediction_exports/stage03_citation_intent_evidence.jsonl"
    write_jsonl(intent_path, values["intents"])
    atomic_json(
        intent_path.with_name(intent_path.name + ".manifest.json"),
        {
            "stage": 4,
            "run_id": "run-intent",
            "dataset_fingerprint": "synthetic",
            "records": len(intents),
        },
    )
    atomic_json(
        s4 / "run_manifest.json",
        {
            "stage": 4,
            "status": "COMPLETE",
            "run_id": "run-intent",
            "created_at": produced,
            "dataset_manifest": {"dataset_fingerprint": "synthetic"},
        },
    )
    refresh(s4, "run_manifest.json")
    return Stage06Config.model_validate(
        {
            "inputs": {
                "stage03_root": s3.parent,
                "stage03_run": s3,
                "stage05_root": s5.parent,
                "topic_run": s5,
                "intent_export": intent_path,
            },
            "output_dir": tmp_path / "graphs",
            "report_dir": tmp_path / "reports",
            "batch_size": 1,
        }
    )


def change_config(config, **changes):
    from ci_hgnn_rgd.graph.stage06_config import Stage06Config

    values = config.model_dump(mode="python")
    for key, value in changes.items():
        if isinstance(value, dict):
            values[key].update(value)
        else:
            values[key] = value
    return Stage06Config.model_validate(values)
