"""Synthetic Stage 05 software fixtures, never scholarly results or human annotation."""

import json
from datetime import UTC, datetime

from ci_hgnn_rgd.data.provenance import file_sha256
from ci_hgnn_rgd.nlp.topics.config import Stage05Config
from ci_hgnn_rgd.nlp.topics.data import prepare_paper

POLICY = "EXPLORATORY_NO_TEMPORAL_CLAIM"


def source_row(index=0, **changes):
    category = "physics" if index % 2 == 0 else "biology"
    return {
        "canonical_paper_id": f"synthetic:p{index}",
        "source_dataset": "synthetic",
        "source_record_id": str(index),
        "record_key": f"fixture:{index}",
        "dataset_version": "fixture-v1",
        "title": f"{category} fixture paper {index}",
        "abstract": (
            f"We investigate {category} mechanisms with controlled experiments. "
            f"These {category} measurements provide experimental analysis."
        ),
        "publication_year": 2020,
        "publication_date": "2020-02-01",
        "date_precision": "day",
        "available_at": "2020-02-02",
        "categories": [category],
        "provenance": [{"synthetic": True, "source_dataset": "synthetic", "record": index}],
        **changes,
    }


def configuration(tmp_path, **changes):
    values = {
        "temporal": {"inherit_stage03_policy": False, "policy": POLICY},
        "dataset": {"stage03_root": tmp_path / "stage03", "max_records": 100},
        "topic_model": {
            "n_topics": 2,
            "min_df": 1,
            "min_topic_size": 2,
            "umap_neighbors": 2,
            "hdbscan_min_samples": 1,
            "max_iter": 100,
        },
        "output_root": tmp_path / "stage05",
        "model_root": tmp_path / "models",
        "investigation_model_root": tmp_path / "investigation_models",
        "report_root": tmp_path / "reports",
        "cache_root": tmp_path / "cache",
    }
    values.update(changes)
    return Stage05Config.model_validate(values)


def papers(config=None, count=8):
    config = config or Stage05Config()
    return tuple(
        prepare_paper(source_row(index), config, policy=POLICY, cutoff=None)
        for index in range(count)
    )


def stage03_fixture(tmp_path, rows=None):
    root = tmp_path / "stage03"
    run = root / "fixture-run"
    run.mkdir(parents=True)
    rows = rows if rows is not None else [source_row(i) for i in range(8)]
    checksums = {}
    for table, records in (("papers", rows), ("source_local_papers", [])):
        path = run / (table + ".jsonl")
        path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        checksums[path.name] = file_sha256(path)
    (run / "integration_manifest.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "run_id": "synthetic-stage03",
                "real_sources": [],
                "synthetic_sources": ["synthetic"],
                "output_checksums": checksums,
                "config": {"temporal_integrity": {"policy": POLICY}},
            }
        ),
        encoding="utf-8",
    )
    (root / "latest.json").write_text(json.dumps({"run_dir": str(run)}), encoding="utf-8")
    return configuration(tmp_path), run


def annotation(paper, topic, annotator, label="INVESTIGATES_TOPIC", split="train"):
    return {
        "paper_key": paper.paper_key,
        "canonical_paper_id": paper.canonical_paper_id,
        "topic_id": topic.topic_id,
        "topic_definition": topic.provisional_label,
        "title": paper.original_title,
        "abstract": paper.original_abstract,
        "label": label,
        "annotator_id": annotator,
        "annotation_timestamp": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        "guideline_version": "stage05-v1",
        "confidence": 0.8,
        "text_field": "abstract" if label != "UNKNOWN" else None,
        "relevant_text_span": paper.original_abstract if label != "UNKNOWN" else None,
        "start_offset": 0 if label != "UNKNOWN" else None,
        "end_offset": len(paper.original_abstract) if label != "UNKNOWN" else None,
        "split": split,
        "synthetic": True,
    }
