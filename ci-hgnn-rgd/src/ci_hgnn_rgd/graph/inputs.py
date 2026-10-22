"""Manifest-checked adapters; IDs rather than independent file order define all joins."""

import json
from dataclasses import dataclass
from pathlib import Path

from ..data.provenance import file_sha256
from ..data.temporal_integrity import TemporalPolicy, parse_cutoff
from .stage06_config import Stage06Config


def document(path: Path) -> dict:
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError(f"metadata exceeds the 64 MiB JSON budget: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def rows(path: Path, *, batch_size=256, max_record_bytes=2097152):
    if not path.exists():
        return
    if path.is_dir():
        for part in sorted(path.glob("*.parquet")):
            yield from rows(part, batch_size=batch_size, max_record_bytes=max_record_bytes)
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                if (
                    "record_json" in row
                    and len(row["record_json"].encode("utf-8")) > max_record_bytes
                ):
                    raise ValueError(f"Parquet record exceeds configured byte limit: {path}")
                value = json.loads(row["record_json"]) if "record_json" in row else row
                yield value
    else:
        with path.open("rb") as stream:
            while line := stream.readline(max_record_bytes + 1):
                if len(line) > max_record_bytes:
                    raise ValueError(f"record exceeds configured byte limit: {path}")
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"invalid table row: {path}")
                    yield value


def table(root: Path, name: str) -> Path:
    for suffix in (".jsonl", ".parquet"):
        path = root / (name + suffix)
        if path.exists():
            return path
    return root / (name + ".jsonl")


def verify_file(path: Path, root: Path, manifest: dict, *, required=True) -> str | None:
    if not path.exists():
        if required:
            raise ValueError(f"required input is unavailable: {path}")
        return None
    checksums = manifest.get("artifact_checksums", manifest.get("output_checksums", {}))
    if path.is_dir():
        for part in sorted(path.glob("*.parquet")):
            verify_file(part, root, manifest)
        return None
    key = path.relative_to(root).as_posix()
    normalized = {k.replace("\\", "/"): v for k, v in checksums.items()}
    digest = file_sha256(path)
    if key not in normalized or normalized[key] != digest:
        raise ValueError(f"missing or mismatched input checksum: {path}")
    return digest


def published(root: Path, filename: str, stage: int) -> dict:
    if root.name.startswith(".pending-") or (root / "failure.json").exists():
        raise ValueError(f"input run is not published: {root}")
    manifest = document(root / filename)
    if manifest.get("stage", stage) != stage or manifest.get("status") != "COMPLETE":
        raise ValueError(f"invalid stage {stage} manifest: {root}")
    return manifest


@dataclass
class GraphInputBundle:
    stage03: Path
    stage03_manifest: dict
    topic_run: Path | None
    topic_manifest: dict | None
    topic_export: Path | None
    export_manifest: dict | None
    embedding_run: Path | None
    embedding_manifest: dict | None
    intent_export: Path | None
    intent_manifest: dict | None
    policy: str
    cutoff: str | None
    snapshot_id: str
    blocked: list[str]
    checksums: dict[str, str]


def inspect_inputs(config: Stage06Config) -> GraphInputBundle:
    from ..pipeline import fingerprint

    inputs = config.inputs
    if inputs.stage03_run:
        stage03 = inputs.stage03_run
    else:
        stage03 = Path(document(inputs.stage03_root / "latest.json")["run_dir"])
    if not stage03.resolve().is_relative_to(inputs.stage03_root.resolve()):
        raise ValueError("Stage 03 run must remain inside configured Stage 03 root")
    s3 = published(stage03, "integration_manifest.json", 3)
    blocked, checksums = [], {}

    def verify(path, root, manifest, required=True):
        digest = verify_file(path, root, manifest, required=required)
        if digest:
            checksums[str(path)] = digest

    for name in (
        "papers",
        "source_local_papers",
        "authors",
        "authorships",
        "citations",
        "source_fields",
        "temporal_eligibility",
        "identifier_mappings",
    ):
        verify(table(stage03, name), stage03, s3, required=name == "papers")
    policy = config.temporal.policy or s3.get("temporal", {}).get("policy")
    if policy is None:
        raise ValueError("Stage 03 temporal policy missing; configure it explicitly")
    policy = TemporalPolicy(policy).value
    cutoff = config.temporal.cutoff or s3.get("temporal", {}).get("cutoff")
    if cutoff:
        # Older manifests use a space before their timezone; normalize once.
        cutoff = parse_cutoff(cutoff).isoformat()
    if policy != TemporalPolicy.EXPLORATORY_NO_TEMPORAL_CLAIM and cutoff is None:
        raise ValueError("historical graph construction requires a cutoff")
    topic_run, topic_manifest = inputs.topic_run, None
    if config.include_topics and topic_run is None:
        for file in sorted(inputs.stage05_root.glob("run-*/stage05_manifest.json"), reverse=True):
            candidate = document(file)
            if (
                candidate.get("status") == "COMPLETE"
                and candidate.get("execution_status") == "TOPIC_MODELS_FITTED"
                and candidate.get("primary_algorithm") == "bertopic"
            ):
                corpus = document(file.parent / "training_corpus_manifest.json")
                if corpus.get("stage03_run_id") != s3.get("run_id"):
                    continue
                topic_run = file.parent
                break
    export, export_manifest, embedding, embedding_manifest = None, None, None, None
    if config.include_topics and topic_run:
        topic_manifest = published(topic_run, "stage05_manifest.json", 5)
        corpus = document(topic_run / "training_corpus_manifest.json")
        if corpus.get("stage03_run_id") != s3.get("run_id"):
            raise ValueError("topic corpus belongs to a different Stage 03 run")
        for source in corpus.get("tables", []):
            source_path = (stage03 / source["path"]).resolve()
            if not source_path.is_relative_to(stage03.resolve()):
                raise ValueError("topic corpus source path escapes its Stage 03 run")
            if file_sha256(source_path) != source["sha256"]:
                raise ValueError("topic training corpus Stage 03 checksum mismatch")
        export = inputs.topic_export or topic_run
        if inputs.topic_export is None:
            for file in sorted(
                inputs.stage05_root.glob("run-*/stage05_manifest.json"), reverse=True
            ):
                candidate = document(file)
                if (
                    candidate.get("status") == "COMPLETE"
                    and candidate.get("kind") == "export-graph-evidence"
                    and candidate.get("parent_run_id") == topic_manifest.get("run_id")
                ):
                    export = file.parent
                    break
        export_manifest = published(export, "stage05_manifest.json", 5)
        if export != topic_run and export_manifest.get("parent_run_id") != topic_manifest["run_id"]:
            raise ValueError("Stage 05 evidence export belongs to a different topic model")
        if export_manifest.get("corpus_sha256") != topic_manifest.get("corpus_sha256"):
            raise ValueError("Stage 05 export corpus fingerprint mismatch")
        for name in ("topic_catalog", "paper_topic_assignments", "paper_topic_evidence"):
            verify(
                table(export, name),
                export,
                export_manifest,
                required=name != "paper_topic_evidence",
            )
        for name in ("training_corpus_manifest.json", "temporal_snapshot.json", "papers.jsonl"):
            verify(topic_run / name, topic_run, topic_manifest)
        embedding = inputs.embedding_run or topic_run
    elif config.include_topics:
        blocked.append("Stage 05 fitted BERTopic catalog unavailable")
    if inputs.embedding_run:
        embedding = inputs.embedding_run
    if embedding is None and config.features.paper_embedding != "none":
        for file in sorted(inputs.stage05_root.glob("run-*/stage05_manifest.json"), reverse=True):
            candidate = document(file)
            if (
                candidate.get("status") != "COMPLETE"
                or not (file.parent / "paper_embeddings.npy").exists()
            ):
                continue
            corpus = document(file.parent / "training_corpus_manifest.json")
            if corpus.get("stage03_run_id") == s3.get("run_id"):
                encoder = document(file.parent / "embedding_model_manifest.json")["encoder"]
                if encoder.get("backend") == config.features.paper_embedding:
                    embedding = file.parent
                    break
    if embedding and config.features.paper_embedding != "none":
        embedding_manifest = published(embedding, "stage05_manifest.json", 5)
        corpus = document(embedding / "training_corpus_manifest.json")
        if corpus.get("stage03_run_id") != s3.get("run_id"):
            raise ValueError("embedding corpus belongs to a different Stage 03 run")
        for name in (
            "paper_embeddings.npy",
            "paper_embedding_index.jsonl",
            "embedding_model_manifest.json",
            "papers.jsonl",
            "training_corpus_manifest.json",
        ):
            verify(embedding / name, embedding, embedding_manifest)
        encoder = document(embedding / "embedding_model_manifest.json")["encoder"]
        if encoder.get("backend") != config.features.paper_embedding or not encoder.get(
            "model_revision"
        ):
            raise ValueError("embedding model backend/revision incompatible with graph config")
    else:
        embedding = None
        if config.features.paper_embedding != "none":
            blocked.append("scientific embeddings unavailable; features explicitly masked")
    intent, intent_manifest = inputs.intent_export, None
    if intent is None:
        candidates = sorted(
            inputs.intent_model_root.glob(
                "run-*/prediction_exports/stage03_citation_intent_evidence.jsonl"
            ),
            reverse=True,
        )
        intent = candidates[0] if candidates else None
    if intent:
        intent_manifest = document(intent.with_name(intent.name + ".manifest.json"))
        run = intent.parent.parent
        im = published(run, "run_manifest.json", 4)
        intent_manifest["_model_created_at"] = im.get("created_at") or im.get("completed_at")
        if intent_manifest.get("run_id") != im.get("run_id") or intent_manifest.get(
            "dataset_fingerprint"
        ) != im.get("dataset_manifest", {}).get("dataset_fingerprint"):
            raise ValueError("citation-intent export model/data manifest mismatch")
        verify(intent, run, im)
    else:
        blocked.append("citation-intent prediction export unavailable")
    snapshot_id = config.temporal.snapshot_id or "snapshot:" + fingerprint([policy, cutoff])[:20]
    return GraphInputBundle(
        stage03,
        s3,
        topic_run,
        topic_manifest,
        export,
        export_manifest,
        embedding,
        embedding_manifest,
        intent,
        intent_manifest,
        policy,
        cutoff,
        snapshot_id,
        blocked,
        checksums,
    )


def input_report(bundle: GraphInputBundle) -> dict:
    return {
        "stage03_run": str(bundle.stage03),
        "stage03_counts": bundle.stage03_manifest.get("table_counts", {}),
        "topic_run": str(bundle.topic_run) if bundle.topic_run else None,
        "topic_export": str(bundle.topic_export) if bundle.topic_export else None,
        "embedding_run": str(bundle.embedding_run) if bundle.embedding_run else None,
        "intent_export": str(bundle.intent_export) if bundle.intent_export else None,
        "intent_records": (bundle.intent_manifest or {}).get("records"),
        "policy": bundle.policy,
        "cutoff": bundle.cutoff,
        "snapshot_id": bundle.snapshot_id,
        "blocked_dependencies": bundle.blocked,
        "verified_input_files": len(bundle.checksums),
        "graph_created": False,
    }
