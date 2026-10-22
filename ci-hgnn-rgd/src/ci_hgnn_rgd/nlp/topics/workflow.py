"""Stage 05 workflows. Every fitted representation is local to one frozen corpus snapshot."""

import json
import time
from importlib import metadata
from itertools import islice

from ...data.manifests import atomic_json
from ...data.temporal_integrity import parse_cutoff
from ..intent_contracts import CitationIntentEvidence
from ..intent_transformer import _available_system_memory_gib
from .artifacts import RunBuilder, iter_rows, resolve_run, write_rows
from .contracts import (
    InvestigationAnnotation,
    PaperTopicEvidence,
    PreparedPaper,
    TopicAssignment,
)
from .data import prepare_corpus, prepare_paper
from .embeddings import embed_corpus, encoder_identity
from .investigation import (
    adjudicate,
    agreement_report,
    annotation_sample,
    candidate_evidence,
    validate_annotations,
)
from .models import FittedTopics, fit_topics
from .quality import (
    align_topics,
    embedding_neighbors,
    expert_review_template,
    redundancy,
    topic_quality,
)
from .reporting import generate_reports


def estimate_resources(config, view):
    n, dim = len(view.papers), config.embedding.dimension
    installed = {}
    for package in (
        "torch",
        "transformers",
        "adapters",
        "bertopic",
        "umap-learn",
        "hdbscan",
        "pyarrow",
    ):
        try:
            installed[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed[package] = None
    ram = _available_system_memory_gib()
    blockers = []
    if not n:
        blockers.append("no_eligible_scholarly_paper_text")
    if ram is not None and ram < config.embedding.min_available_ram_gib:
        blockers.append("insufficient_available_RAM_for_encoder")
    required = ["torch", "transformers", "pyarrow"] + (
        ["adapters"] if config.embedding.backend == "specter2" else []
    )
    blockers.extend("missing_dependency:" + p for p in required if installed[p] is None)
    hardware = {
        "torch_version": installed["torch"],
        "cuda_available": False,
        "gpu_memory_gib": None,
    }
    if installed["torch"] and "+cpu" in installed["torch"]:
        hardware["device"] = "cpu"
        if config.embedding.device not in {"auto", "cpu"}:
            blockers.append("requested_accelerator_unavailable")
    elif installed["torch"]:
        import torch

        from ...runtime import detect_device

        hardware["device"] = detect_device(config.embedding.device).device
        hardware["cuda_available"] = torch.cuda.is_available()
        if hardware["cuda_available"]:
            free, total = torch.cuda.mem_get_info()
            hardware["gpu_memory_gib"] = {"free": free / 1024**3, "total": total / 1024**3}
    return {
        "papers": n,
        "float32_embedding_bytes": n * dim * 4,
        "dimension": dim,
        "available_system_ram_gib": ram,
        "packages": installed,
        "encoder_minimum_available_ram_gib": config.embedding.min_available_ram_gib,
        "encoder_quantization": config.embedding.quantization,
        "encoder_batch_size": config.embedding.batch_size,
        "encoder_max_length": config.embedding.max_length,
        "encoder_cpu_threads": config.embedding.cpu_threads,
        "ready_for_embedding": not blockers,
        "blockers": blockers,
        "hardware": hardware,
        "topic_memory_note": (
            "UMAP/HDBSCAN need additional data-dependent memory beyond "
            "embeddings; max_records is bounded. No full pairwise matrix."
        ),
        "downloads": "offline by default; resource estimation never fetches model weights",
    }


def _write_view(builder, view):
    atomic_json(builder.pending / "input_data_profile.json", view.profile)
    atomic_json(builder.pending / "training_corpus_manifest.json", view.manifest)
    atomic_json(builder.pending / "temporal_snapshot.json", view.temporal_snapshot)
    assessments = [p.temporal_assessment["assessments"] for p in view.papers]
    atomic_json(
        builder.pending / "topic_temporal_audit.json",
        {
            **view.temporal_snapshot,
            "papers_assessed": len(view.papers),
            "unknown_availability_papers": sum(
                any(item["unknown_availability"] for item in group) for group in assessments
            ),
            "future_feature_papers_at_current_cutoff": sum(
                any(reason.startswith("future_") for item in group for reason in item["reasons"])
                for group in assessments
            )
            if view.temporal_snapshot["cutoff"] is not None
            else None,
            "current_cutoff_available": view.temporal_snapshot["cutoff"] is not None,
            "source_stage03_audit": view.manifest.get("stage03_temporal_audit", {}),
            "interpretation": (
                "Source Stage 03 counts describe its input corpus and cutoff. "
                "Current counts describe this selected corpus; without a cutoff, "
                "future violations are not evaluable. Ingestion is not historical availability."
            ),
        },
    )
    write_rows(builder.pending, "papers", view.papers)


def _history_guard(config, snapshot):
    if snapshot["policy"] == "STRICT_POINT_IN_TIME":
        available = config.embedding.verified_model_available_at
        if available is None or parse_cutoff(available) > parse_cutoff(snapshot["cutoff"]):
            raise ValueError(
                "strict historical embeddings need verified_model_available_at at/before "
                "cutoff; corpus eligibility alone does not establish encoder eligibility"
            )


def _embeddings(builder, config, view, embedding_run_id=None):
    import numpy as np

    _history_guard(config, view.temporal_snapshot)
    if embedding_run_id:
        path, manifest = resolve_run(config.output_root, embedding_run_id)
        if manifest.get("corpus_sha256") != view.manifest["corpus_sha256"]:
            raise ValueError("precomputed embedding corpus differs from topic training corpus")
        stored = json.loads((path / "embedding_model_manifest.json").read_text(encoding="utf-8"))
        if stored["encoder"] != encoder_identity(config.embedding):
            raise ValueError("precomputed embedding model/tokenizer/adapter settings differ")
        # Copy the safe numeric mmap file and its verified row index, not model objects.
        import shutil

        shutil.copy2(path / "paper_embeddings.npy", builder.pending / "paper_embeddings.npy")
        info = stored
    else:
        info = embed_corpus(
            view.papers,
            builder.pending / "paper_embeddings.npy",
            config.embedding,
            config.cache_root,
            offline=config.offline,
        )
    atomic_json(builder.pending / "embedding_config.json", config.embedding.model_dump(mode="json"))
    atomic_json(builder.pending / "embedding_model_manifest.json", info)
    atomic_json(builder.pending / "embedding_quality.json", info["quality"])
    write_rows(builder.pending, "paper_embedding_index", info["index"])
    matrix = np.load(builder.pending / "paper_embeddings.npy", allow_pickle=False, mmap_mode="r")
    atomic_json(
        builder.pending / "embedding_neighbors.json", embedding_neighbors(matrix, view.papers)
    )
    return matrix


def run_corpus_action(config, root, action, *, embedding_run_id=None):
    builder = RunBuilder(config.output_root, config, kind=action, project_root=root)
    started = time.perf_counter()
    matrix = None
    try:
        view = prepare_corpus(config, exclusions_path=builder.pending / "excluded_records.jsonl")
        _write_view(builder, view)
        common = {
            "corpus_sha256": view.manifest["corpus_sha256"],
            "temporal_snapshot": view.temporal_snapshot,
            "paper_count": len(view.papers),
            "execution_status": view.profile["status"],
            "investigation_training_status": "NOT_TRAINED",
            "limitations": [
                "No human annotations or expert assessments are implied.",
                "Pretrained encoder temporal coverage is separate from corpus eligibility.",
            ],
        }
        if action in {"inspect-data", "audit-temporal", "estimate-resources"}:
            resource = estimate_resources(config, view) if action == "estimate-resources" else None
            if resource:
                atomic_json(builder.pending / "resource_estimate.json", resource)
                common["resources"] = resource
            result = builder.publish(common)
            result["reports"] = generate_reports(builder.final, result, config.report_root)
            return result, 0 if view.papers or action == "estimate-resources" else 1
        if not view.papers:
            raise ValueError(view.profile["blocker"])
        matrix = (
            _embeddings(builder, config, view, embedding_run_id)
            if action in {"embed", "train"}
            else None
        )
        if action == "embed":
            matrix._mmap.close()
            matrix = None
            result = builder.publish(
                {
                    **common,
                    "execution_status": "EMBEDDINGS_GENERATED",
                    "runtime_seconds": time.perf_counter() - started,
                }
            )
        else:
            algorithms = ("nmf", "lda") if action == "train-baselines" else ("bertopic",)
            comparisons = []
            for index, algorithm in enumerate(algorithms):
                begin = time.perf_counter()
                model = fit_topics(
                    view.papers,
                    config.topic_model,
                    algorithm=algorithm,
                    version=builder.run_id + ":" + algorithm,
                    snapshot=view.temporal_snapshot["snapshot_id"],
                    seed=config.seed,
                    embeddings=matrix,
                )
                model.save(builder.pending / "models" / algorithm)
                quality = topic_quality(model, view.papers)
                quality["redundancy"] = redundancy(model.catalog)
                prefix = "" if index == 0 else algorithm + "_"
                write_rows(builder.pending, prefix + "topic_catalog", model.catalog)
                write_rows(builder.pending, prefix + "paper_topic_assignments", model.assignments)
                write_rows(
                    builder.pending,
                    prefix + "topic_keywords",
                    (
                        {"topic_id": topic.topic_id, "rank": rank, "keyword": word}
                        for topic in model.catalog
                        for rank, word in enumerate(topic.keywords, 1)
                    ),
                )
                write_rows(
                    builder.pending,
                    prefix + "representative_papers",
                    (
                        {"topic_id": topic.topic_id, "paper_key": key}
                        for topic in model.catalog
                        for key in topic.representative_paper_keys
                    ),
                )
                atomic_json(builder.pending / (prefix + "topic_quality.json"), quality)
                write_rows(
                    builder.pending,
                    prefix + "topic_quality_metrics",
                    [{"model": algorithm, "metrics": quality}],
                )
                if index == 0:
                    write_rows(
                        builder.pending,
                        "topic_expert_review",
                        expert_review_template(model.catalog),
                    )
                    write_rows(
                        builder.pending,
                        "topic_temporal_metadata",
                        (
                            {
                                "topic_id": topic.topic_id,
                                "publication_year_distribution": (
                                    topic.publication_year_distribution
                                ),
                                "interpretation": (
                                    "Retrospective counts under a frozen "
                                    "representation; no prospective claim."
                                ),
                            }
                            for topic in model.catalog
                        ),
                    )
                comparisons.append(
                    {
                        "model": algorithm
                        + (":" + config.embedding.backend if algorithm == "bertopic" else ""),
                        "topics": len(model.catalog),
                        "diversity": quality["topic_diversity"],
                        "npmi": quality["coherence"]["mean"],
                        "outlier_rate": quality["outlier_rate"],
                        "runtime_seconds": time.perf_counter() - begin,
                    }
                )
            atomic_json(
                builder.pending / "topic_stability.json",
                {
                    "status": "NOT_EVALUABLE",
                    "reason": "No repeated fits requested; use topics evaluate --stability",
                },
            )
            atomic_json(
                builder.pending / "topic_assignment_config.json",
                {
                    "training_assignments": "original_fitted_model_assignments",
                    "new_bertopic_documents": "frozen_centroid_with_train_only_rejection",
                    "scores_calibrated": False,
                    "outliers_forced_into_topics": False,
                },
            )
            if matrix is not None:
                matrix._mmap.close()
                matrix = None
            result = builder.publish(
                {
                    **common,
                    "execution_status": "TOPIC_MODELS_FITTED",
                    "primary_algorithm": algorithms[0],
                    "model_comparison": comparisons,
                    "embedding_backend": config.embedding.backend if action == "train" else None,
                    "runtime_seconds": time.perf_counter() - started,
                },
                fitted_topic_model=True,
            )
        result["reports"] = generate_reports(builder.final, result, config.report_root)
        return result, 0
    except Exception as error:
        builder.fail(error)
        raise
    finally:
        if matrix is not None:
            matrix._mmap.close()


def load_topic_run(config, run_id, *, model_name=None):
    path, manifest = resolve_run(config.output_root, run_id)
    algorithm = model_name or manifest.get("primary_algorithm")
    if not algorithm:
        raise ValueError("run contains no fitted topic model")
    model = FittedTopics.load(path / "models" / algorithm)
    papers = tuple(PreparedPaper.model_validate(row) for row in iter_rows(path / "papers.jsonl"))
    return path, manifest, model, papers


def downstream_action(config, root, args):
    path, parent, model, papers = load_topic_run(
        config, args.run_id, model_name=getattr(args, "model_name", None)
    )
    action = args.topic_command
    if action == "report":
        return {"reports": generate_reports(path, parent, config.report_root)}, 0
    builder = RunBuilder(config.output_root, config, kind=action, project_root=root)
    try:
        snapshot = parent["temporal_snapshot"]
        assignments = tuple(
            TopicAssignment.model_validate(row)
            for row in iter_rows(
                path
                / (
                    (
                        ""
                        if model.algorithm == parent["primary_algorithm"]
                        else model.algorithm + "_"
                    )
                    + "paper_topic_assignments.jsonl"
                )
            )
        )
        details = {
            "parent_run_id": args.run_id,
            "temporal_snapshot": snapshot,
            "corpus_sha256": parent["corpus_sha256"],
            "execution_status": "COMPLETED",
            "limitations": ["No heterogeneous graph or research-gap score is constructed."],
        }
        if action == "assign":
            assigned_count, seen = 0, set()
            cutoff = config.temporal.cutoff or snapshot["cutoff"]
            if config.temporal.policy and config.temporal.policy != snapshot["policy"]:
                raise ValueError("inference cannot change the frozen topic temporal policy")
            if snapshot["policy"] != "EXPLORATORY_NO_TEMPORAL_CLAIM" and (
                parse_cutoff(cutoff) < parse_cutoff(snapshot["cutoff"])
            ):
                raise ValueError(
                    "future-fitted topic model cannot serve an earlier historical cutoff"
                )
            encoder = None
            if model.algorithm == "bertopic":
                from .embeddings import ScientificEncoder

                stored = json.loads((path / "embedding_model_manifest.json").read_text())
                if stored["encoder"] != encoder_identity(config.embedding):
                    raise ValueError("assignment encoder differs from fitted topic encoder")
                encoder = ScientificEncoder(
                    config.embedding, offline=config.offline, cache_dir=config.cache_root / "models"
                )

            def assignments_stream():
                nonlocal assigned_count
                source = iter(iter_rows(args.input))
                batch_index = 0
                while rows := list(islice(source, config.embedding.batch_size)):
                    incoming = tuple(
                        prepare_paper(row, config, policy=snapshot["policy"], cutoff=cutoff)
                        for row in rows
                    )
                    for paper in incoming:
                        if paper.paper_key in seen:
                            raise ValueError("duplicate paper identities in assignment input")
                        seen.add(paper.paper_key)
                        if len(seen) > config.dataset.max_records:
                            raise ValueError(
                                "assignment exceeds max_records; split input or raise the limit"
                            )
                    matrix = None
                    if encoder is not None:
                        import numpy as np

                        target = builder.pending / f"inference_embeddings_{batch_index:06d}.npy"
                        embed_corpus(
                            incoming,
                            target,
                            config.embedding,
                            config.cache_root,
                            offline=config.offline,
                            encoder=encoder,
                        )
                        matrix = np.load(target, mmap_mode="r", allow_pickle=False)
                    output = model.assign(incoming, embeddings=matrix)
                    assigned_count += sum(not item.outlier_status for item in output)
                    yield from output
                    batch_index += 1

            write_rows(builder.pending, "paper_topic_assignments", assignments_stream())
            details.update(papers=len(seen), assigned=assigned_count, topic_model_refitted=False)
        elif action == "export-catalog":
            write_rows(builder.pending, "topic_catalog", model.catalog)
        elif action in {"generate-investigation-candidates", "predict-investigation"}:
            paper_map = {p.paper_key: p for p in papers}
            topic_map = {t.topic_id: t for t in model.catalog}
            evidence = []
            pairs = (
                list(iter_rows(args.input))
                if getattr(args, "input", None)
                else [
                    {"paper_key": a.paper_key, "topic_id": a.topic_id}
                    for a in assignments
                    if a.topic_id
                ]
            )
            for pair in pairs:
                if pair.get("paper_key") not in paper_map:
                    supplied = pair.get("paper", pair)
                    paper = prepare_paper(
                        supplied,
                        config,
                        policy=snapshot["policy"],
                        cutoff=config.temporal.cutoff or snapshot["cutoff"],
                    )
                    if pair.get("paper_key") and pair["paper_key"] != paper.paper_key:
                        raise ValueError("paper key differs from source identity")
                    pair["paper_key"] = paper.paper_key
                    paper_map[paper.paper_key] = paper
                paper, topic = paper_map[pair["paper_key"]], topic_map[pair["topic_id"]]
                evidence.append(candidate_evidence(paper, topic))
            write_rows(
                builder.pending,
                "prediction_papers",
                (paper_map[pair["paper_key"]] for pair in pairs),
            )
            if getattr(args, "rank_evidence", False):
                from .embeddings import ScientificEncoder
                from .investigation import rank_evidence_spans

                if model.algorithm != "bertopic":
                    raise ValueError(
                        "scientific evidence ranking requires a fitted scientific BERTopic centroid"
                    )
                _history_guard(config, snapshot)
                stored = json.loads((path / "embedding_model_manifest.json").read_text())
                if stored["encoder"] != encoder_identity(config.embedding):
                    raise ValueError("evidence-ranking encoder differs from topic encoder")
                encoder = ScientificEncoder(
                    config.embedding, offline=config.offline, cache_dir=config.cache_root / "models"
                )
                centroids = {
                    local: model.arrays["centroids"][index]
                    for index, local in enumerate(model.metadata["local_topic_ids"])
                }
                evidence = [
                    rank_evidence_spans(
                        paper_map[pair["paper_key"]],
                        topic_map[pair["topic_id"]],
                        centroids[topic_map[pair["topic_id"]].local_topic_id],
                        encoder,
                    )
                    for pair in pairs
                ]
            if getattr(args, "classifier_run_id", None):
                from types import SimpleNamespace

                from .investigation_model import predict_investigation_model

                classifier_path, classifier_manifest = resolve_run(
                    config.output_root, args.classifier_run_id
                )
                if classifier_manifest.get("investigation_training_status") != "TRAINED":
                    raise ValueError(
                        "selected investigation model is not a validated real trained run"
                    )
                inputs = [
                    SimpleNamespace(
                        title=paper_map[p["paper_key"]].title,
                        abstract=paper_map[p["paper_key"]].abstract,
                        topic_definition=topic_map[p["topic_id"]].provisional_label,
                    )
                    for p in pairs
                ]
                probabilities = predict_investigation_model(
                    classifier_path,
                    inputs,
                    device=config.embedding.device,
                    batch_size=config.investigation.batch_size,
                )
                from .contracts import LABELS

                evidence = [
                    PaperTopicEvidence.model_validate(
                        {
                            **item.model_dump(mode="python"),
                            "evidence_source": "model",
                            "evidence_label": LABELS[max(range(3), key=values.__getitem__)],
                            "evidence_method": "supervised_scibert_paper_topic",
                            "evidence_score": max(values),
                            "score_semantics": "uncalibrated_softmax",
                            "review_status": "model_predicted",
                            "annotator_or_model_id": args.classifier_run_id,
                            "model_version": args.classifier_run_id,
                            "uncertainty_status": "unvalidated_model_prediction",
                            "evidence_text_span": None,
                            "evidence_start_offset": None,
                            "evidence_end_offset": None,
                            "text_field": None,
                            "candidate_label": None,
                        }
                    )
                    for item, values in zip(evidence, probabilities, strict=True)
                ]
            write_rows(builder.pending, "investigation_candidates", evidence)
            write_rows(builder.pending, "paper_topic_evidence", evidence)
            details["evidence_count"] = len(evidence)
        elif action == "export-annotation-sample":
            sample = annotation_sample(
                papers,
                model.catalog,
                assignments,
                size=config.investigation.annotation_sample_size,
                seed=config.seed,
                guideline_version=config.investigation.guideline_version,
            )
            write_rows(builder.pending, "annotation_tasks", sample)
            details.update(
                tasks=len(sample), real_annotations=0, investigation_training_status="NOT_TRAINED"
            )
        elif action in {
            "import-annotations",
            "evaluate-annotations",
            "train-investigation-classifier",
        }:
            if not getattr(args, "input", None):
                raise ValueError(
                    "NOT_TRAINED: provide independently completed annotation JSONL via --input"
                )
            annotations = validate_annotations(
                list(iter_rows(args.input)),
                papers,
                model.catalog,
                guideline_version=config.investigation.guideline_version,
            )
            write_rows(builder.pending, "investigation_annotations", annotations)
            report = agreement_report(annotations)
            atomic_json(builder.pending / "annotation_report.json", report)
            adjudications = (
                list(iter_rows(args.adjudications)) if getattr(args, "adjudications", None) else []
            )
            finals = adjudicate(annotations, adjudications=adjudications)
            write_rows(builder.pending, "validated_investigation_labels", finals)
            human_evidence = []
            paper_map, topic_map = (
                {p.paper_key: p for p in papers},
                {t.topic_id: t for t in model.catalog},
            )
            for final in finals:
                annotation = InvestigationAnnotation.model_validate(final["annotation"])
                paper, topic = paper_map[annotation.paper_key], topic_map[annotation.topic_id]
                human_evidence.append(
                    PaperTopicEvidence(
                        paper_key=paper.paper_key,
                        canonical_paper_id=paper.canonical_paper_id,
                        topic_id=topic.topic_id,
                        evidence_label=annotation.label,
                        evidence_source="human",
                        evidence_text_span=annotation.relevant_text_span,
                        evidence_start_offset=annotation.start_offset,
                        evidence_end_offset=annotation.end_offset,
                        text_field="original_" + annotation.text_field
                        if annotation.text_field
                        else None,
                        evidence_method="independent_annotation_consensus_or_adjudication",
                        evidence_score=annotation.confidence,
                        score_semantics="human_self_reported_confidence",
                        review_status="human_verified",
                        annotator_or_model_id=final["adjudicator_id"]
                        or "+".join(final["independent_annotator_ids"]),
                        model_version="human:" + config.investigation.guideline_version,
                        topic_model_version=topic.model_version,
                        source_provenance=paper.provenance,
                        temporal_snapshot_id=topic.temporal_snapshot_id,
                        uncertainty_status="human_annotation_not_infallible",
                    )
                )
            write_rows(builder.pending, "paper_topic_evidence", human_evidence)
            details.update(
                annotations=len(annotations),
                validated_pairs=len(finals),
                investigation_training_status="NOT_TRAINED",
            )
            if action == "train-investigation-classifier":
                from .investigation_model import train_investigation

                result = train_investigation(finals, builder.pending, config)
                details["investigation_training_status"] = result["status"]
                details["training"] = result
        elif action == "evaluate":
            quality = topic_quality(model, papers)
            atomic_json(builder.pending / "topic_quality.json", quality)
            if getattr(args, "compare_run_id", None):
                _, other_meta, other, other_papers = load_topic_run(config, args.compare_run_id)
                if parent["corpus_sha256"] != other_meta["corpus_sha256"]:
                    raise ValueError("topic comparison requires identical frozen corpora")
                other_quality = topic_quality(other, other_papers)
                if (
                    quality["coherence"]["tokenization"]
                    != other_quality["coherence"]["tokenization"]
                ):
                    raise ValueError("topic coherence comparison tokenizers differ")
                atomic_json(
                    builder.pending / "topic_comparison.json",
                    {"left": quality, "right": other_quality},
                )
            if getattr(args, "stability", False):
                import numpy as np

                from .config import ModelSettings

                settings = ModelSettings.model_validate(model.metadata["settings"])
                matrix = (
                    np.load(path / "paper_embeddings.npy", mmap_mode="r", allow_pickle=False)
                    if model.algorithm == "bertopic"
                    else None
                )
                stability = []
                for seed in settings.stability_seeds:
                    repeated = fit_topics(
                        papers,
                        settings,
                        algorithm=model.algorithm,
                        version=builder.run_id + ":seed:" + str(seed),
                        snapshot=snapshot["snapshot_id"],
                        seed=seed,
                        embeddings=matrix,
                    )
                    stability.append(
                        {"seed": seed, "alignment": align_topics(model.catalog, repeated.catalog)}
                    )
                atomic_json(builder.pending / "topic_stability.json", {"repeated_fits": stability})
        elif action == "export-graph-evidence":
            write_rows(builder.pending, "topic_catalog", model.catalog)
            write_rows(builder.pending, "paper_topic_assignments", assignments)
            evidence = []
            if getattr(args, "evidence_run_id", None):
                evidence_path, evidence_meta = resolve_run(config.output_root, args.evidence_run_id)
                if evidence_meta.get("parent_run_id") != args.run_id:
                    raise ValueError("paper-topic evidence belongs to a different frozen topic run")
                evidence = [
                    PaperTopicEvidence.model_validate(row)
                    for row in iter_rows(evidence_path / "paper_topic_evidence.jsonl")
                ]
                valid_topics = {topic.topic_id for topic in model.catalog}
                if any(
                    item.topic_id not in valid_topics
                    or item.topic_model_version != model.version
                    or item.temporal_snapshot_id != model.snapshot
                    for item in evidence
                ):
                    raise ValueError("evidence references a different topic model or snapshot")
            else:
                pmap, tmap = (
                    {p.paper_key: p for p in papers},
                    {t.topic_id: t for t in model.catalog},
                )
                evidence = [
                    candidate_evidence(pmap[a.paper_key], tmap[a.topic_id])
                    for a in assignments
                    if a.topic_id
                ]
            write_rows(builder.pending, "paper_topic_evidence", evidence)
            source = getattr(args, "citation_intent_input", None)
            count = (
                write_rows(
                    builder.pending,
                    "citation_intent_evidence",
                    (CitationIntentEvidence.model_validate(row) for row in iter_rows(source)),
                )
                if source and source.exists()
                else 0
            )
            details.update(
                stage04_records=count,
                stage04_status="PRESERVED_SEPARATELY" if count else "UNAVAILABLE_OR_NOT_SUPPLIED",
                graph_created=False,
                missing_endpoints="remain_null",
                stage06_export_schema="stage05_evidence_v1",
            )
        else:
            raise ValueError("unsupported downstream Stage 05 action")
        result = builder.publish(
            details, fitted_investigation_model=action == "train-investigation-classifier"
        )
        result["reports"] = generate_reports(builder.final, result, config.report_root)
        return result, 0
    except Exception as error:
        builder.fail(error)
        raise
