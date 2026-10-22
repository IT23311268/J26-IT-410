"""Stage 05 end-to-end synthetic workflows and explicit CLI failure boundaries."""

import argparse
import json
from pathlib import Path

import pytest
from conftest import PROJECT
from stage05_helpers import stage03_fixture

from ci_hgnn_rgd.cli import main
from ci_hgnn_rgd.nlp.topics.artifacts import iter_rows, latest_topic_run_id, resolve_run
from ci_hgnn_rgd.nlp.topics.data import prepare_corpus
from ci_hgnn_rgd.nlp.topics.workflow import downstream_action, run_corpus_action


@pytest.fixture
def topic_run(tmp_path, monkeypatch):
    config, _ = stage03_fixture(tmp_path)
    original = prepare_corpus

    def software_fixture(*args, **kwargs):
        return original(*args, **kwargs, allow_synthetic_fixtures=True)

    monkeypatch.setattr("ci_hgnn_rgd.nlp.topics.workflow.prepare_corpus", software_fixture)
    result, status = run_corpus_action(config, tmp_path, "train-baselines")
    assert status == 0
    return config, result, tmp_path


def args(run_id, command, **kwargs):
    return argparse.Namespace(run_id=run_id, topic_command=command, model_name=None, **kwargs)


def test_complete_baseline_export_annotation_and_report_workflow(topic_run):
    config, result, root = topic_run
    assert len(result["model_comparison"]) == 2
    assert Path(result["model_artifact_dir"]).is_dir()
    original, _ = resolve_run(config.output_root, result["run_id"])
    for name in (
        "topic_catalog",
        "paper_topic_assignments",
        "topic_keywords",
        "representative_papers",
    ):
        assert (original / (name + ".parquet")).is_file()
    evidence, status = downstream_action(
        config, root, args(result["run_id"], "generate-investigation-candidates")
    )
    assert status == 0
    rows = list(iter_rows(Path(evidence["run_dir"]) / "paper_topic_evidence.parquet"))
    assert rows and {row["evidence_label"] for row in rows} == {"UNKNOWN"}
    sample, status = downstream_action(
        config, root, args(result["run_id"], "export-annotation-sample")
    )
    assert (
        sample["real_annotations"] == 0 and sample["investigation_training_status"] == "NOT_TRAINED"
    )
    exported, status = downstream_action(
        config,
        root,
        args(
            result["run_id"],
            "export-graph-evidence",
            evidence_run_id=evidence["run_id"],
            citation_intent_input=None,
        ),
    )
    assert exported["graph_created"] is False and exported["stage04_records"] == 0
    assert exported["stage06_export_schema"] == "stage05_evidence_v1"
    assert not (root / "graphs").exists()
    reported, _ = downstream_action(config, root, args(result["run_id"], "report"))
    report_dir = Path(reported["reports"]["readiness"]).parent
    assert (report_dir / "topic_quality.json").exists()
    assert not (report_dir / "investigation_confusion_matrix.png").exists()


@pytest.mark.parametrize("action", ["generate-investigation-candidates", "export-graph-evidence"])
def test_cli_without_run_id_uses_fitted_model_despite_newer_inspections_and_exports(
    topic_run, monkeypatch, capsys, action
):
    config, fitted, root = topic_run
    inspected, _ = run_corpus_action(config, root, "inspect-data")
    assert inspected["run_id"] != fitted["run_id"]
    monkeypatch.setattr("ci_hgnn_rgd.nlp.topics.cli._configuration", lambda _: (root, config))
    for _ in range(2):
        assert main(["topics", action]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["parent_run_id"] == fitted["run_id"]
        assert result["execution_status"] == "COMPLETED"
    assert latest_topic_run_id(config.output_root, model_name="lda") == fitted["run_id"]
    with pytest.raises(ValueError, match="No completed fitted topic model for bertopic"):
        latest_topic_run_id(config.output_root, model_name="bertopic")


@pytest.mark.parametrize("action", ["generate-investigation-candidates", "export-graph-evidence"])
def test_cli_without_fitted_model_explains_missing_prerequisite(tmp_path, monkeypatch, action):
    config, _ = stage03_fixture(tmp_path, rows=[])
    run_corpus_action(config, tmp_path, "inspect-data")
    monkeypatch.setattr("ci_hgnn_rgd.nlp.topics.cli._configuration", lambda _: (tmp_path, config))
    # Argument parsing succeeds, and the workflow returns a normal prerequisite
    # failure without manufacturing empty candidates or graph evidence.
    assert main(["topics", action]) == 1
    with pytest.raises(ValueError, match="No completed fitted topic model"):
        latest_topic_run_id(config.output_root)
    assert not config.model_root.exists()


def test_latest_model_checksum_failure_does_not_fall_back_to_older_model(topic_run, monkeypatch):
    from ci_hgnn_rgd.nlp.topics.cli import _run_command

    config, older, root = topic_run
    newer, _ = run_corpus_action(config, root, "train-baselines")
    catalog = Path(newer["run_dir"]) / "topic_catalog.jsonl"
    catalog.write_text(catalog.read_text() + "\n", encoding="utf-8")
    monkeypatch.setattr("ci_hgnn_rgd.nlp.topics.cli._configuration", lambda _: (root, config))
    with pytest.raises(ValueError, match="checksum mismatch"):
        _run_command(args(None, "generate-investigation-candidates"))
    # Explicit selection still permits a verified older model.
    result, status = _run_command(args(older["run_id"], "generate-investigation-candidates"))
    assert status == 0 and result["parent_run_id"] == older["run_id"]


def test_published_profile_points_to_existing_exclusions_file(topic_run):
    _, result, _ = topic_run
    run = Path(result["run_dir"])
    profile = json.loads((run / "input_data_profile.json").read_text())
    assert Path(profile["full_exclusions_path"]) == run / "excluded_records.jsonl"
    assert Path(profile["full_exclusions_path"]).is_file()


def test_failed_corpus_run_retains_valid_pending_exclusions_path(tmp_path):
    config, _ = stage03_fixture(tmp_path, rows=[])
    with pytest.raises(ValueError, match="Supply Stage 03"):
        run_corpus_action(config, tmp_path, "embed")
    profile_file = next(config.output_root.glob(".pending-*/input_data_profile.json"))
    profile = json.loads(profile_file.read_text())
    assert Path(profile["full_exclusions_path"]) == profile_file.parent / "excluded_records.jsonl"
    assert Path(profile["full_exclusions_path"]).is_file()


@pytest.mark.parametrize("action", ["embed", "train"])
def test_embedding_mmap_is_closed_before_windows_directory_publication(
    topic_run, monkeypatch, action
):
    import numpy as np

    from ci_hgnn_rgd.nlp.topics.artifacts import RunBuilder
    from ci_hgnn_rgd.nlp.topics.models import fit_topics

    config, _, root = topic_run
    tracked = []

    def mapped_embeddings(builder, _config, view, _run_id):
        matrix = np.lib.format.open_memmap(
            builder.pending / "paper_embeddings.npy",
            mode="w+",
            dtype="float32",
            shape=(len(view.papers), 3),
        )
        matrix[:] = 1
        matrix.flush()
        tracked.append(matrix)
        return matrix

    publish = RunBuilder.publish

    def windows_publication(builder, *args, **kwargs):
        if any(not item._mmap.closed for item in tracked):
            raise PermissionError("Windows directory publication blocked by open mmap")
        return publish(builder, *args, **kwargs)

    monkeypatch.setattr("ci_hgnn_rgd.nlp.topics.workflow._embeddings", mapped_embeddings)
    monkeypatch.setattr(RunBuilder, "publish", windows_publication)
    if action == "train":

        class SyntheticTopicFixture:
            def fit_transform(self, docs, embeddings):
                return [i % 2 for i in range(len(docs))], np.ones(len(docs))

            def get_topic(self, topic):
                return [("physics" if topic == 0 else "biology", 1.0), ("experiment", 0.3)]

        monkeypatch.setattr(
            "ci_hgnn_rgd.nlp.topics.workflow.fit_topics",
            lambda *a, **kw: fit_topics(*a, **kw, bertopic_model=SyntheticTopicFixture()),
        )
    result, status = run_corpus_action(config, root, action)
    assert status == 0
    assert (Path(result["run_dir"]) / "paper_embeddings.npy").exists()
    if action == "train":
        assert result["embedding_backend"] == config.embedding.backend


def test_failed_pending_manifest_cannot_be_resolved_as_published(topic_run):
    import shutil

    config, fitted, _ = topic_run
    source = Path(fitted["run_dir"])
    pending = source.parent / (".pending-" + source.name)
    shutil.copytree(source, pending)
    (pending / "failure.json").write_text('{"status":"FAILED"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not completely published"):
        resolve_run(config.output_root, pending.name)


def test_windows_transient_publication_lock_is_retried_without_partial_output(
    tmp_path, monkeypatch
):
    import os

    from ci_hgnn_rgd.nlp.topics.artifacts import _publish_directory

    pending, final = tmp_path / ".pending-model", tmp_path / "model"
    pending.mkdir()
    (pending / "weights.json").write_text("{}", encoding="utf-8")
    rename, attempts, waits = os.rename, [], []

    def scanner_lock(source, target):
        attempts.append((source, target))
        if len(attempts) < 3:
            assert not final.exists()
            error = PermissionError("synthetic Windows scanner lock")
            error.winerror = 32
            raise error
        rename(source, target)

    monkeypatch.setattr("ci_hgnn_rgd.nlp.topics.artifacts.os.rename", scanner_lock)
    monkeypatch.setattr("ci_hgnn_rgd.nlp.topics.artifacts.time.sleep", waits.append)
    _publish_directory(pending, final)
    assert len(attempts) == 3 and len(waits) == 2
    assert (final / "weights.json").is_file() and not pending.exists()


def test_persistent_windows_publication_error_remains_a_failure(tmp_path, monkeypatch):
    from ci_hgnn_rgd.nlp.topics.artifacts import _publish_directory

    attempts = []

    def denied(*args):
        attempts.append(args)
        error = PermissionError("synthetic persistent denial")
        error.winerror = 5
        raise error

    monkeypatch.setattr("ci_hgnn_rgd.nlp.topics.artifacts.os.rename", denied)
    monkeypatch.setattr("ci_hgnn_rgd.nlp.topics.artifacts.time.sleep", lambda _: None)
    with pytest.raises(PermissionError, match="persistent denial"):
        _publish_directory(tmp_path / "pending", tmp_path / "final")
    assert len(attempts) == 8


def test_temporal_report_keeps_unknown_availability_and_source_cutoff_findings(
    tmp_path, monkeypatch
):
    from stage05_helpers import source_row

    config, stage03 = stage03_fixture(
        tmp_path,
        [source_row(0, available_at=None, update_date="2022-01-01"), source_row(1)],
    )
    source_path = stage03 / "integration_manifest.json"
    source = json.loads(source_path.read_text())
    source["temporal"] = {"records_assessed": 2, "unknown_availability": 1, "future_updates": 1}
    source_path.write_text(json.dumps(source), encoding="utf-8")
    values = config.model_dump()
    values["temporal"]["cutoff"] = "2021-12-31T23:59:59Z"
    config = type(config).model_validate(values)
    original = prepare_corpus
    monkeypatch.setattr(
        "ci_hgnn_rgd.nlp.topics.workflow.prepare_corpus",
        lambda *a, **kw: original(*a, **kw, allow_synthetic_fixtures=True),
    )
    result, status = run_corpus_action(config, tmp_path, "audit-temporal")
    report = json.loads(Path(result["reports"]["topic_temporal_audit"]).read_text())
    assert status == 0
    assert report["papers_assessed"] == 2
    assert report["unknown_availability_papers"] == 1
    assert report["future_feature_papers_at_current_cutoff"] == 1
    assert report["source_stage03_audit"] == source["temporal"]
    assert report["strict_historical_claim"] is False


def test_temporal_report_without_cutoff_does_not_claim_zero_future_violations(topic_run):
    _, result, _ = topic_run
    report = json.loads(Path(result["reports"]["topic_temporal_audit"]).read_text())
    assert report["papers_assessed"] == 8
    assert report["future_feature_papers_at_current_cutoff"] is None
    assert report["current_cutoff_available"] is False


def test_stage04_contract_is_preserved_and_never_used_as_supervision(topic_run):
    from ci_hgnn_rgd.nlp.intent_contracts import CitationIntentEvidence
    from ci_hgnn_rgd.nlp.topics.contracts import utc_now

    config, result, root = topic_run
    prediction = CitationIntentEvidence(
        citation_context_id="synthetic:c1",
        source_dataset="synthetic",
        predicted_intent="result",
        prob_background=0.1,
        prob_method=0.1,
        prob_result=0.8,
        confidence=0.8,
        predictive_entropy=0.639,
        calibration_status="uncalibrated",
        model_id="fixture",
        model_version="stage04-fixture",
        prediction_timestamp=utc_now(),
        provenance=({"synthetic": True},),
    )
    source = root / "stage04.jsonl"
    source.write_text(prediction.model_dump_json() + "\n", encoding="utf-8")
    exported, _ = downstream_action(
        config,
        root,
        args(
            result["run_id"],
            "export-graph-evidence",
            evidence_run_id=None,
            citation_intent_input=source,
        ),
    )
    path = Path(exported["run_dir"])
    copied = list(iter_rows(path / "citation_intent_evidence.parquet"))[0]
    assert copied["predicted_intent"] == "result"
    assert copied["citing_paper_id"] is None and copied["cited_paper_id"] is None
    assert copied["provenance"] == [{"synthetic": True}]
    assert {row["evidence_label"] for row in iter_rows(path / "paper_topic_evidence.jsonl")} == {
        "UNKNOWN"
    }


def test_supervised_cli_path_stops_before_model_creation_without_annotations(topic_run):
    config, result, root = topic_run
    with pytest.raises(ValueError, match="NOT_TRAINED"):
        downstream_action(
            config, root, args(result["run_id"], "train-investigation-classifier", input=None)
        )
    assert not config.investigation_model_root.exists()


def test_frozen_assignment_and_explicit_seed_stability(topic_run):
    from stage05_helpers import source_row

    config, result, root = topic_run
    source = root / "new.jsonl"
    source.write_text(json.dumps(source_row(100)) + "\n", encoding="utf-8")
    assigned, _ = downstream_action(config, root, args(result["run_id"], "assign", input=source))
    assert assigned["topic_model_refitted"] is False
    evaluated, _ = downstream_action(
        config, root, args(result["run_id"], "evaluate", stability=True, compare_run_id=None)
    )
    stability = json.loads((Path(evaluated["run_dir"]) / "topic_stability.json").read_text())
    assert len(stability["repeated_fits"]) == 2


def test_cli_commands_registered_and_invalid_arguments_rejected(capsys):
    with pytest.raises(SystemExit) as result:
        main(["topics", "--help"])
    assert result.value.code == 0
    output = capsys.readouterr().out
    for command in (
        "embed",
        "train",
        "train-baselines",
        "assign",
        "audit-temporal",
        "export-annotation-sample",
        "train-investigation-classifier",
        "export-graph-evidence",
    ):
        assert command in output
    with pytest.raises(SystemExit) as result:
        main(["topics", "embed", "--model", "specter2", "--batch-size", "0"])
    assert result.value.code == 2
    assert (
        main(
            [
                "topics",
                "assign",
                "--project-root",
                str(PROJECT),
                "--run-id",
                "missing-run",
                "--input",
                "missing.jsonl",
                "--offline",
            ]
        )
        == 1
    )


def test_low_memory_cli_records_quantization_and_bounded_encoder_settings():
    from ci_hgnn_rgd.nlp.topics.cli import _configuration, register_parser

    parser = argparse.ArgumentParser()
    register_parser(parser.add_subparsers(dest="command"))
    parsed = parser.parse_args(
        ["topics", "embed", "--model", "specter2", "--low-memory", "--project-root", str(PROJECT)]
    )
    _, config = _configuration(parsed)
    assert config.embedding.quantization == "int8_dynamic"
    assert config.embedding.device == "cpu"
    assert config.embedding.batch_size == 1
    assert config.embedding.max_length == 256
    assert config.embedding.cpu_threads == 2
    assert config.embedding.min_available_ram_gib == 0.5


def test_old_facade_preserves_unknown_and_requires_nonempty_inputs(synthetic_corpus):
    from ci_hgnn_rgd.nlp.investigation_detection import InvestigationDetector
    from ci_hgnn_rgd.nlp.topic_discovery import TopicDiscoverer

    predictions = InvestigationDetector().detect(synthetic_corpus.papers, synthetic_corpus.topics)
    assert {p.relationship for p in predictions} == {"unknown"}
    assert all(p.confidence is None for p in predictions)
    with pytest.raises(ValueError):
        TopicDiscoverer().fit((), None)


def test_assignment_document_limit_is_enforced_without_silent_dropping(topic_run):
    from stage05_helpers import source_row

    config, result, root = topic_run
    values = config.model_dump()
    values["dataset"]["max_records"] = 1
    limited = type(config).model_validate(values)
    source = root / "too_many.jsonl"
    source.write_text("".join(json.dumps(source_row(i)) + "\n" for i in (90, 91)), encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds max_records"):
        downstream_action(limited, root, args(result["run_id"], "assign", input=source))


def test_rule_prediction_for_new_paper_requires_source_provenance(topic_run):
    from stage05_helpers import source_row

    config, result, root = topic_run
    original = Path(result["run_dir"])
    topic = next(iter_rows(original / "topic_catalog.jsonl"))
    source = root / "new_pair.jsonl"
    source.write_text(
        json.dumps({**source_row(100), "topic_id": topic["topic_id"]}) + "\n", encoding="utf-8"
    )
    output, status = downstream_action(
        config, root, args(result["run_id"], "predict-investigation", input=source)
    )
    record = next(iter_rows(Path(output["run_dir"]) / "paper_topic_evidence.jsonl"))
    assert record["paper_key"] == "synthetic:p100" and record["evidence_label"] == "UNKNOWN"
    assert record["source_provenance"][0]["record"] == 100
