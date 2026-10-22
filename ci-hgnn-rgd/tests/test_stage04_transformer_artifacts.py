"""Offline tiny-model tests for Stage 04 transformer and artifact behavior.

No pretrained SciBERT files are requested by these tests.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ci_hgnn_rgd.nlp.intent_artifacts import (
    IntentArtifactError,
    IntentRunPublication,
    resolve_intent_run,
    update_run_manifest,
)
from ci_hgnn_rgd.nlp.intent_config import Stage04Config
from ci_hgnn_rgd.nlp.intent_contracts import LABEL_TO_ID, IntentDataError, IntentDatasetRecord
from ci_hgnn_rgd.nlp.intent_transformer import (
    TransformerPredictor,
    TransformerResourceError,
    _collator,
    _load_pretrained,
    _make_optimizer,
    _optimizer_state_bytes,
    effective_batch_settings,
    resource_preflight,
    train_scibert,
    validate_resume_compatibility,
)

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast  # noqa: E402


def record(split: str, label: str, identifier: str) -> IntentDatasetRecord:
    text = f"{label} fixture alpha beta {identifier}"
    return IntentDatasetRecord(
        citation_context_id=f"synthetic:{identifier}",
        source_dataset="synthetic-fixture",
        source_record_id=identifier,
        dataset_version="fixture-v1",
        source_split=split,
        original_text=text,
        model_text=text,
        original_label=label,
        normalized_label=label,
        label_id=LABEL_TO_ID[label],
        identity_resolution_status="UNRESOLVED",
    )


def tokenizer(tmp_path: Path) -> BertTokenizerFast:
    vocabulary = tmp_path / "vocab.txt"
    vocabulary.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nbackground\nmethod\nresult\nfixture\nalpha\nbeta\n",
        encoding="utf-8",
    )
    return BertTokenizerFast(vocab_file=str(vocabulary), do_lower_case=True)


def configuration(tmp_path: Path) -> Stage04Config:
    return Stage04Config.model_validate(
        {
            "model": {"max_length": 16},
            "training": {
                "mode": "quick",
                "epochs": 1,
                "train_batch_size": 2,
                "eval_batch_size": 2,
                "device": "cpu",
                "precision": "fp32",
                "early_stopping_patience": 0,
                "min_available_ram_gib": 0,
                "seed": 17,
            },
            "outputs": {
                "model_root": tmp_path / "models",
                "report_root": tmp_path / "reports",
                "work_root": tmp_path / "work",
            },
        }
    )


def tiny_model(*, dropout: float = 0.1) -> BertForSequenceClassification:
    return BertForSequenceClassification(
        BertConfig(
            vocab_size=11,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_labels=3,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
    )


def test_dynamic_padding_three_logits_and_cross_entropy(tmp_path):
    local_tokenizer = tokenizer(tmp_path)
    collate = _collator(local_tokenizer, 16, torch)
    batch = collate(
        [
            {"text": "background fixture", "label": 0, "context_id": "one"},
            {"text": "method fixture alpha beta", "label": 1, "context_id": "two"},
        ]
    )
    assert batch["inputs"]["input_ids"].shape[0] == 2
    assert batch["inputs"]["input_ids"].shape[1] > 2
    assert batch["inputs"]["attention_mask"][0].sum() < batch["inputs"]["attention_mask"][1].sum()
    model = tiny_model()
    logits = model(**batch["inputs"]).logits
    assert logits.shape == (2, 3)
    assert torch.isfinite(torch.nn.functional.cross_entropy(logits, batch["labels"]))


def test_offline_pretrained_loader_requests_local_files_only(tmp_path):
    config = configuration(tmp_path)
    values = config.model_dump(mode="python")
    values["offline"] = True
    config = Stage04Config.model_validate(values)
    calls = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(("tokenizer", args, kwargs))
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(("model", args, kwargs))
            return cls()

    with patch(
        "ci_hgnn_rgd.nlp.intent_transformer._require_transformer_stack",
        return_value=(torch, FakeTokenizer, FakeModel, object()),
    ):
        _load_pretrained(config)
    assert all(call[2]["local_files_only"] is True for call in calls)
    assert all(call[2]["trust_remote_code"] is False for call in calls)


def test_tiny_local_training_checkpoint_and_safe_reload(tmp_path):
    config = configuration(tmp_path)
    local_tokenizer = tokenizer(tmp_path)
    train = tuple(
        record("train", label, f"train-{label}-{index}")
        for label in ("background", "method", "result")
        for index in range(2)
    )
    dev = tuple(
        record("dev", label, f"dev-{label}") for label in ("background", "method", "result")
    )
    run_dir = tmp_path / "tiny-run"
    dataset_manifest = {"dataset_fingerprint": "synthetic-tiny", "label_mapping": LABEL_TO_ID}
    result = train_scibert(
        train,
        dev,
        run_dir,
        config,
        dataset_manifest,
        model=tiny_model(),
        tokenizer=local_tokenizer,
        enforce_resource_preflight=False,
    )
    assert result.best_epoch == 1
    assert len(result.training_history) == 1
    assert (result.model_dir / "model.safetensors").is_file()
    assert not list(result.model_dir.glob("*.bin"))
    predictor = TransformerPredictor.load(run_dir, device="cpu")
    output = predictor.predict([dev[0].model_text, dev[1].model_text], batch_size=1)
    assert len(output["logits"]) == 2
    assert all(len(row) == 3 and pytest.approx(sum(row)) == 1 for row in output["probabilities"])


def test_low_memory_training_updates_encoder_and_head_with_smaller_optimizer_state(tmp_path):
    from copy import deepcopy

    config = configuration(tmp_path).model_dump(mode="python")
    config["training"].update(optimizer="adafactor", gradient_checkpointing=True, cpu_threads=1)
    config = Stage04Config.model_validate(config)
    model = tiny_model()
    before_head = model.classifier.weight.detach().clone()
    before_encoder = model.bert.embeddings.word_embeddings.weight.detach().clone()
    train = tuple(
        record("train", label, f"train-{label}-{i}") for label in LABEL_TO_ID for i in range(2)
    )
    dev = tuple(record("dev", label, f"dev-{label}") for label in LABEL_TO_ID)
    result = train_scibert(
        train,
        dev,
        tmp_path / "memory-run",
        config,
        {"dataset_fingerprint": "synthetic-memory"},
        model=model,
        tokenizer=tokenizer(tmp_path),
        enforce_resource_preflight=False,
    )
    assert model.is_gradient_checkpointing
    assert not torch.equal(before_head, model.classifier.weight)
    assert not torch.equal(before_encoder, model.bert.embeddings.word_embeddings.weight)
    progress = json.loads((tmp_path / "memory-run/training_progress.json").read_text())
    assert progress["status"] == "TRAINED" and progress["examples_seen"] == 6
    assert progress["test_split_used"] is False
    adam_model = deepcopy(model)
    adam = _make_optimizer(torch, adam_model, configuration(tmp_path))
    for parameter in adam_model.parameters():
        parameter.grad = torch.ones_like(parameter)
    adam.step()
    assert 0 < result.resource_preflight["optimizer_state_bytes"] < _optimizer_state_bytes(adam) / 2


def resume_fixture(tmp_path):
    config = configuration(tmp_path)
    run = tmp_path / "resume"
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps({"run_type": "scibert", "config": config.model_dump(mode="json")}),
        encoding="utf-8",
    )
    (run / "dataset_manifest.json").write_text(
        json.dumps({"dataset_fingerprint": "same"}), encoding="utf-8"
    )
    (run / "label_mapping.json").write_text(json.dumps(LABEL_TO_ID), encoding="utf-8")
    (run / "model_metadata.json").write_text(
        json.dumps(
            {
                "tokenizer_id": config.model.name,
                "tokenizer_revision": config.model.revision,
                "max_length": config.model.max_length,
                "num_labels": 3,
                "label_order": list(LABEL_TO_ID),
            }
        ),
        encoding="utf-8",
    )
    return config, run


def test_resume_rejects_incompatible_labels(tmp_path):
    config, run = resume_fixture(tmp_path)
    validate_resume_compatibility(
        run, dataset_manifest={"dataset_fingerprint": "same"}, config=config
    )
    (run / "label_mapping.json").write_text(
        json.dumps({"background": 1, "method": 0, "result": 2}), encoding="utf-8"
    )
    with pytest.raises(IntentDataError, match="label mapping"):
        validate_resume_compatibility(
            run, dataset_manifest={"dataset_fingerprint": "same"}, config=config
        )


def test_resume_cache_location_is_not_part_of_model_identity(tmp_path):
    config, run = resume_fixture(tmp_path)
    path = run / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["model"]["cache_dir"] = str(tmp_path / "previous-cache")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    validate_resume_compatibility(
        run, dataset_manifest={"dataset_fingerprint": "same"}, config=config
    )


def test_atomic_model_run_publication_rejects_path_escape_and_tracks_updates(tmp_path):
    root = tmp_path / "models"
    publication = IntentRunPublication(root, config={"synthetic": True}, project_root=tmp_path)
    publication.write_json("metadata.json", {"synthetic": True})
    final = publication.publish({"run_type": "synthetic-test"})
    assert resolve_intent_run(root, final.name) == final
    (final / "evaluation.json").write_text("{}", encoding="utf-8")
    manifest = update_run_manifest(final, {"kind": "synthetic_test_update"})
    assert manifest["updates"][-1]["kind"] == "synthetic_test_update"
    assert "evaluation.json" in manifest["artifact_checksums"]


def test_model_publication_retries_transient_windows_directory_lock(tmp_path):
    publication = IntentRunPublication(
        tmp_path / "models", config={"synthetic": True}, project_root=tmp_path
    )
    rename = os.rename
    attempts = []

    def locked_once(source, target):
        attempts.append((source, target))
        if len(attempts) == 1:
            error = PermissionError("synthetic transient directory lock")
            error.winerror = 32
            raise error
        rename(source, target)

    with (
        patch("ci_hgnn_rgd.nlp.intent_artifacts.os.rename", side_effect=locked_once),
        patch("ci_hgnn_rgd.nlp.intent_artifacts.time.sleep") as sleep,
    ):
        final = publication.publish({"run_type": "synthetic-test"})
    assert len(attempts) == 2 and sleep.call_count == 1
    assert resolve_intent_run(publication.root, final.name) == final


def test_pending_model_with_complete_manifest_is_not_resolvable(tmp_path):
    publication = IntentRunPublication(
        tmp_path / "models", config={"synthetic": True}, project_root=tmp_path
    )
    publication.write_json("run_manifest.json", {"stage": 4, "status": "COMPLETE"})
    with pytest.raises(IntentArtifactError, match="not completely published"):
        resolve_intent_run(publication.root, publication.pending.name)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("test_evaluated", True, "held-out test"),
        ("run_type", "baselines", "SciBERT"),
    ],
)
def test_resume_rejects_observed_test_or_wrong_architecture(tmp_path, field, value, message):
    config, run = resume_fixture(tmp_path)
    path = run / "run_manifest.json"
    document = json.loads(path.read_text())
    document[field] = value
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(IntentDataError, match=message):
        validate_resume_compatibility(
            run, dataset_manifest={"dataset_fingerprint": "same"}, config=config
        )


def test_resume_rejects_changed_data_or_tokenizer(tmp_path):
    config, run = resume_fixture(tmp_path)
    with pytest.raises(IntentDataError, match="dataset"):
        validate_resume_compatibility(
            run, dataset_manifest={"dataset_fingerprint": "different"}, config=config
        )
    path = run / "model_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["tokenizer_revision"] = "incompatible"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(IntentDataError, match="tokenizer"):
        validate_resume_compatibility(
            run, dataset_manifest={"dataset_fingerprint": "same"}, config=config
        )


def test_resource_adaptation_preserves_effective_batch_size():
    config = Stage04Config()
    cpu = effective_batch_settings(config, {"selected_device": "cpu", "gpu": None})
    assert cpu["train_batch_size"] == 2
    assert cpu["effective_train_batch_size"] == 16
    gpu = effective_batch_settings(
        config,
        {
            "selected_device": "cuda",
            "gpu": {
                "available_memory_gib": 4,
            },
        },
    )
    assert gpu["train_batch_size"] == 2
    assert gpu["gradient_accumulation_steps"] == 8


@pytest.mark.parametrize("available_commit,ready", [(1.0, False), (5.0, True)])
def test_low_memory_full_preflight_requires_commit_headroom(available_commit, ready):
    config = Stage04Config.model_validate(
        {
            "training": {
                "device": "cpu",
                "min_available_ram_gib": 0.25,
                "min_available_commit_gib": 3.0,
            }
        }
    )
    with (
        patch("ci_hgnn_rgd.nlp.intent_transformer._available_system_memory_gib", return_value=0.35),
        patch(
            "ci_hgnn_rgd.nlp.intent_transformer._available_commit_memory_gib",
            return_value=available_commit,
        ),
    ):
        report = resource_preflight(config)
    assert report["ready"] is ready
    assert ("insufficient_available_commit_memory" in report["blockers"]) is not ready


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resource_failure_explains_memory_and_preserves_report_without_loading(tmp_path, device):
    """A blocked real run must explain the remedy without allocating pretrained weights."""
    config = Stage04Config()
    train = tuple(record("train", label, label) for label in LABEL_TO_ID)
    dev = tuple(record("dev", label, label) for label in LABEL_TO_ID)
    is_cpu = device == "cpu"
    preflight = {
        "selected_device": device,
        "available_system_memory_gib": 0.75 if is_cpu else 4.0,
        "minimum_available_ram_gib": 3.0,
        "gpu": None if is_cpu else {"available_memory_gib": 1.25, "minimum_available_gpu_gib": 2.0},
        "blockers": [
            "insufficient_available_system_memory"
            if is_cpu
            else "insufficient_available_gpu_memory"
        ],
        "ready": False,
    }
    run_dir = tmp_path / "blocked-run"
    with (
        patch("ci_hgnn_rgd.nlp.intent_transformer.resource_preflight", return_value=preflight),
        patch("ci_hgnn_rgd.nlp.intent_transformer._load_pretrained") as load,
        pytest.raises(TransformerResourceError) as failure,
    ):
        train_scibert(train, dev, run_dir, config, {"dataset_fingerprint": "synthetic-blocked"})
    message = str(failure.value)
    assert ("0.75 GiB" if is_cpu else "1.25 GiB") in message
    assert ("3.00 GiB" if is_cpu else "2.00 GiB") in message
    assert f"selected device: {device}" in message
    assert "then rerun" in message
    report = run_dir / "resource_preflight.json"
    assert str(report) in message
    assert json.loads(report.read_text()) == preflight
    load.assert_not_called()
    assert not (run_dir / "checkpoints").exists()


@pytest.mark.parametrize("weighting", ["none", "inverse"])
def test_partial_accumulation_matches_large_batch(tmp_path, weighting):
    """An uneven last microbatch and unequal weights must match full-batch gradients."""
    from copy import deepcopy

    torch.set_num_threads(1)
    local_tokenizer = tokenizer(tmp_path)
    base = configuration(tmp_path).model_dump(mode="python")
    base["training"].update(
        {
            "auto_adjust_batch_size": False,
            "class_weighting": weighting,
            "train_batch_size": 5,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.001,
        }
    )
    train = tuple(
        record("train", label, str(i))
        for i, label in enumerate(["background", "background", "background", "method", "result"])
    )
    dev = tuple(record("dev", label, label) for label in LABEL_TO_ID)
    model = tiny_model(dropout=0.0)
    # Dropout changes random draws with batch shape, so disable it for this
    # numerical equivalence test. Real training retains configured dropout.
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0
    manifest = {"dataset_fingerprint": "synthetic-accumulation"}
    full = train_scibert(
        train,
        dev,
        tmp_path / "full",
        Stage04Config.model_validate(base),
        manifest,
        model=deepcopy(model),
        tokenizer=local_tokenizer,
        enforce_resource_preflight=False,
    )
    base["training"].update({"train_batch_size": 2, "gradient_accumulation_steps": 4})
    partial = train_scibert(
        train,
        dev,
        tmp_path / "partial",
        Stage04Config.model_validate(base),
        manifest,
        model=deepcopy(model),
        tokenizer=local_tokenizer,
        enforce_resource_preflight=False,
    )
    left = BertForSequenceClassification.from_pretrained(full.model_dir, use_safetensors=True)
    right = BertForSequenceClassification.from_pretrained(partial.model_dir, use_safetensors=True)
    for a, b in zip(left.parameters(), right.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)
    assert full.training_history[0]["train_loss"] == pytest.approx(
        partial.training_history[0]["train_loss"], rel=1e-5
    )


def test_collator_records_truncation(tmp_path):
    batch = _collator(tokenizer(tmp_path), 16, torch)(
        [
            {"text": "alpha " * 30, "label": 0, "context_id": "long"},
            {"text": "method fixture", "label": 1, "context_id": "short"},
        ]
    )
    assert batch["truncated"] == [True, False]


def test_best_checkpoint_and_early_stopping_use_validation_only(tmp_path):
    config = configuration(tmp_path).model_dump(mode="python")
    config["training"].update({"epochs": 8, "quick_epochs": 8, "early_stopping_patience": 2})
    train = tuple(record("train", label, label) for label in LABEL_TO_ID)
    dev = tuple(record("dev", label, label) for label in LABEL_TO_ID)
    scores = iter([0.2, 0.5, 0.1, 0.1])

    def validation(*args, **kwargs):
        return {"macro_f1": next(scores)}, []

    with patch("ci_hgnn_rgd.nlp.intent_transformer._evaluate", side_effect=validation):
        result = train_scibert(
            train,
            dev,
            tmp_path / "selection",
            Stage04Config.model_validate(config),
            {"dataset_fingerprint": "synthetic-selection"},
            model=tiny_model(),
            tokenizer=tokenizer(tmp_path),
            enforce_resource_preflight=False,
        )
    assert result.best_epoch == 2
    assert len(result.training_history) == 4
    metadata = json.loads((result.model_dir / "checkpoint_metadata.json").read_text())
    assert metadata["epoch"] == 2
    history = json.loads((tmp_path / "selection" / "training_history.json").read_text())
    assert history["best_epoch"] == 2 and len(history["epochs"]) == 4


def test_optional_seeds_reproduce_python_numpy_and_torch():
    import random

    import numpy as np

    from ci_hgnn_rgd.runtime import set_random_seeds

    set_random_seeds(93, include_optional=True)
    first = (random.random(), np.random.random(), torch.rand(3))
    set_random_seeds(93, include_optional=True)
    second = (random.random(), np.random.random(), torch.rand(3))
    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2], atol=0, rtol=0)
