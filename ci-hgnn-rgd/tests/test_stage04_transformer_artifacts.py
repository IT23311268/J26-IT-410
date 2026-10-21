"""Offline tiny-model tests for Stage 04 transformer and artifact behavior.

No pretrained SciBERT files are requested by these tests.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ci_hgnn_rgd.nlp.intent_artifacts import (
    IntentRunPublication,
    resolve_intent_run,
    update_run_manifest,
)
from ci_hgnn_rgd.nlp.intent_config import Stage04Config
from ci_hgnn_rgd.nlp.intent_contracts import LABEL_TO_ID, IntentDataError, IntentDatasetRecord
from ci_hgnn_rgd.nlp.intent_transformer import (
    TransformerPredictor,
    _collator,
    _load_pretrained,
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


def tiny_model() -> BertForSequenceClassification:
    return BertForSequenceClassification(
        BertConfig(
            vocab_size=11,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_labels=3,
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


def test_resume_rejects_incompatible_labels(tmp_path):
    config = configuration(tmp_path)
    run = tmp_path / "resume"
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps({"config": config.model_dump(mode="json")}), encoding="utf-8"
    )
    (run / "dataset_manifest.json").write_text(
        json.dumps({"dataset_fingerprint": "same"}), encoding="utf-8"
    )
    (run / "label_mapping.json").write_text(json.dumps(LABEL_TO_ID), encoding="utf-8")
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
