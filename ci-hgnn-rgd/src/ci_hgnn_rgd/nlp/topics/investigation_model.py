"""Independent three-class SciBERT pair classifier, gated on validated annotations."""

import json
import shutil
from pathlib import Path

from ...data.manifests import atomic_json
from ...pipeline import fingerprint
from ...runtime import detect_device, set_random_seeds
from ..intent_baselines import calculate_class_weights
from ..intent_metrics import evaluate_predictions
from ..intent_transformer import _available_system_memory_gib
from .contracts import LABEL_MAPPING, LABELS
from .investigation import validated_training_splits


def investigation_metrics(truth, predicted, probabilities=None):
    metrics = evaluate_predictions(truth, predicted, probabilities)
    previous = metrics["label_order"]
    metrics["per_class"] = {
        label: metrics["per_class"][old] for label, old in zip(LABELS, previous, strict=True)
    }
    metrics["label_order"] = list(LABELS)
    return metrics


def _encode(tokenizer, rows, max_length):
    return tokenizer(
        [row.title + " " + (row.abstract or "") for row in rows],
        [row.topic_definition for row in rows],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def _predict(model, tokenizer, rows, *, device, batch_size, max_length):
    import torch

    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            encoded = _encode(tokenizer, rows[start : start + batch_size], max_length)
            logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
            if logits.shape[-1] != 3:
                raise ValueError("investigation classifier must have three logits")
            probabilities.extend(logits.float().softmax(-1).cpu().tolist())
    return probabilities


def train_investigation(
    finals, run_dir: Path, config, *, model=None, tokenizer=None, allow_synthetic=False
):
    """Fit train only; dev selects checkpoints; test is never predicted in this function."""
    splits = validated_training_splits(
        finals,
        min_train=config.investigation.min_train_annotations,
        allow_synthetic=allow_synthetic,
    )
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if model is None:
        ram = _available_system_memory_gib()
        if ram is not None and ram < 3:
            raise RuntimeError(
                "investigation training blocked: at least 3 GiB available RAM required"
            )
    set_random_seeds(config.seed, include_optional=True)
    settings = config.investigation
    device = detect_device(config.embedding.device).device
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.embedding.scibert_model,
            revision=config.embedding.scibert_revision,
            num_labels=3,
            label2id=LABEL_MAPPING,
            id2label=dict(enumerate(LABELS)),
            local_files_only=config.offline,
            use_safetensors=True,
            trust_remote_code=False,
        )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            config.embedding.scibert_model,
            revision=config.embedding.scibert_revision,
            local_files_only=config.offline,
            trust_remote_code=False,
        )
    model.config.label2id = LABEL_MAPPING
    model.config.id2label = dict(enumerate(LABELS))
    model.to(device)
    train = splits["train"]
    weights = calculate_class_weights(
        [LABEL_MAPPING[row.label] for row in train], strategy=settings.class_weighting
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([weights[i] for i in range(3)], device=device) if weights else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    history, best, best_epoch = [], -1.0, 0
    for epoch in range(1, settings.epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        for start in range(0, len(order), settings.batch_size):
            rows = [train[index] for index in order[start : start + settings.batch_size]]
            inputs = _encode(tokenizer, rows, settings.max_length)
            labels = torch.tensor([LABEL_MAPPING[row.label] for row in rows], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                model(**{key: value.to(device) for key, value in inputs.items()}).logits, labels
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        probabilities = _predict(
            model,
            tokenizer,
            splits["dev"],
            device=device,
            batch_size=settings.batch_size,
            max_length=settings.max_length,
        )
        metrics = investigation_metrics(
            [LABEL_MAPPING[row.label] for row in splits["dev"]],
            [max(range(3), key=row.__getitem__) for row in probabilities],
            probabilities,
        )
        history.append({"epoch": epoch, "validation": metrics})
        if metrics["macro_f1"] > best:
            best, best_epoch = metrics["macro_f1"], epoch
            checkpoint = run_dir / "checkpoints" / str(epoch)
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint / "tokenizer")
            atomic_json(run_dir / "validation_metrics.json", metrics)
        atomic_json(run_dir / "training_history.json", history)
        if epoch - best_epoch >= 2:
            break
    shutil.copytree(run_dir / "checkpoints" / str(best_epoch), run_dir / "model")
    atomic_json(run_dir / "label_mapping.json", LABEL_MAPPING)
    atomic_json(
        run_dir / "annotation_manifest.json",
        {
            "annotation_sha256": fingerprint(finals),
            "split_counts": {key: len(rows) for key, rows in splits.items()},
            "test_evaluated": False,
            "best_epoch": best_epoch,
            "synthetic_fixture": allow_synthetic,
            "model_id": config.embedding.scibert_model,
            "revision": config.embedding.scibert_revision,
            "max_length": settings.max_length,
        },
    )
    return {
        "status": "TRAINED_SYNTHETIC_FIXTURE" if allow_synthetic else "TRAINED",
        "best_epoch": best_epoch,
        "validation_macro_f1": best,
        "test_evaluated": False,
    }


def predict_investigation_model(run_dir: Path, rows, *, device="auto", batch_size=4):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    manifest = json.loads((run_dir / "annotation_manifest.json").read_text(encoding="utf-8"))
    mapping = json.loads((run_dir / "label_mapping.json").read_text(encoding="utf-8"))
    if mapping != LABEL_MAPPING:
        raise ValueError("investigation checkpoint label mapping mismatch")
    selected = detect_device(device).device
    model = AutoModelForSequenceClassification.from_pretrained(
        run_dir / "model", local_files_only=True, use_safetensors=True, trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        run_dir / "model/tokenizer", local_files_only=True, trust_remote_code=False
    )
    model.to(selected)
    return _predict(
        model,
        tokenizer,
        rows,
        device=selected,
        batch_size=batch_size,
        max_length=manifest["max_length"],
    )
