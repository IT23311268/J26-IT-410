"""Lazy SciBERT fine-tuning and safe checkpoint loading for Stage 04."""

from __future__ import annotations

import ctypes
import json
import logging
import math
import os
import shutil
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data.manifests import atomic_json
from ..pipeline import fingerprint
from ..runtime import detect_device, set_random_seeds
from .intent_baselines import calculate_class_weights
from .intent_calibration import TemperatureScaling, apply_temperature_to_probabilities
from .intent_config import Stage04Config
from .intent_contracts import INTENT_LABELS, IntentDataError, IntentDatasetRecord
from .intent_metrics import evaluate_predictions

LOGGER = logging.getLogger(__name__)


class TransformerDependencyError(RuntimeError):
    """Raised for an explicitly requested transformer action without its optional stack."""


class TransformerResourceError(RuntimeError):
    """Raised before training when configured resource floors are not available."""


def _require_transformer_stack():
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as error:
        raise TransformerDependencyError(
            "SciBERT training requires the optional Stage 04 torch and transformers dependencies"
        ) from error
    try:
        import safetensors  # noqa: F401
    except ImportError as error:
        raise TransformerDependencyError("SciBERT checkpoints require safetensors") from error
    return torch, AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup


def _windows_memory_status():
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status
    return None


def _available_system_memory_gib() -> float | None:
    """Best-effort available RAM without adding a runtime psutil dependency."""
    if os.name == "nt":
        status = _windows_memory_status()
        return status.ullAvailPhys / 1024**3 if status is not None else None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        return (page_size * available_pages) / 1024**3
    except (AttributeError, OSError, ValueError):
        return None


def _available_commit_memory_gib() -> float | None:
    status = _windows_memory_status()
    return status.ullAvailPageFile / 1024**3 if status is not None else None


def resource_preflight(config: Stage04Config) -> dict:
    """Report actual resource availability and fail only configured full-training floors."""
    torch, *_ = _require_transformer_stack()
    device_info = detect_device(config.training.device)
    available_ram = _available_system_memory_gib()
    available_commit = _available_commit_memory_gib()
    report: dict[str, Any] = {
        "requested_device": config.training.device,
        "selected_device": device_info.device,
        "torch_installed": device_info.torch_installed,
        "available_system_memory_gib": available_ram,
        "minimum_available_ram_gib": config.training.min_available_ram_gib,
        "available_commit_memory_gib": available_commit,
        "minimum_available_commit_gib": config.training.min_available_commit_gib,
        "gpu": None,
        "blockers": [],
    }
    if config.training.mode == "full" and available_ram is not None:
        if available_ram < config.training.min_available_ram_gib:
            report["blockers"].append("insufficient_available_system_memory")
    if (
        config.training.mode == "full"
        and available_commit is not None
        and available_commit < config.training.min_available_commit_gib
    ):
        report["blockers"].append("insufficient_available_commit_memory")
    if device_info.device == "cuda":
        free, total = torch.cuda.mem_get_info()
        gpu = {
            "name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "available_memory_gib": free / 1024**3,
            "total_memory_gib": total / 1024**3,
            "minimum_available_gpu_gib": config.training.min_available_gpu_gib,
        }
        report["gpu"] = gpu
        if (
            config.training.mode == "full"
            and gpu["available_memory_gib"] < config.training.min_available_gpu_gib
        ):
            report["blockers"].append("insufficient_available_gpu_memory")
    report["ready"] = not report["blockers"]
    return report


def _precision(config: Stage04Config, device: str):
    requested = config.training.precision
    if requested == "auto":
        return "fp16" if device == "cuda" else "fp32"
    if requested == "fp16" and device != "cuda":
        raise TransformerResourceError(
            "fp16 precision requires CUDA in this Stage 04 implementation"
        )
    if requested == "bf16" and device not in {"cuda", "cpu"}:
        raise TransformerResourceError("bf16 precision is unavailable on the selected device")
    return requested


def effective_batch_settings(config: Stage04Config, preflight: dict) -> dict:
    """Conservatively reduce microbatches while preserving the requested effective batch."""
    requested = config.training.train_batch_size
    cap = requested
    if config.training.auto_adjust_batch_size:
        if preflight["selected_device"] == "cpu":
            cap = config.training.cpu_train_batch_size_cap
        elif preflight.get("gpu"):
            # A starting heuristic, not a memory guarantee; record its decision.
            free = preflight["gpu"]["available_memory_gib"]
            cap = 2 if free < 6 else 4 if free < 10 else 8 if free < 16 else requested
    microbatch = max(value for value in range(1, min(cap, requested) + 1) if requested % value == 0)
    accumulation = config.training.gradient_accumulation_steps * (requested // microbatch)
    return {
        "requested_train_batch_size": requested,
        "train_batch_size": microbatch,
        "eval_batch_size": min(config.training.eval_batch_size, microbatch * 2)
        if microbatch < requested
        else config.training.eval_batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_train_batch_size": microbatch * accumulation,
        "policy": "conservative_device_memory_heuristic"
        if microbatch < requested
        else "configured",
    }


def _truncation_flags(tokenizer, texts: Sequence[str], max_length: int) -> list[bool]:
    flags = []
    for text in texts:
        encoded = tokenizer(text, truncation=False, add_special_tokens=True)
        flags.append(len(encoded["input_ids"]) > max_length)
    return flags


def _seed_worker(worker_id: int) -> None:
    # Torch has already assigned each worker a deterministic seed.
    import random

    import torch

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:
        pass


class _ContextDataset:
    def __init__(self, records: Sequence[IntentDatasetRecord]):
        self.records = tuple(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return {
            "text": record.model_text,
            "label": record.label_id,
            "context_id": record.citation_context_id,
        }


@dataclass
class _ContextCollator:
    """Top-level callable so DataLoader workers can use Windows spawn."""

    tokenizer: Any
    max_length: int

    def __call__(self, rows):
        import torch

        texts = [row["text"] for row in rows]
        try:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_length=True,
            )
        except TypeError:  # tiny injected test tokenizers need only the standard inputs.
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        lengths = encoded.pop("length", None)
        return {
            "inputs": dict(encoded),
            "labels": torch.tensor([row["label"] for row in rows], dtype=torch.long),
            "context_ids": [row["context_id"] for row in rows],
            "lengths": lengths,
            "truncated": _truncation_flags(self.tokenizer, texts, self.max_length),
        }


def _collator(tokenizer, max_length: int, torch):
    return _ContextCollator(tokenizer, max_length)


def _move_inputs(inputs: dict, device):
    return {name: value.to(device) for name, value in inputs.items()}


def _autocast(torch, device: str, precision: str):
    if precision == "fp16":
        return torch.autocast(device_type=device, dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type=device, dtype=torch.bfloat16)
    return nullcontext()


def _evaluate(
    model, loader, *, torch, device, precision, class_weights=None
) -> tuple[dict, list[list[float]]]:
    model.eval()
    labels, predictions, logits = [], [], []
    loss_sum = 0.0
    loss_denominator = 0.0
    truncated = 0
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="sum")
    with torch.inference_mode():
        for batch in loader:
            targets = batch["labels"].to(device)
            inputs = _move_inputs(batch["inputs"], device)
            with _autocast(torch, str(device).split(":", 1)[0], precision):
                output = model(**inputs)
                batch_logits = output.logits
                loss = criterion(batch_logits, targets)
            values = batch_logits.float().cpu().tolist()
            logits.extend([[float(item) for item in row] for row in values])
            labels.extend(int(value) for value in targets.cpu().tolist())
            truncated += sum(batch["truncated"])
            predictions.extend(int(value) for value in batch_logits.argmax(dim=-1).cpu().tolist())
            loss_sum += float(loss.detach().cpu())
            loss_denominator += (
                float(class_weights[targets].sum().detach().cpu())
                if class_weights is not None
                else len(targets)
            )
    metrics = evaluate_predictions(labels, predictions)
    metrics["loss"] = loss_sum / loss_denominator if loss_denominator else None
    metrics["truncated_contexts"] = truncated
    return metrics, logits


def _load_pretrained(config: Stage04Config, *, resume_run: Path | None = None):
    _, AutoTokenizer, AutoModelForSequenceClassification, _ = _require_transformer_stack()
    if resume_run is not None:
        model_dir = Path(resume_run) / "model"
        tokenizer_dir = Path(resume_run) / "tokenizer"
        return (
            AutoTokenizer.from_pretrained(
                tokenizer_dir, local_files_only=True, trust_remote_code=False
            ),
            AutoModelForSequenceClassification.from_pretrained(
                model_dir, local_files_only=True, use_safetensors=True, trust_remote_code=False
            ),
        )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        use_fast=True,
        local_files_only=config.offline,
        trust_remote_code=False,
        cache_dir=str(config.model.cache_dir),
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        num_labels=3,
        id2label={index: label for index, label in enumerate(INTENT_LABELS)},
        label2id={label: index for index, label in enumerate(INTENT_LABELS)},
        local_files_only=config.offline,
        trust_remote_code=False,
        use_safetensors=True,
        cache_dir=str(config.model.cache_dir),
    )
    return tokenizer, model


def validate_resume_compatibility(
    resume_run: Path,
    *,
    dataset_manifest: dict,
    config: Stage04Config,
    training_context_ids_sha256: str | None = None,
) -> None:
    """Validate a model-only warm start; this does not restore training state.

    Configuration and data compatibility are strict. A run whose final test has
    been observed cannot become a new model-selection starting point.
    """
    resume_run = Path(resume_run)
    manifest_path = resume_run / "run_manifest.json"
    dataset_path = resume_run / "dataset_manifest.json"
    labels_path = resume_run / "label_mapping.json"
    tokenizer_meta = resume_run / "model_metadata.json"
    if not all(
        path.is_file() for path in (manifest_path, dataset_path, labels_path, tokenizer_meta)
    ):
        raise IntentDataError("resume run lacks required Stage 04 compatibility artifacts")
    prior = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior_data = json.loads(dataset_path.read_text(encoding="utf-8"))
    prior_labels = json.loads(labels_path.read_text(encoding="utf-8"))
    details = json.loads(tokenizer_meta.read_text(encoding="utf-8"))
    if prior.get("run_type") != "scibert":
        raise IntentDataError("resume requires a SciBERT training run")
    test_dir = resume_run / "evaluations" / "test"
    test_observed = prior.get("test_evaluated") is True or any(
        update.get("test_split_used") is True or update.get("split") == "test"
        for update in prior.get("updates", [])
    )
    if test_observed or (test_dir.exists() and any(test_dir.rglob("*.json"))):
        raise IntentDataError("cannot resume a run whose held-out test has already been evaluated")
    prior_model = prior.get("config", {}).get("model", {})
    if {key: value for key, value in prior_model.items() if key != "cache_dir"} != (
        config.model.model_dump(mode="json", exclude={"cache_dir"})
    ):
        raise IntentDataError(
            "resume model architecture, revision, or sequence length is incompatible"
        )
    if prior_labels != {label: index for index, label in enumerate(INTENT_LABELS)}:
        raise IntentDataError("resume label mapping is incompatible")
    if (
        not dataset_manifest.get("dataset_fingerprint")
        or prior_data.get("dataset_fingerprint") != dataset_manifest["dataset_fingerprint"]
    ):
        raise IntentDataError("resume dataset or official split manifest is incompatible")
    if prior.get("config", {}).get("training") != config.training.model_dump(mode="json"):
        raise IntentDataError("resume training configuration is incompatible")
    if (
        training_context_ids_sha256 is not None
        and prior.get("training_view", {}).get("training_context_ids_sha256")
        != training_context_ids_sha256
    ):
        raise IntentDataError("resume training record selection or ordering is incompatible")
    if (
        details.get("tokenizer_id") != config.model.name
        or details.get("tokenizer_revision") != config.model.revision
        or details.get("max_length") != config.model.max_length
    ):
        raise IntentDataError(
            "resume tokenizer identity, revision, or sequence length is incompatible"
        )
    if details.get("num_labels") != 3 or details.get("label_order") != list(INTENT_LABELS):
        raise IntentDataError("resume classifier label order is incompatible")


@dataclass(frozen=True)
class TransformerTrainingResult:
    model_dir: Path
    tokenizer_dir: Path
    best_epoch: int
    validation_metrics: dict
    training_history: tuple[dict, ...]
    resource_preflight: dict


def _make_optimizer(torch, model, config):
    if config.training.optimizer == "adafactor":
        from transformers import Adafactor

        return Adafactor(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )


def _optimizer_state_bytes(optimizer) -> int:
    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if hasattr(value, "numel")
    )


def train_scibert(
    train_records: Sequence[IntentDatasetRecord],
    dev_records: Sequence[IntentDatasetRecord],
    run_dir: Path,
    config: Stage04Config,
    dataset_manifest: dict,
    *,
    model=None,
    tokenizer=None,
    resume_run: Path | None = None,
    enforce_resource_preflight: bool = True,
) -> TransformerTrainingResult:
    """Fine-tune a three-logit classifier, selecting a checkpoint only on dev macro F1."""
    if not train_records or not dev_records:
        raise IntentDataError("SciBERT training requires nonempty official train and dev splits")
    if any(row.source_split != "train" for row in train_records):
        raise IntentDataError("SciBERT fitting accepts only official train records")
    if any(row.source_split != "dev" for row in dev_records):
        raise IntentDataError("SciBERT validation accepts only official dev records")
    if set(row.label_id for row in train_records) != {0, 1, 2}:
        raise IntentDataError("SciBERT training requires all three SciCite labels")
    torch, _, _, get_linear_schedule_with_warmup = _require_transformer_stack()
    preflight = resource_preflight(config)
    if enforce_resource_preflight and not preflight["ready"]:
        report_path = Path(run_dir) / "resource_preflight.json"
        atomic_json(report_path, preflight)
        details = []
        if "insufficient_available_system_memory" in preflight["blockers"]:
            details.append(
                f"available system RAM {preflight['available_system_memory_gib']:.2f} GiB "
                "is below the configured minimum "
                f"{preflight['minimum_available_ram_gib']:.2f} GiB; "
                "close memory-heavy applications or use a machine with more free RAM, then rerun"
            )
        if "insufficient_available_gpu_memory" in preflight["blockers"]:
            gpu = preflight["gpu"]
            details.append(
                f"available GPU memory {gpu['available_memory_gib']:.2f} GiB "
                f"is below the configured minimum {gpu['minimum_available_gpu_gib']:.2f} GiB; "
                "free GPU memory or select a device with sufficient memory, then rerun"
            )
        if "insufficient_available_commit_memory" in preflight["blockers"]:
            details.append(
                f"available Windows commit memory {preflight['available_commit_memory_gib']:.2f} "
                "GiB is below the configured minimum "
                f"{preflight['minimum_available_commit_gib']:.2f} GiB; "
                "free system memory or use a machine with sufficient commit capacity, then rerun"
            )
        raise TransformerResourceError(
            "full training blocked by configured resource floor: "
            + ", ".join(preflight["blockers"])
            + f" (selected device: {preflight['selected_device']}). "
            + ". ".join(details)
            + f". Resource report: {report_path}"
        )
    selected_device = preflight["selected_device"]
    if selected_device == "cpu" and config.training.cpu_threads is not None:
        torch.set_num_threads(config.training.cpu_threads)
    batches = effective_batch_settings(config, preflight)
    preflight["effective_batch_settings"] = batches
    preflight["optimizer"] = config.training.optimizer
    preflight["gradient_checkpointing"] = config.training.gradient_checkpointing
    atomic_json(Path(run_dir) / "resource_preflight.json", preflight)
    precision = _precision(config, selected_device)
    set_random_seeds(
        config.training.seed,
        include_optional=True,
        deterministic=config.training.deterministic,
    )
    if resume_run is not None:
        validate_resume_compatibility(
            resume_run,
            dataset_manifest=dataset_manifest,
            config=config,
            training_context_ids_sha256=fingerprint(
                [row.citation_context_id for row in train_records]
            ),
        )
    if tokenizer is None or model is None:
        loaded_tokenizer, loaded_model = _load_pretrained(config, resume_run=resume_run)
        tokenizer = tokenizer or loaded_tokenizer
        model = model or loaded_model
    device = torch.device(selected_device)
    model.to(device)
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    preflight["model_parameter_bytes"] = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    if getattr(model.config, "num_labels", 3) != 3:
        raise IntentDataError("SciBERT classifier must expose exactly three output logits")
    train_dataset = _ContextDataset(train_records)
    dev_dataset = _ContextDataset(dev_records)
    generator = torch.Generator()
    generator.manual_seed(config.training.seed)
    collate = _collator(tokenizer, config.model.max_length, torch)
    if config.training.weighted_sampling:
        weights = calculate_class_weights(
            [row.label_id for row in train_records],
            strategy="inverse",
            smoothing=config.training.class_weight_smoothing,
        )
        sample_weights = torch.DoubleTensor([weights[row.label_id] for row in train_records])
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True, generator=generator
        )
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batches["train_batch_size"],
        sampler=sampler,
        collate_fn=collate,
        num_workers=config.training.num_workers,
        worker_init_fn=_seed_worker if config.training.num_workers else None,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=batches["eval_batch_size"],
        shuffle=False,
        collate_fn=collate,
        num_workers=config.training.num_workers,
        worker_init_fn=_seed_worker if config.training.num_workers else None,
    )
    class_weights = calculate_class_weights(
        [row.label_id for row in train_records],
        strategy=config.training.class_weighting,
        smoothing=config.training.class_weight_smoothing,
    )
    weight_tensor = (
        torch.tensor(
            [class_weights[index] for index in range(3)], dtype=torch.float32, device=device
        )
        if class_weights is not None
        else None
    )
    # Sum each microbatch, then normalize the accumulated gradients by the
    # effective batch denominator. This also handles partial final batches and
    # unequal class-weight sums, matching one larger cross-entropy batch.
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor, reduction="sum")
    optimizer = _make_optimizer(torch, model, config)
    accumulation_steps = batches["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    epochs = (
        min(config.training.epochs, config.training.quick_epochs)
        if config.training.mode == "quick"
        else config.training.epochs
    )
    preflight["effective_epochs"] = epochs
    total_steps = max(1, steps_per_epoch * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * config.training.warmup_ratio),
        num_training_steps=total_steps,
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    else:  # pragma: no cover - maintained for older supported torch releases.
        scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16")
    run_dir = Path(run_dir)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=False)
    history = []
    best_epoch = -1
    best_metrics = None
    no_improvement = 0
    started = time.perf_counter()
    completed_optimizer_steps = 0
    examples_seen = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        loss_denominator = 0.0
        window_denominator = 0.0
        truncated = 0
        epoch_started = time.perf_counter()
        for step, batch in enumerate(train_loader, start=1):
            labels = batch["labels"].to(device)
            inputs = _move_inputs(batch["inputs"], device)
            with _autocast(torch, selected_device, precision):
                logits = model(**inputs).logits
                if logits.shape[-1] != 3:
                    raise IntentDataError("SciBERT model output must contain exactly three logits")
                loss = criterion(logits, labels)
            denominator = (
                float(weight_tensor[labels].sum().detach().cpu())
                if weight_tensor is not None
                else len(labels)
            )
            window_denominator += denominator
            scaler.scale(loss).backward()
            if step % accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(window_denominator)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                prior_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= prior_scale:
                    scheduler.step()
                    completed_optimizer_steps += 1
                window_denominator = 0.0
            accumulated_loss += float(loss.detach().cpu())
            loss_denominator += denominator
            truncated += sum(batch["truncated"])
            examples_seen += len(labels)
            if step == 1 or step % 16 == 0 or step == len(train_loader):
                elapsed = time.perf_counter() - started
                progress = {
                    "pid": os.getpid(),
                    "run_dir": str(run_dir),
                    "status": "TRAINING",
                    "epoch": epoch,
                    "epochs": epochs,
                    "microbatches_completed": step,
                    "microbatches_per_epoch": len(train_loader),
                    "examples_seen": examples_seen,
                    "training_records": len(train_records),
                    "optimizer_steps": completed_optimizer_steps,
                    "optimizer": config.training.optimizer,
                    "optimizer_state_bytes": _optimizer_state_bytes(optimizer),
                    "elapsed_seconds": elapsed,
                    "estimated_epoch_seconds": (time.perf_counter() - epoch_started)
                    * len(train_loader)
                    / step,
                    "train_loss": accumulated_loss / loss_denominator,
                    "test_split_used": False,
                }
                atomic_json(run_dir / "training_progress.json", progress)
                LOGGER.info("SciBERT progress: %s", progress)
        progress["status"] = "VALIDATING"
        atomic_json(run_dir / "training_progress.json", progress)
        LOGGER.info("SciBERT validation: epoch %s, %s dev contexts", epoch, len(dev_records))
        validation, _ = _evaluate(
            model,
            dev_loader,
            torch=torch,
            device=device,
            precision=precision,
            class_weights=weight_tensor,
        )
        event = {
            "epoch": epoch,
            "train_loss": accumulated_loss / loss_denominator if loss_denominator else None,
            "validation": validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - started,
            "train_truncated_contexts": truncated,
        }
        history.append(event)
        candidate = validation["macro_f1"]
        prior = best_metrics["macro_f1"] if best_metrics is not None else None
        if prior is None or candidate > prior + config.training.min_delta:
            best_epoch = epoch
            best_metrics = validation
            no_improvement = 0
            checkpoint = checkpoints / f"epoch-{epoch:02d}"
            model.save_pretrained(checkpoint, safe_serialization=True)
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(checkpoint / "tokenizer")
            atomic_json(
                checkpoint / "checkpoint_metadata.json", {"epoch": epoch, "validation": validation}
            )
        else:
            no_improvement += 1
        # Preserve progress after every completed epoch if the next is interrupted.
        atomic_json(
            run_dir / "training_history.json", {"epochs": history, "best_epoch": best_epoch}
        )
        if no_improvement and no_improvement >= max(1, config.training.early_stopping_patience):
            break
    if best_epoch < 0 or best_metrics is None:
        raise RuntimeError("SciBERT training produced no validation checkpoint")
    best_checkpoint = checkpoints / f"epoch-{best_epoch:02d}"
    model_dir = run_dir / "model"
    tokenizer_dir = run_dir / "tokenizer"
    shutil.copytree(best_checkpoint, model_dir)
    checkpoint_tokenizer = best_checkpoint / "tokenizer"
    if checkpoint_tokenizer.exists():
        shutil.copytree(checkpoint_tokenizer, tokenizer_dir)
    elif hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(tokenizer_dir)
    atomic_json(run_dir / "training_history.json", {"epochs": history, "best_epoch": best_epoch})
    atomic_json(run_dir / "validation_metrics.json", best_metrics)
    atomic_json(
        run_dir / "training_progress.json",
        {
            "status": "TRAINED",
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "examples_seen": examples_seen,
            "optimizer_steps": completed_optimizer_steps,
            "optimizer_state_bytes": _optimizer_state_bytes(optimizer),
            "test_split_used": False,
        },
    )
    preflight["optimizer_state_bytes"] = _optimizer_state_bytes(optimizer)
    atomic_json(
        run_dir / "model_metadata.json",
        {
            "model_id": config.model.name,
            "model_revision": config.model.revision,
            "tokenizer_id": config.model.name,
            "tokenizer_revision": config.model.revision,
            "num_labels": 3,
            "label_order": list(INTENT_LABELS),
            "max_length": config.model.max_length,
            "dataset_fingerprint": dataset_manifest["dataset_fingerprint"],
            "best_epoch": best_epoch,
            "resume": str(resume_run) if resume_run else None,
            "resume_optimizer_state": (
                "not_restored; model-only safetensors resume avoids unsafe pickle"
            ),
        },
    )
    atomic_json(run_dir / "resource_preflight.json", preflight)
    return TransformerTrainingResult(
        model_dir=model_dir,
        tokenizer_dir=tokenizer_dir,
        best_epoch=best_epoch,
        validation_metrics=best_metrics,
        training_history=tuple(history),
        resource_preflight=preflight,
    )


@dataclass
class TransformerPredictor:
    """Loaded local safetensors checkpoint used for batched inference only."""

    model: Any
    tokenizer: Any
    device: Any
    max_length: int
    calibration: TemperatureScaling | None
    metadata: dict

    @classmethod
    def load(cls, run_dir: Path, *, device: str = "auto") -> TransformerPredictor:
        torch, AutoTokenizer, AutoModelForSequenceClassification, _ = _require_transformer_stack()
        run_dir = Path(run_dir)
        metadata = json.loads((run_dir / "model_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("label_order") != list(INTENT_LABELS) or metadata.get("num_labels") != 3:
            raise IntentDataError("checkpoint has an incompatible SciCite output shape")
        selected = detect_device(device).device
        model = AutoModelForSequenceClassification.from_pretrained(
            run_dir / "model", local_files_only=True, use_safetensors=True, trust_remote_code=False
        )
        tokenizer = AutoTokenizer.from_pretrained(
            run_dir / "tokenizer", local_files_only=True, trust_remote_code=False
        )
        model.to(selected)
        model.eval()
        calibration = None
        path = run_dir / "calibration.json"
        if path.is_file():
            document = json.loads(path.read_text(encoding="utf-8"))
            if document.get("method") == "temperature_scaling":
                calibration = TemperatureScaling(
                    temperature=float(document["temperature"]),
                    fitted_on_split=document["fitted_on_split"],
                    examples=int(document["examples"]),
                )
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=torch.device(selected),
            max_length=int(metadata["max_length"]),
            calibration=calibration,
            metadata=metadata,
        )

    def _truncated_flags(self, texts: Sequence[str]) -> list[bool]:
        return _truncation_flags(self.tokenizer, texts, self.max_length)

    def predict(self, texts: Sequence[str], *, batch_size: int = 32) -> dict:
        if batch_size < 1:
            raise IntentDataError("inference batch size must be positive")
        torch, *_ = _require_transformer_stack()
        logits: list[list[float]] = []
        truncated = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            truncated.extend(self._truncated_flags(batch))
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = _move_inputs(dict(encoded), self.device)
            with torch.inference_mode():
                values = self.model(**inputs).logits.float().cpu().tolist()
            logits.extend([[float(value) for value in row] for row in values])
        probabilities, calibration_status = apply_temperature_to_probabilities(
            logits, self.calibration
        )
        return {
            "predicted_ids": [max(range(3), key=row.__getitem__) for row in probabilities],
            "logits": logits,
            "probabilities": probabilities,
            "truncated": truncated,
            "calibration_status": calibration_status,
        }
