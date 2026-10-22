"""Explicit command-line entrypoints for the Stage 04 citation-intent subsystem."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from ..config import ConfigurationError
from ..data.manifests import atomic_json
from ..pipeline import canonical_json, fingerprint
from .intent_artifacts import (
    IntentRunPublication,
    resolve_intent_run,
    update_run_manifest,
)
from .intent_baselines import BaselinePredictor, evaluate_baseline, train_baselines
from .intent_calibration import fit_temperature_scaling
from .intent_config import Stage04Config, load_stage04_config
from .intent_contracts import ID_TO_LABEL
from .intent_data import (
    PreparedIntentData,
    leakage_controlled_train_view,
    prepare_intent_data,
    quick_training_subset,
)
from .intent_inference import (
    InferenceInputError,
    iter_context_predictions,
    iter_prediction_inputs,
    load_predictor,
    predict_contexts,
    write_prediction_export,
)
from .intent_metrics import bootstrap_macro_f1, evaluate_predictions
from .intent_reporting import generate_run_reports
from .intent_transformer import TransformerPredictor, train_scibert


class IntentCliError(ValueError):
    """Raised for invalid Stage 04 command combinations before a run is started."""


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _add_config_options(parser: argparse.ArgumentParser, *, offline: bool = True) -> None:
    parser.add_argument("--config", type=Path, help="Stage 04 citation-intent YAML")
    parser.add_argument("--project-root", type=Path, help="ci-hgnn-rgd component root")
    if offline:
        parser.add_argument(
            "--offline",
            action="store_true",
            help="forbid model-hub access; locally cached model artifacts remain usable",
        )


def register_parser(commands: argparse._SubParsersAction) -> None:
    """Register only Stage 04 commands below the existing top-level CLI."""
    parser = commands.add_parser(
        "citation-intent",
        help="Stage 04 SciCite citation-intent modelling, evaluation, and exports",
    )
    actions = parser.add_subparsers(dest="intent_command", required=True)

    inspect = actions.add_parser(
        "inspect-data", help="validate and profile the immutable Stage 03 view"
    )
    _add_config_options(inspect)
    inspect.add_argument(
        "--max-records", type=_positive, help="maximum context IDs to include in output sample"
    )

    audit = actions.add_parser("audit-leakage", help="audit official splits without changing them")
    _add_config_options(audit)

    baselines = actions.add_parser(
        "train-baselines", help="fit train-only TF-IDF baselines and select by dev macro F1"
    )
    _add_config_options(baselines)
    baselines.add_argument("--seed", type=int)
    baselines.add_argument("--output-dir", type=Path, help="Stage 04 model-run root")

    train = actions.add_parser("train", help="fine-tune the local SciBERT three-class classifier")
    _add_config_options(train)
    train.add_argument("--model", choices=("scibert",), required=True)
    train.add_argument("--mode", choices=("quick", "full"))
    train.add_argument(
        "--low-memory",
        action="store_true",
        help="CPU microbatch 1, gradient checkpointing and factored Adafactor optimizer state",
    )
    train.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    train.add_argument("--batch-size", type=_positive)
    train.add_argument("--seed", type=int)
    train.add_argument(
        "--search", action="store_true", help="run the configured bounded dev search"
    )
    train.add_argument("--output-dir", type=Path, help="Stage 04 model-run root")
    train.add_argument("--resume", help="compatible untested run for a model-only warm start")
    train.add_argument("--max-records", type=_positive, help="quick-mode train subset cap")

    evaluate = actions.add_parser(
        "evaluate", help="evaluate a frozen run on dev or explicit held-out test"
    )
    _add_config_options(evaluate)
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--split", choices=("dev", "test"), required=True)
    evaluate.add_argument(
        "--model",
        choices=("majority", "tfidf_logistic_regression", "tfidf_linear_svm", "scibert"),
        help="one selected baseline; omitted evaluates every selected baseline or SciBERT",
    )
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    evaluate.add_argument("--batch-size", type=_positive)

    calibrate = actions.add_parser(
        "calibrate", help="fit temperature scaling on dev predictions only"
    )
    _add_config_options(calibrate)
    calibrate.add_argument("--run-id", required=True)
    calibrate.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    calibrate.add_argument("--batch-size", type=_positive)

    predict = actions.add_parser(
        "predict", help="run order-preserving inference for one citation context"
    )
    _add_config_options(predict)
    predict.add_argument("--run-id", required=True)
    predict.add_argument("--model", help="selected baseline name, when the run contains baselines")
    predict.add_argument("--text", required=True)
    predict.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    predict.add_argument("--batch-size", type=_positive)

    predict_file = actions.add_parser(
        "predict-file", help="predict JSONL or Parquet records without changing source records"
    )
    _add_config_options(predict_file)
    predict_file.add_argument("--run-id", required=True)
    predict_file.add_argument("--model", help="selected baseline name, when applicable")
    predict_file.add_argument("--input", type=Path, required=True)
    predict_file.add_argument("--output", type=Path, required=True)
    predict_file.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    predict_file.add_argument("--batch-size", type=_positive)

    export = actions.add_parser(
        "export-graph-evidence",
        help="export unlinked-or-linked citation-context evidence; never create a graph",
    )
    _add_config_options(export)
    export.add_argument("--run-id", required=True)
    export.add_argument("--model", help="selected baseline name, when applicable")
    export.add_argument("--output", type=Path, help="JSONL or Parquet target inside the project")
    export.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    export.add_argument("--batch-size", type=_positive)

    report = actions.add_parser("report", help="generate reports from actual run artifacts only")
    _add_config_options(report)
    report.add_argument("--run-id", required=True)
    report.add_argument("--output-dir", type=Path, help="Stage 04 report-root override")


def _project_root(args: argparse.Namespace) -> Path:
    if args.project_root is not None:
        root = Path(args.project_root).resolve()
        if (root / "configs" / "citation_intent_stage04.yaml").is_file():
            return root
        raise ConfigurationError("--project-root must contain configs/citation_intent_stage04.yaml")
    candidates: list[Path] = []
    if args.config is not None:
        candidates.append(Path(args.config).resolve().parent.parent)
    current = Path.cwd().resolve()
    candidates.extend((current, current / "ci-hgnn-rgd"))
    for candidate in candidates:
        if (candidate / "configs" / "citation_intent_stage04.yaml").is_file():
            return candidate
    raise ConfigurationError(
        "cannot find configs/citation_intent_stage04.yaml; pass --project-root or --config"
    )


def _inside_project(path: Path, root: Path, *, option: str) -> Path:
    resolved = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise IntentCliError(f"{option} must remain inside the project root")
    return resolved


def _load_config(args: argparse.Namespace, root: Path) -> Stage04Config:
    config = load_stage04_config(args.config, root)
    values = config.model_dump(mode="python")
    if getattr(args, "low_memory", False):
        values["training"].update(
            optimizer="adafactor",
            gradient_checkpointing=True,
            cpu_threads=2,
            device="cpu",
            precision="fp32",
            cpu_train_batch_size_cap=1,
            eval_batch_size=1,
            min_available_ram_gib=0.25,
            min_available_commit_gib=3.0,
        )
    if getattr(args, "offline", False):
        values["offline"] = True
    if getattr(args, "seed", None) is not None:
        values["training"]["seed"] = args.seed
        values["training"]["multiple_seeds"] = ()
    if getattr(args, "device", None) is not None:
        values["training"]["device"] = args.device
    if getattr(args, "batch_size", None) is not None:
        values["training"]["train_batch_size"] = args.batch_size
        values["training"]["eval_batch_size"] = args.batch_size
        values["inference"]["batch_size"] = args.batch_size
    if getattr(args, "mode", None) is not None:
        values["training"]["mode"] = args.mode
    if getattr(args, "search", False):
        values["search"]["enabled"] = True
    if (
        getattr(args, "max_records", None) is not None
        and getattr(args, "intent_command", None) == "train"
    ):
        if values["training"]["mode"] != "quick":
            raise IntentCliError(
                "--max-records requires --mode quick; full training uses all records"
            )
        values["training"]["quick_train_limit"] = args.max_records
    if getattr(args, "output_dir", None) is not None:
        output = _inside_project(args.output_dir, root, option="--output-dir")
        if args.intent_command in {"train", "train-baselines"}:
            values["outputs"]["model_root"] = output
        elif args.intent_command == "report":
            values["outputs"]["report_root"] = output
    return Stage04Config.model_validate(values)


def _prepare(config: Stage04Config, root: Path) -> PreparedIntentData:
    return prepare_intent_data(config, root)


def _manifest(run_dir: Path) -> dict:
    path = Path(run_dir) / "run_manifest.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntentCliError(f"cannot read Stage 04 run manifest: {type(error).__name__}") from None
    if document.get("stage") != 4 or document.get("status") != "COMPLETE":
        raise IntentCliError("run manifest is not a published Stage 04 artifact")
    return document


def _verify_dataset_compatibility(manifest: dict, prepared: PreparedIntentData) -> None:
    expected = manifest.get("dataset_manifest", {}).get("dataset_fingerprint")
    observed = prepared.dataset_manifest["dataset_fingerprint"]
    if expected != observed:
        raise IntentCliError(
            "published model and current Stage 03 dataset fingerprint differ; "
            "refusing evaluation or export"
        )
    if manifest.get("label_mapping") != prepared.dataset_manifest["label_mapping"]:
        raise IntentCliError("published model and current fixed label mapping differ")


def _write_run_inputs(publication: IntentRunPublication, prepared: PreparedIntentData) -> None:
    publication.write_json("config.json", publication.config)
    publication.write_json("label_mapping.json", prepared.dataset_manifest["label_mapping"])
    publication.write_json("dataset_manifest.json", prepared.dataset_manifest)
    publication.write_json("dataset_profile.json", prepared.dataset_profile)
    publication.write_json("leakage_audit.json", prepared.leakage_audit)
    publication.write_json("temporal_validation.json", prepared.temporal_validation)


def _training_manifest(prepared: PreparedIntentData, *, run_type: str) -> dict:
    return {
        "run_type": run_type,
        "dataset_manifest": prepared.dataset_manifest,
        "label_mapping": prepared.dataset_manifest["label_mapping"],
        "official_splits": {
            "train": "fit",
            "dev": "selection_and_early_stopping",
            "test": "not_used_during_training_or_selection",
        },
        "test_evaluated": False,
        "temporal_validation": prepared.temporal_validation,
    }


def _training_view(prepared: PreparedIntentData, config: Stage04Config) -> tuple[tuple, dict]:
    records = prepared.split("train")
    sensitivity = config.evaluation.sensitivity_exclude_cross_split_text_overlap
    if sensitivity:
        records = leakage_controlled_train_view(
            records, prepared.split("dev") + prepared.split("test")
        )
    if config.training.mode == "quick":
        records = quick_training_subset(records, limit=config.training.quick_train_limit)
    return records, {
        "kind": "text_overlap_sensitivity" if sensitivity else "official_benchmark",
        "quick_mode": config.training.mode == "quick",
        "official_train_records": len(prepared.split("train")),
        "training_records": len(records),
        "training_context_ids_sha256": fingerprint([row.citation_context_id for row in records]),
        "test_labels_used": False,
        "test_text_used_for_overlap_filter": sensitivity,
        "official_assignments_modified": False,
    }


def _train_baselines(config: Stage04Config, root: Path) -> tuple[dict, int]:
    prepared = _prepare(config, root)
    train_records, training_view = _training_view(prepared, config)
    publication = IntentRunPublication(
        config.outputs.model_root, config=config.model_dump(mode="json"), project_root=root
    )
    try:
        _write_run_inputs(publication, prepared)
        results = train_baselines(
            train_records,
            prepared.split("dev"),
            publication.pending / "baselines",
            config.baselines,
            seed=config.training.seed,
        )
        publication.write_json("baseline_results.json", results)
        run_dir = publication.publish(
            {
                **_training_manifest(prepared, run_type="baselines"),
                "selected_models": results["selected_models"],
                "baseline_training_seconds": results["training_seconds"],
                "baseline_serialization": results["serialization_policy"],
                "training_view": training_view,
            }
        )
    except Exception as error:
        publication.fail(error)
        raise
    return {
        "status": "trained_baselines",
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "training_split": "train",
        "selection_split": "dev",
        "test_split_used": False,
        "selected_models": results["selected_models"],
        "dataset_profile": prepared.dataset_profile,
        "training_view": training_view,
    }, 0


def _training_trials(config: Stage04Config) -> list[tuple[int, Stage04Config]]:
    candidates = config.search.candidates if config.search.enabled else (None,)
    trials = []
    for index, candidate in enumerate(candidates):
        for seed in config.training.multiple_seeds or (config.training.seed,):
            values = config.model_dump(mode="python")
            values["training"]["seed"] = seed
            values["training"]["multiple_seeds"] = ()
            if candidate is not None:
                overrides = candidate.model_dump(exclude_none=True)
                if "max_length" in overrides:
                    values["model"]["max_length"] = overrides.pop("max_length")
                values["training"].update(overrides)
            trials.append((index, Stage04Config.model_validate(values)))
    return trials


def _train_scibert(config: Stage04Config, root: Path, args: argparse.Namespace) -> tuple[dict, int]:
    trials = _training_trials(config)
    if args.resume and (len(trials) != 1 or config.search.enabled):
        raise IntentCliError("--resume requires a single run with search disabled")
    prepared = _prepare(config, root)
    train_records, training_view = _training_view(prepared, config)
    completed = []
    for candidate_index, seed_config in trials:
        resume_run = (
            resolve_intent_run(seed_config.outputs.model_root, args.resume) if args.resume else None
        )
        publication = IntentRunPublication(
            seed_config.outputs.model_root,
            config=seed_config.model_dump(mode="json"),
            project_root=root,
        )
        try:
            _write_run_inputs(publication, prepared)
            result = train_scibert(
                train_records,
                prepared.split("dev"),
                publication.pending,
                seed_config,
                prepared.dataset_manifest,
                resume_run=resume_run,
            )
            run_dir = publication.publish(
                {
                    **_training_manifest(prepared, run_type="scibert"),
                    "training_mode": seed_config.training.mode,
                    "training_records": len(train_records),
                    "validation_records": len(prepared.split("dev")),
                    "best_epoch": result.best_epoch,
                    "validation_metrics": result.validation_metrics,
                    "resource_preflight": result.resource_preflight,
                    "training_view": training_view,
                    "search_pending": config.search.enabled,
                }
            )
        except Exception as error:
            publication.fail(error)
            raise
        completed.append(
            {
                "seed": seed_config.training.seed,
                "candidate_index": candidate_index,
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "best_epoch": result.best_epoch,
                "validation_metrics": result.validation_metrics,
            }
        )
    summaries = []
    for index in sorted({item["candidate_index"] for item in completed}):
        runs = [item for item in completed if item["candidate_index"] == index]
        scores = [item["validation_metrics"]["macro_f1"] for item in runs]
        summaries.append(
            {
                "candidate_index": index,
                "parameters": config.search.candidates[index].model_dump(exclude_none=True)
                if config.search.enabled
                else {},
                "seeds": [item["seed"] for item in runs],
                "run_ids": [item["run_id"] for item in runs],
                "mean_dev_macro_f1": fmean(scores),
                "std_dev_macro_f1": pstdev(scores) if len(scores) > 1 else 0.0,
            }
        )
    selected = max(summaries, key=lambda item: item["mean_dev_macro_f1"])
    selection = {
        "metric": "mean_dev_macro_f1",
        "candidates": summaries,
        "selected_candidate_index": selected["candidate_index"],
        "selected_run_id": selected["run_ids"][0],
        "seed_selection_policy": "first configured seed; never select a seed using test results",
        "tie_break": "first configured candidate",
        "test_split_used": False,
        "mode": config.training.mode,
        "training_view": training_view,
    }
    if config.search.enabled:
        for run in completed:
            run_dir = Path(run["run_dir"])
            atomic_json(run_dir / "search_selection.json", selection)
            update_run_manifest(run_dir, {"kind": "dev_search_selection", "test_split_used": False})
    return {
        "status": "trained_scibert",
        "runs": completed,
        "mode": config.training.mode,
        "training_records": len(train_records),
        "validation_records": len(prepared.split("dev")),
        "test_split_used": False,
        "selection": selection,
        "training_view": training_view,
    }, 0


def _predict_with_backend(predictor: Any, texts: list[str], batch_size: int) -> dict:
    if isinstance(predictor, BaselinePredictor):
        return predictor.predict(texts)
    return predictor.predict(texts, batch_size=batch_size)


def _evaluate_one(
    run_dir: Path,
    manifest: dict,
    prepared: PreparedIntentData,
    *,
    model_name: str,
    split: str,
    device: str,
    batch_size: int,
    ece_bins: int,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
    seed: int,
) -> dict:
    records = prepared.split(split)  # type: ignore[arg-type]
    if manifest.get("run_type") == "baselines":
        selected = manifest.get("selected_models", {})
        if model_name not in selected:
            raise IntentCliError("requested model is not a selected baseline in this run")
        metrics, output = evaluate_baseline(
            run_dir / "baselines" / selected[model_name]["artifact"], records, ece_bins=ece_bins
        )
        model_version = manifest["run_id"]
    else:
        if model_name != "scibert":
            raise IntentCliError("SciBERT runs can only evaluate the scibert model")
        predictor = TransformerPredictor.load(run_dir, device=device)
        output = _predict_with_backend(
            predictor, [record.model_text for record in records], batch_size
        )
        metrics = evaluate_predictions(
            [record.label_id for record in records],
            output["predicted_ids"],
            output.get("probabilities"),
            ece_bins=ece_bins,
        )
        model_version = manifest["run_id"]
    if output.get("probabilities") is not None:
        probabilities = output["probabilities"]
    else:
        probabilities = [None] * len(records)
    errors = []
    for record, prediction, probability in zip(
        records, output["predicted_ids"], probabilities, strict=True
    ):
        if prediction == record.label_id:
            continue
        values = probability or (None, None, None)
        errors.append(
            {
                "citation_context_id": record.citation_context_id,
                "source_record_id": record.source_record_id,
                "true_label": record.normalized_label,
                "predicted_label": ID_TO_LABEL[prediction],
                "prob_background": values[0],
                "prob_method": values[1],
                "prob_result": values[2],
                "model_version": model_version,
                "text_reference": record.citation_context_id,
            }
        )
    document = {
        "schema_version": "1.0",
        "stage": 4,
        "run_id": manifest["run_id"],
        "model": model_name,
        "split": split,
        "evaluated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "dataset_fingerprint": prepared.dataset_manifest["dataset_fingerprint"],
        "metrics": metrics,
        "prediction_probability_status": output.get("calibration_status"),
        "test_split_used": split == "test",
        "errors": errors,
    }
    if bootstrap_resamples:
        groups = [
            f"{record.citing_paper_id}\x1f{record.cited_paper_id}"
            if record.citing_paper_id is not None and record.cited_paper_id is not None
            else None
            for record in records
        ]
        document["bootstrap_macro_f1"] = bootstrap_macro_f1(
            [record.label_id for record in records],
            output["predicted_ids"],
            groups=groups,
            seed=seed,
            n_resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
        )
    return document


def _evaluate(config: Stage04Config, root: Path, args: argparse.Namespace) -> tuple[dict, int]:
    run_dir = resolve_intent_run(config.outputs.model_root, args.run_id)
    manifest = _manifest(run_dir)
    if args.split == "test" and manifest.get("search_pending"):
        selection_path = run_dir / "search_selection.json"
        if not selection_path.is_file():
            raise IntentCliError("bounded search is incomplete; held-out test remains locked")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected = selection["candidates"][selection["selected_candidate_index"]]
        if args.run_id not in selected["run_ids"]:
            raise IntentCliError("held-out test is available only for the dev-selected candidate")
    prepared = _prepare(config, root)
    _verify_dataset_compatibility(manifest, prepared)
    if manifest.get("run_type") == "baselines":
        selected = manifest.get("selected_models", {})
        names = [args.model] if args.model else sorted(selected)
        if not names:
            raise IntentCliError("baseline run does not record selected models")
    elif manifest.get("run_type") == "scibert":
        if args.model not in {None, "scibert"}:
            raise IntentCliError("--model must be scibert for a SciBERT run")
        names = ["scibert"]
    else:
        raise IntentCliError("unsupported run type for evaluation")
    for name in names:
        if (run_dir / "evaluations" / args.split / f"{name}.json").exists():
            raise IntentCliError(
                f"{args.split} evaluation already exists for {name}; "
                "read its saved report instead of repeating inference"
            )
    results = []
    for name in names:
        document = _evaluate_one(
            run_dir,
            manifest,
            prepared,
            model_name=name,
            split=args.split,
            device=args.device or config.training.device,
            batch_size=args.batch_size or config.inference.batch_size,
            ece_bins=config.calibration.ece_bins,
            bootstrap_resamples=config.evaluation.bootstrap_resamples,
            bootstrap_confidence=config.evaluation.bootstrap_confidence,
            seed=config.training.seed,
        )
        path = run_dir / "evaluations" / args.split / f"{name}.json"
        atomic_json(path, document)
        update_run_manifest(
            run_dir,
            {
                "kind": "explicit_evaluation",
                "model": name,
                "split": args.split,
                "path": str(path.relative_to(run_dir)),
                "test_split_used": args.split == "test",
            },
        )
        results.append(document)
    return {
        "status": "evaluated",
        "run_id": args.run_id,
        "split": args.split,
        "test_split_used": args.split == "test",
        "evaluations": [
            {
                "model": item["model"],
                "metrics": item["metrics"],
                "path": str(run_dir / "evaluations" / args.split / f"{item['model']}.json"),
            }
            for item in results
        ],
    }, 0


def _calibrate(config: Stage04Config, root: Path, args: argparse.Namespace) -> tuple[dict, int]:
    run_dir = resolve_intent_run(config.outputs.model_root, args.run_id)
    manifest = _manifest(run_dir)
    if manifest.get("run_type") != "scibert":
        raise IntentCliError("temperature scaling is available only for local SciBERT logits")
    if (
        manifest.get("test_evaluated")
        or any(item.get("test_split_used") for item in manifest.get("updates", []))
        or any((run_dir / "evaluations" / "test").glob("*.json"))
    ):
        raise IntentCliError("calibration must be frozen before held-out test evaluation")
    calibration_path = run_dir / "calibration.json"
    if calibration_path.exists():
        raise IntentCliError("this run already has a calibration artifact; it is not overwritten")
    prepared = _prepare(config, root)
    _verify_dataset_compatibility(manifest, prepared)
    predictor = TransformerPredictor.load(run_dir, device=args.device or config.training.device)
    output = predictor.predict(
        [record.model_text for record in prepared.split("dev")],
        batch_size=args.batch_size or config.inference.batch_size,
    )
    scaler, report = fit_temperature_scaling(
        output["logits"],
        [record.label_id for record in prepared.split("dev")],
        split="dev",
        ece_bins=config.calibration.ece_bins,
    )
    atomic_json(calibration_path, {**scaler.to_dict(), "report": report})
    update_run_manifest(
        run_dir,
        {
            "kind": "temperature_calibration",
            "split": "dev",
            "path": "calibration.json",
            "test_split_used": False,
        },
    )
    return {
        "status": "calibrated",
        "run_id": args.run_id,
        "calibration": report,
        "test_split_used": False,
    }, 0


def _load_local_predictor(
    config: Stage04Config, args: argparse.Namespace
) -> tuple[Path, Any, dict]:
    run_dir = resolve_intent_run(config.outputs.model_root, args.run_id)
    predictor, model_info = load_predictor(
        run_dir,
        model_name=getattr(args, "model", None),
        device=args.device or config.training.device,
    )
    return run_dir, predictor, model_info


def _predict(config: Stage04Config, args: argparse.Namespace) -> tuple[dict, int]:
    _, predictor, model_info = _load_local_predictor(config, args)
    predictions = predict_contexts(
        [
            {
                "citation_context_id": "manual-input:0",
                "source_dataset": "manual-input",
                "text": args.text,
            }
        ],
        predictor,
        model_info,
        batch_size=args.batch_size or config.inference.batch_size,
        low_confidence_threshold=config.calibration.low_confidence_threshold,
        include_text_excerpt=config.inference.include_text_excerpt,
        text_excerpt_characters=config.inference.text_excerpt_characters,
    )
    return {"prediction": predictions[0].model_dump(mode="json")}, 0


def _predict_file(config: Stage04Config, root: Path, args: argparse.Namespace) -> tuple[dict, int]:
    _, predictor, model_info = _load_local_predictor(config, args)
    source = _inside_project(args.input, root, option="--input")
    destination = _inside_project(args.output, root, option="--output")
    if destination.exists():
        raise IntentCliError("--output already exists; prediction exports are never overwritten")
    batch_size = args.batch_size or config.inference.batch_size
    rows = iter_prediction_inputs(source, batch_size=batch_size)
    predictions = iter_context_predictions(
        rows,
        predictor,
        model_info,
        batch_size=batch_size,
        low_confidence_threshold=config.calibration.low_confidence_threshold,
        include_text_excerpt=config.inference.include_text_excerpt,
        text_excerpt_characters=config.inference.text_excerpt_characters,
    )
    summary = {"records": 0, "truncated": 0, "skipped_empty_text": 0, "linked": 0}
    output = write_prediction_export(destination, _count_predictions(predictions, summary))
    return {
        "status": "predicted_file",
        "input": str(source),
        "output": str(output),
        "predictions": summary["records"],
        "truncated_contexts": summary["truncated"],
        "skipped_empty_contexts": summary["skipped_empty_text"],
    }, 0


def _count_predictions(predictions, summary: dict):
    for item in predictions:
        summary["records"] += 1
        summary["truncated"] += item.truncation_applied
        summary["skipped_empty_text"] += item.prediction_status == "skipped_empty_text"
        summary["linked"] += item.citing_paper_id is not None and item.cited_paper_id is not None
        yield item


def _export_graph_evidence(
    config: Stage04Config, root: Path, args: argparse.Namespace
) -> tuple[dict, int]:
    run_dir, predictor, model_info = _load_local_predictor(config, args)
    manifest = _manifest(run_dir)
    prepared = _prepare(config, root)
    _verify_dataset_compatibility(manifest, prepared)
    destination = (
        _inside_project(args.output, root, option="--output")
        if args.output is not None
        else run_dir / "prediction_exports" / "stage03_citation_intent_evidence.jsonl"
    )
    if destination.exists():
        raise IntentCliError("graph-evidence export already exists; it is never overwritten")
    predictions = iter_context_predictions(
        (row for split in ("train", "dev", "test") for row in prepared.split(split)),
        predictor,
        model_info,
        batch_size=args.batch_size or config.inference.batch_size,
        low_confidence_threshold=config.calibration.low_confidence_threshold,
        include_text_excerpt=config.inference.include_text_excerpt,
        text_excerpt_characters=config.inference.text_excerpt_characters,
    )
    summary = {"records": 0, "truncated": 0, "skipped_empty_text": 0, "linked": 0}
    output = write_prediction_export(destination, _count_predictions(predictions, summary))
    linked = summary["linked"]
    export_manifest = {
        "schema_version": "1.0",
        "stage": 4,
        "run_id": manifest["run_id"],
        "dataset_fingerprint": prepared.dataset_manifest["dataset_fingerprint"],
        "records": summary["records"],
        "truncated_contexts": summary["truncated"],
        "skipped_empty_contexts": summary["skipped_empty_text"],
        "verified_paper_pair_links": linked,
        "unlinked_contexts": summary["records"] - linked,
        "graph_created": False,
        "integration_policy": (
            "Each record is citation-context evidence. Later stages may attach a record "
            "only to its verified matching paper pair and must preserve repeated contexts."
        ),
        "citation_intent_interpretation": (
            "Predicted intent is not direct-investigation or research-gap evidence."
        ),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    atomic_json(manifest_path, export_manifest)
    if output.is_relative_to(run_dir):
        update_run_manifest(
            run_dir,
            {
                "kind": "graph_evidence_export",
                "path": str(output.relative_to(run_dir)),
                "manifest": str(manifest_path.relative_to(run_dir)),
            },
        )
    return {
        "status": "exported_graph_evidence",
        "run_id": manifest["run_id"],
        "output": str(output),
        "manifest": str(manifest_path),
        "records": summary["records"],
        "truncated_contexts": summary["truncated"],
        "verified_paper_pair_links": linked,
        "unlinked_contexts": summary["records"] - linked,
        "graph_created": False,
    }, 0


def _inspect_data(config: Stage04Config, root: Path, args: argparse.Namespace) -> tuple[dict, int]:
    prepared = _prepare(config, root)
    sample_limit = args.max_records or 0
    sample = [
        {
            "citation_context_id": record.citation_context_id,
            "source_record_id": record.source_record_id,
            "source_split": record.source_split,
            "label": record.normalized_label,
            "linked_to_verified_paper_pair": bool(record.citing_paper_id and record.cited_paper_id),
        }
        for record in prepared.all_records[:sample_limit]
    ]
    return {
        "status": "inspected",
        "dataset_profile": prepared.dataset_profile,
        "dataset_manifest": prepared.dataset_manifest,
        "temporal_validation": prepared.temporal_validation,
        "sample": sample,
    }, 0


def _audit_leakage(config: Stage04Config, root: Path) -> tuple[dict, int]:
    prepared = _prepare(config, root)
    sensitivity = None
    if config.evaluation.sensitivity_exclude_cross_split_text_overlap:
        evaluation = prepared.split("dev") + prepared.split("test")
        train = leakage_controlled_train_view(prepared.split("train"), evaluation)
        sensitivity = {
            "enabled": True,
            "official_train_records": len(prepared.split("train")),
            "leakage_controlled_train_records": len(train),
            "official_assignments_modified": False,
            "purpose": "optional sensitivity view; not the official SciCite benchmark",
        }
    return {
        "status": "audited",
        "leakage_audit": prepared.leakage_audit,
        "temporal_validation": prepared.temporal_validation,
        "sensitivity": sensitivity,
    }, 0


def _report(config: Stage04Config, root: Path, args: argparse.Namespace) -> tuple[dict, int]:
    prepared = _prepare(config, root)
    run_dir = resolve_intent_run(config.outputs.model_root, args.run_id)
    _verify_dataset_compatibility(_manifest(run_dir), prepared)
    generated = generate_run_reports(
        config.outputs.model_root, args.run_id, config.outputs.report_root, prepared
    )
    return {"status": "reported", "run_id": args.run_id, "reports": generated}, 0


def run_command(args: argparse.Namespace) -> tuple[dict, int]:
    """Execute one explicit Stage 04 action and return JSON-ready output plus exit code."""
    root = _project_root(args)
    config = _load_config(args, root)
    command = args.intent_command
    if command == "inspect-data":
        return _inspect_data(config, root, args)
    if command == "audit-leakage":
        return _audit_leakage(config, root)
    if command == "train-baselines":
        return _train_baselines(config, root)
    if command == "train":
        return _train_scibert(config, root, args)
    if command == "evaluate":
        return _evaluate(config, root, args)
    if command == "calibrate":
        return _calibrate(config, root, args)
    if command == "predict":
        return _predict(config, args)
    if command == "predict-file":
        return _predict_file(config, root, args)
    if command == "export-graph-evidence":
        return _export_graph_evidence(config, root, args)
    if command == "report":
        return _report(config, root, args)
    raise IntentCliError(f"unsupported citation-intent command: {command}")


def run_and_render(args: argparse.Namespace) -> int:
    """Compatibility adapter for direct testing of the Stage 04 parser handler."""
    try:
        report, code = run_command(args)
        print(canonical_json(report))
        return code
    except (OSError, ValueError, KeyError, InferenceInputError) as error:
        print(canonical_json({"status": "failed", "error": str(error)}))
        return 1
