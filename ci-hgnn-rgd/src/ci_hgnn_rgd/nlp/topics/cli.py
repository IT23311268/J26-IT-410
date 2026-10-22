"""Explicit Stage 05 CLI. Online model access is opt-in after resource checks."""

import argparse
import json
from pathlib import Path

from ...data.manifests import atomic_json
from .artifacts import RunBuilder, iter_rows, latest_topic_run_id, resolve_run
from .config import Stage05Config, load_config
from .workflow import downstream_action, run_corpus_action

ACTIONS = (
    "inspect-data",
    "estimate-resources",
    "embed",
    "train",
    "train-baselines",
    "evaluate",
    "assign",
    "export-catalog",
    "generate-investigation-candidates",
    "export-annotation-sample",
    "import-annotations",
    "evaluate-annotations",
    "train-investigation-classifier",
    "predict-investigation",
    "evaluate-investigation-classifier",
    "audit-temporal",
    "export-graph-evidence",
    "report",
)

AUTO_TOPIC_RUN_ACTIONS = {"generate-investigation-candidates", "export-graph-evidence"}


def _positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def register_parser(commands):
    parser = commands.add_parser(
        "topics", help="Stage 05 scientific topics and independent investigation evidence"
    )
    actions = parser.add_subparsers(dest="topic_command", required=True)
    for name in ACTIONS:
        action = actions.add_parser(name)
        action.add_argument("--config", type=Path)
        action.add_argument("--project-root", type=Path)
        action.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
        action.add_argument("--batch-size", type=_positive)
        action.add_argument("--max-records", type=_positive)
        action.add_argument(
            "--low-memory",
            action="store_true",
            help="CPU dynamic int8 encoder, batch 1, at most 256 tokens, and two CPU threads",
        )
        action.add_argument("--seed", type=int)
        access = action.add_mutually_exclusive_group()
        access.add_argument("--offline", action="store_true")
        access.add_argument("--allow-model-download", action="store_true")
        action.add_argument(
            "--output-dir", type=Path, help="versioned Stage 05 data root inside project"
        )
        action.add_argument("--stage03-run", type=Path)
        action.add_argument("--temporal-cutoff")
        action.add_argument(
            "--temporal-policy",
            choices=(
                "STRICT_POINT_IN_TIME",
                "PUBLICATION_DATE_PROXY",
                "EXPLORATORY_NO_TEMPORAL_CLAIM",
            ),
        )
        if name == "embed":
            action.add_argument("--model", choices=("specter2", "scibert"), required=True)
        if name == "train":
            action.add_argument("--model", choices=("bertopic",), default="bertopic")
            action.add_argument("--embedding-model", choices=("specter2", "scibert"))
            action.add_argument("--embeddings-run-id")
        if name not in {
            "inspect-data",
            "estimate-resources",
            "embed",
            "train",
            "train-baselines",
            "audit-temporal",
        }:
            action.add_argument(
                "--run-id",
                required=name not in AUTO_TOPIC_RUN_ACTIONS,
                help="fitted topic run; defaults to the latest completed fitted topic model"
                if name in AUTO_TOPIC_RUN_ACTIONS
                else "published Stage 05 run ID",
            )
            action.add_argument("--model-name", choices=("bertopic", "nmf", "lda"))
        if name in {"assign", "import-annotations"}:
            action.add_argument("--input", type=Path, required=True)
        elif name in {
            "evaluate-annotations",
            "train-investigation-classifier",
            "predict-investigation",
        }:
            action.add_argument("--input", type=Path)
        if name in {"import-annotations", "evaluate-annotations", "train-investigation-classifier"}:
            action.add_argument("--adjudications", type=Path)
        if name == "predict-investigation":
            action.add_argument("--classifier-run-id")
        if name in {"predict-investigation", "generate-investigation-candidates"}:
            action.add_argument(
                "--rank-evidence",
                action="store_true",
                help="encode source spans and rank against scientific topic centroids",
            )
        if name == "evaluate":
            action.add_argument("--compare-run-id")
            action.add_argument("--stability", action="store_true")
        if name == "export-graph-evidence":
            action.add_argument("--evidence-run-id")
            action.add_argument("--citation-intent-input", type=Path)
        if name == "evaluate-investigation-classifier":
            action.add_argument("--split", choices=("dev", "test"), required=True)


def _configuration(args):
    candidates = (
        [args.project_root] if args.project_root else [Path.cwd(), Path.cwd() / "ci-hgnn-rgd"]
    )
    root = next(
        (
            Path(p).resolve()
            for p in candidates
            if (Path(p) / "configs/topic_model_stage05.yaml").is_file()
        ),
        None,
    )
    if root is None:
        raise ValueError("cannot locate Stage 05 configuration; pass --project-root")
    config = load_config(args.config, root)
    values = config.model_dump(mode="python")
    if args.low_memory:
        values["embedding"].update(
            device="cpu",
            batch_size=1,
            max_length=min(values["embedding"]["max_length"], 256),
            precision="fp32",
            quantization="int8_dynamic",
            cpu_threads=2,
            min_available_ram_gib=0.5,
        )
    for argument, section, key in (
        ("device", "embedding", "device"),
        ("batch_size", "embedding", "batch_size"),
        ("max_records", "dataset", "max_records"),
        ("stage03_run", "dataset", "stage03_run"),
        ("temporal_cutoff", "temporal", "cutoff"),
        ("temporal_policy", "temporal", "policy"),
    ):
        if getattr(args, argument, None) is not None:
            value = getattr(args, argument)
            if argument == "temporal_cutoff":
                from ...data.temporal_integrity import parse_cutoff

                value = parse_cutoff(value)
            values[section][key] = value
    if args.seed is not None:
        values["seed"] = args.seed
    if args.batch_size is not None:
        values["investigation"]["batch_size"] = args.batch_size
    if args.output_dir:
        output = (root / args.output_dir).resolve()
        if not output.is_relative_to(root):
            raise ValueError("--output-dir must stay inside project")
        values["output_root"] = output
    if args.allow_model_download:
        values["offline"] = False
    if args.offline:
        values["offline"] = True
    if args.topic_command == "embed":
        values["embedding"]["backend"] = args.model
    if getattr(args, "embedding_model", None):
        values["embedding"]["backend"] = args.embedding_model
    return root, Stage05Config.model_validate(values)


def evaluate_investigation(config, root, args):
    from .contracts import LABEL_MAPPING
    from .investigation import validated_training_splits
    from .investigation_model import investigation_metrics, predict_investigation_model

    path, manifest = resolve_run(config.output_root, args.run_id)
    if manifest.get("investigation_training_status") != "TRAINED":
        raise ValueError("NOT_TRAINED: no validated real investigation classifier in this run")
    finals = list(iter_rows(path / "validated_investigation_labels.jsonl"))
    splits = validated_training_splits(finals, min_train=config.investigation.min_train_annotations)
    saved = json.loads((path / "annotation_manifest.json").read_text(encoding="utf-8"))
    from ...pipeline import fingerprint

    if saved["annotation_sha256"] != fingerprint(finals):
        raise ValueError("frozen annotation manifest differs from evaluation input")
    rows = splits[args.split]
    probabilities = predict_investigation_model(
        path, rows, device=config.embedding.device, batch_size=config.investigation.batch_size
    )
    metrics = investigation_metrics(
        [LABEL_MAPPING[row.label] for row in rows],
        [max(range(3), key=row.__getitem__) for row in probabilities],
        probabilities,
    )
    builder = RunBuilder(config.output_root, config, kind=args.topic_command, project_root=root)
    atomic_json(
        builder.pending / "investigation_model_metrics.json",
        {"split": args.split, "metrics": metrics},
    )
    result = builder.publish(
        {
            "parent_run_id": args.run_id,
            "evaluation_split": args.split,
            "test_evaluated": args.split == "test",
            "execution_status": "EVALUATED",
            "investigation_training_status": "TRAINED",
        }
    )
    from .reporting import generate_reports

    result["reports"] = generate_reports(builder.final, result, config.report_root)
    return result, 0


def _run_command(args):
    root, config = _configuration(args)
    if args.topic_command == "report":
        from .reporting import generate_reports

        path, manifest = resolve_run(config.output_root, args.run_id)
        return {"reports": generate_reports(path, manifest, config.report_root)}, 0
    if args.topic_command == "evaluate-investigation-classifier":
        return evaluate_investigation(config, root, args)
    if args.topic_command in {
        "inspect-data",
        "estimate-resources",
        "embed",
        "train",
        "train-baselines",
        "audit-temporal",
    }:
        return run_corpus_action(
            config,
            root,
            args.topic_command,
            embedding_run_id=getattr(args, "embeddings_run_id", None),
        )
    if args.topic_command in AUTO_TOPIC_RUN_ACTIONS and args.run_id is None:
        args.run_id = latest_topic_run_id(config.output_root, model_name=args.model_name)
    for name in ("input", "adjudications", "citation_intent_input"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, (root / value).resolve())
    return downstream_action(config, root, args)


def run_command(args):
    try:
        return _run_command(args)
    except ImportError as error:
        raise RuntimeError(
            f"Stage 05 optional dependency unavailable ({error.name}); "
            "install the project's topics extra in a suitable environment"
        ) from None
