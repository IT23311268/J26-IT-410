"""Progressive offline Stage 03 processing commands and read-only audit commands."""

import json
from pathlib import Path

from .integration import ALIASES, PHASES, run_stage03
from .integration_manifest import resolve_run, trace_record
from .stage03_config import InputSource, Stage03Config, load_stage03_config
from .stage03_reporting import publish_reports, validate_run
from .temporal_integrity import parse_cutoff

AUDIT_ACTIONS = {"audit-leakage", "validate-integrated", "integration-report", "trace-record"}
ACTIONS = set(PHASES) | AUDIT_ACTIONS


def register_stage03(actions):
    for name in sorted(ACTIONS):
        parser = actions.add_parser(name, help="Stage 03 offline " + name)
        parser.add_argument("--config", type=Path, help="existing base project YAML")
        parser.add_argument("--project-root", type=Path)
        parser.add_argument("--stage03-config", type=Path, help="Stage 03 integration YAML")
        parser.add_argument("--offline", action="store_true", help="Stage 03 always runs offline")
        parser.add_argument("--dataset", required=name == "clean")
        parser.add_argument("--path", type=Path)
        parser.add_argument("--input-format", choices=("raw", "stage02", "graph_tables"))
        parser.add_argument("--source-version")
        parser.add_argument("--synthetic", action="store_true", help="mark software-test fixtures")
        parser.add_argument("--max-records", type=int)
        parser.add_argument("--batch-size", type=int)
        parser.add_argument("--memory-budget-mb", type=int)
        parser.add_argument("--seed", type=int)
        parser.add_argument(
            "--temporal-cutoff", help="timezone-aware timestamp or inclusive UTC date"
        )
        parser.add_argument(
            "--temporal-policy",
            choices=(
                "STRICT_POINT_IN_TIME",
                "PUBLICATION_DATE_PROXY",
                "EXPLORATORY_NO_TEMPORAL_CLAIM",
            ),
        )
        parser.add_argument("--output-dir", type=Path, help="versioned Stage 03 output root")
        parser.add_argument("--report-dir", type=Path)
        parser.add_argument("--storage-format", choices=("auto", "jsonl", "parquet"))
        parser.add_argument("--run-dir", type=Path, help="published run for audit commands")
        parser.add_argument("--dry-run", action="store_true", help="show plan without writes")
        if name == "trace-record":
            parser.add_argument("--id", required=True, help="canonical or source record key")
            parser.add_argument("--limit", type=int, default=100)


def run_command(args, project):
    root = project.paths.processed.parent.parent
    config = load_stage03_config(args.stage03_config, root)
    values = config.model_dump()
    for option in ("max_records", "batch_size", "memory_budget_mb", "seed", "storage_format"):
        value = getattr(args, option)
        if value is not None:
            values["integration"][option] = value
    for option in ("output_dir", "report_dir"):
        value = getattr(args, option)
        if value is not None:
            values["integration"][option] = value.resolve()
    if args.temporal_cutoff:
        values["temporal_integrity"]["cutoff"] = parse_cutoff(args.temporal_cutoff)
    if args.temporal_policy:
        values["temporal_integrity"]["policy"] = args.temporal_policy
    dataset = ALIASES.get(args.dataset, args.dataset)
    if args.path is not None and dataset is None:
        raise ValueError("--path requires --dataset")
    if args.input_format or args.source_version or args.synthetic:
        if dataset is None:
            raise ValueError("source format/version/synthetic options require --dataset")
        configured = next(
            (source for source in config.integration.inputs if source.dataset == dataset), None
        )
        if args.path is None and configured is None:
            from .integration import select_sources

            selected, _ = select_sources(project, config, dataset=dataset)
            configured = selected[0]
        source = InputSource(
            dataset=dataset,
            path=args.path.resolve() if args.path else configured.path,
            version=args.source_version or (configured.version if configured else None),
            format=args.input_format or (configured.format if configured else "raw"),
            synthetic=args.synthetic or (configured.synthetic if configured else False),
        )
        values["integration"]["inputs"] = [source.model_dump()]
    config = Stage03Config.model_validate(values)
    if args.data_command in PHASES:
        return run_stage03(
            project,
            config,
            action=args.data_command,
            dataset=dataset,
            path=args.path,
            dry_run=args.dry_run,
        ), 0
    if args.dry_run:
        return {
            "dry_run": True,
            "action": args.data_command,
            "output_root": str(config.integration.output_dir),
            "writes": False,
        }, 0
    run = resolve_run(config.integration.output_dir, args.run_dir)
    if args.data_command == "trace-record":
        if not 1 <= args.limit <= 1000:
            raise ValueError("trace limit must be between 1 and 1000")
        return trace_record(run, args.id, limit=args.limit), 0
    if args.data_command == "validate-integrated":
        result = validate_run(run)
        return result, 0 if result["valid"] else 1
    if args.data_command == "audit-leakage":
        return json.loads((run / "stage03_temporal_leakage.json").read_text(encoding="utf-8")), 0
    return {"run_dir": str(run), "reports": publish_reports(run, config.integration.report_dir)}, 0
