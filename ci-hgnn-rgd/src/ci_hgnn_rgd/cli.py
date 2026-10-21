"""CI-HGNN-RGD foundation and explicit Stage 02 data commands. No model training."""

import argparse
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .config import ConfigurationError, load_config
from .data.registry import builtin_registry
from .logging_utils import configure_logging
from .pipeline import STAGES, canonical_json, fingerprint
from .runtime import check_environment


def _default_config() -> Path:
    for candidate in (Path("configs/base.yaml"), Path("ci-hgnn-rgd/configs/base.yaml")):
        if candidate.is_file():
            return candidate
    raise ConfigurationError(
        "cannot find configs/base.yaml; use --config or run in the project directory"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ci-hgnn-rgd", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    env = commands.add_parser(
        "check-env", help="report runtime, optional packages, and CPU/GPU availability"
    )
    env.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    config_parser = commands.add_parser(
        "validate-config", help="validate YAML and paths without executing stages"
    )
    config_parser.add_argument("--config", type=Path)
    config_parser.add_argument("--project-root", type=Path)
    commands.add_parser(
        "show-pipeline", help="show the proposed twelve-stage workflow and implementation status"
    )
    from .data.cli import register_parser
    from .nlp.intent_cli import register_parser as register_intent_parser

    register_parser(commands)
    register_intent_parser(commands)
    args = parser.parse_args(argv)
    logger = configure_logging()
    try:
        if args.command == "check-env":
            report = check_environment(args.device)
            print(canonical_json(report))
            return 0 if report["ready"] else 1
        if args.command == "show-pipeline":
            print(canonical_json({"stages": [asdict(stage) for stage in STAGES]}))
            return 0
        if args.command == "citation-intent":
            from .nlp.intent_cli import run_command

            report, code = run_command(args)
            print(canonical_json(report))
            return code
        config = load_config(args.config or _default_config(), project_root=args.project_root)
        logger = configure_logging(config.run.log_level)
        if args.command == "data":
            from .data.cli import run

            report, code = run(args, config)
            print(canonical_json(report))
            return code
        registry = builtin_registry()
        for selection in config.data.datasets:
            metadata = registry.get(selection.dataset_id, selection.version_id)
            if selection.sha256 is not None and selection.sha256 != metadata.sha256:
                raise ValueError(
                    "configured dataset checksum does not match the registered version"
                )
        report = {
            "status": "valid",
            "config_sha256": fingerprint(config),
            "config": config.model_dump(mode="json"),
            "datasets_acquired": False,
            "scientific_stages_executed": [],
        }
        logger.info(
            "Stage 01 configuration validated; no data acquired or scientific stages executed"
        )
        print(canonical_json(report))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        logger.error("Command failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
