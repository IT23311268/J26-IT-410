"""Argparse wiring for explicit Stage 02 data operations."""

from pathlib import Path

from .reporting import generate_reports
from .service import DataService


def positive(value):
    number = int(value)
    if number < 1:
        raise ValueError("value must be positive")
    return number


def register_parser(commands):
    parser = commands.add_parser(
        "data", help="Stage 02 acquisition, inspection, quality and compatibility"
    )
    actions = parser.add_subparsers(dest="data_command", required=True)
    for name in (
        "list",
        "inspect",
        "acquire",
        "validate",
        "audit-identifiers",
        "report",
        "readiness",
    ):
        leaf = actions.add_parser(name)
        leaf.add_argument("--config", type=Path)
        leaf.add_argument("--project-root", type=Path)
        leaf.add_argument(
            "--data-dir",
            type=Path,
            help="raw data root; explicit relative paths use current directory",
        )
        leaf.add_argument("--output-dir", type=Path, help="registry, staging and report root")
        leaf.add_argument(
            "--offline",
            action="store_true",
            help="forbid network acquisition; local inspection never auto-downloads",
        )
        if name in {"inspect", "acquire", "validate"}:
            leaf.add_argument("--dataset", required=True)
            leaf.add_argument(
                "--path",
                type=Path,
                help="manually downloaded source file/directory or graph cache root",
            )
        if name == "acquire":
            leaf.add_argument(
                "--expected-sha256", help="independently trusted expected archive checksum"
            )
        if name in {"inspect", "validate"}:
            leaf.add_argument("--max-records", type=positive)
            leaf.add_argument(
                "--sample-size",
                type=positive,
                help=(
                    "deterministic reservoir size; scans full input unless --max-records "
                    "limits scope"
                ),
            )
            leaf.add_argument("--seed", type=int)
            leaf.add_argument(
                "--version-id", help="verified upstream release identifier; unknown remains null"
            )
            leaf.add_argument(
                "--synthetic",
                action="store_true",
                help="explicitly label software-test fixtures; never real readiness",
            )
            leaf.add_argument(
                "--trusted-cache",
                action="store_true",
                help="allow explicitly verified graph cache; may invoke official deserialization",
            )
            leaf.add_argument("--max-edges", type=positive, default=100000)

    from .stage03_cli import register_stage03

    register_stage03(actions)


def run(args, config):
    from .stage03_cli import ACTIONS, run_command

    if args.data_command in ACTIONS:
        return run_command(args, config)
    service = DataService(config, data_dir=args.data_dir, output_dir=args.output_dir)
    if args.data_command == "list":
        return {
            "datasets": [
                state.model_dump(mode="json") for state in service.registry.states.values()
            ]
        }, 0
    if args.data_command == "acquire":
        state = service.acquire(
            args.dataset, path=args.path, offline=args.offline, expected_sha256=args.expected_sha256
        )
        return state.model_dump(mode="json"), 0
    if args.data_command in {"inspect", "validate"}:
        result = service.inspect(
            args.dataset,
            path=args.path,
            max_records=args.max_records,
            sample_size=args.sample_size,
            seed=args.seed,
            version=args.version_id,
            synthetic=args.synthetic,
            offline=True,
            trusted_cache=args.trusted_cache,
            max_edges=args.max_edges,
        )
        code = 0
        if args.data_command == "validate":
            quality = result.get("quality", {})
            if (
                quality.get("invalid_record_count", 0)
                or quality.get("cross_split_exact_text_overlap_groups", 0)
                or quality.get("duplicate_records", 0)
                or result.get("issues")
            ):
                code = 1
        return result, code
    return generate_reports(service), 0
