"""Explicit Stage 06 commands, independent of the frozen foundation runtime guard."""

import argparse
from collections import Counter
from pathlib import Path

from ..data.temporal_integrity import TemporalPolicy
from .inputs import document, input_report, inspect_inputs, rows, table
from .stage06_config import Stage06Config, load_graph_config

ACTIONS = (
    "inspect-inputs",
    "validate-contracts",
    "estimate-resources",
    "build",
    "build-snapshot",
    "validate",
    "statistics",
    "export-pyg",
    "trace-node",
    "trace-edge",
    "audit-temporal",
    "report",
)


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def register_parser(commands):
    graph = commands.add_parser("graph", help="Stage 06 evidence-preserving graph construction")
    actions = graph.add_subparsers(dest="graph_action", required=True)
    for name in ACTIONS:
        command = actions.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--project-root", type=Path)
        command.add_argument("--output-dir", type=Path)
        command.add_argument("--offline", action="store_true")
        if name in {"build", "build-snapshot", "estimate-resources", "validate-contracts"}:
            command.add_argument("--max-papers", type=positive)
            command.add_argument("--field")
            command.add_argument("--start-year", type=int)
            command.add_argument("--end-year", type=int)
            command.add_argument("--temporal-policy", choices=[p.value for p in TemporalPolicy])
            command.add_argument("--snapshot-id")
            command.add_argument("--cutoff", required=name == "build-snapshot")
            command.add_argument("--dry-run", action="store_true")
            command.add_argument("--mask-relation", action="append", default=[])
        if name in {
            "validate",
            "statistics",
            "export-pyg",
            "trace-node",
            "trace-edge",
            "audit-temporal",
            "report",
        }:
            command.add_argument("--graph-version")
        if name in {"trace-node", "trace-edge"}:
            command.add_argument("--id", required=True)


def _configuration(args):
    root = args.project_root
    if root is None:
        root = next(
            (
                p
                for p in (Path.cwd(), Path.cwd() / "ci-hgnn-rgd")
                if (p / "configs/graph_stage06.yaml").exists()
            ),
            None,
        )
    if root is None:
        raise ValueError("cannot find configs/graph_stage06.yaml; pass --project-root")
    root = root.resolve()
    path = args.config or root / "configs/graph_stage06.yaml"
    config = load_graph_config(path, project_root=root)
    values = config.model_dump(mode="python")
    for name in ("max_papers", "field", "start_year", "end_year", "output_dir"):
        value = getattr(args, name, None)
        if value is not None:
            if name == "output_dir":
                value = (root / value).resolve()
                if not value.is_relative_to(root):
                    raise ValueError("graph output must remain inside project root")
            values[name] = value
    for name, option in (
        ("policy", "temporal_policy"),
        ("cutoff", "cutoff"),
        ("snapshot_id", "snapshot_id"),
    ):
        value = getattr(args, option, None)
        if value is not None:
            values["temporal"][name] = value
    if args.graph_action == "build-snapshot" and getattr(args, "temporal_policy", None) is None:
        values["temporal"]["policy"] = "STRICT_POINT_IN_TIME"
    if getattr(args, "mask_relation", None):
        values["masked_relations"] = tuple(args.mask_relation)
    return Stage06Config.model_validate(values)


def validate_contracts(config):
    from ..nlp.intent_contracts import CitationIntentEvidence
    from ..nlp.topics.contracts import PaperTopicEvidence, TopicAssignment, TopicRecord

    bundle = inspect_inputs(config)
    counts = Counter()
    if bundle.intent_export:
        for row in rows(bundle.intent_export):
            prediction = CitationIntentEvidence.model_validate(row)
            if prediction.model_version != bundle.intent_manifest["run_id"]:
                raise ValueError("citation-intent model version mismatch")
            counts["citation_intent_predictions"] += 1
    if bundle.topic_export:
        models = {}
        for row in rows(table(bundle.topic_export, "topic_catalog")):
            record = TopicRecord.model_validate(row)
            models[record.topic_id] = (record.model_version, record.temporal_snapshot_id)
            counts["topics"] += 1
        for name, model in (
            ("paper_topic_assignments", TopicAssignment),
            ("paper_topic_evidence", PaperTopicEvidence),
        ):
            for row in rows(table(bundle.topic_export, name)):
                record = model.model_validate(row)
                if record.topic_id and models.get(record.topic_id) != (
                    record.topic_model_version,
                    record.temporal_snapshot_id,
                ):
                    raise ValueError("topic assignment/evidence namespace mismatch")
                counts[name] += 1
    return {"status": "VALIDATED", **input_report(bundle), "validated_records": dict(counts)}


def run_command(args):
    from .workflow import build_graph, estimate_resources, graph_path, trace_record, validate_graph

    config = _configuration(args)
    action = args.graph_action
    if action == "inspect-inputs":
        return input_report(inspect_inputs(config)), 0
    if action == "validate-contracts":
        return validate_contracts(config), 0
    if action == "estimate-resources":
        return estimate_resources(config), 0
    if action in {"build", "build-snapshot"}:
        return build_graph(config, dry_run=args.dry_run), 0
    version = getattr(args, "graph_version", None)
    if action in {"trace-node", "trace-edge"}:
        return trace_record(config, args.id, version=version, edge=action == "trace-edge"), 0
    if action == "validate":
        return validate_graph(config, version), 0
    path = graph_path(config, version)
    if action == "export-pyg":
        validate_graph(config, version)
        return {
            "graph_version": path.name,
            "format": "safetensors_plus_json",
            "tensor_path": str(path / "tensors/heterodata.safetensors"),
            "loader": "ci_hgnn_rgd.graph.pyg.load_hetero_graph",
        }, 0
    name = {"statistics": "graph_statistics", "audit-temporal": "temporal_integrity"}.get(action)
    if name:
        return document(path / "reports" / (name + ".json")), 0
    return {
        "graph_version": path.name,
        "report_dir": str(path / "reports"),
        "readiness": document(path / "reports/stage06_readiness.json"),
    }, 0
