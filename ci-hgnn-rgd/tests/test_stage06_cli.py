"""Synthetic offline CLI integration; actual data builds are separately opt-in."""

import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch_geometric")

from stage06_helpers import fixture_config  # noqa: E402

from ci_hgnn_rgd.cli import main  # noqa: E402
from ci_hgnn_rgd.graph.workflow import build_graph  # noqa: E402


@pytest.fixture
def cli_graph(tmp_path):
    config = fixture_config(tmp_path)
    result = build_graph(config)
    path = tmp_path / "graph-config.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    return tmp_path, path, result


@pytest.mark.parametrize(
    "action",
    [
        "inspect-inputs",
        "validate-contracts",
        "estimate-resources",
        "statistics",
        "export-pyg",
        "validate",
        "trace-node",
        "trace-edge",
        "audit-temporal",
        "report",
    ],
)
def test_graph_cli_commands(cli_graph, capsys, action):
    root, config, _ = cli_graph
    args = ["graph", action, "--config", str(config), "--project-root", str(root)]
    if action in {"trace-node", "trace-edge"}:
        args += ["--id", "p:1" if action == "trace-node" else "c:0"]
    assert main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output


@pytest.mark.parametrize("action", ["build", "build-snapshot"])
def test_graph_cli_build_actions(cli_graph, capsys, action):
    root, config, _ = cli_graph
    args = [
        "graph",
        action,
        "--config",
        str(config),
        "--project-root",
        str(root),
        "--max-papers",
        "2",
    ]
    if action == "build-snapshot":
        args += ["--cutoff", "2022-12-31"]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["statistics"]["node_counts"]["paper"] == 2


def test_graph_cli_informative_failure_and_argument_validation(cli_graph, capsys):
    root, config, _ = cli_graph
    assert (
        main(
            [
                "graph",
                "trace-node",
                "--id",
                "absent",
                "--config",
                str(config),
                "--project-root",
                str(root),
            ]
        )
        == 1
    )
    with pytest.raises(SystemExit) as error:
        main(["graph", "build", "--max-papers", "0"])
    assert error.value.code == 2
    assert (
        main(
            [
                "graph",
                "build",
                "--config",
                str(config),
                "--project-root",
                str(root),
                "--output-dir",
                "../escape",
            ]
        )
        == 1
    )
    assert not (Path(root).parent / "escape").exists()
    capsys.readouterr()
