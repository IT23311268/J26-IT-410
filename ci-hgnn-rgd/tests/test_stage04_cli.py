"""CLI surface tests; no command here trains or downloads a model."""

import json
import subprocess
import sys
from argparse import Namespace
from unittest.mock import patch

import pytest
from conftest import PROJECT

from ci_hgnn_rgd.nlp.intent_cli import IntentCliError, _calibrate, _evaluate, _training_trials
from ci_hgnn_rgd.nlp.intent_config import Stage04Config


def run_cli(*arguments):
    return subprocess.run(
        [sys.executable, "-m", "ci_hgnn_rgd", *arguments],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


def test_stage04_commands_are_registered_without_import_time_model_downloads():
    result = run_cli("citation-intent", "--help")
    assert result.returncode == 0, result.stderr
    for command in (
        "inspect-data",
        "audit-leakage",
        "train-baselines",
        "train",
        "evaluate",
        "calibrate",
        "predict",
        "predict-file",
        "export-graph-evidence",
        "report",
    ):
        assert command in result.stdout


def test_stage04_cli_returns_nonzero_for_unknown_published_run():
    result = run_cli(
        "citation-intent",
        "predict",
        "--run-id",
        "run-does-not-exist",
        "--text",
        "synthetic input only",
        "--offline",
    )
    assert result.returncode == 1
    assert result.stdout == ""
    log = json.loads(result.stderr)
    assert log["level"] == "ERROR"
    assert "not found" in log["message"]


def test_cli_rejects_invalid_batch_size_and_full_mode_subset():
    invalid = run_cli("citation-intent", "train", "--model", "scibert", "--batch-size", "0")
    assert invalid.returncode == 2
    invalid = run_cli("citation-intent", "train", "--model", "scibert", "--max-records", "5")
    assert invalid.returncode == 1
    assert "requires --mode quick" in invalid.stderr


def test_calibration_cannot_change_a_tested_run(tmp_path):
    config = Stage04Config()
    with (
        patch("ci_hgnn_rgd.nlp.intent_cli.resolve_intent_run", return_value=tmp_path),
        patch(
            "ci_hgnn_rgd.nlp.intent_cli._manifest",
            return_value={"run_type": "scibert", "test_evaluated": True},
        ),
        patch("ci_hgnn_rgd.nlp.intent_cli._prepare") as prepare,
    ):
        with pytest.raises(IntentCliError, match="before held-out test"):
            _calibrate(config, tmp_path, Namespace(run_id="fixture"))
        prepare.assert_not_called()


def test_incomplete_search_keeps_test_locked(tmp_path):
    with (
        patch("ci_hgnn_rgd.nlp.intent_cli.resolve_intent_run", return_value=tmp_path),
        patch(
            "ci_hgnn_rgd.nlp.intent_cli._manifest",
            return_value={"search_pending": True},
        ),
        patch("ci_hgnn_rgd.nlp.intent_cli._prepare") as prepare,
    ):
        with pytest.raises(IntentCliError, match="test remains locked"):
            _evaluate(Stage04Config(), tmp_path, Namespace(run_id="fixture", split="test"))
        prepare.assert_not_called()


def test_bounded_search_crosses_candidates_and_seeds_without_using_test():
    config = Stage04Config.model_validate(
        {
            "training": {"multiple_seeds": [7, 13]},
            "search": {
                "enabled": True,
                "max_runs": 4,
                "candidates": [
                    {"class_weighting": "none"},
                    {"class_weighting": "inverse", "max_length": 128},
                ],
            },
        }
    )
    trials = _training_trials(config)
    assert [(index, trial.training.seed) for index, trial in trials] == [
        (0, 7),
        (0, 13),
        (1, 7),
        (1, 13),
    ]
    assert trials[2][1].model.max_length == 128
    assert trials[2][1].training.class_weighting == "inverse"
    with pytest.raises(ValueError, match="exceed"):
        Stage04Config.model_validate(
            {
                "training": {"multiple_seeds": [7, 13]},
                "search": {"enabled": True, "max_runs": 2},
            }
        )
