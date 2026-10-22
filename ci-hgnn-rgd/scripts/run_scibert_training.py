"""Run real-data SciBERT training with durable stdout, logs and completion status."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from ci_hgnn_rgd.data.manifests import atomic_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = args.report_dir
    report = (root / report).resolve() if not report.is_absolute() else report.resolve()
    if not report.is_relative_to(root):
        parser.error("training report must remain inside project")
    report.mkdir(parents=True, exist_ok=True)
    if (report / "status.json").exists():
        parser.error("training report already exists; choose a new report directory")
    command = [
        sys.executable,
        "-u",
        "-m",
        "ci_hgnn_rgd",
        "citation-intent",
        "train",
        "--model",
        "scibert",
        "--mode",
        args.mode,
        "--offline",
    ]
    environment = dict(os.environ)
    environment.update(
        HF_HOME=str(root / "data/interim/huggingface"),
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
    )
    state = {
        "status": "STARTING",
        "mode": args.mode,
        "supervisor_pid": os.getpid(),
        "started_at": datetime.now(UTC).isoformat(),
        "command": command,
        "log_file": str(report / "training.log"),
        "test_split_used": False,
    }
    atomic_json(report / "status.json", state)
    try:
        with (
            (report / "stdout.json").open("w", encoding="utf-8") as stdout,
            (report / "training.log").open("w", encoding="utf-8") as stderr,
        ):
            process = subprocess.Popen(
                command, cwd=root, env=environment, stdout=stdout, stderr=stderr
            )
            state.update(status="RUNNING", pid=process.pid)
            atomic_json(report / "status.json", state)
            exit_code = process.wait()
        output = (report / "stdout.json").read_text(encoding="utf-8")
        try:
            result = json.loads(output) if output.strip() else None
        except json.JSONDecodeError:
            result = {"error": "Command output was not JSON; inspect stdout.json and training.log"}
        state.update(
            status="TRAINED" if exit_code == 0 else "FAILED", exit_code=exit_code, result=result
        )
    except (OSError, subprocess.SubprocessError) as error:
        exit_code = 1
        state.update(status="FAILED", exit_code=exit_code, error=str(error))
    state["finished_at"] = datetime.now(UTC).isoformat()
    atomic_json(report / "status.json", state)
    print(json.dumps(state, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
