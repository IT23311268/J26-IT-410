"""Hash existing code and published inputs without importing the ML pipeline."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

PROTOTYPE = Path(__file__).resolve().parents[1]
COMPONENT = PROTOTYPE.parent
REPO = COMPONENT.parent
BASELINE = PROTOTYPE / "reports/isolation-before.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def paths():
    listed = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=REPO
        )
        .decode()
        .split("\0")
    )
    selected = {
        (REPO / p).resolve()
        for p in listed
        if p
        and not (REPO / p).resolve().is_relative_to(PROTOTYPE)
        and (REPO / p).is_file()
        and "__pycache__" not in p
        and not p.endswith(".pyc")
    }
    # Hash every published Stage 06 artifact and each consumed input.
    pointer = json.loads((COMPONENT / "data/processed/stage06/latest.json").read_text())
    graph = COMPONENT / "data/processed/stage06" / pointer["graph_version"]
    selected.update(p.resolve() for p in graph.rglob("*") if p.is_file())
    manifest = json.loads((graph / "manifests/graph_manifest.json").read_text())
    selected.update(Path(p).resolve() for p in manifest["input_checksums"])
    for root in (COMPONENT / "artifacts/models/citation_intent").glob("run-*"):
        # Published model checkpoints; active .pending training is independent.
        if (root / "run_manifest.json").is_file():
            selected.update(p.resolve() for p in root.rglob("*") if p.is_file())
    return sorted(selected)


def snapshot():
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    if BASELINE.exists():
        raise SystemExit("Isolation baseline already exists; will not overwrite it")
    result = {str(p.relative_to(REPO)): digest(p) for p in paths()}
    BASELINE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Recorded {len(result)} pre-existing file hashes")


def verify():
    original = json.loads(BASELINE.read_text(encoding="utf-8"))
    changed = [
        name
        for name, sha in original.items()
        if not (REPO / name).is_file() or digest(REPO / name) != sha
    ]
    report = {
        "checked_files": len(original),
        "changed_files": changed,
        "status": "PASSED" if not changed else "FAILED",
    }
    (PROTOTYPE / "reports/isolation-after.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report))
    if changed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["snapshot", "verify"])
    args = parser.parse_args()
    snapshot() if args.action == "snapshot" else verify()
