"""Reports contain measured artifacts and explicit unexecuted/annotation statuses."""

import csv
import json

from ...data.manifests import atomic_json
from .artifacts import iter_rows


def generate_reports(run_dir, manifest, report_root):
    output = report_root / manifest["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    generated = {}
    metrics_path = run_dir / "investigation_model_metrics.json"
    if metrics_path.exists():
        document = json.loads(metrics_path.read_text(encoding="utf-8"))
        matrix = document["metrics"]["confusion_matrix"]
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            from .contracts import LABELS

            figure, axis = plt.subplots(figsize=(8, 6))
            image = axis.imshow(matrix, cmap="Blues")
            figure.colorbar(image, ax=axis)
            axis.set_xticks(range(3), LABELS, rotation=20, ha="right")
            axis.set_yticks(range(3), LABELS)
            axis.set_xlabel("Predicted investigation evidence")
            axis.set_ylabel("Human annotation")
            axis.set_title(f"{document['split']}: {document['metrics']['samples']} annotated pairs")
            for row in range(3):
                for column in range(3):
                    axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
            figure.tight_layout()
            figure.savefig(output / "investigation_confusion_matrix.png", dpi=180)
            plt.close(figure)
            generated["investigation_confusion_matrix"] = str(
                output / "investigation_confusion_matrix.png"
            )
        except ImportError:
            pass
    for source, target in (
        ("input_data_profile.json", "input_data_profile"),
        ("embedding_quality.json", "embedding_quality"),
        ("topic_quality.json", "topic_quality"),
        ("topic_stability.json", "topic_stability"),
        ("temporal_snapshot.json", "topic_temporal_audit"),
        ("investigation_model_metrics.json", "investigation_model_metrics"),
    ):
        if source == "temporal_snapshot.json" and (run_dir / "topic_temporal_audit.json").is_file():
            source = "topic_temporal_audit.json"
        if (run_dir / source).is_file():
            value = json.loads((run_dir / source).read_text(encoding="utf-8"))
            atomic_json(output / (target + ".json"), value)
            (output / (target + ".md")).write_text(
                f"# {target.replace('_', ' ')}\n\n```json\n"
                + json.dumps(value, indent=2)
                + "\n```\n",
                encoding="utf-8",
            )
            generated[target] = str(output / (target + ".json"))
    if (run_dir / "topic_catalog.jsonl").is_file():
        rows = list(iter_rows(run_dir / "topic_catalog.jsonl"))
        (output / "topic_catalog.md").write_text(
            "# Provisional topics\n\n"
            + "\n".join(
                f"- `{r['topic_id']}`: {r['provisional_label']} "
                f"({r['paper_count']} papers; requires review)."
                for r in rows
            )
            + "\n",
            encoding="utf-8",
        )
        generated["topic_catalog"] = str(output / "topic_catalog.md")
    comparisons = manifest.get("model_comparison", [])
    if comparisons:
        with (output / "topic_model_comparison.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "model",
                    "topics",
                    "diversity",
                    "npmi",
                    "outlier_rate",
                    "runtime_seconds",
                ],
            )
            writer.writeheader()
            writer.writerows(comparisons)
    annotation_path = run_dir / "annotation_report.json"
    annotation = (
        json.loads(annotation_path.read_text(encoding="utf-8"))
        if annotation_path.exists()
        else {"status": "NO_HUMAN_ANNOTATIONS", "supervised_status": "NOT_TRAINED"}
    )
    (output / "investigation_annotation_report.md").write_text(
        "# Investigation annotation status\n\n```json\n"
        + json.dumps(annotation, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    readiness = {
        "stage": 5,
        "kind": manifest["kind"],
        "scientific_results": manifest.get("execution_status"),
        "investigation_classifier": manifest.get("investigation_training_status", "NOT_TRAINED"),
        "temporal_claim": manifest.get("temporal_snapshot", {}).get(
            "strict_historical_claim", False
        ),
        "stage06_handoff": "versioned contracts available; graph not constructed",
        "limitations": manifest.get("limitations", []),
        "interpretation": (
            "Topic membership, citation intent and candidate scores "
            "do not establish direct investigation."
        ),
    }
    (output / "stage05_readiness.md").write_text(
        "# Stage 05 readiness\n\n```json\n" + json.dumps(readiness, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    generated["readiness"] = str(output / "stage05_readiness.md")
    return generated
