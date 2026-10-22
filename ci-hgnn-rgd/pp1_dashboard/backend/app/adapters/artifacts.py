"""Read published JSON/SQLite contracts; never import or execute the ML pipeline."""

import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from ..schemas import DEMO_NOTICE

NODE_TYPES = {"paper", "author", "topic", "research_field"}
EDGE_TYPES = {
    "cites",
    "authored_by",
    "associated_with_topic",
    "mentions_topic",
    "investigates_topic",
    "classified_in",
}
STAGES = [
    "Project foundation",
    "Dataset acquisition",
    "Cleaning & entity resolution",
    "Citation-intent classification",
    "Topics & investigation evidence",
    "Heterogeneous knowledge graph",
    "Graph representation learning",
    "Research-gap scoring",
    "Baselines & ablations",
    "Robustness analysis",
    "Temporal & expert evaluation",
    "Evidence explanations & reporting",
]
LIMITATIONS = [
    "Author counts represent source-local records, not verified unique people.",
    "Topic membership is not a topic mention or direct-investigation ground truth.",
    "Result citation intent is not direct-investigation ground truth.",
    "The exploratory graph does not establish historical availability or leakage-free forecasting.",
    "Reverse message-passing edges are excluded from scholarly relationship counts.",
]


class ArtifactUnavailable(Exception):
    pass


def document(path):
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ArtifactUnavailable("Artifact metadata exceeds the viewer resource limit.")
    return json.loads(path.read_text(encoding="utf-8"))


def synthetic(value):
    if isinstance(value, list):
        return any(synthetic(item) for item in value)
    if isinstance(value, dict):
        return (
            value.get("synthetic") is True
            or bool(value.get("synthetic_sources"))
            or any(synthetic(item) for item in value.values())
        )
    return False


def safe_path(root, value):
    path = (root / str(value).replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ArtifactUnavailable("Artifact path is outside its published root.")
    return path


def verify(path, expected):
    if not expected or not path.is_file():
        raise ArtifactUnavailable("A required published artifact or checksum is missing.")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ArtifactUnavailable(
            "Artifact checksum verification failed. Restore a trusted published version."
        )


def node(row):
    payload = json.loads(row["payload"])
    if synthetic(payload):
        raise ArtifactUnavailable("Synthetic provenance was rejected in real mode.")
    return {
        "id": row["key"],
        "type": row["kind"],
        "label": payload.get("title")
        or payload.get("display_name")
        or payload.get("provisional_label")
        or payload.get("label")
        or row["key"],
        "year": payload.get("publication_year"),
        "abstract": payload.get("abstract"),
        "keywords": payload.get("keywords", []),
        "provenance": payload.get("provenance", []),
        "metadata": {
            key: payload[key]
            for key in (
                "arxiv_id",
                "doi",
                "authors",
                "categories",
                "identity_status",
                "temporal_eligibility",
                "model_version",
                "representative_paper_keys",
                "topic_id",
            )
            if key in payload
        },
        "synthetic": False,
    }


def edge(row):
    payload = json.loads(row["payload"])
    if synthetic(payload):
        raise ArtifactUnavailable("Synthetic provenance was rejected in real mode.")
    return {
        "id": row["id"],
        "source": row["s"],
        "target": row["t"],
        "relation": row["rel"],
        "provenance": payload.get("provenance", []),
        "synthetic": False,
    }


class RealReader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.watched = {}
        self.contexts = {}
        self.metrics = []
        pointer = self.read(self.root / "data/processed/stage06/latest.json")
        self.graph = safe_path(self.root / "data/processed/stage06", pointer["graph_version"])
        if self.graph.name.startswith(".pending"):
            raise ArtifactUnavailable("Pending graph builds are not published artifacts.")
        self.manifest = self.read(self.graph / "manifests/graph_manifest.json")
        if (
            self.manifest.get("status") != "COMPLETE"
            or self.manifest.get("schema_version") != "stage06-v1"
            or synthetic(self.manifest)
        ):
            raise ArtifactUnavailable("A complete, real stage06-v1 graph is required.")
        self.version = self.manifest["graph_version"]
        self.db = self.graph / "graph.sqlite"
        self.checked(self.db, self.manifest["artifact_checksums"].get("graph.sqlite"))
        integrity = self.published("reports/graph_integrity.json")
        if integrity.get("status") != "PASSED" or integrity.get("critical_violations") != 0:
            raise ArtifactUnavailable("The published graph integrity check has not passed.")
        self.integrity = integrity
        self.stats_report = self.published("reports/graph_statistics.json")
        # Check published consumed inputs, without loading arrays or model checkpoints.
        for path, checksum in self.manifest.get("input_checksums", {}).items():
            self.checked(safe_path(self.root, path), checksum)
        with self.connection() as conn:
            self.node_counts = dict(conn.execute("SELECT kind,count(*) FROM nodes GROUP BY kind"))
            self.edge_counts = dict(
                conn.execute("SELECT rel,count(*) FROM edges WHERE derived=0 GROUP BY rel")
            )
            if (
                self.node_counts != self.stats_report["node_counts"]
                or self.edge_counts != self.stats_report["scholarly_edge_counts"]
            ):
                raise ArtifactUnavailable("Graph counts disagree with its published statistics.")
            # Reject synthetic records even outside the initially visible neighborhood.
            for table in ("nodes", "edges", "evidence"):
                for row in conn.execute(f"SELECT payload FROM {table}"):
                    if synthetic(json.loads(row[0])):
                        raise ArtifactUnavailable("Synthetic provenance was rejected in real mode.")
        self._load_intents()

    def watch(self, path):
        stat = path.stat()
        self.watched[path] = (stat.st_mtime_ns, stat.st_size)

    def read(self, path):
        value = document(path)
        self.watch(path)
        return value

    def checked(self, path, checksum):
        verify(path, checksum)
        self.watch(path)
        return path

    def published(self, name):
        path = safe_path(self.graph, name)
        self.checked(path, self.manifest["artifact_checksums"].get(name))
        return self.read(path)

    def assert_unchanged(self):
        for path, stamp in self.watched.items():
            if not path.is_file() or (path.stat().st_mtime_ns, path.stat().st_size) != stamp:
                raise ArtifactUnavailable(
                    "Published inputs changed after verification. Restart the viewer to revalidate."
                )

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.db.as_uri() + "?mode=ro&immutable=1", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        budget = [0]

        def guard():
            budget[0] += 1
            return int(budget[0] > 5000)

        conn.set_progress_handler(guard, 10000)
        try:
            yield conn
        finally:
            conn.close()

    def _load_intents(self):
        inputs = self.manifest.get("inputs", {})
        export = inputs.get("intent_export")
        self.intent_manifest = None
        if export:
            run = safe_path(self.root, export).parents[1]
            try:
                model = self.read(run / "run_manifest.json")
                if model.get("status") != "COMPLETE" or synthetic(model):
                    raise ArtifactUnavailable("The intent model run is incomplete or synthetic.")
                dataset = model["dataset_manifest"]
                context = (
                    safe_path(self.root, dataset["stage03_run_dir"])
                    / dataset["citation_context_table"]
                )
                self.checked(context, dataset["citation_context_table_sha256"])
                with context.open(encoding="utf-8") as stream:
                    for line in stream:
                        if len(line) > 2_000_000:
                            raise ArtifactUnavailable("Context record exceeds the resource limit.")
                        row = json.loads(line)
                        if synthetic(row):
                            raise ArtifactUnavailable("Synthetic citation contexts were rejected.")
                        self.contexts[row["context_id"]] = {
                            "text": row.get("original_context"),
                            "original_label": row.get("original_intent_label"),
                            "source_citing_id": row.get("citing_identifier", {}).get("raw"),
                            "source_cited_id": row.get("cited_identifier", {}).get("raw"),
                        }
                self.intent_manifest = model
                for name, sha in model.get("artifact_checksums", {}).items():
                    if name.replace("\\", "/").startswith("evaluations/") and name.endswith(
                        ".json"
                    ):
                        path = self.checked(safe_path(run, name), sha)
                        result = self.read(path)
                        metrics = result.get("metrics", result)
                        for metric in ("accuracy", "macro_f1", "weighted_f1"):
                            if isinstance(metrics.get(metric), (int, float)):
                                self.metrics.append(
                                    {
                                        "model": path.stem,
                                        "split": path.parent.name,
                                        "metric": metric,
                                        "value": metrics[metric],
                                        "samples": metrics.get("samples"),
                                        "source": str(path.relative_to(self.root)),
                                    }
                                )
            except (OSError, ValueError, KeyError, ArtifactUnavailable):
                self.contexts = {}
                self.metrics = []
                self.intent_manifest = None
        self.intents = []
        with self.connection() as conn:
            for row in conn.execute(
                "SELECT id,payload FROM evidence WHERE "
                "kind='citation_context_predictions' ORDER BY id LIMIT 100001"
            ):
                p = json.loads(row["payload"])
                if len(self.intents) >= 100000:
                    raise ArtifactUnavailable(
                        "Citation predictions exceed the prototype's 100,000-record budget."
                    )
                if p.get("predicted_intent") not in ("background", "method", "result"):
                    raise ArtifactUnavailable("Unsupported citation-intent taxonomy.")
                self.intents.append(
                    {
                        "id": p.get("citation_context_id", row["id"]),
                        "text": p.get("text_excerpt"),
                        "label": p["predicted_intent"],
                        "confidence": p.get("confidence"),
                        "source": p.get("source_dataset", "Unknown"),
                        "split": p.get("source_split"),
                        "citing_paper_id": p.get("citing_paper_id"),
                        "cited_paper_id": p.get("cited_paper_id"),
                        "model_id": p.get("model_id"),
                        "model_version": p.get("model_version"),
                        "provenance": p.get("provenance", []),
                        "synthetic": False,
                        "original_label": None,
                        **self.contexts.get(p.get("citation_context_id"), {}),
                    }
                )

    def graph_stats(self):
        return {
            "node_counts": self.node_counts,
            "edge_counts": self.edge_counts,
            "components": self.stats_report.get("weak_connected_components"),
            "limitations": LIMITATIONS,
        }

    def datasets(self):
        items = [
            {
                "name": "arXiv · admitted graph corpus",
                "records": self.node_counts.get("paper", 0),
                "status": "Verified Data",
                "detail": (
                    "Counts describe the published bounded corpus, not the complete arXiv archive."
                ),
                "source": str(self.manifest["inputs"].get("stage03_run")),
            },
            {
                "name": "SciCite · exported context predictions",
                "records": len(self.intents),
                "status": "Model Prediction",
                "detail": "Independent corpus; unresolved endpoints do not become citation facts.",
                "source": self.manifest["inputs"].get("intent_export"),
            },
        ]
        registry = self.root / "artifacts/reports/dataset_registry.json"
        if registry.is_file():
            for key, value in document(registry).get("datasets", {}).items():
                items.append(
                    {
                        "name": key + " · acquisition registry",
                        "records": value.get("valid_record_count"),
                        "status": value.get("status", "Validation Pending"),
                        "detail": (
                            "Historical registry state; may predate separately published runs. "
                        )
                        + "; ".join(value.get("errors", []))[:400],
                        "source": value.get("source_url"),
                    }
                )
        return {"items": items}

    def progress(self):
        stages = []
        for index, name in enumerate(STAGES, 1):
            verified = index == 6 or (index == 4 and self.intent_manifest is not None)
            stages.append(
                {
                    "number": index,
                    "name": name,
                    "status": "Not Yet Implemented"
                    if index > 6
                    else "Verified Data"
                    if verified
                    else "Validation Pending",
                    "detail": "Planned stage; this prototype does not execute it."
                    if index > 6
                    else (
                        "Published graph, checksum, measured counts and integrity report "
                        "verified; scientific limitations remain."
                    )
                    if index == 6
                    else (
                        "Published intent export and context-table checksum verified. "
                        "Predictions are model outputs."
                    )
                    if index == 4
                    else (
                        "Downstream consumed artifacts are present; a complete independent "
                        "stage acceptance audit is pending."
                    ),
                    "evidence": [str(self.graph.relative_to(self.root))] if index <= 6 else [],
                }
            )
        return {
            "stages": stages,
            "engineering_note": (
                "Artifact availability is reported separately from full stage "
                "acceptance. No overall completion percentage is inferred."
            ),
            "scientific_note": (
                "Scientific gap validity, expert review and temporal "
                "generalization remain unverified."
            ),
        }

    def overview(self):
        with self.connection() as conn:
            years = [
                {"label": str(row[0]) if row[0] is not None else "Unknown", "count": row[1]}
                for row in conn.execute(
                    "SELECT json_extract(payload,'$.publication_year'),count(*) FROM "
                    "nodes WHERE kind='paper' GROUP BY 1 ORDER BY 1"
                )
            ]
        coverage = self.stats_report.get("paper_topic_coverage", {})
        return {
            "counts": {
                "papers": self.node_counts.get("paper", 0),
                "authors": self.node_counts.get("author", 0),
                "citations": self.edge_counts.get("cites", 0),
                "topics": self.node_counts.get("topic", 0),
            },
            "publication_years": years,
            "intent_distribution": self.citation_intents()["distribution"] if self.intents else [],
            "topic_coverage": {
                "assigned": coverage.get("assigned_papers", 0),
                "papers": coverage.get("papers", 0),
                "fraction": coverage.get("fraction", 0),
            },
            "graph": {
                "nodes": sum(self.node_counts.values()),
                "edges": sum(self.edge_counts.values()),
                "components": self.stats_report.get("weak_connected_components"),
            },
            "datasets": self.datasets()["items"],
            "stages": self.progress()["stages"],
            "provenance": [
                str(self.graph.relative_to(self.root)),
                "graph.sqlite SHA-256 verified against the published graph manifest",
                "Input paths and SHA-256 values verified; model weights are never loaded",
                "Temporal policy: " + self.manifest["policy"],
            ],
            "limitations": LIMITATIONS,
            "intent_model": self.intents[0]["model_id"] if self.intents else None,
            "created_at": self.manifest.get("created_at"),
        }

    def citation_intents(self, q="", label=None, limit=20, offset=0):
        items = [
            p
            for p in self.intents
            if (not label or p["label"] == label)
            and (not q or q.casefold() in ((p["text"] or "") + " " + p["id"]).casefold())
        ]
        counts = Counter(p["label"] for p in self.intents)
        first = self.intents[0] if self.intents else {}
        return {
            "items": items[offset : offset + limit],
            "total": len(items),
            "limit": limit,
            "offset": offset,
            "distribution": [
                {"label": label, "count": counts[label]}
                for label in ("background", "method", "result")
            ],
            "model_id": first.get("model_id"),
            "model_version": first.get("model_version"),
            "calibration": (
                "Uncalibrated model confidence; not a validated probability of correctness."
            ),
        }

    def nodes(self, q="", node_type=None, limit=20, offset=0):
        where = "(? IS NULL OR kind=?) AND (?='' OR instr(lower(payload),lower(?))>0)"
        args = (node_type, node_type, q, q)
        with self.connection() as conn:
            total = conn.execute(f"SELECT count(*) FROM nodes WHERE {where}", args).fetchone()[0]
            items = [
                node(row)
                for row in conn.execute(
                    f"SELECT * FROM nodes WHERE {where} ORDER BY kind,key LIMIT ? OFFSET ?",
                    (*args, limit, offset),
                )
            ]
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    def node_detail(self, node_id):
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE key=? LIMIT 1", (node_id,)).fetchone()
            if row is None:
                raise KeyError(node_id)
            relations = [
                edge(r)
                for r in conn.execute(
                    "SELECT * FROM edges WHERE derived=0 AND (s=? OR t=?) ORDER BY id LIMIT 50",
                    (node_id, node_id),
                )
            ]
        return {"node": node(row), "relations": relations}

    def topics(self, q="", limit=20, offset=0):
        page = self.nodes(q=q, node_type="topic", limit=limit, offset=offset)
        with self.connection() as conn:
            items = []
            for n in page["items"]:
                payload = json.loads(
                    conn.execute("SELECT payload FROM nodes WHERE key=?", (n["id"],)).fetchone()[0]
                )
                count = conn.execute(
                    (
                        "SELECT count(DISTINCT s) FROM edges WHERE t=? AND derived=0 AND "
                        "rel='associated_with_topic'"
                    ),
                    (n["id"],),
                ).fetchone()[0]
                items.append(
                    {
                        "id": n["id"],
                        "label": n["label"],
                        "keywords": n["keywords"],
                        "paper_count": count,
                        "coverage": count / max(1, self.node_counts.get("paper", 0)),
                        "model_version": payload.get("model_version", "Unavailable"),
                        "review_status": payload.get("review_status", "Validation Pending"),
                        "synthetic": False,
                    }
                )
        return {**page, "items": items}

    def topic_detail(self, topic_id):
        topic = next(
            (t for t in self.topics(q=topic_id, limit=1)["items"] if t["id"] == topic_id), None
        )
        if topic is None:
            raise KeyError(topic_id)
        n = self.node_detail(topic_id)["node"]
        with self.connection() as conn:
            papers = [
                node(r)
                for r in conn.execute(
                    (
                        "SELECT n.* FROM nodes n JOIN edges e ON n.key=e.s AND n.kind=e.sk "
                        "WHERE e.t=? AND e.derived=0 AND e.rel='associated_with_topic' "
                        "ORDER BY n.key LIMIT 20"
                    ),
                    (topic_id,),
                )
            ]
            representative = n["metadata"].get("representative_paper_keys", [])
            for paper_id in representative[:10]:
                if any(p["id"] == paper_id for p in papers):
                    continue
                row = conn.execute(
                    "SELECT n.* FROM nodes n JOIN edges e ON n.key=e.s "
                    "WHERE n.kind='paper' AND n.key=? AND e.t=? AND e.derived=0 "
                    "AND e.rel='associated_with_topic' LIMIT 1",
                    (paper_id, topic_id),
                ).fetchone()
                if row:
                    papers.insert(0, node(row))
            papers.sort(key=lambda p: p["id"] not in representative)
            papers = papers[:20]
            for paper in papers:
                paper["metadata"]["representative"] = paper["id"] in representative
            evidence = []
            for row in conn.execute(
                (
                    "SELECT payload FROM evidence WHERE kind='paper_topic_evidence' "
                    "AND json_extract(payload,'$.topic_id')=? LIMIT 20"
                ),
                (n["metadata"].get("topic_id"),),
            ):
                p = json.loads(row[0])
                evidence.append(
                    {
                        "tier": p.get("evidence_tier", "UNKNOWN"),
                        "label": p.get("evidence_label", "UNKNOWN"),
                        "paper_id": p.get("paper_key"),
                        "span": p.get("evidence_text_span"),
                        "uncertainty": p.get("uncertainty_status", "Validation Pending"),
                        "synthetic": False,
                    }
                )
        return {
            "topic": topic,
            "papers": papers,
            "evidence": evidence,
            "provenance": n["provenance"],
            "membership_note": (
                "Topic membership is an unsupervised association. It does not "
                "establish a topic mention or direct investigation. Rule "
                "candidates require independent review."
            ),
        }

    def neighborhood(self, seed=None, depth=1, limit=70, node_types=None, edge_types=None):
        node_types = NODE_TYPES if node_types is None else node_types
        edge_types = EDGE_TYPES if edge_types is None else edge_types
        with self.connection() as conn:
            if seed is None:
                row = conn.execute(
                    "SELECT n.key FROM nodes n LEFT JOIN edges e ON e.t=n.key AND "
                    "e.derived=0 WHERE n.kind='topic' GROUP BY n.key ORDER BY "
                    "count(e.id) DESC,n.key LIMIT 1"
                ).fetchone()
                if row is None:
                    row = conn.execute("SELECT key FROM nodes ORDER BY kind,key LIMIT 1").fetchone()
                seed = row[0] if row else None
            start = conn.execute("SELECT * FROM nodes WHERE key=? LIMIT 1", (seed,)).fetchone()
            if start is None or start["kind"] not in node_types:
                return {
                    "nodes": [],
                    "edges": [],
                    "seed": seed,
                    "depth": depth,
                    "limit": limit,
                    "truncated": False,
                }
            nodes = {seed: node(start)}
            edges = {}
            frontier = [seed]
            truncated = False
            for hop in range(depth):
                next_frontier = []
                # Reserve room for the next hop instead of exhausting the node
                # budget on a high-degree seed's immediate paper neighbors.
                hop_limit = (
                    min(limit, max(1, limit // 3))
                    if depth > 1 and hop == 0 and "author" in node_types
                    else limit
                )
                for key in frontier:
                    if not edge_types or not node_types:
                        continue

                    # Filter before capping fanout so rare relations remain discoverable.
                    def placeholders(values):
                        return ",".join("?" for _ in values)

                    clause = (
                        f"AND rel IN ({placeholders(edge_types)}) "
                        f"AND sk IN ({placeholders(node_types)}) "
                        f"AND tk IN ({placeholders(node_types)}) "
                    )
                    parameters = (*sorted(edge_types), *sorted(node_types), *sorted(node_types))
                    rows = list(
                        conn.execute(
                            "SELECT * FROM edges WHERE derived=0 AND s=? "
                            + clause
                            + "ORDER BY id LIMIT 401",
                            (key, *parameters),
                        )
                    )
                    rows += list(
                        conn.execute(
                            "SELECT * FROM edges WHERE derived=0 AND t=? "
                            + clause
                            + "ORDER BY id LIMIT 401",
                            (key, *parameters),
                        )
                    )
                    if len(rows) > 400:
                        truncated = True
                    for r in rows:
                        if (
                            r["rel"] not in edge_types
                            or r["sk"] not in node_types
                            or r["tk"] not in node_types
                        ):
                            continue
                        other = r["t"] if r["s"] == key else r["s"]
                        if other not in nodes:
                            if len(nodes) >= hop_limit:
                                truncated = True
                                continue
                            n = conn.execute(
                                "SELECT * FROM nodes WHERE key=? AND kind=?",
                                (other, r["tk"] if r["s"] == key else r["sk"]),
                            ).fetchone()
                            if n is None:
                                continue
                            nodes[other] = node(n)
                            next_frontier.append(other)
                        if len(edges) < 400:
                            edges[r["id"]] = edge(r)
                        else:
                            truncated = True
                frontier = next_frontier
        return {
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
            "seed": seed,
            "depth": depth,
            "limit": limit,
            "truncated": truncated,
        }

    def validation(self):
        return {
            "checks": [
                {
                    "name": "Published graph integrity",
                    "status": "Verified Data",
                    "detail": (
                        "Checksum verified; zero critical violations in the published "
                        "Stage 06 report."
                    ),
                },
                {
                    "name": "Citation linkage",
                    "status": "Validation Pending",
                    "detail": (
                        str(
                            self.stats_report.get("citation_intent_coverage", {}).get(
                                "linked_contexts", 0
                            )
                        )
                        + " linked contexts. Unresolved SciCite endpoints are retained separately."
                    ),
                },
                {
                    "name": "Investigation evidence",
                    "status": "Validation Pending",
                    "detail": json.dumps(self.stats_report.get("investigation_evidence_tiers", {})),
                },
                {
                    "name": "Context text & evaluation sidecars",
                    "status": "Verified Data" if self.intent_manifest else "Validation Pending",
                    "detail": (
                        "Published model run, context checksum and available evaluation "
                        "checksums checked."
                    )
                    if self.intent_manifest
                    else (
                        "No compatible verified model sidecars available; context "
                        "text/metrics omitted."
                    ),
                },
            ],
            "metrics": self.metrics,
            "plans": [
                {
                    "name": "Technical validation",
                    "status": "Planned",
                    "detail": (
                        "Verify contracts, provenance, bounds and read-only behavior "
                        "independently of scientific quality."
                    ),
                },
                {
                    "name": "Temporal backtesting",
                    "status": "Not Yet Implemented",
                    "detail": (
                        "Verify historical source availability; freeze train cutoffs, "
                        "topic fits and features; mask prediction targets before "
                        "chronological evaluation."
                    ),
                },
                {
                    "name": "Expert evaluation",
                    "status": "Not Yet Implemented",
                    "detail": (
                        "Collect independent, blinded investigation and gap judgments; "
                        "document agreement, adjudication and uncertainty."
                    ),
                },
            ],
            "missing_evidence": [
                "Verified citation links between SciCite contexts and admitted papers",
                "Human investigation annotations and independently validated positive evidence",
                "Expert topic and research-gap validation",
                "Verified historical source snapshots",
                "Stage 07 GNN and Stage 08 scoring outputs",
            ],
            "temporal_policy": self.manifest.get("policy"),
        }


class DemoReader:
    """Small in-memory fixture; no real reader or pipeline filesystem access."""

    version = "synthetic-ui-v1"

    def __init__(self):
        self.data = document(Path(__file__).resolve().parents[3] / "demo_data/dataset.json")
        self.nodes_data = self.data["nodes"]
        self.edges_data = self.data["edges"]
        self.intents = self.data["intents"]
        for item in self.nodes_data + self.edges_data + self.intents:
            item.update(synthetic=True, notice=DEMO_NOTICE)

    def assert_unchanged(self):
        pass

    def nodes(self, q="", node_type=None, limit=20, offset=0):
        items = [
            n
            for n in self.nodes_data
            if (not node_type or n["type"] == node_type)
            and q.casefold() in (n["label"] + n["id"]).casefold()
        ]
        return {
            "items": items[offset : offset + limit],
            "total": len(items),
            "limit": limit,
            "offset": offset,
        }

    def node_detail(self, node_id):
        n = next((n for n in self.nodes_data if n["id"] == node_id), None)
        if n is None:
            raise KeyError(node_id)
        return {
            "node": n,
            "relations": [e for e in self.edges_data if node_id in (e["source"], e["target"])],
        }

    def graph_stats(self):
        return {
            "node_counts": dict(Counter(n["type"] for n in self.nodes_data)),
            "edge_counts": dict(Counter(e["relation"] for e in self.edges_data)),
            "components": None,
            "limitations": [DEMO_NOTICE, *LIMITATIONS],
        }

    def datasets(self):
        return {
            "items": [
                {
                    "name": "Synthetic interface fixture",
                    "records": len(self.nodes_data),
                    "status": "Conceptual Preview",
                    "detail": DEMO_NOTICE,
                    "source": "pp1_dashboard/demo_data/dataset.json",
                    "synthetic": True,
                    "notice": DEMO_NOTICE,
                }
            ]
        }

    def progress(self):
        return {
            "stages": [
                {
                    "number": i,
                    "name": name,
                    "status": "Validation Pending" if i <= 6 else "Not Yet Implemented",
                    "detail": DEMO_NOTICE,
                    "evidence": [],
                    "synthetic": True,
                    "notice": DEMO_NOTICE,
                }
                for i, name in enumerate(STAGES, 1)
            ],
            "engineering_note": "Demo fixtures cannot establish engineering stage acceptance.",
            "scientific_note": "No scientific results or evaluation metrics exist in demo mode.",
        }

    def citation_intents(self, q="", label=None, limit=20, offset=0):
        items = [
            r
            for r in self.intents
            if (not label or r["label"] == label)
            and q.casefold() in (r["text"] + r["id"]).casefold()
        ]
        counts = Counter(r["label"] for r in self.intents)
        return {
            "items": items[offset : offset + limit],
            "total": len(items),
            "limit": limit,
            "offset": offset,
            "distribution": [
                {"label": k, "count": v, "synthetic": True, "notice": DEMO_NOTICE}
                for k, v in counts.items()
            ],
            "model_id": "No model · synthetic UI fixture",
            "model_version": "synthetic-ui-v1",
            "calibration": "No trained model or confidence scores.",
        }

    def topics(self, q="", limit=20, offset=0):
        page = self.nodes(q=q, node_type="topic", limit=limit, offset=offset)
        return {
            **page,
            "items": [
                {
                    "id": n["id"],
                    "label": n["label"],
                    "keywords": n["keywords"],
                    "paper_count": sum(e["target"] == n["id"] for e in self.edges_data),
                    "coverage": sum(e["target"] == n["id"] for e in self.edges_data) / 6,
                    "model_version": "No fitted model",
                    "review_status": "Synthetic UI fixture",
                    "synthetic": True,
                    "notice": DEMO_NOTICE,
                }
                for n in page["items"]
            ],
        }

    def topic_detail(self, topic_id):
        t = next((t for t in self.topics()["items"] if t["id"] == topic_id), None)
        if t is None:
            raise KeyError(topic_id)
        ids = {e["source"] for e in self.edges_data if e["target"] == topic_id}
        return {
            "topic": t,
            "papers": [n for n in self.nodes_data if n["id"] in ids],
            "evidence": [],
            "provenance": [
                {"synthetic": True, "source": "Hand-authored UI fixture", "notice": DEMO_NOTICE}
            ],
            "membership_note": (
                "Synthetic membership illustrates the UI only; no "
                "direct-investigation evidence is asserted."
            ),
        }

    def neighborhood(self, seed=None, depth=1, limit=70, node_types=None, edge_types=None):
        seed = seed or "demo:topic:1"
        types = NODE_TYPES if node_types is None else node_types
        relations = EDGE_TYPES if edge_types is None else edge_types
        allowed = {n["id"]: n for n in self.nodes_data if n["type"] in types}
        chosen = {seed} if seed in allowed else set()
        for _ in range(depth):
            frontier = set(chosen)
            for e in self.edges_data:
                if (
                    e["relation"] in relations
                    and e["source"] in allowed
                    and e["target"] in allowed
                    and (e["source"] in frontier or e["target"] in frontier)
                ):
                    chosen.update((e["source"], e["target"]))
        ids = list(sorted(chosen, key=lambda k: (k != seed, k)))[:limit]
        return {
            "nodes": [allowed[k] for k in ids],
            "edges": [
                e
                for e in self.edges_data
                if e["relation"] in relations and e["source"] in ids and e["target"] in ids
            ][:400],
            "seed": seed,
            "depth": depth,
            "limit": limit,
            "truncated": len(chosen) > limit,
        }

    def overview(self):
        s = self.graph_stats()
        return {
            "counts": {
                "papers": s["node_counts"].get("paper", 0),
                "authors": s["node_counts"].get("author", 0),
                "citations": s["edge_counts"].get("cites", 0),
                "topics": s["node_counts"].get("topic", 0),
            },
            "publication_years": [
                {"label": str(k), "count": v}
                for k, v in sorted(
                    Counter(n["year"] for n in self.nodes_data if n["type"] == "paper").items()
                )
            ],
            "intent_distribution": self.citation_intents()["distribution"],
            "topic_coverage": {"assigned": 6, "papers": 6, "fraction": 1},
            "graph": {
                "nodes": len(self.nodes_data),
                "edges": len(self.edges_data),
                "components": None,
            },
            "datasets": self.datasets()["items"],
            "stages": self.progress()["stages"],
            "provenance": [DEMO_NOTICE, "Hand-authored fixture; no pipeline execution"],
            "limitations": s["limitations"],
            "intent_model": None,
            "created_at": None,
        }

    def validation(self):
        return {
            "checks": [
                {"name": "Synthetic fixture", "status": "Conceptual Preview", "detail": DEMO_NOTICE}
            ],
            "metrics": [],
            "plans": [
                {
                    "name": "Independent research validation",
                    "status": "Not Yet Implemented",
                    "detail": (
                        "Use genuine data and reviewed protocols; synthetic examples "
                        "provide no scientific evidence."
                    ),
                }
            ],
            "missing_evidence": ["Demo mode contains no scientific evaluation results."],
            "temporal_policy": None,
        }
