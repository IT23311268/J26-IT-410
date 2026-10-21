"""Official OGB API boundary plus chunked array validation. Never converts to NetworkX."""

import csv
import gzip
from importlib.util import find_spec
from pathlib import Path

from ..download_manager import AcquisitionBlocked

SOURCE_URL = "https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv"


def shape(value):
    return tuple(int(v) for v in value.shape) if value is not None else None


def _integer_array(array):
    dtype = str(getattr(array, "dtype", ""))
    return "int" in dtype and "uint" not in dtype or "uint" in dtype


def edge_bounds(edge_index, source_count, target_count, *, chunk_size=100000, max_edges=None):
    issues = []
    dimensions = shape(edge_index)
    if dimensions is None or len(dimensions) != 2 or dimensions[0] != 2:
        return {
            "issues": ["edge_index_must_have_shape_2_by_E"],
            "checked_edges": 0,
            "total_edges": None,
        }
    if not _integer_array(edge_index):
        issues.append("edge_indices_must_be_integer")
    count = dimensions[1]
    checked = min(count, max_edges) if max_edges is not None else count
    bounds = []
    for start in range(0, checked, chunk_size):
        part = edge_index[:, start : min(start + chunk_size, checked)]
        if part.shape[1]:
            low, high = (
                [int(part[row].min()) for row in (0, 1)],
                [int(part[row].max()) for row in (0, 1)],
            )
            bounds.append((low, high))
            if low[0] < 0 or high[0] >= source_count or low[1] < 0 or high[1] >= target_count:
                issues.append("edge_endpoint_out_of_bounds")
    return {
        "issues": sorted(set(issues)),
        "checked_edges": checked,
        "total_edges": count,
        "index_min": [min(b[0][row] for b in bounds) for row in (0, 1)] if bounds else None,
        "index_max": [max(b[1][row] for b in bounds) for row in (0, 1)] if bounds else None,
    }


def validate_splits(splits, num_nodes):
    """Validate indices in chunks with a one-byte-per-node membership array."""
    seen = bytearray(num_nodes)
    issues, report = [], {}
    for number, (name, indices) in enumerate(splits.items(), 1):
        if len(shape(indices) or ()) != 1 or not _integer_array(indices):
            issues.append(f"invalid_split_shape_or_dtype:{name}")
            continue
        count = int(indices.shape[0])
        report[name] = count
        for start in range(0, count, 100000):
            for raw in indices[start : start + 100000]:
                index = int(raw)
                if not 0 <= index < num_nodes:
                    issues.append(f"split_index_out_of_bounds:{name}")
                elif seen[index] & (1 << (number - 1)):
                    issues.append(f"duplicate_split_index:{name}")
                elif seen[index]:
                    issues.append(f"overlapping_split_index:{name}")
                if 0 <= index < num_nodes:
                    seen[index] |= 1 << (number - 1)
        if len(issues) > 1000:
            issues = sorted(set(issues))
    if not {"train", "valid", "test"} <= splits.keys():
        issues.append("missing_official_split")
    return {"counts": report, "issues": sorted(set(issues)), "membership_memory_bytes": num_nodes}


class OGBNArxivAdapter:
    def iter_edges(self, dataset, provenance, *, max_edges=None):
        from ..staging import StagedCitationEdge

        graph, _ = dataset[0]
        edge_index = graph["edge_index"]
        count = int(edge_index.shape[1])
        years = graph.get("node_year")
        for index in range(min(count, max_edges) if max_edges is not None else count):
            source, target = int(edge_index[0, index]), int(edge_index[1, index])
            year = None
            if years is not None:
                value = years[source]
                if getattr(value, "shape", ()):
                    value = value[0]
                year = int(value)
            yield StagedCitationEdge(
                source_dataset="ogbn-arxiv",
                source_node_id=source,
                target_node_id=target,
                identifier_namespace="ogbn-arxiv-node-index",
                publication_year=year,
                provenance=provenance,
            )

    def inspect(self, dataset, *, max_edges=None):
        graph, labels = dataset[0]
        nodes = int(graph["num_nodes"])
        edges = edge_bounds(graph.get("edge_index"), nodes, nodes, max_edges=max_edges)
        issues = list(edges["issues"])
        features, years = graph.get("node_feat"), graph.get("node_year")
        for name, value in (("features", features), ("labels", labels), ("years", years)):
            dimensions = shape(value)
            if dimensions is None or not dimensions or dimensions[0] != nodes:
                issues.append(f"{name}_row_count_mismatch_or_missing")
        splits = validate_splits(dataset.get_idx_split(), nodes)
        issues.extend(splits["issues"])
        arrays = (graph.get("edge_index"), features, labels, years)
        memory = sum(int(getattr(value, "nbytes", 0)) for value in arrays if value is not None)
        return {
            "dataset_id": "ogbn-arxiv",
            "node_types": ["paper"],
            "edge_types": ["paper:cites:paper"],
            "node_count": nodes,
            "edge_count": edges["total_edges"],
            "edge_integrity": edges,
            "feature_shape": shape(features),
            "label_shape": shape(labels),
            "year_shape": shape(years),
            "year_range": [int(years.min()), int(years.max())]
            if years is not None and nodes
            else None,
            "official_splits": splits,
            "array_memory_bytes": memory,
            "direction": (
                "citing_to_cited; supplied edge order retained; official master "
                "add_inverse_edge=False"
            ),
            "identifier_namespace": "ogbn-arxiv-node-index",
            "citation_intent": "unavailable",
            "issues": sorted(set(issues)),
            "partial": max_edges is not None
            and edges["checked_edges"] < (edges["total_edges"] or 0),
        }

    def load(
        self,
        root: Path,
        *,
        offline=True,
        trusted_cache=False,
        memory_budget_bytes=512 * 1024**2,
        factory=None,
    ):
        if factory is not None:
            return factory(name="ogbn-arxiv", root=str(root))
        if find_spec("ogb") is None or find_spec("torch") is None:
            raise AcquisitionBlocked(
                "official OGB API requires optional ogb and torch; neither was installed "
                "for Stage 02"
            )
        directory = root / "ogbn_arxiv"
        if not (directory / "raw/edge.csv.gz").is_file():
            raise AcquisitionBlocked(
                "official OGB raw files absent; acquire verified release first (no "
                "implicit API download)"
            )
        if not (directory / "RELEASE_v1.txt").is_file():
            raise AcquisitionBlocked(
                "release marker absent; refusing OGB update prompt/deletion behavior"
            )
        if not trusted_cache:
            raise AcquisitionBlocked(
                "OGB API may deserialize cached objects; explicitly verified local cache required"
            )
        if memory_budget_bytes < 256 * 1024**2:
            raise AcquisitionBlocked(
                "configured graph memory budget is below conservative 256 MiB loading estimate"
            )
        from ogb.nodeproppred import NodePropPredDataset

        return NodePropPredDataset(name="ogbn-arxiv", root=str(root))

    def inspect_mapping(self, root: Path, *, max_rows=1000):
        files = sorted(root.rglob("*nodeidx*paper*.csv.gz"))
        if not files:
            return {"status": "unavailable", "files": [], "external_id_namespace": None}
        reports = []
        for path in files:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                count = 0
                for _ in reader:
                    count += 1
                    if count >= max_rows:
                        break
                reports.append(
                    {"path": str(path), "fields": reader.fieldnames, "inspected_rows": count}
                )
        return {
            "status": "observed_header_only",
            "files": reports,
            "external_id_namespace": "unverified; requires authoritative mapping semantics",
        }
