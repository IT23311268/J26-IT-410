"""H2GB MAGDataset boundary. No invented node schema, automatic processing, or pickle loading."""

from importlib.util import find_spec
from pathlib import Path

from ..download_manager import AcquisitionBlocked
from .ogbn_arxiv import edge_bounds, shape

SOURCE_URL = "https://github.com/SpaceLearner/H2GB"
ALTERNATIVE_SOURCE = "https://github.com/junhongmit/H2GB"
VERIFIED_CODE = (
    "https://raw.githubusercontent.com/junhongmit/H2GB/main/H2GB/datasets/mag_dataset.py"
)


def _get(store, key):
    return store.get(key) if hasattr(store, "get") else getattr(store, key, None)


class MAGYearAdapter:
    def iter_edges(self, dataset, provenance, *, max_edges=100000):
        from ..staging import StagedCitationEdge

        graph = dataset[0]
        emitted = 0
        for source, relation, target in graph.edge_types:
            edge_index = _get(graph[source, relation, target], "edge_index")
            if edge_index is None:
                continue
            for index in range(int(edge_index.shape[1])):
                if emitted >= max_edges:
                    return
                yield StagedCitationEdge(
                    source_dataset="mag-year-h2gb",
                    source_node_id=int(edge_index[0, index]),
                    target_node_id=int(edge_index[1, index]),
                    identifier_namespace="h2gb-mag-year-node-index",
                    edge_type=f"{source}:{relation}:{target}",
                    provenance=provenance,
                )
                emitted += 1

    def load(self, root: Path, *, offline=True, trusted_cache=False, factory=None):
        if factory is not None:
            return factory(root=str(root), name="mag-year", rand_split=False)
        missing = [
            name
            for name in ("torch", "torch_geometric", "torch_sparse", "H2GB", "ogb")
            if find_spec(name) is None
        ]
        if missing:
            raise AcquisitionBlocked(
                "MAG-Year optional dependencies unavailable: " + ", ".join(missing)
            )
        if not trusted_cache or not (root / "mag-year/processed/data.pt").exists():
            raise AcquisitionBlocked(
                "H2GB loads data.pt with torch.load; requires verified existing trusted "
                "cache; automatic processing disabled"
            )
        from H2GB.datasets import MAGDataset

        return MAGDataset(root=str(root), name="mag-year", rand_split=False)

    def inspect(self, dataset, *, max_edges=100000):
        graph = dataset[0]
        node_types = list(graph.node_types)
        edge_types = list(graph.edge_types)
        nodes, relations, issues = {}, [], []
        memory = 0
        for name in node_types:
            store = graph[name]
            count = int(store.num_nodes)
            features, labels, years = _get(store, "x"), _get(store, "y"), _get(store, "node_year")
            if years is None:
                years = _get(store, "year")
            masks = {}
            membership = bytearray(count)
            for index, field in enumerate(("train_mask", "val_mask", "test_mask"), 1):
                mask = _get(store, field)
                if mask is None:
                    continue
                if shape(mask) != (count,) or "bool" not in str(mask.dtype):
                    issues.append(f"invalid_split_mask:{name}:{field}")
                    continue
                selected = 0
                for start in range(0, count, 100000):
                    for offset, flag in enumerate(mask[start : start + 100000]):
                        if bool(flag):
                            selected += 1
                            if membership[start + offset]:
                                issues.append(f"overlapping_split_masks:{name}")
                            membership[start + offset] = index
                masks[field] = selected
            for field, value in (("features", features), ("labels", labels), ("years", years)):
                dimensions = shape(value)
                if dimensions is not None and dimensions[0] != count:
                    issues.append(f"row_count_mismatch:{name}:{field}")
                if value is not None:
                    memory += (
                        int(value.numel() * value.element_size())
                        if hasattr(value, "numel")
                        else int(getattr(value, "nbytes", 0))
                    )
            nodes[name] = {
                "count": count,
                "feature_shape": shape(features),
                "label_shape": shape(labels),
                "year_shape": shape(years),
                "splits": masks,
                "raw_paper_identifiers": "not verified",
            }
        for source, relation, target in edge_types:
            if source not in nodes or target not in nodes:
                issues.append("unknown_relation_endpoint_type")
                continue
            checked = edge_bounds(
                _get(graph[source, relation, target], "edge_index"),
                nodes[source]["count"],
                nodes[target]["count"],
                max_edges=max_edges,
            )
            issues.extend(checked["issues"])
            relations.append({"source": source, "relation": relation, "target": target, **checked})
        return {
            "dataset_id": "mag-year-h2gb",
            "node_types": node_types,
            "edge_types": [list(item) for item in edge_types],
            "nodes": nodes,
            "relations": relations,
            "issues": sorted(set(issues)),
            "array_memory_bytes_lower_bound": memory,
            "citation_intent": "unverified; not assumed",
            "identifier_namespace": "h2gb-mag-year-node-index",
            "partial": any(
                r["checked_edges"] < r["total_edges"]
                for r in relations
                if r["total_edges"] is not None
            ),
            "limitations": [
                "Code documentation is not observed dataset schema.",
                (
                    "Maintainer code symmetrizes some edges, adds reverse relations, derives "
                    "year labels and random splits; no prospective temporal safety is "
                    "assumed."
                ),
                "No cross-source integer mapping is assumed.",
            ],
        }
