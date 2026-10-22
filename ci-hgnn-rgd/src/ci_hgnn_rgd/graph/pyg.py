"""CPU HeteroData tensors serialized as safetensors plus JSON, never executable pickle."""

import json
from pathlib import Path

from ..data.manifests import atomic_json
from ..data.provenance import file_sha256
from .inputs import document
from .temporal import assessment, split_visibility


def require_pyg():
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError as error:
        raise RuntimeError(
            "Stage 06 requires the optional graph extra (torch-geometric)"
        ) from error
    return torch, HeteroData


def make_heterodata(store, bundle, config):
    import numpy as np

    torch, HeteroData = require_pyg()
    data = HeteroData()
    encoder, matrix = {}, None
    if bundle.embedding_run:
        encoder = document(bundle.embedding_run / "embedding_model_manifest.json")["encoder"]
        matrix = np.load(
            bundle.embedding_run / "paper_embeddings.npy", mmap_mode="r", allow_pickle=False
        )
        expected = store.aux("embedding_rows", "count")["count"]
        if matrix.ndim != 2 or matrix.shape != (expected, encoder.get("dimension")):
            matrix._mmap.close()
            raise ValueError("embedding dimension/row count differs from its manifest/index")
        if matrix.dtype != np.float32:
            matrix._mmap.close()
            raise ValueError("graph scientific embeddings must have recorded float32 dtype")
    dimension = encoder.get("dimension", 0)
    n_papers = store.db.execute("SELECT COUNT(*) FROM nodes WHERE kind='paper'").fetchone()[0]
    n_edges = store.db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    n_nodes = store.db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    estimate = n_papers * (dimension * 4 + 40) + n_edges * 80 + n_nodes * 48
    if estimate > config.features.max_tensor_bytes:
        if matrix is not None:
            matrix._mmap.close()
        raise ValueError("graph tensors exceed configured byte budget; reduce graph filters")
    feature_counts = {
        "embedding_present": 0,
        "embedding_missing": 0,
        "embedding_excluded_temporal": 0,
        "embedding_dimension": dimension,
        "encoder": encoder,
        "placeholder_semantics": "zero vectors are missing, use has_embedding mask",
        "normalization": "upstream recorded encoder normalization only; no fitted scaler",
    }
    kinds = {kind for (kind,) in store.db.execute("SELECT DISTINCT kind FROM nodes")} | {"paper"}
    try:
        for kind in sorted(kinds):
            count = store.db.execute("SELECT COUNT(*) FROM nodes WHERE kind=?", (kind,)).fetchone()[
                0
            ]
            target = data[kind]
            target.num_nodes = count
            target.original_index = torch.arange(count, dtype=torch.long)
            target.temporal_eligible = torch.ones(count, dtype=torch.bool)
            target.x = torch.zeros(
                (count, dimension if kind == "paper" else 1), dtype=torch.float32
            )
            for split in config.temporal.split_cutoffs:
                target[split + "_mask"] = torch.zeros(count, dtype=torch.bool)
                if kind == "paper":
                    target[split + "_embedding_mask"] = torch.zeros(count, dtype=torch.bool)
            if kind == "paper":
                target.has_embedding = torch.zeros(count, dtype=torch.bool)
                target.text_available = torch.zeros(count, dtype=torch.bool)
                target.publication_year = torch.zeros(count, dtype=torch.float32)
                target.has_publication_year = torch.zeros(count, dtype=torch.bool)
            for node in store.node_rows(kind):
                idx, key = node["graph_index"], node["stable_id"]
                for split, visible in node.get("split_visibility", {}).items():
                    target[split + "_mask"][idx] = visible
                if kind == "paper":
                    target.text_available[idx] = node["text_available"]
                    year = node.get("publication_year")
                    if config.features.include_publication_year and year is not None:
                        target.publication_year[idx] = year
                        target.has_publication_year[idx] = True
                    feature = store.db.execute(
                        "SELECT row_number,payload FROM features WHERE key=?", (key,)
                    ).fetchone()
                    temporal = None
                    if feature:
                        temporal = assessment(
                            node,
                            bundle,
                            kind="feature",
                            produced_at=bundle.embedding_manifest.get("created_at"),
                        )
                    if feature and temporal["eligible"]:
                        number = feature[0]
                        if not 0 <= number < len(matrix):
                            raise ValueError("embedding row is out of matrix bounds")
                        vector = np.array(matrix[number], copy=True)
                        if not np.isfinite(vector).all():
                            raise ValueError("non-finite scientific feature vector")
                        target.x[idx] = torch.from_numpy(vector)
                        target.has_embedding[idx] = True
                        store.evidence(
                            "feature_provenance",
                            key,
                            {
                                "stable_id": key,
                                "embedding_run": str(bundle.embedding_run),
                                "index": json.loads(feature[1]),
                                "encoder": encoder,
                                "temporal_eligibility": temporal,
                                "split_visibility": split_visibility(
                                    node,
                                    bundle,
                                    config,
                                    kind="feature",
                                    produced_at=bundle.embedding_manifest.get("created_at"),
                                ),
                            },
                        )
                        for split, visible in split_visibility(
                            node,
                            bundle,
                            config,
                            kind="feature",
                            produced_at=bundle.embedding_manifest.get("created_at"),
                        ).items():
                            target[split + "_embedding_mask"][idx] = visible
                        feature_counts["embedding_present"] += 1
                    else:
                        feature_counts["embedding_missing"] += 1
                        if feature:
                            feature_counts["embedding_excluded_temporal"] += 1
                        if config.features.missing_embedding_strategy == "error":
                            raise ValueError("paper embedding missing or not temporally eligible")
                elif kind in {"author", "topic"}:
                    rel = "authored_by" if kind == "author" else "associated_with_topic"
                    value = store.db.execute(
                        "SELECT COUNT(*) FROM edges WHERE rel=? AND t=?", (rel, key)
                    ).fetchone()[0]
                    target.x[idx, 0] = value
                    if kind == "author":
                        for split in config.temporal.split_cutoffs:
                            eligible = store.db.execute(
                                "SELECT 1 FROM edges WHERE rel='authored_by' AND t=? AND "
                                "json_extract(payload,?)=1 LIMIT 1",
                                (key, "$.split_visibility." + split),
                            ).fetchone()
                            target[split + "_mask"][idx] = bool(eligible)
                else:
                    # No fabricated vector: this column records an observed node's presence.
                    target.x[idx, 0] = 1
        for sk, relation, tk, count in store.db.execute(
            "SELECT sk,rel,tk,COUNT(*) FROM edges GROUP BY sk,rel,tk ORDER BY sk,rel,tk"
        ):
            target = data[sk, relation, tk]
            target.edge_index = torch.empty((2, count), dtype=torch.long)
            target.edge_attr = torch.zeros((count, 6), dtype=torch.float32)
            target.missing_intent = torch.ones(count, dtype=torch.bool)
            target.derived = torch.zeros(count, dtype=torch.bool)
            for split in config.temporal.split_cutoffs:
                target[split + "_mask"] = torch.zeros(count, dtype=torch.bool)
                target[split + "_intent"] = torch.zeros((count, 3), dtype=torch.float32)
                target[split + "_context_count"] = torch.zeros(count, dtype=torch.float32)
                target[split + "_missing_intent"] = torch.ones(count, dtype=torch.bool)
            for i, edge in enumerate(store.edge_rows(relation)):
                target.edge_index[:, i] = torch.tensor([edge["source_index"], edge["target_index"]])
                target.derived[i] = edge["derived"]
                target.edge_attr[i, :3] = torch.tensor(
                    edge.get("intent_probabilities", [0.0, 0.0, 0.0])
                )
                target.edge_attr[i, 3] = edge.get("context_count", 0)
                score = edge.get("assignment_score")
                if score is not None:
                    target.edge_attr[i, 4] = score
                    target.edge_attr[i, 5] = 1
                target.missing_intent[i] = edge.get("missing_intent", True)
                for split, visible in edge.get("split_visibility", {}).items():
                    target[split + "_mask"][i] = visible
                for split in config.temporal.split_cutoffs:
                    probabilities = edge.get("split_intent_probabilities", {}).get(
                        split, [0.0, 0.0, 0.0]
                    )
                    is_edge_intent = config.citation_intent_representation != "context_nodes"
                    target[split + "_intent"][i] = torch.tensor(
                        probabilities if is_edge_intent else [0.0, 0.0, 0.0]
                    )
                    count = edge.get("split_context_counts", {}).get(split, 0)
                    target[split + "_context_count"][i] = count
                    target[split + "_missing_intent"][i] = not count or not is_edge_intent
        feature_counts["tensor_bytes"] = sum(
            t.numel() * t.element_size()
            for storage in data.stores
            for t in storage.values()
            if isinstance(t, torch.Tensor)
        )
        denominator = feature_counts["embedding_present"] + feature_counts["embedding_missing"]
        feature_counts["embedding_missing_fraction"] = (
            feature_counts["embedding_missing"] / denominator if denominator else None
        )
        feature_counts["publication_year_missing"] = int(
            (~data["paper"].has_publication_year).sum()
        )
        feature_counts["scientific_text_missing"] = int((~data["paper"].text_available).sum())
        if feature_counts["tensor_bytes"] > config.features.max_tensor_bytes:
            raise ValueError("actual tensor payload exceeds graph byte budget")
        feature_counts["edge_attribute_columns"] = [
            "prob_background",
            "prob_method",
            "prob_result",
            "context_count",
            "score",
            "has_score",
        ]
        feature_counts["score_comparability"] = (
            "scores retain per-edge method semantics in sidecars"
        )
        data.schema_version = "stage06-v1"
        data.snapshot_id = bundle.snapshot_id
        return data, feature_counts
    finally:
        if matrix is not None:
            matrix._mmap.close()


def validate_heterodata(data):
    torch, _ = require_pyg()
    if not data.validate(raise_on_error=False):
        raise ValueError("PyG graph index/type validation failed")
    for kind in data.node_types:
        storage = data[kind]
        if storage.x.ndim != 2 or storage.x.shape[0] != storage.num_nodes:
            raise ValueError("node feature rows are misaligned")
        if not torch.isfinite(storage.x).all():
            raise ValueError("non-finite node feature")
        for name, value in storage.items():
            if isinstance(value, torch.Tensor) and value.shape[0] != storage.num_nodes:
                raise ValueError(f"node feature/mask alignment mismatch: {kind}/{name}")
        if kind == "paper":
            if storage.has_embedding.dtype != torch.bool:
                raise ValueError("paper missing-feature mask must be boolean")
            if torch.any(storage.x[~storage.has_embedding] != 0):
                raise ValueError("missing embedding placeholder must be a masked zero vector")
    for key in data.edge_types:
        storage = data[key]
        index = storage.edge_index
        if index.dtype != torch.long or index.ndim != 2 or index.shape[0] != 2:
            raise ValueError("edge_index must be int64 with shape [2,E]")
        if storage.edge_attr.ndim != 2 or storage.edge_attr.shape[1] != 6:
            raise ValueError("edge attributes must retain the documented six columns")
        for name, value in storage.items():
            if name != "edge_index" and isinstance(value, torch.Tensor):
                if len(value) != index.shape[1] or (
                    value.is_floating_point() and not torch.isfinite(value).all()
                ):
                    raise ValueError("edge attribute alignment/non-finite value")
        probs = storage.edge_attr[:, :3]
        present = ~storage.missing_intent
        if torch.any(probs[~present] != 0):
            raise ValueError("missing intent must retain a zero placeholder vector")
        if (
            torch.any(probs < 0)
            or torch.any(probs > 1)
            or not torch.allclose(probs[present].sum(1), torch.ones(int(present.sum())), atol=1e-5)
        ):
            raise ValueError("citation probability vectors are misaligned/invalid")
    return {
        "status": "PASSED",
        "node_stores": len(data.node_types),
        "edge_stores": len(data.edge_types),
        "pyg_validate": True,
    }


def export_heterodata(data, directory: Path):
    from safetensors.torch import save_file

    validate_heterodata(data)
    directory.mkdir(parents=True, exist_ok=True)
    tensors, stores = {}, []
    for i, key in enumerate([*data.node_types, *data.edge_types]):
        storage = data[key]
        attributes = {}
        for name, value in storage.items():
            if hasattr(value, "numel"):
                tensor_key = f"store_{i}__{name}"
                tensors[tensor_key] = value.contiguous().cpu()
                attributes[name] = tensor_key
        stores.append(
            {
                "key": list(key) if isinstance(key, tuple) else key,
                "num_nodes": storage.num_nodes if isinstance(key, str) else None,
                "attributes": attributes,
            }
        )
    save_file(tensors, str(directory / "heterodata.safetensors"))
    atomic_json(
        directory / "heterodata.json",
        {
            "schema_version": "stage06-v1",
            "snapshot_id": data.snapshot_id,
            "stores": stores,
            "serialization": "safetensors_plus_json_no_pickle",
        },
    )
    return str(directory / "heterodata.safetensors")


def load_hetero_graph(graph_version, snapshot_id=None, *, root=None, masked_relations=()):
    from safetensors.torch import load_file

    _, HeteroData = require_pyg()
    graph = Path(graph_version) if root is None else Path(root) / graph_version
    graph = graph.resolve()
    if root and not graph.is_relative_to(Path(root).resolve()):
        raise ValueError("graph version escapes configured output root")
    if graph.name.startswith(".pending-") or (graph / "failure.json").exists():
        raise ValueError("graph is not atomically published")
    manifest = document(graph / "manifests/graph_manifest.json")
    if manifest.get("status") != "COMPLETE" or manifest.get("schema_version") != "stage06-v1":
        raise ValueError("unsupported or incomplete graph manifest")
    if snapshot_id and snapshot_id != manifest["snapshot_id"]:
        raise ValueError("requested graph snapshot differs from published graph")
    for relative, digest in manifest["artifact_checksums"].items():
        path = (graph / relative).resolve()
        if not path.is_relative_to(graph) or file_sha256(path) != digest:
            raise ValueError("graph artifact checksum/path mismatch")
    metadata = document(graph / "tensors/heterodata.json")
    tensors = load_file(str(graph / "tensors/heterodata.safetensors"), device="cpu")
    data = HeteroData()
    for store in metadata["stores"]:
        key = tuple(store["key"]) if isinstance(store["key"], list) else store["key"]
        relation = key[1] if isinstance(key, tuple) else None
        # Reverse edges must also be hidden to avoid revealing a held-out target.
        original = manifest.get("reverse_relations", {}).get(relation, relation)
        if "cites" in masked_relations and original in {
            "originates_from",
            "refers_to",
            "expresses_intent_about",
        }:
            continue
        if relation in masked_relations or original in masked_relations:
            continue
        if isinstance(key, str):
            data[key].num_nodes = store["num_nodes"]
        for name, tensor_key in store["attributes"].items():
            data[key][name] = tensors[tensor_key]
    data.schema_version, data.snapshot_id = metadata["schema_version"], metadata["snapshot_id"]
    for kind, relation in (("author", "authored_by"), ("topic", "associated_with_topic")):
        if kind in data.node_types and relation in masked_relations:
            data[kind].x.zero_()
    validate_heterodata(data)
    return data


def temporal_view(data, split: str):
    """A non-mutating visibility view; zero all hidden features and remove hidden edges."""
    result = data.clone()
    key = split + "_mask"
    for kind in result.node_types:
        storage = result[kind]
        if key not in storage:
            raise ValueError("split cutoff visibility was not constructed for this graph")
        visible = storage[key]
        storage.x[~visible] = 0
        if kind == "paper":
            feature_visible = visible & storage.get(split + "_embedding_mask", visible)
            storage.x[~feature_visible] = 0
            storage.has_embedding &= feature_visible
            storage.publication_year[~visible] = 0
            storage.has_publication_year &= visible
    for edge_type in result.edge_types:
        storage = result[edge_type]
        visible = storage.get(key)
        if visible is None:
            raise ValueError("edge split visibility unavailable")
        visible = (
            visible
            & result[edge_type[0]][key][storage.edge_index[0]]
            & (result[edge_type[2]][key][storage.edge_index[1]])
        )
        for name, tensor in list(storage.items()):
            storage[name] = tensor[:, visible] if name == "edge_index" else tensor[visible]
        if split + "_intent" in storage:
            storage.edge_attr[:, :3] = storage[split + "_intent"]
            storage.edge_attr[:, 3] = storage[split + "_context_count"]
            storage.missing_intent = storage[split + "_missing_intent"]
    result = result.subgraph({kind: result[kind][key] for kind in result.node_types})
    for storage in result.stores:
        for name in list(storage):
            if any(name.startswith(period + "_") for period in ("train", "dev", "test")):
                del storage[name]
    for kind, relation in (("author", "authored_by"), ("topic", "associated_with_topic")):
        if kind in result.node_types:
            result[kind].x.zero_()
            edge_type = ("paper", relation, kind)
            if edge_type in result.edge_types:
                index = result[edge_type].edge_index[1]
                result[kind].x[:, 0].scatter_add_(0, index, result[kind].x.new_ones(len(index)))
    validate_heterodata(result)
    return result
