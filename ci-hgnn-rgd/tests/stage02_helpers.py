"""Miniature SYNTHETIC graph interfaces; not real graph observations or benchmarks."""


class Array:
    def __init__(self, values, dtype="int64"):
        self.values, self.dtype = values, dtype
        self.shape = (
            (len(values), len(values[0]))
            if values and isinstance(values[0], list)
            else (len(values),)
        )
        self.nbytes = len(values) * (len(values[0]) if len(self.shape) == 2 else 1) * 8

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, columns = key
            selected = self.values[rows]
            if isinstance(rows, int):
                return (
                    Array(selected[columns], self.dtype)
                    if isinstance(columns, slice)
                    else selected[columns]
                )
            return Array([row[columns] for row in selected], self.dtype)
        result = self.values[key]
        return (
            Array(result, self.dtype)
            if isinstance(key, slice) or isinstance(result, list)
            else result
        )

    def __iter__(self):
        return iter(self.values)

    def min(self):
        return (
            min(v for row in self.values for v in row) if len(self.shape) == 2 else min(self.values)
        )

    def max(self):
        return (
            max(v for row in self.values for v in row) if len(self.shape) == 2 else max(self.values)
        )


class OGBFixture:
    def __init__(self):
        self.graph = {
            "num_nodes": 3,
            "edge_index": Array([[0, 1], [1, 2]]),
            "node_feat": Array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], "float32"),
            "node_year": Array([[2020], [2021], [2022]]),
        }
        self.labels = Array([[0], [1], [0]])
        self.splits = {"train": Array([0]), "valid": Array([1]), "test": Array([2])}

    def __getitem__(self, index):
        assert index == 0
        return self.graph, self.labels

    def get_idx_split(self):
        return self.splits


class Store(dict):
    def __init__(self, num_nodes, **values):
        super().__init__(values)
        self.num_nodes = num_nodes


class HeteroFixture:
    node_types = ["paper", "author"]
    edge_types = [("author", "writes", "paper")]

    def __init__(self):
        self.stores = {
            "paper": Store(
                2,
                x=Array([[1.0], [2.0]], "float32"),
                y=Array([0, 1]),
                train_mask=Array([True, False], "bool"),
                val_mask=Array([False, True], "bool"),
                test_mask=Array([False, False], "bool"),
            ),
            "author": Store(1),
            ("author", "writes", "paper"): {"edge_index": Array([[0], [1]])},
        }

    def __getitem__(self, key):
        return self.stores[key]


def scicite_record(**updates):
    values = {
        "id": "synthetic-record",
        "unique_id": "synthetic-annotation",
        "string": "SYNTHETIC citation context for software testing only.",
        "label": "background",
        "citingPaperId": "synthetic-citing-id",
        "citedPaperId": "synthetic-cited-id",
    }
    return {**values, **updates}


def arxiv_record(**updates):
    values = {
        "id": "2001.00001v2",
        "title": "SYNTHETIC paper fixture",
        "abstract": "SYNTHETIC abstract; not scholarly evidence.",
        "authors": "Synthetic Author",
        "categories": "cs.AI",
        "doi": "https://doi.org/10.1234/SYNTHETIC",
        "versions": [{"version": "v1", "created": "Wed, 01 Jan 2020 00:00:00 GMT"}],
        "update_date": "2020-01-02",
    }
    return {**values, **updates}
