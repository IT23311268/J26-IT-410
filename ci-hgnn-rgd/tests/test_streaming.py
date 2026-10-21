import io
import json
import tracemalloc

from ci_hgnn_rgd.data.streaming import Selection, jsonl_events


def test_streams_individual_lines_and_invalid_encoding():
    events = list(jsonl_events(io.BytesIO(b'{"id":"synthetic"}\n\xff\n'), "synthetic.jsonl"))
    assert events[0].record == {"id": "synthetic"}
    assert events[1].error == "invalid_encoding"


def test_overlong_line_is_consumed_without_materializing_it():
    events = list(
        jsonl_events(io.BytesIO(b"x" * 10000 + b'\n{"id":1}\n'), "synthetic", max_line_bytes=100)
    )
    assert len(events) == 2
    assert events[0].error == "line_exceeds_size_limit"
    assert events[1].record == {"id": 1}


def test_reservoir_memory_is_bounded_by_sample_not_stream_size():
    class RepeatedLines:
        def __init__(self, count):
            self.count = count

        def readline(self, limit):
            if not self.count:
                return b""
            self.count -= 1
            return b'{"id":"SYNTHETIC fixture", "title":"synthetic text"}\n'

    tracemalloc.start()
    selection = Selection(sample_size=16, seed=42)
    selected = list(selection.select(jsonl_events(RepeatedLines(10000), "synthetic")))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(selected) == 16
    assert selection.describe()["total_records"] == 10000
    assert peak < 2_000_000  # Python allocations only; not an empirical dataset memory benchmark.


def test_duplicate_json_keys_are_explicitly_invalid():
    event = next(jsonl_events(io.BytesIO(b'{"id":1,"id":2}\n'), "synthetic"))
    assert event.record is None and event.error == "malformed_json_or_object"


def test_scaffold_files_are_not_misrepresented_as_intent_splits(tmp_path):
    import tarfile

    from ci_hgnn_rgd.data.streaming import events

    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name in ("scicite/train.jsonl", "scicite/scaffolds/sections-scaffold-train.jsonl"):
            payload = json.dumps({"string": "SYNTHETIC", "label": "background"}).encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            stream.addfile(info, io.BytesIO(payload))
    assert len(list(events(archive, "scicite"))) == 1
