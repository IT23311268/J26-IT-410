"""Bounded JSONL events, archive streams, prefix limits, and deterministic reservoir sampling."""

import hashlib
import json
import random
import tarfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class RecordEvent:
    source: str
    line: int
    split: str | None
    sha256: str
    record: dict | None
    error: str | None = None


def jsonl_events(stream, source, split=None, *, max_line_bytes=4 * 1024**2):
    line_number = 0
    while raw := stream.readline(max_line_bytes + 1):
        line_number += 1
        digest = hashlib.sha256(raw)
        if len(raw) > max_line_bytes:
            while not raw.endswith(b"\n"):
                raw = stream.readline(max_line_bytes)
                if not raw:
                    break
                digest.update(raw)
            yield RecordEvent(
                source, line_number, split, digest.hexdigest(), None, "line_exceeds_size_limit"
            )
            continue
        if not raw.strip():
            continue
        try:

            def unique_pairs(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("duplicate JSON field")
                    result[key] = value
                return result

            constants = []

            def invalid_constant(value, constants=constants):
                constants.append(value)
                return float(value)

            record = json.loads(
                raw.decode("utf-8-sig"),
                object_pairs_hook=unique_pairs,
                parse_constant=invalid_constant,
            )
            if not isinstance(record, dict):
                raise ValueError("expected a JSON object per line")
            yield RecordEvent(
                source,
                line_number,
                split,
                digest.hexdigest(),
                record,
                "nonstandard_json_constant" if constants else None,
            )
        except (UnicodeError, ValueError, RecursionError) as error:
            yield RecordEvent(
                source,
                line_number,
                split,
                digest.hexdigest(),
                None,
                "invalid_encoding"
                if isinstance(error, UnicodeError)
                else "malformed_json_or_object",
            )


def _split(name):
    stem = Path(name).stem.lower()
    return stem if stem in {"train", "dev", "valid", "validation", "test"} else None


@contextmanager
def source_streams(path: Path, dataset: str, *, max_expanded_bytes=1024**3):
    """No extraction or pickle. Archive member names are retained in provenance."""
    if path.is_dir():
        paths = sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl"}
        )
        if dataset == "scicite":
            paths = [p for p in paths if _split(p.name) is not None]
        if not paths:
            raise ValueError("no supported source JSONL files found")

        def directory_iterator():
            for item in paths:
                with item.open("rb") as stream:
                    yield str(item), _split(item.name), stream

        yield directory_iterator()
    elif tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            if sum(member.size for member in members) > max_expanded_bytes:
                raise ValueError("archive expansion exceeds inspection budget")

            def tar_iterator():
                count = 0
                for member in members:
                    if PurePosixPath(member.name).suffix not in {".json", ".jsonl"}:
                        continue
                    split = _split(member.name)
                    if dataset == "scicite" and split is None:
                        continue  # Auxiliary scaffold taxonomies are not the SciCite intent task.
                    count += 1
                    with archive.extractfile(member) as stream:
                        yield f"{path}!{member.name}", split, stream
                if not count:
                    raise ValueError("no principal JSONL split files in archive")

            yield tar_iterator()
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            if sum(entry.file_size for entry in entries) > max_expanded_bytes:
                raise ValueError("archive expansion exceeds inspection budget")

            def zip_iterator():
                for member in entries:
                    if PurePosixPath(member.filename).suffix in {".json", ".jsonl"}:
                        with archive.open(member) as stream:
                            yield f"{path}!{member.filename}", _split(member.filename), stream

            yield zip_iterator()
    else:
        with path.open("rb") as stream:
            yield iter(((str(path), _split(path.name), stream),))


def events(path, dataset, *, max_line_bytes=4 * 1024**2):
    with source_streams(Path(path), dataset) as streams:
        for source, split, stream in streams:
            yield from jsonl_events(stream, source, split, max_line_bytes=max_line_bytes)


class Selection:
    def __init__(self, *, max_records=None, sample_size=None, seed=42):
        if (
            max_records is not None
            and max_records < 1
            or sample_size is not None
            and sample_size < 1
        ):
            raise ValueError("record limits and sample sizes must be positive")
        self.max_records, self.sample_size, self.seed = max_records, sample_size, seed
        self.seen = 0
        self.exhausted = False

    def select(self, records):
        if self.sample_size is None:
            for record in records:
                if self.max_records is not None and self.seen >= self.max_records:
                    return
                self.seen += 1
                yield record
            self.exhausted = True
            return
        reservoir = []
        rng = random.Random(self.seed)
        for record in records:
            if self.max_records is not None and self.seen >= self.max_records:
                break
            self.seen += 1
            if len(reservoir) < self.sample_size:
                reservoir.append(record)
            elif (index := rng.randrange(self.seen)) < self.sample_size:
                reservoir[index] = record
        else:
            self.exhausted = True
        yield from sorted(reservoir, key=lambda item: (item.source, item.line))

    def describe(self):
        return {
            "strategy": "reservoir"
            if self.sample_size
            else "prefix"
            if self.max_records
            else "full",
            "seed": self.seed,
            "max_records": self.max_records,
            "sample_size": self.sample_size,
            "records_scanned": self.seen,
            "input_exhausted": self.exhausted,
            "total_records": self.seen if self.exhausted else None,
        }
