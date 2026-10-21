"""Directly observed schema frequencies, bounded nested exploration and field count."""

import math
import re
from collections import Counter, defaultdict


class SchemaInspector:
    def __init__(self, *, max_fields=512, max_depth=4):
        self.max_fields, self.max_depth = max_fields, max_depth
        self.objects = 0
        self.present = Counter()
        self.empty = Counter()
        self.types = defaultdict(Counter)
        self.date_formats = Counter()
        self.truncated = False

    def observe(self, record):
        self.objects += 1

        def visit(value, prefix="", depth=0):
            if depth > self.max_depth:
                return
            if isinstance(value, dict):
                for key, child in value.items():
                    path = f"{prefix}.{key}" if prefix else key
                    if path not in self.types and len(self.types) >= self.max_fields:
                        self.truncated = True
                        continue
                    self.types[path][type(child).__name__] += 1
                    if not prefix:
                        self.present[path] += 1
                        if (
                            child is None
                            or child == ""
                            or child == []
                            or isinstance(child, float)
                            and not math.isfinite(child)
                        ):
                            self.empty[path] += 1
                    if isinstance(child, str) and ("date" in key.lower() or key == "created"):
                        pattern = (
                            "ISO_date"
                            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", child)
                            else "RFC_like_datetime"
                            if "GMT" in child
                            else "unclassified_string"
                        )
                        self.date_formats[path + ":" + pattern] += 1
                    visit(child, path, depth + 1)
            elif isinstance(value, list):
                for child in value[:16]:
                    visit(child, prefix + "[]", depth + 1)

        visit(record)

    def report(self):
        return {
            "parsed_objects": self.objects,
            "observed_date_formats": dict(self.date_formats),
            "fields": {
                key: {
                    "types": dict(self.types[key]),
                    "present": self.present[key],
                    "absent": self.objects - self.present[key],
                    "null_or_empty": self.empty[key],
                    "completeness": (self.present[key] - self.empty[key]) / self.objects
                    if self.objects
                    else None,
                }
                for key in sorted(self.present)
            },
            "nested_observations": {
                key: dict(counts)
                for key, counts in sorted(self.types.items())
                if key not in self.present
            },
            "bounded_nested_inspection": {
                "max_fields": self.max_fields,
                "max_depth": self.max_depth,
                "list_elements_per_record": 16,
                "truncated": self.truncated,
            },
        }
