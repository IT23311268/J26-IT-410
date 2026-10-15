"""
package_reader.py - TAF task 1: "Receive verified content from other AI agents".

In the full group pipeline, this component does NOT read PDFs itself:
  * Member 1 (ingestion) extracts section-aware text and figures,
  * Member 3 (knowledge graph) adds cross-paper insights,
  * Member 2 (grounding gatekeeper) verifies everything with Ragas,
and the verified result is handed to this engine as a CONTENT PACKAGE.

This file defines that package - a JSON file - and turns it into the same
StructuredPaper the PDF route produces. Everything after this (mapping,
composition, rendering) works the same whichever route the content came in by.

The format (only "title" and "sections" are required):

{
  "schema": "ieee-composition/verified-content/v1",
  "title": "Paper title",
  "authors": ["K. M. Perera", "S. Fernando"],
  "affiliations": ["University of Colombo, Sri Lanka"],
  "abstract": "...",
  "keywords": ["lung cancer", "deep learning"],
  "sections": [
    {"heading": "Background", "level": 1,
     "content": [
        {"type": "paragraph", "text": "...", "verified": true},
        {"type": "figure", "number": 1, "caption": "System overview.", "image": "figures/fig1.png"},
        {"type": "table",  "number": 1, "caption": "Detection results."},
        {"type": "equation", "text": "L = a + b (1)"}
     ]}
  ],
  "references": ["Sung, H., ... (2021). Global cancer statistics ...", "..."],
  "verification": {"verified_by": "Member 2 - Ragas", "faithfulness": 0.94}
}

Any content item marked "verified": false is LEFT OUT and reported - this
engine composes only content the gatekeeper has approved.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from ..models import Element, Section, StructuredPaper

SCHEMA = "ieee-composition/verified-content/v1"


class PackageError(ValueError):
    """The content package is not in the expected format."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PackageError(f"Content package problem: {message}")


def _item_to_element(item: Dict[str, Any], base_dir: str) -> Element:
    kind = item.get("type", "paragraph")
    if kind == "paragraph":
        return Element(kind="paragraph", text=str(item.get("text", "")).strip())
    if kind in ("figure", "table"):
        number = item.get("number", "")
        label = "Figure" if kind == "figure" else "Table"
        caption = str(item.get("caption", "")).strip()
        image = item.get("image", "")
        if image and not os.path.isabs(image):
            image = os.path.join(base_dir, image)
        return Element(kind="caption", text=f"{label} {number}. {caption}".strip(), image=image)
    if kind == "equation":
        return Element(kind="equation", text=str(item.get("text", "")).strip())
    if kind == "code":
        return Element(kind="code", text=str(item.get("text", "")))
    raise PackageError(f"Content package problem: unknown content type '{kind}'")


def read_package(path: str) -> StructuredPaper:
    """Load a verified content package (.json) as a StructuredPaper."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as error:
        raise PackageError(f"Content package is not valid JSON: {error}") from error

    _require(isinstance(data, dict), "the file must contain one JSON object")
    _require(bool(data.get("title")), '"title" is missing')
    _require(isinstance(data.get("sections"), list) and data["sections"],
             '"sections" must be a non-empty list')
    if data.get("schema") not in (None, SCHEMA):
        raise PackageError(f'Content package problem: unsupported schema "{data["schema"]}" '
                           f'(expected "{SCHEMA}")')

    base_dir = os.path.dirname(os.path.abspath(path))
    paper = StructuredPaper(
        source_path=path,
        title=str(data["title"]).strip(),
        authors=[str(a) for a in data.get("authors", [])],
        affiliations=[str(a) for a in data.get("affiliations", [])],
        abstract=str(data.get("abstract", "")).strip(),
        keywords=[str(k) for k in data.get("keywords", [])],
        references=[str(r) for r in data.get("references", [])],
    )

    held_back: List[str] = []
    for i, raw in enumerate(data["sections"], start=1):
        _require(isinstance(raw, dict) and raw.get("heading") is not None,
                 f"section {i} needs a \"heading\"")
        level = int(raw.get("level", 1))
        _require(level in (1, 2, 3), f"section {i} has level {level}; use 1, 2 or 3")
        section = Section(heading=str(raw["heading"]).strip(), number=str(raw.get("number", "")),
                          level=level, in_appendix=bool(raw.get("appendix", False)))
        for item in raw.get("content", []):
            if item.get("verified") is False:
                held_back.append(f'"{section.heading}": '
                                 f'{str(item.get("text") or item.get("caption", ""))[:50]}...')
                continue
            element = _item_to_element(item, base_dir)
            if element.text or element.image:
                section.elements.append(element)
        paper.sections.append(section)

    paper.label_counts = {"sections": len(paper.sections),
                          "content items": sum(len(s.elements) for s in paper.sections)}
    if held_back:
        paper.warnings.append(f"{len(held_back)} item(s) marked as NOT verified by the grounding "
                              "gatekeeper were left out: " + "; ".join(held_back[:3]))
    verification = data.get("verification")
    if isinstance(verification, dict) and verification:
        summary = ", ".join(f"{k}: {v}" for k, v in verification.items())
        paper.ai_log.append(f"Content received as verified package ({summary}).")
    return paper
