"""
pipeline.py - the whole engine as ONE function.

Both front ends call this:
    main.py   - the command line (prints each stage as a table)
    app.py    - the Streamlit web page (shows each stage as it finishes)

so the two can never drift apart: whatever the command line does, the web
page does too.

    result = run_pipeline("data/input/sample_paper.pdf")
    result.docx_path          -> the IEEE Word file
    result.compliance         -> the before / after score

`on_stage` is an optional function that is called after every stage with
(stage_name, result_so_far). The command line uses it to print; the web page
uses it to tick the stage off on screen.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .compose import compose, make_llm
from .evaluate import evaluate
from .ingest import PackageError, load_document, read_package
from .placement import place_visuals
from .render import ieee_style, render_docx
from .rubric import build_rubric, save_rubric
from .structure import map_to_ieee, structure_document
from .visuals import export_visuals

# The stages in the order they run, with a short name for each.
STAGES = [
    ("ingest", "Read the file"),
    ("structure", "Rebuild the paper"),
    ("map", "Map to the IEEE skeleton"),
    ("compose", "Compose in IEEE style"),
    ("visuals", "Place figures and tables"),
    ("render", "Write the Word document"),
    ("score", "Score IEEE compliance"),
]


@dataclass
class PipelineResult:
    input_path: str
    outdir: str
    stem: str
    is_package: bool = False
    source_doc: object = None                 # ParsedDocument (None for a package)
    paper: object = None                      # StructuredPaper, finished
    source_paper: object = None               # StructuredPaper as it came in
    llm: object = None
    exported: Dict[str, int] = field(default_factory=dict)
    assets_dir: str = ""
    docx_path: str = ""
    render_notes: List[str] = field(default_factory=list)
    compliance: object = None                 # ComplianceReport
    rubric: List[object] = field(default_factory=list)    # RubricItem per rulebook rule
    annotated_path: str = ""                  # the Word copy with rule comments in the margin
    saved: List[Tuple[str, str]] = field(default_factory=list)


Callback = Optional[Callable[[str, PipelineResult], None]]


def run_pipeline(input_path: str, outdir: str = "data/output", offline: bool = False,
                 polish: bool = False, on_stage: Callback = None,
                 llm=None, paper_size: str = "letter") -> PipelineResult:
    """Run every stage on one input file.

    Raises FileNotFoundError / ValueError / PackageError for a bad input, and
    PermissionError if the Word file is open in Word.
    """
    def tell(stage: str) -> None:
        if on_stage:
            on_stage(stage, result)

    ieee_style.use_paper(paper_size)          # rule LAY-01: US Letter (default) or A4
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    result = PipelineResult(input_path=input_path, outdir=outdir, stem=stem)

    if input_path.lower().endswith(".json"):
        # Route B: verified content handed over by the other agents (TAF task 1)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"No such file: {input_path}")
        result.is_package = True
        result.paper = read_package(input_path)
        tell("ingest")
        tell("structure")
    else:
        # Route A: read the paper itself (stand-in for Member 1's ingestion)
        result.source_doc = load_document(input_path)
        tell("ingest")
        result.paper = structure_document(result.source_doc)
        lines_path = os.path.join(outdir, f"{stem}_stage2_labelled_lines.json")
        result.source_doc.to_json(lines_path)
        result.saved.append((lines_path, "every line with its label"))
        tell("structure")

    result.paper = map_to_ieee(result.paper)
    result.source_paper = copy.deepcopy(result.paper)     # kept for the "before" score
    tell("map")

    result.llm = llm if llm is not None else make_llm(offline=offline)
    result.paper = compose(result.paper, result.llm, polish=polish)
    tell("compose")

    place_visuals(result.paper)
    result.assets_dir = os.path.join(outdir, f"{stem}_assets")
    result.exported = export_visuals(result.paper, input_path, result.assets_dir)
    tell("visuals")

    result.docx_path = os.path.join(outdir, f"{stem}_IEEE.docx")
    result.render_notes = render_docx(result.paper, result.docx_path)
    paper_path = os.path.join(outdir, f"{stem}_ieee_paper.json")
    result.paper.to_json(paper_path)                     # after rendering: all receipts included
    result.saved.append((paper_path, "the finished paper, as data"))
    tell("render")

    result.compliance = evaluate(result.source_doc, result.source_paper,
                                 result.paper, result.docx_path)
    compliance_path = os.path.join(outdir, f"{stem}_compliance.json")
    result.compliance.to_json(compliance_path)
    result.saved.append((compliance_path, "the compliance score, check by check"))
    # the rubric: every rulebook rule, whether it applies, where it was applied
    result.rubric = build_rubric(result.paper, result.compliance, result.is_package)
    rubric_path = os.path.join(outdir, f"{stem}_rubric.json")
    save_rubric(result.rubric, rubric_path)
    result.saved.append((rubric_path, "the IEEE rubric: every rule, where it was applied"))
    result.annotated_path = os.path.join(outdir, f"{stem}_IEEE_annotated.docx")
    render_docx(result.paper, result.annotated_path, annotate=True)
    result.saved.append((result.annotated_path, "the manuscript with each IEEE rule as a "
                                                 "margin comment"))
    if any(result.exported.values()):
        result.saved.append((result.assets_dir, "the figure, table and equation images"))
    tell("score")
    return result


__all__ = ["run_pipeline", "PipelineResult", "STAGES", "PackageError"]
