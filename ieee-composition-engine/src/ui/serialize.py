"""
serialize.py - turn a finished engine run (PipelineResult) into JSON for the React page.

Nothing here changes the paper or the scores: it only reads what the engine
produced (run_pipeline in src/pipeline.py) and lays it out for the browser.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, List

from ..rubric.rubric import rubric_score, summary
from ..rules import ieee_rules as R
from . import views as V


def _asset_url(job_id: str, path: str) -> str:
    return f"/api/jobs/{job_id}/asset/{os.path.basename(path)}" if path else ""


def rulebook() -> dict:
    return {"summary": R.summary(),
            "rules": [{"id": r.id, "group": r.group, "rule": r.rule, "value": r.value,
                       "status": r.status, "source": R.SOURCES[r.source][0],
                       "source_url": R.SOURCES[r.source][1], "section": r.section,
                       "check": r.check} for r in R.RULES]}


def serialize(result, job_id: str) -> dict:
    paper, report = result.paper, result.compliance
    summary_ = paper.placement_summary or {}

    mapping = []
    for s in paper.sections:
        original = f"{s.number} {s.heading}".strip() or "(text before the first heading)"
        if s.level == 1:
            ieee = (f"{s.ieee_number}. " if s.ieee_number else "") + s.ieee_heading.upper()
        else:
            ieee = f"{s.ieee_number}. {s.ieee_heading}"
        mapping.append({"original": original, "ieee": ieee, "level": s.level,
                        "confidence": round(s.mapping_confidence or 0, 2) if s.level == 1 else None,
                        "reason": s.mapping_reason if s.level == 1 else ""})

    visuals = []
    for card in V.visual_cards(paper):
        visuals.append({"label": card["label"], "method": card["method"],
                        "moved": card["method"].startswith(("after its", "semantic")),
                        "distance": V.distance_text(card),
                        "distance_before": card.get("distance_before"),
                        "distance_after": card.get("distance_after"),
                        "image": _asset_url(job_id, card["image"]) if card["image"] else "",
                        "caption": card["caption"]})

    records = paper.reference_records or []
    references = []
    for n, ref in enumerate(paper.references, start=1):
        record = records[n - 1] if n - 1 < len(records) else {}
        references.append({"n": n, "segments": record.get("segments") or [[ref, False]],
                           "source": record.get("source", "")})

    llm = result.llm
    downloads = {
        "docx": f"/api/jobs/{job_id}/file/docx",
        "annotated": f"/api/jobs/{job_id}/file/annotated" if result.annotated_path else "",
        "compliance": f"/api/jobs/{job_id}/file/compliance",
        "paper": f"/api/jobs/{job_id}/file/paper",
        "images": f"/api/jobs/{job_id}/file/images" if any(result.exported.values()) else "",
    }

    rubric_items = [asdict(i) for i in result.rubric]
    ticked, applicable, pct = rubric_score(result.rubric)
    return {
        "job_id": job_id,
        "input_name": os.path.basename(result.input_path),
        "is_package": result.is_package,
        "title": paper.title or os.path.basename(result.input_path),
        "authors": paper.authors,
        "score": {"before": report.before_score, "after": report.after_score},
        "placement": summary_,
        "references_count": len(paper.references),
        "exported": {k: v for k, v in (result.exported or {}).items() if v},
        "compliance": [{**asdict(c)} for c in report.checks],
        "mapping": mapping,
        "missing_sections": paper.missing_sections,
        "visuals": visuals,
        "equations": [_asset_url(job_id, p) for p in V.equation_images(paper)],
        "references": references,
        "compose_report": paper.compose_report or [],
        "ai_log": paper.ai_log or [],
        "notes": list(result.render_notes or []) + list(paper.warnings or []),
        "llm": {"name": getattr(llm, "name", ""), "available": getattr(llm, "available", False),
                "reason": getattr(llm, "reason", ""), "calls": getattr(llm, "calls_made", 0),
                "cache_hits": getattr(llm, "cache_hits", 0)},
        "rubric": rubric_items,
        "rubric_summary": summary(result.rubric),
        "rubric_suggested": {"ticked": ticked, "applicable": applicable, "percent": pct},
        "rulebook": rulebook(),
        "downloads": downloads,
    }


def stage_list(done: Dict[str, str], running: str = "", failed: str = "") -> List[dict]:
    out = []
    for key, title in V.STAGES:
        state = "done" if key in done else "failed" if key == failed else \
            "running" if key == running else "waiting"
        out.append({"key": key, "title": title, "state": state, "note": done.get(key, "")})
    return out
