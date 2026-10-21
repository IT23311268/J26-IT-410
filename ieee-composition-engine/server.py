"""
server.py - starts the React web page for the IEEE Manuscript Composition Engine.

    python server.py

Opens http://localhost:8000 in your browser.

What this file does: it is a small web API around the engine. The React page
(frontend/) runs in the browser and cannot run Python itself, so it asks this
server to do the work. The server only CALLS the existing engine - the same
run_pipeline() that main.py and the Streamlit page use - and sends back the
results as JSON. No engine logic lives here.

    GET  /api/settings                 Gemini key found? papers in data/input
    POST /api/upload?filename=x.pdf    save an uploaded paper in data/uploads
    POST /api/jobs                     start composing a paper -> {job_id}
    GET  /api/jobs/{id}                progress: the seven stages, live
    GET  /api/jobs/{id}/result         everything the page shows, as JSON
    GET  /api/jobs/{id}/preview        the IEEE manuscript with one rule highlighted
    GET  /api/jobs/{id}/file/{kind}    downloads: docx, annotated, compliance, paper, images
    POST /api/jobs/{id}/rubric.xlsx    the filled rubric, with the reviewer's answers
    GET  /api/jobs/{id}/asset/{name}   a figure / table / equation image

The built React page (frontend/dist) is served at "/".
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
import traceback
import uuid
import webbrowser
from typing import Dict

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import (FileResponse, HTMLResponse, JSONResponse, Response)
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from src.compose.llm import NoLLM
from src.pipeline import PackageError, run_pipeline
from src.ui import views as V
from src.ui.highlights import highlights_for
from src.ui.preview import build_preview, preview_images
from src.ui.serialize import rulebook, serialize, stage_list

ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(ROOT, "data", "input")
UPLOAD_DIR = os.path.join(ROOT, "data", "uploads")
OUTPUT_DIR = os.environ.get("IEEE_OUTPUT_DIR") or os.path.join(ROOT, "data", "output")
DIST = os.path.join(ROOT, "frontend", "dist")
INPUT_TYPES = (".pdf", ".docx", ".json")
PORT = int(os.environ.get("IEEE_PORT", "8000"))

JOBS: Dict[str, dict] = {}
RUN_LOCK = threading.Lock()          # one paper at a time (the paper size setting is shared)

DESCRIPTIONS = {
    "sample_paper.pdf": "Messy 3-page sample paper with a figure and a table",
    "sample_draft.docx": "Word draft using heading styles",
    "sample_verified_content.json": "Verified content package from the other agents",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gemini_key_found() -> bool:
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
    except ImportError:
        pass
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return bool(key) and key != "your-key-here"


def papers() -> list:
    if not os.path.isdir(INPUT_DIR):
        return []
    names = [n for n in os.listdir(INPUT_DIR) if n.lower().endswith(INPUT_TYPES)]
    first = {"sample_paper.pdf": 0, "sample_draft.docx": 1, "sample_verified_content.json": 2}
    names.sort(key=lambda n: (first.get(n, 3), n.lower()))
    kinds = {".pdf": "PDF paper", ".docx": "Word document", ".json": "Content package"}
    return [{"name": n, "description": DESCRIPTIONS.get(n, kinds[os.path.splitext(n)[1].lower()])}
            for n in names]


def resolve_input(name: str) -> str:
    """'sample_paper.pdf' -> data/input/..., 'uploads/x.pdf' -> data/uploads/x.pdf.
    Only plain file names are accepted, never paths."""
    folder = INPUT_DIR
    if name.startswith("uploads/"):
        folder, name = UPLOAD_DIR, name[len("uploads/"):]
    if not name or os.path.basename(name) != name:
        raise ValueError("Unknown paper.")
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {name}")
    return path


def error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def job_or_404(request: Request):
    job = JOBS.get(request.path_params["job_id"])
    return job


# ---------------------------------------------------------------------------
# Running a paper (in the background, so the page can show live progress)
# ---------------------------------------------------------------------------

def run_job(job_id: str, path: str, use_gemini: bool, polish: bool, paper_size: str) -> None:
    job = JOBS[job_id]
    done: Dict[str, str] = {}
    current = {"stage": "ingest"}

    def on_stage(stage, result):
        done[stage] = V.stage_fact(stage, result)
        current["stage"] = V.next_stage(stage) or ""
        job["stages"] = stage_list(done, running=current["stage"])

    job["stages"] = stage_list(done, running="ingest")
    try:
        with RUN_LOCK:
            llm = None if use_gemini else NoLLM(
                "Gemini is switched off in Settings" if gemini_key_found()
                else "no GEMINI_API_KEY in .env")
            result = run_pipeline(path, outdir=OUTPUT_DIR, polish=polish and use_gemini,
                                  on_stage=on_stage, llm=llm, paper_size=paper_size)
            job["result"] = result
            job["images"] = preview_images(result.paper)
            job["json"] = serialize(result, job_id)
        job["status"] = "done"
    except PermissionError as exc:
        job.update(status="error", error=f"Cannot write {os.path.basename(exc.filename or 'the Word file')}. "
                                         "It is probably open in Word: close it, then compose again.")
    except (FileNotFoundError, ValueError, PackageError) as exc:
        job.update(status="error", error=str(exc))
    except Exception as exc:                            # anything unexpected: show the details
        job.update(status="error", error=f"The engine stopped with an unexpected error: {exc}",
                   details=traceback.format_exc())
    if job["status"] == "error":
        job["stages"] = stage_list(done, failed=current["stage"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

async def api_settings(request: Request):
    return JSONResponse({"gemini_key": gemini_key_found(), "papers": papers(),
                         "paper_sizes": [{"key": "letter", "label": "US Letter"},
                                         {"key": "a4", "label": "A4"}]})


async def api_rules(request: Request):
    return JSONResponse(rulebook())


async def api_upload(request: Request):
    filename = request.query_params.get("filename", "")
    name = re.sub(r"[^\w.\-]+", "_", os.path.basename(filename))
    if not name.lower().endswith(INPUT_TYPES):
        return error("Upload a PDF, a Word file (.docx) or a content package (.json).")
    data = await request.body()
    if not data:
        return error("The file is empty.")
    if len(data) > 60 * 1024 * 1024:
        return error("The file is larger than 60 MB.")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with open(os.path.join(UPLOAD_DIR, name), "wb") as f:
        f.write(data)
    return JSONResponse({"name": f"uploads/{name}", "label": name})


async def api_start(request: Request):
    body = await request.json()
    try:
        path = resolve_input(str(body.get("paper", "")))
    except (FileNotFoundError, ValueError) as exc:
        return error(str(exc))
    use_gemini = bool(body.get("gemini")) and gemini_key_found()
    paper_size = body.get("paper_size") if body.get("paper_size") in ("letter", "a4") else "letter"
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "stages": stage_list({}, running="ingest"),
                    "error": "", "details": ""}
    threading.Thread(target=run_job, daemon=True,
                     args=(job_id, path, use_gemini, bool(body.get("polish")), paper_size)).start()
    return JSONResponse({"job_id": job_id})


async def api_job(request: Request):
    job = job_or_404(request)
    if job is None:
        return error("Unknown job.", 404)
    return JSONResponse({"status": job["status"], "stages": job["stages"],
                         "error": job.get("error", ""), "details": job.get("details", "")})


async def api_result(request: Request):
    job = job_or_404(request)
    if job is None or job["status"] != "done":
        return error("The result is not ready.", 404)
    return JSONResponse(job["json"])


async def api_preview(request: Request):
    job = job_or_404(request)
    if job is None or job["status"] != "done":
        return HTMLResponse("<p>Not ready.</p>", status_code=404)
    result = job["result"]
    items = {i.id: i for i in result.rubric}
    item = items.get(request.query_params.get("rule", ""))
    if item is None:
        page = build_preview(result.paper, [], job["images"], theme="light")
        return HTMLResponse(page)
    place = int(request.query_params.get("place", "0") or 0)
    place = max(0, min(place, max(0, len(item.evidence) - 1)))
    e = item.evidence[place] if item.evidence else None
    banner = f"{item.id}: {e['where']}" if e else f"{item.id}: look at the highlighted part"
    if e and (e.get("anchor") or "document") == "document":
        banner = f"{item.id} applies to the whole page: {e.get('after') or e.get('before')}"
    return HTMLResponse(build_preview(result.paper, highlights_for(item, place), job["images"],
                                      banner=banner, theme="light"))


async def api_file(request: Request):
    job = job_or_404(request)
    if job is None or job["status"] != "done":
        return error("Not ready.", 404)
    r = job["result"]
    kind = request.path_params["kind"]
    paths = {
        "docx": r.docx_path,
        "annotated": r.annotated_path,
        "compliance": os.path.join(r.outdir, f"{r.stem}_compliance.json"),
        "paper": os.path.join(r.outdir, f"{r.stem}_ieee_paper.json"),
    }
    if kind == "images":
        data = V.zip_folder(r.assets_dir)
        if data is None:
            return error("No images.", 404)
        return Response(data, media_type="application/zip", headers={
            "Content-Disposition": f'attachment; filename="{r.stem}_assets.zip"'})
    path = paths.get(kind)
    if not path or not os.path.exists(path):
        return error("No such file.", 404)
    return FileResponse(path, filename=os.path.basename(path))


async def api_rubric_xlsx(request: Request):
    job = job_or_404(request)
    if job is None or job["status"] != "done":
        return error("Not ready.", 404)
    from src.rubric.export import rubric_workbook
    r = job["result"]
    body = await request.json()
    decisions = {k: bool(v) for k, v in (body.get("decisions") or {}).items()}
    ticks = {i.id: decisions.get(i.id, False) for i in r.rubric}
    data = rubric_workbook(r.rubric, ticks, r.paper.title, r.compliance.after_score,
                           reviewed=decisions)
    return Response(data, media_type="application/vnd.openxmlformats-officedocument."
                                     "spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{r.stem}_IEEE_rubric.xlsx"'})


async def api_asset(request: Request):
    job = job_or_404(request)
    if job is None or job["status"] != "done":
        return error("Not ready.", 404)
    name = os.path.basename(request.path_params["name"])
    path = os.path.join(job["result"].assets_dir, name)
    if not name.lower().endswith(".png") or not os.path.exists(path):
        return error("No such image.", 404)
    return FileResponse(path, media_type="image/png")


async def no_frontend(request: Request):
    return HTMLResponse("<h2>The React page has not been built.</h2><p>Run "
                        "<code>cd frontend</code>, <code>npm install</code>, "
                        "<code>npm run build</code>, then start <code>python server.py</code> "
                        "again.</p>", status_code=503)


routes = [
    Route("/api/settings", api_settings),
    Route("/api/rules", api_rules),
    Route("/api/upload", api_upload, methods=["POST"]),
    Route("/api/jobs", api_start, methods=["POST"]),
    Route("/api/jobs/{job_id}", api_job),
    Route("/api/jobs/{job_id}/result", api_result),
    Route("/api/jobs/{job_id}/preview", api_preview),
    Route("/api/jobs/{job_id}/file/{kind}", api_file),
    Route("/api/jobs/{job_id}/rubric.xlsx", api_rubric_xlsx, methods=["POST"]),
    Route("/api/jobs/{job_id}/asset/{name}", api_asset),
]
if os.path.isdir(DIST):
    routes.append(Mount("/", app=StaticFiles(directory=DIST, html=True), name="frontend"))
else:
    routes.append(Route("/", no_frontend))

app = Starlette(routes=routes)


if __name__ == "__main__":
    import uvicorn
    url = f"http://localhost:{PORT}"
    print(f"\n  IEEE Manuscript Composition Engine\n  Open {url} in your browser."
          "\n  Press Ctrl+C to stop.\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
