"""
app.py - Step 7: the web page (front end) for the IEEE Manuscript Composition Engine.

Start it with:
    streamlit run app.py

Your browser opens at http://localhost:8501. Pick a paper (or drop your own),
press "Compose IEEE manuscript", watch the seven stages tick off, then see
the compliance score, the section mapping, where every figure went, the IEEE
references, and download the Word file.

This page does NOT contain the engine. It calls run_pipeline() from
src/pipeline.py - exactly the same code main.py runs on the command line -
and only shows the results. The HTML pieces are built in src/ui/views.py and
the dark theme lives in src/ui/theme.css and .streamlit/config.toml.
"""

from __future__ import annotations

import os
import re
import traceback

import streamlit as st

from src.compose.llm import NoLLM
from src.pipeline import PackageError, run_pipeline
from src.ui import views as V

ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(ROOT, "data", "input")
UPLOAD_DIR = os.path.join(ROOT, "data", "uploads")
# IEEE_OUTPUT_DIR lets the tests write somewhere else
OUTPUT_DIR = os.environ.get("IEEE_OUTPUT_DIR") or os.path.join(ROOT, "data", "output")
UPLOAD_CHOICE = "Upload my own file"
INPUT_TYPES = ("pdf", "docx", "json")


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def gemini_key_found() -> bool:
    """True if a real GEMINI_API_KEY is in .env (the key itself is never shown)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
    except ImportError:
        pass
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return bool(key) and key != "your-key-here"


def papers_in_input_folder() -> list:
    """Every paper in data/input - drop a PDF there and it shows up here."""
    if not os.path.isdir(INPUT_DIR):
        return []
    names = [n for n in os.listdir(INPUT_DIR) if n.lower().endswith(INPUT_TYPES)]
    first = {"sample_paper.pdf": 0, "sample_draft.docx": 1, "sample_verified_content.json": 2}
    return sorted(names, key=lambda n: (first.get(n, 3), n.lower()))


def describe(name: str) -> str:
    known = {
        "sample_paper.pdf": "Messy 3-page sample paper with a figure and a table",
        "sample_draft.docx": "Word draft using heading styles",
        "sample_verified_content.json": "Verified content package from the other agents",
    }
    return known.get(name, {".pdf": "PDF paper", ".docx": "Word document",
                            ".json": "Content package"}[os.path.splitext(name)[1].lower()])


def save_upload(uploaded) -> str:
    """Write the uploaded file into data/uploads and return its path."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    name = re.sub(r"[^\w.\-]+", "_", os.path.basename(uploaded.name)) or "upload.pdf"
    path = os.path.join(UPLOAD_DIR, name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="IEEE Manuscript Engine", page_icon="📄",
                   layout="wide", initial_sidebar_state="auto")
html(V.theme_css())

state = st.session_state
state.setdefault("result", None)        # the last finished run
state.setdefault("stages", {})          # stage -> one-line fact
state.setdefault("error", None)

# ---------------------------------------------------------------------------
# Sidebar: settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Settings")
    key_ok = gemini_key_found()
    use_gemini = st.toggle(
        "Use Gemini", value=key_ok, disabled=not key_ok,
        help="Gemini reads the references and drafts the abstract and index terms. "
             "Every answer is checked against the paper before it is used.")
    if key_ok:
        html('<p class="key-ok">Gemini key found in .env</p>')
    else:
        html('<p class="key-missing">No Gemini key in .env. The engine uses its rules only.</p>')
    polish = st.toggle(
        "Polish the wording", value=False, disabled=not use_gemini,
        help="Gemini rewrites each section in IEEE style. A rewrite is kept only if its "
             "numbers and citations are unchanged. Uses one request per section.")
    paper_size = st.radio(
        "Paper size", ["letter", "a4"], horizontal=True,
        format_func=lambda k: {"letter": "US Letter", "a4": "A4"}[k],
        help="From the IEEE rulebook, rule LAY-01. Most IEEE conferences use US Letter.")
    html("""<div class="side-id">
        <b>IEEE Manuscript Composition Engine</b><br>
        Component 4 of project J26-IT-410,<br>An AI Research Agent Pipeline Using RAG<br><br>
        Senarath S.P.J.H, IT21245442
        </div>""")

# ---------------------------------------------------------------------------
# Header and input
# ---------------------------------------------------------------------------

html("""<header class="masthead">
  <h1>IEEE Manuscript Composition Engine</h1>
  <p>Turns an unstructured research paper into an IEEE manuscript: sections mapped to the
  IEEE skeleton, references and citations in IEEE style, every figure and table placed
  after the paragraph that first mentions it, and a compliance score to prove it.</p>
</header>""")

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("#### Choose a paper")
    choices = [UPLOAD_CHOICE] + papers_in_input_folder()
    choice = st.radio(
        "Paper", choices, index=1 if len(choices) > 1 else 0, label_visibility="collapsed",
        captions=["PDF, Word or a verified content package (.json)"]
                 + [describe(n) for n in choices[1:]])
    input_path = None
    if choice == UPLOAD_CHOICE:
        uploaded = st.file_uploader("Drop a paper here", type=list(INPUT_TYPES))
        if uploaded is not None:
            input_path = save_upload(uploaded)
    else:
        input_path = os.path.join(INPUT_DIR, choice)

    run = st.button("Compose IEEE manuscript", type="primary", width="stretch",
                    disabled=input_path is None)

with right:
    st.markdown("#### Stages")
    stage_box = st.empty()
    stage_box.markdown(V.stage_list(state.stages), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Running the engine
# ---------------------------------------------------------------------------

if run and input_path:
    done = {}
    current = {"stage": "ingest"}                  # the stage that is running now
    state.result, state.error = None, None
    stage_box.markdown(V.stage_list(done, running="ingest"), unsafe_allow_html=True)

    def on_stage(stage, result):
        done[stage] = V.stage_fact(stage, result)
        current["stage"] = V.next_stage(stage)
        stage_box.markdown(V.stage_list(done, running=current["stage"]),
                           unsafe_allow_html=True)

    try:
        # Gemini off -> rules only (None means: let the engine make the Gemini client)
        llm = None if use_gemini else NoLLM(
            "Gemini is switched off in Settings" if key_ok else "no GEMINI_API_KEY in .env")
        state.result = run_pipeline(input_path, outdir=OUTPUT_DIR, polish=polish and use_gemini,
                                    on_stage=on_stage, llm=llm, paper_size=paper_size)
    except PermissionError as error:
        state.error = (f"Cannot write {os.path.basename(error.filename or 'the Word file')}. "
                       "It is probably open in Word: close it, then compose again.", None)
    except (FileNotFoundError, ValueError, PackageError) as error:
        state.error = (str(error), None)
    except Exception as error:                     # anything unexpected: show the details
        state.error = (f"The engine stopped with an unexpected error: {error}",
                       traceback.format_exc())
    state.stages = done
    if state.error:                                # show where it stopped
        stage_box.markdown(V.stage_list(done, failed=current["stage"]), unsafe_allow_html=True)

if state.error:
    message, details = state.error
    st.error(message)
    if details:
        with st.expander("Technical details"):
            st.code(details)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

r = state.result
if r is not None:
    paper, report = r.paper, r.compliance
    st.divider()
    st.markdown(f"## {paper.title or os.path.basename(r.input_path)}")
    html(V.score_track(report.before_score, report.after_score))
    html(V.headline_facts(r))

    # downloads
    cols = st.columns(4)
    with open(r.docx_path, "rb") as f:
        cols[0].download_button("Download Word file", f.read(), file_name=os.path.basename(r.docx_path),
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                type="primary", width="stretch", on_click="ignore")
    images_zip = V.zip_folder(r.assets_dir)
    cols[1].download_button("Figure images (.zip)", images_zip or b"", file_name=f"{r.stem}_assets.zip",
                            mime="application/zip", width="stretch", on_click="ignore",
                            disabled=images_zip is None)
    compliance_json = os.path.join(r.outdir, f"{r.stem}_compliance.json")
    with open(compliance_json, "rb") as f:
        cols[2].download_button("Compliance report (.json)", f.read(), file_name=os.path.basename(compliance_json),
                                mime="application/json", width="stretch", on_click="ignore")
    paper_json = os.path.join(r.outdir, f"{r.stem}_ieee_paper.json")
    with open(paper_json, "rb") as f:
        cols[3].download_button("Paper data (.json)", f.read(), file_name=os.path.basename(paper_json),
                                mime="application/json", width="stretch", on_click="ignore")
    html(f'<p class="quiet">Also saved on this computer in {V.esc(os.path.relpath(r.outdir, ROOT))}.</p>')

    cards = V.visual_cards(paper)
    tabs = st.tabs(["Compliance checks", "Section mapping",
                    f"Figures and tables ({len(cards)})",
                    f"References ({len(paper.references)})", "What changed",
                    f"IEEE rulebook ({len(V.R.RULES)})"])

    with tabs[0]:
        html('<p class="quiet">Each check is run on the paper as uploaded '
             '(<span style="color:var(--amber)">before</span>) and on the IEEE manuscript '
             '(<span style="color:var(--cobalt)">after</span>). Checks that cannot be measured '
             'for this input show n/a and are left out of the average.</p>')
        html(V.compliance_rows(report))

    with tabs[1]:
        html('<p class="quiet">How the author\'s sections were matched onto the IEEE skeleton, '
             'and why.</p>')
        html(V.mapping_table(paper))

    with tabs[2]:
        summary = paper.placement_summary or {}
        b, a = summary.get("before", {}), summary.get("after", {})
        if not cards:
            html('<p class="quiet">This paper has no figures or tables to place.</p>')
        else:
            if b.get("mean_distance") is not None:
                m = st.columns(3)
                m[0].metric("Paragraphs to first mention", a["mean_distance"],
                            delta=round(a["mean_distance"] - b["mean_distance"], 2),
                            delta_color="inverse", help="Average distance; 0 means right after it.")
                m[1].metric("After first mention", f"{a['after_first_mention']}%",
                            delta=f"{a['after_first_mention'] - b['after_first_mention']} points")
                m[2].metric("Directly after first mention", f"{a['adjacent']}%",
                            delta=f"{a['adjacent'] - b['adjacent']} points")
            grid = st.columns(3, gap="medium")
            for i, card in enumerate(cards):
                with grid[i % 3]:
                    if card["image"]:
                        st.image(card["image"], width="stretch")
                    else:
                        why = ("the visual area could not be found on the page"
                               if r.input_path.lower().endswith(".pdf")
                               else "images are copied from PDF papers only")
                        html(f'<div class="vis-missing">No image: {why}, so the Word file '
                             f'shows a placeholder.</div>')
                    moved = card["method"].startswith(("after its", "semantic"))
                    html(f'<p class="vis-label">{V.esc(card["label"])}</p>'
                         f'<p class="vis-meta"><span class="{"moved" if moved else ""}">'
                         f'{V.esc(card["method"].capitalize())}</span><br>'
                         f'{V.esc(V.distance_text(card))}</p>')
        equations = V.equation_images(paper)
        if equations:
            with st.expander(f"Equations copied as images ({len(equations)})"):
                for path in equations:
                    st.image(path)

    with tabs[3]:
        if paper.references:
            html(V.reference_list(paper))
        else:
            html('<p class="quiet">No references were found.</p>')

    with tabs[4]:
        st.markdown("##### Composition")
        html(V.bullet_list(paper.compose_report or ["Nothing to report."]))
        llm = r.llm
        if getattr(llm, "calls_made", 0) or getattr(llm, "cache_hits", 0):
            html(f'<p class="quiet">Gemini requests: {llm.calls_made} new, '
                 f'{llm.cache_hits} reused from the cache.</p>')
        elif not getattr(llm, "available", False):
            html(f'<p class="quiet">Rules only: {V.esc(getattr(llm, "reason", ""))}.</p>')
        if paper.ai_log:
            st.markdown("##### Written or changed by AI or automatic extraction")
            html(V.bullet_list(paper.ai_log))
        notes = list(r.render_notes or []) + list(paper.warnings or [])
        if notes:
            st.markdown("##### Notes")
            html(V.bullet_list(notes))

    with tabs[5]:
        counts = V.R.summary()
        html(f'<p class="quiet">Every IEEE rule the engine follows, from '
             f'<code>src/rules/ieee_rules.py</code>. <b>{counts["enforced"]}</b> are applied '
             f'automatically, <b>{counts["checked"]}</b> are checked and reported, and '
             f'<b>{counts["author"]}</b> need the author\'s judgement. The compliance checks above '
             f'show which rule each one tests.</p>')
        html(V.rulebook_table())
