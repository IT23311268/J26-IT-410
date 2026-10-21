"""
preview.py - the finished IEEE manuscript as a web page, with one rule's places highlighted.

The rubric walkthrough (rubric_tab.py) shows this next to each rule, so the
reviewer can SEE exactly where the rule was applied in the output before
ticking it. It is built from the same finished paper the Word file is built
from, in the same order (ordered_sections), with the same labels
("Fig. 1.", "TABLE I", "Abstract—", "Index Terms—").

A highlight is a dict:
    anchor   where:  "element:u12"  a paragraph / caption / equation
                     "section:u3"   a heading
                     "reference:7"  one reference;  "references"  the list
                     "front:title" | "front:authors" | "front:abstract" | "front:index_terms"
                     "document"     the whole page
                     "all:paragraph" | "all:heading" | "all:caption" | "all:figure" |
                     "all:table" | "all:equation" | "all:visual"        every block of a kind
                     "all:citation" | "all:mention" | "all:number" | "all:acronym"
                                                                     every match of a pattern
                     "slot:acknowledgment" | "slot:appendix"        those sections
    text     the exact words to mark inside that place ("" = mark the whole place)
    segment  for a reference: "authors" | "title" | "venue" | "tail"
    current  True for the place the reviewer is looking at (scrolled into view)
    related / related_text   a second place to mark, e.g. the paragraph that
             mentions a moved figure
"""

from __future__ import annotations

import base64
import html as html_lib
import io
import os
import re
from typing import Dict, List, Optional, Tuple

from ..placement.contextual import caption_label
from ..render import ieee_style as S
from ..render.docx_renderer import CAPTION_PARTS, EQUATION_NUMBER, ordered_sections
from ..rubric.evidence import anchor_of
from ..structure.ieee_mapper import to_roman

_UNITS = "|".join(re.escape(u) for u in sorted(__import__("src.rules.ieee_rules", fromlist=["x"])
                                               .UNITS, key=len, reverse=True))
PATTERNS = {
    "all:citation": r"\[\d+\]",
    "all:mention": r"\b(?:Figs?\.|Tables?|Algorithm)\s+[\dIVXLC]+(?:\([a-z]\))?",
    "all:number": rf"(?<![\w.])\d+(?:\.\d+)?\s?(?:%|{_UNITS})(?![A-Za-z])|(?<![\w.])0\.\d+",
    "all:acronym": r"\b[A-Z]{2,}s?\b",
}
KIND_ANCHORS = {"all:paragraph": "paragraph", "all:heading": "heading", "all:caption": "caption",
                "all:figure": "figure", "all:table": "table", "all:equation": "equation",
                "all:visual": "visual"}
IMAGE_WIDTH = 760


def esc(text) -> str:
    return html_lib.escape(str(text if text is not None else ""), quote=True)


def dom_id(anchor: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", anchor)


# ---------------------------------------------------------------------------
# Images: shrunk and embedded, so the preview works offline and fast
# ---------------------------------------------------------------------------

def preview_images(paper) -> Dict[str, str]:
    """uid -> data URI for every figure, table and equation image."""
    from PIL import Image

    out: Dict[str, str] = {}
    for s in paper.sections:
        for e in s.elements:
            if not e.image or not os.path.exists(e.image):
                continue
            try:
                with Image.open(e.image) as im:
                    im = im.convert("RGB")
                    if im.width > IMAGE_WIDTH:
                        im = im.resize((IMAGE_WIDTH, int(im.height * IMAGE_WIDTH / im.width)))
                    buffer = io.BytesIO()
                    im.save(buffer, "JPEG", quality=82)
                out[anchor_of(e)] = "data:image/jpeg;base64," + \
                    base64.b64encode(buffer.getvalue()).decode()
            except Exception:
                continue
    return out


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------

class _Preview:
    def __init__(self, paper, highlights: List[dict], images: Dict[str, str], banner: str,
                 theme: str = "dark"):
        self.theme = theme
        self.paper = paper
        self.images = images
        self.banner = banner
        self.current_placed = False
        self.at: Dict[str, List[dict]] = {}
        self.related: Dict[str, List[dict]] = {}
        self.kinds: Dict[str, bool] = {}          # block kind -> current?
        self.patterns: List[Tuple[str, bool]] = []
        self.slots: Dict[str, bool] = {}
        self.page = False
        for h in highlights:
            a = h.get("anchor") or "document"
            if a == "document":
                self.page = True
            elif a in KIND_ANCHORS:
                self.kinds[KIND_ANCHORS[a]] = self.kinds.get(KIND_ANCHORS[a], False) or h.get("current")
            elif a in PATTERNS:
                self.patterns.append((PATTERNS[a], bool(h.get("current"))))
            elif a.startswith("slot:"):
                self.slots[a[5:]] = self.slots.get(a[5:], False) or bool(h.get("current"))
            else:
                self.at.setdefault(a, []).append(h)
            if h.get("related"):
                self.related.setdefault(h["related"], []).append(
                    {"text": h.get("related_text", ""), "current": h.get("current")})

    # -- marking ------------------------------------------------------------

    def _current_id(self) -> str:
        if self.current_placed:
            return ""
        self.current_placed = True
        return ' id="hl-current"'

    def marked(self, anchor: str, text: str, pattern_scope: bool = False) -> Tuple[str, bool, bool]:
        """Escape `text` and wrap the highlighted words in <mark>.
        Returns (html, outline_whole_block?, block_is_current?)."""
        spans: List[Tuple[int, int, str, bool]] = []
        outline, current = False, False

        def add(needle: str, cls: str, cur: bool) -> bool:
            needle = (needle or "").strip().rstrip("…").strip()
            if not needle:
                return False
            found = False
            low_text, low_needle = text.lower(), needle.lower()
            start = 0
            while True:
                i = text.find(needle, start)
                if i < 0:
                    i = low_text.find(low_needle, start)
                if i < 0:
                    break
                spans.append((i, i + len(needle), cls, cur))
                found = True
                start = i + len(needle)
            return found

        for h in self.at.get(anchor, []):
            if not add(h.get("text", ""), "hl", bool(h.get("current"))):
                outline = True
                current = current or bool(h.get("current"))
        for r in self.related.get(anchor, []):
            if not add(r["text"], "hl-rel", False):
                outline = True
        if pattern_scope:
            for pattern, cur in self.patterns:
                for m in re.finditer(pattern, text):
                    spans.append((m.start(), m.end(), "hl", cur))

        # non-overlapping, left to right
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        out, pos = [], 0
        for start, end, cls, cur in spans:
            if start < pos:
                continue
            out.append(esc(text[pos:start]))
            ident = self._current_id() if cur else ""
            out.append(f'<mark class="{cls}{" hl-cur" if cur else ""}"{ident}>'
                       f'{esc(text[start:end])}</mark>')
            pos = end
        out.append(esc(text[pos:]))
        return "".join(out), outline, current

    def block_class(self, kind: str, outline: bool, current: bool,
                    slot: str = "") -> Tuple[str, str]:
        """CSS classes and id attribute for a block."""
        classes = []
        if kind in self.kinds or (slot and slot in self.slots):
            classes.append("hl-soft")
            current = current or self.kinds.get(kind, False) or self.slots.get(slot, False)
        if outline:
            classes.append("hl-block")
        ident = ""
        if current and (outline or classes):
            classes.append("hl-block-cur")
            ident = self._current_id()
        return " ".join(classes), ident

    # -- the manuscript -------------------------------------------------------

    def build(self) -> str:
        p = self.paper
        parts: List[str] = []

        # title
        text, outline, cur = self.marked("front:title", p.title or "Untitled Manuscript")
        cls, ident = self.block_class("title", outline, cur)
        parts.append(f'<h1 class="m-title {cls}"{ident or self._idattr("front:title")}>{text}</h1>')

        # author block (TTL-06)
        if p.authors:
            links = p.author_affiliations if len(p.author_affiliations) == len(p.authors) \
                else [[] for _ in p.authors]
            cells = []
            for name, link in zip(p.authors, links):
                affs = "".join(f'<span class="m-aff">{esc(p.affiliations[i])}</span>'
                               for i in link if 0 <= i < len(p.affiliations))
                cells.append(f'<div class="m-author"><span class="m-name">{esc(name)}</span>{affs}</div>')
            _, outline, cur = self.marked("front:authors", "")
            outline = outline or bool(self.at.get("front:authors"))
            cls, ident = self.block_class("authors", outline, cur or any(
                h.get("current") for h in self.at.get("front:authors", [])))
            parts.append(f'<div class="m-authors {cls}"{ident}>{"".join(cells)}</div>')
            shown = {i for link in links for i in link}
            for i, aff in enumerate(p.affiliations):
                if i not in shown:
                    parts.append(f'<p class="m-aff-line">{esc(aff)}</p>')

        body: List[str] = []
        # abstract and index terms
        if p.abstract:
            label = f"{S.ABSTRACT_LABEL}"
            text, outline, cur = self.marked("front:abstract", label + p.abstract)
            cls, ident = self.block_class("abstract", outline, cur)
            text = text.replace(esc(label), f'<i>{esc(label)}</i>', 1) \
                if text.startswith(esc(label)) else text
            body.append(f'<p class="m-abstract {cls}"{ident}>{text}</p>')
        if p.keywords:
            terms = ", ".join(p.keywords)
            terms = terms[:1].upper() + terms[1:]
            label = S.INDEX_TERMS_LABEL
            text, outline, cur = self.marked("front:index_terms", label + terms)
            cls, ident = self.block_class("index", outline, cur)
            text = text.replace(esc(label), f'<i>{esc(label)}</i>', 1) \
                if text.startswith(esc(label)) else text
            body.append(f'<p class="m-abstract {cls}"{ident}>{text}</p>')

        # sections, in the same order as the Word file
        run_in = None
        for section in ordered_sections(p):
            slot = section.ieee_slot if section.level == 1 else ""
            anchor = anchor_of(section)
            label = self._heading_text(section)
            first = section.elements[0] if section.elements else None
            if section.level >= 3 and first is not None and first.kind == "paragraph":
                run_in = (anchor, label, section.level)
            else:
                text, outline, cur = self.marked(anchor, label)
                cls, ident = self.block_class("heading", outline, cur, slot)
                level = min(section.level, 3)
                body.append(f'<div class="m-h{level} {cls}"{ident}>{text}</div>')
            previous = ""
            for element in section.elements:
                body.append(self._element(element, previous, run_in, slot))
                run_in = None if element.kind == "paragraph" else run_in
                previous = element.kind

        # references
        text, outline, cur = self.marked("references", S.REFERENCES_HEADING)
        cls, ident = self.block_class("references", outline, cur)
        body.append(f'<div class="m-h1 {cls}"{ident}>{text}</div>')
        records = p.reference_records or []
        for n, ref in enumerate(p.references, start=1):
            body.append(self._reference(n, ref, records[n - 1] if n - 1 < len(records) else None))

        parts.append(f'<div class="m-cols">{"".join(body)}</div>')
        # a whole-page rule: stay at the top, where the banner explains it
        banner_id = ' id="hl-current"' if self.page and not self.current_placed else ""
        banner = f'<div class="banner"{banner_id}>{esc(self.banner)}</div>' if self.banner else ""
        page_cls = " hl-page" if self.page else ""
        css = CSS + (LIGHT_CSS if self.theme == "light" else "")
        return (f'<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head>'
                f'<body>{banner}<div class="sheet{page_cls}">{"".join(parts)}</div>'
                f'<script>{SCROLL_JS}</script></body></html>')

    def _idattr(self, anchor: str) -> str:
        return f' data-anchor="{esc(anchor)}"'

    @staticmethod
    def _heading_text(section) -> str:
        if section.level == 1:
            return (f"{section.ieee_number}. " if section.ieee_number else "") + section.ieee_heading
        if section.level == 2:
            return f"{section.ieee_number}. {section.ieee_heading}"
        return f"{section.ieee_number} {section.ieee_heading}:"

    def _element(self, element, previous: str, run_in, slot: str) -> str:
        anchor = anchor_of(element)
        kind = element.kind
        if kind == "paragraph":
            text, outline, cur = self.marked(anchor, element.text, pattern_scope=True)
            cls, ident = self.block_class("paragraph", outline, cur, slot)
            lead = ""
            if run_in:
                h_text, h_outline, h_cur = self.marked(run_in[0], run_in[1])
                h_cls, h_ident = self.block_class("heading", h_outline, h_cur)
                lead = f'<span class="m-runin {h_cls}"{h_ident}>{h_text}</span> '
            indent = " noindent" if previous == "equation" and element.text[:1].islower() else ""
            return f'<p class="m-p{indent} {cls}"{ident}>{lead}{text}</p>'
        if kind == "equation":
            match = EQUATION_NUMBER.match(element.text)
            body, number = (match.group(1).strip(), match.group(2).replace(" ", "")) if match \
                else (element.text.strip(), "")
            text, outline, cur = self.marked(anchor, number)
            cls, ident = self.block_class("equation", outline, cur, slot)
            return (f'<div class="m-eq {cls}"{ident}><i>{esc(body)}</i>'
                    f'<span class="m-eqno">{text}</span></div>')
        if kind == "math_region":
            cls, ident = self.block_class("equation", bool(self.at.get(anchor)), any(
                h.get("current") for h in self.at.get(anchor, [])), slot)
            img = self.images.get(anchor)
            inner = f'<img src="{img}" alt="equation">' if img else \
                '<span class="m-missing">[equation image]</span>'
            return f'<div class="m-eqimg {cls}"{ident}>{inner}</div>'
        if kind == "code":
            return f'<pre class="m-code">{esc(element.text)}</pre>'
        if kind == "caption":
            return self._caption(element, slot)
        return ""

    def _caption(self, element, slot: str) -> str:
        anchor = anchor_of(element)
        match = CAPTION_PARTS.match(element.text)
        img = self.images.get(anchor)
        if not match:
            text, outline, cur = self.marked(anchor, element.text)
            cls, ident = self.block_class("caption", outline, cur, slot)
            return f'<p class="m-cap {cls}"{ident}>{text}</p>'
        word, number, rest = match.group(1).lower(), match.group(2), match.group(3).strip()
        rest = rest.rstrip(".")
        picture = f'<img src="{img}" alt="">' if img else \
            '<span class="m-missing">[image not found in the source]</span>'
        if word.startswith("fig"):
            label = f"{S.FIGURE_LABEL} {number}."
            text, outline, cur = self.marked(anchor, f"{label}{S.CAPTION_GAP}{rest}.")
            cls, ident = self.block_class("figure", outline, cur, slot)
            soft = " hl-soft" if "caption" in self.kinds or "visual" in self.kinds else ""
            return (f'<figure class="m-fig {cls}{soft}"{ident}>{picture}'
                    f'<figcaption>{text}</figcaption></figure>')
        if word == "algorithm":
            text, outline, cur = self.marked(anchor, f"{S.ALGORITHM_LABEL} {number}. {rest}")
            cls, ident = self.block_class("figure", outline, cur, slot)
            return f'<figure class="m-fig {cls}"{ident}><p class="m-cap left">{text}</p>{picture}</figure>'
        roman = number if not number.isdigit() else to_roman(int(number))
        label = f"{S.TABLE_LABEL} {roman}"
        text, outline, cur = self.marked(anchor, f"{label}\n{rest}")
        cls, ident = self.block_class("table", outline, cur, slot)
        soft = " hl-soft" if "caption" in self.kinds or "visual" in self.kinds else ""
        label_html, _, title_html = text.partition("\n")
        return (f'<div class="m-tbl {cls}{soft}"{ident}><div class="m-tlabel">{label_html}</div>'
                f'<div class="m-ttitle">{title_html}</div>{picture}</div>')

    def _reference(self, n: int, ref: str, record: Optional[dict]) -> str:
        anchor = f"reference:{n}"
        wanted = {h.get("segment", "") for h in self.at.get(anchor, [])}
        current = any(h.get("current") for h in self.at.get(anchor, []))
        whole = bool(self.at.get(anchor)) and (not wanted - {""} or "" in wanted)
        segments = (record or {}).get("segments") or [(ref, False)]
        pieces = []
        for i, (text, italic) in enumerate(segments):
            kind = "authors" if i == 0 and not italic and "“" not in text else \
                "title" if "“" in text else "venue" if italic else "tail"
            content = f"<i>{esc(text)}</i>" if italic else esc(text)
            if kind in wanted:
                ident = self._current_id() if current else ""
                content = f'<mark class="hl{" hl-cur" if current else ""}"{ident}>{content}</mark>'
            pieces.append(content)
        cls, ident = self.block_class("reference", whole, current and whole)
        return (f'<p class="m-ref {cls}"{ident}><span class="m-refno">[{n}]</span>'
                f'<span>{"".join(pieces)}</span></p>')


def build_preview(paper, highlights: List[dict], images: Optional[Dict[str, str]] = None,
                  banner: str = "", theme: str = "dark") -> str:
    """The manuscript as one HTML page, with these highlights.
    theme: "dark" (Streamlit page) or "light" (React page) - only the background around the sheet."""
    return _Preview(paper, highlights, images or {}, banner, theme).build()


LIGHT_CSS = """
html, body { background: #e9eef5; }
.sheet { box-shadow: 0 1px 3px rgba(16,24,40,.08), 0 8px 24px rgba(16,24,40,.08); }
"""

SCROLL_JS = """
window.addEventListener('load', function () {
  var el = document.getElementById('hl-current');
  if (el && el.classList.contains('banner')) { window.scrollTo(0, 0); }
  else if (el) { el.scrollIntoView({block: 'center'}); }
});
"""

CSS = """
html, body { margin: 0; background: #1a2436; }
body { padding: 14px; font-family: 'Times New Roman', Times, 'Liberation Serif', serif; }
.banner { font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; font-size: 13px; color: #101826;
  background: #ffd84d; border-radius: 6px; padding: 8px 12px; margin: 0 auto 12px; max-width: 820px; }
.sheet { background: #fff; color: #111; max-width: 820px; margin: 0 auto; padding: 44px 46px 56px;
  box-shadow: 0 2px 14px rgba(0,0,0,.35); }
.sheet.hl-page { outline: 4px solid #ffb300; outline-offset: 4px; }
.m-title { font-weight: 400; font-size: 27px; line-height: 1.15; text-align: center; margin: 0 0 14px; }
.m-authors { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px 12px; margin: 0 0 6px;
  text-align: center; }
.m-author { display: flex; flex-direction: column; font-size: 12.5px; }
.m-name { font-size: 13px; }
.m-aff, .m-aff-line { font-style: italic; font-size: 11.5px; }
.m-aff-line { text-align: center; margin: 2px 0; }
.m-cols { column-count: 2; column-gap: 22px; margin-top: 18px; text-align: justify; hyphens: auto; }
.m-abstract { font-weight: 700; font-size: 11px; line-height: 1.3; margin: 0 0 8px; }
.m-p { font-size: 12.5px; line-height: 1.32; margin: 0; text-indent: 1.4em; }
.m-p.noindent { text-indent: 0; }
.m-h1 { font-variant: small-caps; text-align: center; font-size: 12.5px; margin: 12px 0 5px;
  break-after: avoid; }
.m-h2 { font-style: italic; font-size: 12.5px; margin: 8px 0 3px; break-after: avoid; }
.m-h3 { font-style: italic; font-size: 12.5px; margin: 6px 0 2px; }
.m-runin { font-style: italic; }
.m-eq { display: flex; justify-content: center; position: relative; font-size: 12.5px; margin: 6px 0; }
.m-eqno { position: absolute; right: 0; }
.m-eqimg { text-align: center; margin: 6px 0; } .m-eqimg img { max-width: 100%; }
.m-code { font-family: 'Courier New', monospace; font-size: 10px; white-space: pre-wrap; margin: 6px 0; }
.m-fig, .m-tbl { break-inside: avoid; margin: 10px 0; text-align: center; }
.m-fig img, .m-tbl img { max-width: 100%; }
figcaption, .m-cap { font-size: 10.5px; line-height: 1.3; text-align: center; margin: 4px 0 0; }
.m-cap.left { text-align: left; }
.m-tlabel { font-size: 10.5px; } .m-ttitle { font-size: 10.5px; font-variant: small-caps; margin-bottom: 4px; }
.m-ref { font-size: 10.5px; line-height: 1.3; display: grid; grid-template-columns: 2.4em 1fr; margin: 0 0 2px;
  text-align: left; }
.m-missing { color: #888; font-style: italic; font-size: 11px; }
mark.hl { background: #fff1a6; color: inherit; border-radius: 2px; box-shadow: 0 0 0 1px #f1c40f; }
mark.hl-cur { background: #ffd84d; box-shadow: 0 0 0 2px #e0a100; }
mark.hl-rel { background: #d6e4ff; color: inherit; box-shadow: 0 0 0 1px #6c9bff; border-radius: 2px; }
.hl-soft { outline: 1.5px dashed #e0a100; outline-offset: 2px; }
.hl-block { outline: 2px solid #f1c40f; outline-offset: 3px; background: #fffbe0; }
.hl-block-cur { outline: 3px solid #e0a100; background: #fff3b8; }
"""
