"""
compliance.py - Step 6d: the IEEE Format Compliance Score.

Every check tests one or more rules from the IEEE rulebook
(src/rules/ieee_rules.py) and shows their IDs, e.g. "LAY-02". All targets
(sizes, margins, word limits, labels) are read from the rulebook.

The checks, each scored from 0 to 1, run TWICE:
    BEFORE - on the paper as it came in (the source PDF / Word file)
    AFTER  - on the IEEE manuscript the engine produced (the .docx)
The score is the average, as a percentage. "Before 31% -> after 96%" is the
headline number for the evaluation chapter.

Every check reads the rule from the SAME place the renderer does
(render/ieee_style.py), so the checker and the renderer can never disagree
about what "correct" means. Each check also returns a short reason, so a
low score always says WHY.

  LAYOUT         page size, margins, two columns, Times font, 10 pt body,
                 24 pt title
  FRONT MATTER   "Abstract—" label, abstract length, "Index Terms—"
  STRUCTURE      Roman-numeral headings, IEEE section names
  CITATIONS      [n] citations, IEEE reference style, numbered in order of
                 first citation
  VISUALS        "Fig. n." / "TABLE II" captions, "Fig. n" mentions in the
                 text, visuals placed after their first mention

A check that cannot be measured for a given input (e.g. page margins of a
JSON content package) is shown as "n/a" and left out of that average.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..compose.front_matter import abstract_citations, title_is_ieee_case
from ..compose.ieee_text import number_style_counts
from ..compose.references import unabbreviated_words
from ..models import BlockType, ParsedDocument, StructuredPaper
from ..placement.contextual import caption_label, mentions
from ..render import ieee_style as S
from ..rules import ieee_rules as R
from ..structure.ieee_mapper import REQUIRED, SLOTS

Result = Tuple[Optional[float], str]          # (score 0..1 or None, reason)
EMU_PER_INCH = 914400
TIMES = re.compile(r"times|nimbusrom|termes|tinos|liberation ?serif|stix|ptm", re.I)
ROMAN_HEADING = re.compile(r"^[IVXL]+\.?$")
AUTHOR_YEAR = re.compile(r"\([A-Z][^()]*?\b(19|20)\d{2}[a-z]?\)|[A-Z][\w'\-]+ et al\.? \((19|20)\d{2}")
NUMERIC_CITE = re.compile(r"\[\d+\]")
IEEE_REF = re.compile(r"^(\[\d+\]\s*)?([A-Z][a-z]?\.[\s\-]?)+.*?“.+[,?!]”|^(\[\d+\]\s*)?[A-Z][\w&.\- ]+, “")


@dataclass
class Check:
    key: str
    name: str
    group: str
    before: Optional[float] = None
    after: Optional[float] = None
    before_reason: str = ""
    after_reason: str = ""
    rules: List[str] = field(default_factory=list)   # rulebook IDs this check tests
    scored: bool = True                              # False = reported, not in the average


@dataclass
class ComplianceReport:
    checks: List[Check] = field(default_factory=list)
    before_score: Optional[float] = None
    after_score: Optional[float] = None

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _within(value: float, target: float, tolerance: float) -> bool:
    return abs(value - target) <= tolerance


def _share(good: int, total: int) -> Optional[float]:
    return None if total == 0 else round(good / total, 3)


def _paragraph_texts(paper: StructuredPaper) -> List[str]:
    return [e.text for s in paper.sections for e in s.elements if e.kind == "paragraph"]


# ---------------------------------------------------------------------------
# Content checks - the same code judges the source and the output
# ---------------------------------------------------------------------------

def check_abstract_length(paper: StructuredPaper) -> Result:
    low, high = R.ABSTRACT_WORDS
    words = len(paper.abstract.split())
    if words == 0:
        return 0.0, "no abstract"
    if low <= words <= high:
        return 1.0, f"{words} words"
    if low - 50 <= words <= high + 50:
        return 0.5, f"{words} words (IEEE: {low}-{high})"
    return 0.0, f"{words} words (IEEE: {low}-{high})"


def check_abstract_clean(paper: StructuredPaper) -> Result:
    if not paper.abstract:
        return None, "no abstract"
    n = abstract_citations(paper.abstract)
    return (1.0 if n == 0 else 0.0), "no citations in the abstract" if n == 0 \
        else f"{n} citation(s) in the abstract"


def check_title_case(paper: StructuredPaper) -> Result:
    if not paper.title:
        return None, "no title"
    ok = title_is_ieee_case(paper.title)
    return (1.0 if ok else 0.0), "title case" if ok else "title in capitals or marked (Invited)"


def check_venue_abbreviation(paper: StructuredPaper, side: str) -> Result:
    """REF-06. Journal and conference names only (arXiv and web sources have none)."""
    good = total = 0
    for record in paper.reference_records or []:
        if not record.get("venue") or not record.get("segments"):
            continue
        italic = " ".join(text for text, is_italic in record["segments"] if is_italic)
        if not italic:
            continue
        venue = record["venue"] if side == "before" else italic
        total += 1
        good += not unabbreviated_words(venue)
    if total == 0:
        return None, "no journal or conference references"
    return _share(good, total), f"{good}/{total} journal/conference names abbreviated"


def check_number_style(paper: StructuredPaper) -> Result:
    good = bad = 0
    for text in _paragraph_texts(paper):
        g, b = number_style_counts(text)
        good, bad = good + g, bad + b
    if good + bad == 0:
        return None, "no decimals, units or percentages"
    return _share(good, good + bad), f"{good} correct, {bad} to fix (0.25, 10 mm, 2%)"


def check_visual_order(paper: StructuredPaper) -> Result:
    """FIG-06: figures (and tables) are first mentioned in number order."""
    first: Dict[str, List[int]] = {}
    for text in _paragraph_texts(paper):
        for kind, number in mentions(text):
            seq = first.setdefault(kind, [])
            if number not in seq:
                seq.append(number)
    if not first:
        return None, "no figures or tables mentioned"
    ok = [k for k, seq in first.items() if seq == sorted(seq)]
    bad = sorted(set(first) - set(ok))
    return _share(len(ok), len(first)), "first mentioned in number order" if not bad \
        else f"{', '.join(bad)} not first mentioned in number order"


def check_author_block(paper: StructuredPaper, docx_path: str) -> Result:
    """TTL-06 / TTL-07: every author in a cell of their own in the output's author block,
    and nothing that is not a name (no caption, no 'Figure', no long sentence)."""
    if not paper.authors:
        return None, "no authors"
    import docx
    document = docx.Document(docx_path)
    if not document.tables:
        return 0.0, "authors not set out as an IEEE author block"
    cells = [c for row in document.tables[0].rows for c in row.cells if c.text.strip()]
    names = [c.paragraphs[0].text for c in cells]
    clean = [n for n in names if len(n.split()) <= 6 and not CAPTION_START.match(n)]
    linked = sum(1 for c in cells if len(c.paragraphs) > 1)
    ok = len(cells) == len(paper.authors) and len(clean) == len(names)
    note = f"{len(cells)} authors listed separately, {linked} with their affiliation"
    return (1.0 if ok else 0.0), note if ok else f"author block looks wrong: {note}"


CAPTION_START = re.compile(r"(?i)^(fig\.?|figure|table)\s*\d")
ACRONYM = re.compile(r"\b([A-Z]{2,})(s?)\b(?![-\d])")


def check_acronyms(paper: StructuredPaper) -> Result:
    """STY-02 (reported, not scored): every acronym defined at its first use,
    i.e. its first appearance is "full name (ACR)"."""
    text = " ".join([paper.abstract] + _paragraph_texts(paper))
    seen: Dict[str, bool] = {}
    for m in ACRONYM.finditer(text):
        word = m.group(1)
        if word in R.NOT_ACRONYMS or len(word) > 8 or re.fullmatch(r"[IVXLC]+", word):
            continue
        if word not in seen:
            before, after = text[max(0, m.start() - 1):m.start()], text[m.end():m.end() + 1]
            seen[word] = before == "(" and after in (")", ",", ";")
    if not seen:
        return None, "no acronyms"
    undefined = [w for w, ok in seen.items() if not ok]
    note = f"{len(seen) - len(undefined)}/{len(seen)} defined at first use"
    if undefined:
        note += " - check: " + ", ".join(undefined[:6]) + ("..." if len(undefined) > 6 else "")
    return _share(len(seen) - len(undefined), len(seen)), note


def check_heading_numbering(paper: StructuredPaper, use_ieee: bool) -> Result:
    top = [s for s in paper.sections if s.level == 1 and s.heading and not s.in_appendix
           and s.ieee_slot not in ("statement", "acknowledgment", "appendix")]
    if not top:
        return None, "no section headings"
    numbers = [s.ieee_number if use_ieee else s.number for s in top]
    good = sum(1 for n in numbers if ROMAN_HEADING.match(n or ""))
    return _share(good, len(top)), f"{good}/{len(top)} sections numbered I, II, III..."


def check_heading_names(paper: StructuredPaper, use_ieee: bool) -> Result:
    names = []
    for slot in REQUIRED:
        canonical = SLOTS[slot].heading.lower()
        section = next((s for s in paper.sections if s.ieee_slot == slot and s.level == 1), None)
        if section is None:
            continue
        shown = (section.ieee_heading if use_ieee else section.heading).lower().strip()
        names.append(shown == canonical)
    if not names:
        return None, "no required sections found"
    return _share(sum(names), len(names)), \
        f"{sum(names)}/{len(names)} required sections carry the IEEE name"


def check_citation_style(paper: StructuredPaper) -> Result:
    text = " ".join(_paragraph_texts(paper))
    numeric = len(NUMERIC_CITE.findall(text))
    author_year = len(AUTHOR_YEAR.findall(text))
    if numeric + author_year == 0:
        return None, "no citations in the text"
    return _share(numeric, numeric + author_year), \
        f"{numeric} numeric [n], {author_year} author-year"


def check_reference_style(references: List[str]) -> Result:
    if not references:
        return None, "no references"
    good = sum(1 for r in references if IEEE_REF.search(r))
    return _share(good, len(references)), f"{good}/{len(references)} in IEEE form"


def check_reference_order(paper: StructuredPaper) -> Result:
    text = " ".join(_paragraph_texts(paper))
    seen: List[int] = []
    for m in re.finditer(r"\[(\d+)\]", text):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    if not seen:
        return 0.0 if paper.references else None, \
            "references are not numbered by citation" if paper.references else "no citations"
    ok = seen == list(range(1, len(seen) + 1))
    return (1.0 if ok else 0.0), "numbered in order of first citation" if ok \
        else "citation numbers do not follow first appearance"


def check_caption_style(captions: List[str]) -> Result:
    if not captions:
        return None, "no figure or table captions"
    avoid = "|".join(R.CAPTION_AVOID_FIRST)

    def ieee_form(c: str) -> bool:
        figure = re.match(rf"^{re.escape(S.FIGURE_LABEL)} \d+\.{S.CAPTION_GAP}(.)", c)
        if figure:                                         # FIG-02 + FIG-11
            return figure.group(1).isupper() and not re.match(
                rf"^{re.escape(S.FIGURE_LABEL)} \d+\.{S.CAPTION_GAP}(?:{avoid})\s", c)
        return bool(re.match(rf"^{S.TABLE_LABEL} [IVXLC]+$", c)
                    or re.match(rf"^{S.ALGORITHM_LABEL} \d+\. ", c))

    good = sum(1 for c in captions if ieee_form(c))
    return _share(good, len(captions)), f"{good}/{len(captions)} captions in IEEE form"


def check_visual_mentions(paper: StructuredPaper) -> Result:
    text = " ".join(_paragraph_texts(paper))
    ieee = len(re.findall(r"\bFigs?\. \d+", text)) + len(re.findall(r"\bTables? [IVXL]+\b", text))
    other = len(re.findall(r"\bFigures? \d+", text)) + len(re.findall(r"\bTables? \d+", text))
    if ieee + other == 0:
        return None, "the text never mentions a figure or table"
    return _share(ieee, ieee + other), f"{ieee} 'Fig. n'/'Table II', {other} 'Figure n'/'Table 2'"


def check_placement(summary: dict, side: str) -> Result:
    stats = (summary or {}).get(side) or {}
    share = stats.get("after_first_mention")
    if share is None:
        return None, "no visuals mentioned in the text"
    return round(share / 100, 3), \
        f"{share}% after first mention, mean distance {stats.get('mean_distance')} paragraphs"


# ---------------------------------------------------------------------------
# Layout facts - from the source document, and from the output .docx
# ---------------------------------------------------------------------------

def source_layout(doc: Optional[ParsedDocument]) -> Dict[str, Result]:
    """Measure the layout of the INPUT as best we can from its text."""
    if doc is None or not doc.blocks or doc.source_format != "pdf":
        return {}
    facts: Dict[str, Result] = {}
    w, h = doc.page_width / 72, doc.page_height / 72
    ok = _within(w, S.PAPER.width_in, 0.1) and _within(h, S.PAPER.height_in, 0.1)
    facts["page_size"] = (1.0 if ok else 0.0), f"{w:.2f} x {h:.2f} in"

    body = [b for b in doc.blocks if b.block_type is BlockType.BODY and b.word_count >= 6]
    content = [b for b in doc.blocks if b.block_type not in (BlockType.JUNK,)]
    if body and content:
        left = min(b.x0 for b in body) / 72
        right = (doc.page_width - max(b.x1 for b in body)) / 72
        top = min(b.y0 for b in content) / 72
        bottom = (doc.page_height - max(b.y1 for b in content)) / 72
        target = (S.MARGIN_TOP.inches, S.MARGIN_BOTTOM.inches,
                  S.MARGIN_LEFT.inches, S.MARGIN_RIGHT.inches)
        got = (top, bottom, left, right)
        good = sum(1 for g, t in zip(got, target) if _within(g, t, 0.15))
        facts["margins"] = good / 4, "approx. top/bottom/left/right " + \
            "/".join(f"{g:.2f}" for g in got) + " in"
        two = sum(1 for b in body if b.column == 1) / len(body)
        facts["columns"] = (1.0 if two >= 0.2 else 0.0), \
            "two columns" if two >= 0.2 else "single column"
        chars = sum(len(b.text) for b in body)
        times = sum(len(b.text) for b in body if TIMES.search(b.font_name))
        facts["font"] = round(times / chars, 3), f"{100 * times // max(1, chars)}% of body text in Times"
    facts["body_size"] = (1.0 if _within(doc.body_font_size, R.SIZE_PT["body"], 0.25) else 0.0), \
        f"body text {doc.body_font_size} pt"
    titles = [b.font_size for b in doc.blocks if b.block_type is BlockType.TITLE]
    if titles:
        facts["title_size"] = (1.0 if _within(max(titles), R.SIZE_PT["title"], 1) else 0.0), \
            f"title {max(titles)} pt"
    abstract_lines = [b for b in doc.blocks if b.block_type is BlockType.ABSTRACT]
    inline = bool(abstract_lines) and abstract_lines[0].text.startswith(S.ABSTRACT_LABEL)
    facts["abstract_label"] = (1.0 if inline else 0.0), \
        f"'{S.ABSTRACT_LABEL}' run-in label" if inline else f"abstract not introduced by '{S.ABSTRACT_LABEL}'"
    keyword_lines = [b for b in doc.blocks if b.block_type is BlockType.KEYWORDS]
    terms_ok = bool(keyword_lines) and keyword_lines[0].text.startswith("Index Terms")
    facts["index_terms"] = (1.0 if terms_ok else 0.0), \
        f"'{S.INDEX_TERMS_LABEL}' present" if terms_ok else ("keywords not labelled 'Index Terms'"
                                                    if keyword_lines else "no keywords")
    return facts


def output_layout(docx_path: str) -> Dict[str, Result]:
    """Measure the layout of the OUTPUT .docx directly."""
    import docx
    from docx.oxml.ns import qn

    document = docx.Document(docx_path)
    facts: Dict[str, Result] = {}
    sec = document.sections[-1]
    w, h = sec.page_width / EMU_PER_INCH, sec.page_height / EMU_PER_INCH
    facts["page_size"] = (1.0 if _within(w, S.PAPER.width_in, 0.01)
                          and _within(h, S.PAPER.height_in, 0.01) else 0.0), \
        f"{w:.2f} x {h:.2f} in"
    got = [m / EMU_PER_INCH for m in (sec.top_margin, sec.bottom_margin,
                                       sec.left_margin, sec.right_margin)]
    target = (S.MARGIN_TOP.inches, S.MARGIN_BOTTOM.inches, S.MARGIN_LEFT.inches,
              S.MARGIN_RIGHT.inches)
    good = sum(1 for g, t in zip(got, target) if _within(g, t, 0.01))
    facts["margins"] = good / 4, "top/bottom/left/right " + "/".join(f"{g:.3f}" for g in got) + " in"
    cols = [s._sectPr.find(qn("w:cols")) for s in document.sections[1:]]
    two = all(c is not None and c.get(qn("w:num")) == str(S.COLUMN_COUNT) for c in cols) and cols
    facts["columns"] = (1.0 if two else 0.0), "two columns" if two else "not two columns"

    runs = [r for p in document.paragraphs for r in p.runs if r.text.strip()]
    text_runs = [r for r in runs if r.font.name != S.CODE_FONT]
    times = sum(len(r.text) for r in text_runs if r.font.name == S.FONT)
    total = sum(len(r.text) for r in text_runs)
    facts["font"] = _share(times, total), f"{100 * times // max(1, total)}% of text in {S.FONT}"

    body = [p for p in document.paragraphs
            if p.paragraph_format.first_line_indent == S.BODY_FIRST_LINE_INDENT and p.runs]
    sizes = [p.runs[0].font.size for p in body]
    good = sum(1 for z in sizes if z == S.BODY_SIZE)
    facts["body_size"] = _share(good, len(sizes)) if sizes else None, \
        f"{good}/{len(sizes)} body paragraphs at {S.BODY_SIZE.pt:g} pt"
    title = document.paragraphs[0].runs[0].font.size if document.paragraphs[0].runs else None
    facts["title_size"] = (1.0 if title == S.TITLE_SIZE else 0.0), \
        f"title {title.pt if title else '?'} pt"

    texts = [p.text for p in document.paragraphs]
    abstract = next((p for p in document.paragraphs if p.text.startswith(S.ABSTRACT_LABEL)), None)
    ok = abstract is not None and abstract.runs[0].font.bold and abstract.runs[0].font.italic
    facts["abstract_label"] = (1.0 if ok else 0.0), \
        f"'{S.ABSTRACT_LABEL}' bold italic run-in" if ok else f"no '{S.ABSTRACT_LABEL}' label"
    terms = next((t for t in texts if t.startswith(S.INDEX_TERMS_LABEL)), "")
    listed = [x.strip() for x in terms[len(S.INDEX_TERMS_LABEL):].split(",") if x.strip()]
    ok = bool(listed) and "[" not in terms and listed == sorted(listed, key=str.lower)
    # ABS-08: only the first word of the list is capitalised
    ok = ok and listed[0][:1] == listed[0][:1].upper()
    facts["index_terms"] = (1.0 if ok else 0.0), \
        f"'{S.INDEX_TERMS_LABEL}', alphabetical" if ok else "index terms missing or not alphabetical"
    facts["_captions"] = None, "\n".join(
        t for t in texts if re.match(rf"^({re.escape(S.FIGURE_LABEL)} \d+\.|{S.TABLE_LABEL} [IVXLC]+$|"
                                     rf"{S.ALGORITHM_LABEL} \d+\.)", t))
    return facts


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def checks() -> List[Tuple[str, str, str]]:
    """(key, name, group) for every check. Names are built from the rulebook,
    so they always show the values actually being tested."""
    p = S.PAPER
    if p.name == "US Letter":
        size = f"{p.width_in:g} x {p.height_in:g} in"
        margins = f"{p.margin_top_in:g} / {p.margin_bottom_in:g} / {p.margin_side_in:g} in"
    else:
        size = f"{p.width_in * 25.4:.0f} x {p.height_in * 25.4:.0f} mm"
        margins = (f"{p.margin_top_in * 25.4:.0f} / {p.margin_bottom_in * 25.4:.0f} / "
                   f"{p.margin_side_in * 25.4:.0f} mm")
    low, high = R.ABSTRACT_WORDS
    return [
        ("page_size", f"{p.name} page ({size})", "Layout"),
        ("margins", f"IEEE margins ({margins})", "Layout"),
        ("columns", "Two-column body", "Layout"),
        ("font", R.FONT, "Layout"),
        ("body_size", f"{R.SIZE_PT['body']:g} pt body text", "Layout"),
        ("title_size", f"{R.SIZE_PT['title']:g} pt title", "Layout"),
        ("title_case", "Title in title case", "Front matter"),
        ("author_block", "Authors listed separately with affiliations", "Front matter"),
        ("abstract_label", f"'{S.ABSTRACT_LABEL}' label", "Front matter"),
        ("abstract_length", f"Abstract {low}-{high} words", "Front matter"),
        ("abstract_clean", "No citations in the abstract", "Front matter"),
        ("index_terms", f"'{S.INDEX_TERMS_LABEL}', alphabetical", "Front matter"),
        ("heading_numbering", "Headings numbered I, II, III", "Structure"),
        ("heading_names", "IEEE section names", "Structure"),
        ("citation_style", "Numeric [n] citations", "Citations"),
        ("reference_style", "IEEE reference format", "Citations"),
        ("reference_order", "References in order of first citation", "Citations"),
        ("venue_abbreviation", "Journal names abbreviated (Trans., J., Proc.)", "Citations"),
        ("caption_style", f"'{S.FIGURE_LABEL} n.' / '{S.TABLE_LABEL} II' captions", "Visuals"),
        ("visual_mentions", f"'{S.FIGURE_LABEL} n' / 'Table II' in the text", "Visuals"),
        ("visual_order", "Visuals first mentioned in number order", "Visuals"),
        ("placement", "Visuals after their first mention", "Visuals"),
        ("number_style", "Numbers and units (0.25, 10 mm, 2%)", "Writing"),
        ("acronyms", "Acronyms defined at first use (advisory)", "Writing"),
    ]


NOT_SCORED = {"acronyms"}     # judgement calls: reported, but kept out of the average


def evaluate(source_doc: Optional[ParsedDocument], source_paper: StructuredPaper,
             final_paper: StructuredPaper, docx_path: str) -> ComplianceReport:
    before_layout = source_layout(source_doc)
    after_layout = output_layout(docx_path)
    source_captions = [e.text for s in source_paper.sections for e in s.elements
                       if e.kind == "caption"]
    output_captions = [c for c in after_layout.pop("_captions")[1].split("\n") if c]
    summary = final_paper.placement_summary

    content_before: Dict[str, Callable[[], Result]] = {
        "abstract_length": lambda: check_abstract_length(source_paper),
        "abstract_clean": lambda: check_abstract_clean(source_paper),
        "title_case": lambda: check_title_case(source_paper),
        "venue_abbreviation": lambda: check_venue_abbreviation(final_paper, "before"),
        "number_style": lambda: check_number_style(source_paper),
        "visual_order": lambda: check_visual_order(source_paper),
        "acronyms": lambda: check_acronyms(source_paper),
        "heading_numbering": lambda: check_heading_numbering(source_paper, use_ieee=False),
        "heading_names": lambda: check_heading_names(source_paper, use_ieee=False),
        "citation_style": lambda: check_citation_style(source_paper),
        "reference_style": lambda: check_reference_style(source_paper.references),
        "reference_order": lambda: check_reference_order(source_paper),
        "caption_style": lambda: check_caption_style(source_captions),
        "visual_mentions": lambda: check_visual_mentions(source_paper),
        "placement": lambda: check_placement(summary, "before"),
    }
    content_after: Dict[str, Callable[[], Result]] = {
        "abstract_length": lambda: check_abstract_length(final_paper),
        "author_block": lambda: check_author_block(final_paper, docx_path),
        "abstract_clean": lambda: check_abstract_clean(final_paper),
        "title_case": lambda: check_title_case(final_paper),
        "venue_abbreviation": lambda: check_venue_abbreviation(final_paper, "after"),
        "number_style": lambda: check_number_style(final_paper),
        "visual_order": lambda: check_visual_order(final_paper),
        "acronyms": lambda: check_acronyms(final_paper),
        "heading_numbering": lambda: check_heading_numbering(final_paper, use_ieee=True),
        "heading_names": lambda: check_heading_names(final_paper, use_ieee=True),
        "citation_style": lambda: check_citation_style(final_paper),
        "reference_style": lambda: check_reference_style(final_paper.references),
        "reference_order": lambda: check_reference_order(final_paper),
        "caption_style": lambda: check_caption_style(output_captions),
        "visual_mentions": lambda: check_visual_mentions(final_paper),
        "placement": lambda: check_placement(summary, "after"),
    }

    report = ComplianceReport()
    for key, name, group in checks():
        check = Check(key=key, name=name, group=group, rules=R.rules_for_check(key),
                      scored=key not in NOT_SCORED)
        b = content_before[key]() if key in content_before else before_layout.get(key, (None, "n/a"))
        a = content_after[key]() if key in content_after else after_layout.get(key, (None, "n/a"))
        check.before, check.before_reason = b
        check.after, check.after_reason = a
        report.checks.append(check)

    def average(values: List[Optional[float]]) -> Optional[float]:
        values = [v for v in values if v is not None]
        return round(100 * sum(values) / len(values), 1) if values else None

    report.before_score = average([c.before for c in report.checks if c.scored])
    report.after_score = average([c.after for c in report.checks if c.scored])
    return report
