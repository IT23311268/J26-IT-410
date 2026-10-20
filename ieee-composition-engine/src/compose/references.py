"""
references.py - turn any reference list into IEEE style.

Two separate jobs, deliberately kept apart:

  1. PARSE  - split each raw reference into parts: authors, title, venue,
              year, volume, issue, pages, URL.
              Gemini does this when available (it copes with any style);
              rules do it otherwise (they handle APA and the "natbib" style
              most arXiv papers use).

  2. FORMAT - write those parts out in exact IEEE style. Always done by
              plain Python, never by Gemini, so every reference comes out
              formatted identically, and the same input always gives the
              same output.

IEEE reference style, in short:
    [1] Q. Dou, H. Chen, and P. A. Heng, "Title of paper," Journal Name,
        vol. 35, no. 5, pp. 1160-1169, 2016.
  * initials BEFORE the surname
  * more than six authors -> first author + "et al."
  * the title in double quotes, the comma INSIDE the quotes
  * the journal or conference name in italics
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from ..rules import ieee_rules as R
from .llm import BaseLLM, LLMUnavailable

Record = Dict[str, Any]
Segments = List[Tuple[str, bool]]       # (text, italic?) pieces for the renderer

PARTICLES = {"van", "der", "de", "von", "da", "di", "la", "le", "del", "den", "dos", "du", "ter"}
YEAR = re.compile(r"\b(19|20)\d{2}[a-z]?\b")

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def surname_key(family: str) -> str:
    """'van der Schaar' -> 'schaar', 'Côté' -> 'cote'. Used to match citations."""
    words = [w for w in family.split() if w.lower() not in PARTICLES] or family.split()
    return strip_accents(words[-1]).lower() if words else ""


def initials(given: str) -> str:
    """'Marc-Alexandre' -> 'M.-A.'   'Russ R' -> 'R. R.'   'P. A.' -> 'P. A.'"""
    parts = []
    for word in given.replace(".", " ").split():
        hyphenated = [p for p in word.split("-") if p]
        parts.append("-".join(f"{p[0].upper()}." for p in hyphenated))
    return " ".join(parts)


def ieee_name(author: Dict[str, str]) -> str:
    given, family = author.get("given", "").strip(), author.get("family", "").strip()
    if not given:
        return family                     # organisations: "OpenAI"
    return f"{initials(given)} {family}".strip()


def ieee_author_list(authors: List[Dict[str, str]]) -> str:
    names = [ieee_name(a) for a in authors if a.get("family")]
    if not names:
        return ""
    if len(names) > R.MAX_AUTHORS_BEFORE_ET_AL:       # REF-03: more than six -> "et al."
        return f"{names[0]} et al."
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + ", and " + names[-1]


def _clean(text: str) -> str:
    return " ".join((text or "").split()).strip(" .,;")


# ---------------------------------------------------------------------------
# Rule-based parsing
# ---------------------------------------------------------------------------

APA = re.compile(r"^(?P<authors>.+?)\s*\((?P<year>(?:19|20)\d{2}[a-z]?|n\.d\.)\)\.?\s*(?P<rest>.*)$")
APA_AUTHOR = re.compile(r"([^,&]+?),\s*((?:[A-Z][a-z]?\.\s*-?\s*)+)")
JOURNAL_TAIL = re.compile(
    r"^(?P<venue>.+?),\s*(?P<volume>\d+)\s*(?:\((?P<issue>[^)]+)\))?\s*(?:,\s*(?P<pages>[\dA-Za-z]+(?:\s*[-–]\s*[\dA-Za-z]+)?))?\.?$")
PAGES_TAIL = re.compile(r"^(?P<venue>.+?),\s*(?:pp\.\s*)?(?P<pages>\d+\s*[-–]\s*\d+)\.?$")
URL = re.compile(r"(?:URL\s+)?(https?://\S+)")
DOI = re.compile(r"\b(10\.\d{4,9}/\S+?)[.,;]?(?:\s|$)")
ARXIV_ID = re.compile(r"(?:arXiv[:\s]*|arxiv\.org/abs/)(\d{4}\.\d{4,5})(v\d+)?", re.I)
ALREADY_IEEE = re.compile(r"^\s*\[\d+\]\s*|.*[\"“].+,[\"”]")


def _split_given_family(name: str) -> Dict[str, str]:
    """'Mihaela van der Schaar' -> given 'Mihaela', family 'van der Schaar'."""
    words = name.split()
    if len(words) <= 1:
        return {"given": "", "family": name.strip()}
    family_start = len(words) - 1
    while family_start - 1 > 0 and words[family_start - 1].lower() in PARTICLES:
        family_start -= 1
    return {"given": " ".join(words[:family_start]), "family": " ".join(words[family_start:])}


def _parse_apa_authors(text: str) -> List[Dict[str, str]]:
    text = text.replace("&", ",").replace(" and ", ", ")
    authors = []
    for family, given in APA_AUTHOR.findall(text):
        family = family.strip(" ,")
        if family:
            authors.append({"given": given.strip(), "family": family})
    if not authors and text.strip():
        authors.append({"given": "", "family": text.strip(" .,")})
    return authors


def _parse_natbib_authors(text: str) -> List[Dict[str, str]]:
    text = re.sub(r",?\s+and\s+", ", ", text)
    return [_split_given_family(n.strip()) for n in text.split(",") if n.strip()]


def _end_of_author_block(text: str) -> int:
    """Find the full stop that ends the author list in 'A. Smith, B. Jones. Title...'.

    The trick: a full stop after a single capital letter is an initial
    ("Michael Y. Hu"), not the end of the author list.
    """
    for match in re.finditer(r"\.\s", text):
        before = re.search(r"(\S+)$", text[:match.start()])
        token = before.group(1) if before else ""
        if len(token) > 1 or not token.isalpha():
            return match.start()
    return -1


def _repair(text: str) -> str:
    """Undo damage from line-wrapping in the PDF.

    'https://openai.com/index/ gpt-5-system-card/'  -> one URL again
    '4:241– 277'                                    -> '4:241–277'
    """
    for _ in range(3):
        text = re.sub(r"(https?://\S*/)\s+([\w\-.~%/]+)", r"\1\2", text)
    text = re.sub(r"(\d{4}\.)\s+(\d{4,5})", r"\1\2", text)      # arXiv id '2506. 05109'
    text = re.sub(r"(\b10\.\d{4,9}/\S*\.)\s+(\d+)\b", r"\1\2", text)  # DOI '...v39i22. 34505'
    text = re.sub(r"https?:\s+//", lambda m: m.group(0).replace(" ", ""), text)  # 'https: //'
    text = re.sub(r"(https?://\S*\.)\s+([a-z]{2,6}/)", r"\1\2", text)  # 'aclanthology. org/'
    return re.sub(r"(\d)\s*[–-]\s+(\d)", r"\1–\2", text)


def _tidy_fields(record: Record) -> None:
    """Clean up fields however they were parsed (rules or Gemini).

    * repair broken URLs and DOIs ('...v39i22. 34505' -> '...v39i22.34505')
    * a venue must not carry the volume or publisher:
      'Proc. of ICML, volume 235 of Proceedings of Machine Learning Research. PMLR'
      -> 'Proc. of ICML' (the volume moves to its own field)
    """
    for key in ("url", "doi"):
        if record.get(key):
            record[key] = re.sub(r"\s+", "", _repair(str(record[key]))).rstrip(".,")
    venue = str(record.get("venue", ""))
    volume = re.search(r",?\s*\bvolume\s+(\d+)(\s+of\s+[^,.]+)?", venue)
    if volume:
        record["volume"] = record.get("volume") or volume.group(1)
        venue = venue.replace(volume.group(0), "")
    venue = re.split(r"\.\s+(?=[A-Z])", venue)[0]          # drop ". PMLR" etc.
    record["venue"] = _clean(re.sub(r"^\s*In\s+", "", venue))


def _fill_tail(record: Record, rest: str) -> None:
    """Pull volume, issue, pages and venue out of what follows the title.

    Handles both  'Journal, 35(5), 1160-1169'                       (APA)
             and  'In Proc. of X, volume 34, pp. 1-9. PMLR, 2021'    (natbib)
             and  'Advances in X, 35: 6047-6059, 2022'               (natbib journal)
    """
    rest = URL.sub("", rest)
    rest = re.sub(r"\bdoi:\s*\S+", "", rest, flags=re.I)        # the DOI is stored separately
    rest = re.sub(r"^\s*In\s+", "", rest)
    rest = _clean(rest)
    rest = re.sub(r",?\s*(19|20)\d{2}[a-z]?\s*$", "", rest).strip(" .,")

    volume = re.search(r"\bvolume\s+(\d+)(\s+of\s+[^,.]+)?", rest)
    if volume:
        record["volume"] = volume.group(1)
        rest = rest.replace(volume.group(0), "")
    pages = re.search(r"\b(?:pp\.|pages)\s*(\d+\s*[–-]\s*\d+)", rest)
    if pages:
        record["pages"] = pages.group(1)
        rest = rest.replace(pages.group(0), "")
    vol_pages = re.search(r",\s*(\d+)\s*(?:\(([^)]+)\))?\s*:\s*(\d+\s*[–-]\s*\d+)", rest)
    if vol_pages:
        record.update(volume=vol_pages.group(1), issue=vol_pages.group(2) or "",
                      pages=vol_pages.group(3))
        rest = rest.replace(vol_pages.group(0), "")

    rest = re.sub(r"(,\s*)+", ", ", rest).strip(" ,.")
    if not (volume or pages or vol_pages):
        match = JOURNAL_TAIL.match(rest)
        if match and match.group("venue") and not match.group("venue").lower().startswith("arxiv"):
            record.update(venue=_clean(match.group("venue")), volume=match.group("volume") or "",
                          issue=match.group("issue") or "", pages=match.group("pages") or "")
            return
        match = PAGES_TAIL.match(rest)
        if match:
            record.update(venue=_clean(match.group("venue")), pages=match.group("pages"))
            return
    # "Proceedings of X. PMLR" -> the venue is the first sentence; drop the publisher
    record["venue"] = _clean(re.split(r"\.\s+(?=[A-Z])", rest)[0])


def parse_reference_rules(raw: str) -> Record:
    """Parse one reference without AI. Handles APA and natbib styles."""
    text = _repair(re.sub(r"^\s*(\[\d+\]|\d{1,3}\.)\s*", "", raw).strip())
    record: Record = {"raw": raw, "authors": [], "title": "", "venue": "", "year": "",
                      "volume": "", "issue": "", "pages": "", "url": "", "doi": "",
                      "source": "rules"}

    url = URL.search(text)
    if url:
        record["url"] = url.group(1).rstrip(".,")
    doi = DOI.search(text)
    if doi:
        record["doi"] = doi.group(1)

    apa = APA.match(text)
    if apa:                                             # "Dou, Q., ... (2017). Title. Venue, ..."
        record["authors"] = _parse_apa_authors(apa.group("authors"))
        record["year"] = apa.group("year")
        rest = apa.group("rest")
        title_end = re.search(r"(?<=[.?!])\s", rest)
        if title_end:
            record["title"] = _clean(rest[:title_end.start()])
            _fill_tail(record, rest[title_end.end():])
        else:
            record["title"] = _clean(URL.sub("", rest))
    else:                                               # "Victor Barres, ... and K. N. Title. Venue, 2025."
        end = _end_of_author_block(text)
        if end > 0:
            record["authors"] = _parse_natbib_authors(text[:end])
            rest = text[end + 1:].strip()
            title_end = re.search(r"(?<=[.?!])\s", rest)
            if title_end:
                record["title"] = _clean(rest[:title_end.start()])
                _fill_tail(record, rest[title_end.end():])
            else:
                record["title"] = _clean(rest)
        else:
            record["title"] = _clean(text)
        # the year is the last one in the text, ignoring URLs and arXiv numbers
        plain = ARXIV_ID.sub("", URL.sub("", text))
        found = [m.group(0) for m in YEAR.finditer(plain)]
        record["year"] = found[-1] if found else ""
        # natbib puts the year straight after the title when there is no venue:
        # "Hindsight credit assignment, 2019."
        if record["year"] and record["title"].endswith(f", {record['year']}"):
            record["title"] = record["title"][: -len(record["year"]) - 2]

    return record


# ---------------------------------------------------------------------------
# Gemini parsing
# ---------------------------------------------------------------------------

PARSE_PROMPT = """You split academic references into fields. Copy text EXACTLY as it appears;
never invent, correct or complete anything. If a field is absent, use "".

Return ONLY a JSON array with one object per reference, in the same order:
[{{"authors": [{{"given": "Qi", "family": "Dou"}}], "title": "...", "venue": "...",
  "year": "2017", "volume": "", "issue": "", "pages": "559-567", "url": "", "doi": ""}}]

Rules:
- "given" holds first names or initials, "family" the surname (keep particles like
  "van der" in family). An organisation author goes in "family" with "given" empty.
- "venue" is the journal, conference, publisher, or "arXiv preprint arXiv:XXXX.XXXXX".
- Do not include the year, volume or pages inside "venue".

References:
{refs}
"""

BATCH = 20


def _grounded(record: Record, raw: str) -> bool:
    """Reject a Gemini answer that is not supported by the original text.

    The title's words and the year must actually appear in the raw reference.
    This is how we stop the model from inventing details.
    """
    raw_norm = strip_accents(raw).lower()
    title_words = [w for w in re.findall(r"[a-z]{4,}", strip_accents(record.get("title", "")).lower())]
    if not title_words:
        return False
    present = sum(1 for w in title_words if w in raw_norm) / len(title_words)
    year_ok = not record.get("year") or str(record["year"])[:4] in raw
    return present >= 0.8 and year_ok


def parse_references(raws: List[str], llm: BaseLLM) -> Tuple[List[Record], List[str]]:
    """Parse every reference. Gemini first (if available), rules as a backup.

    Returns (records, notes).
    """
    notes: List[str] = []
    records: List[Optional[Record]] = [None] * len(raws)

    if llm.available and raws:
        for start in range(0, len(raws), BATCH):
            chunk = raws[start:start + BATCH]
            # Repair line-wrapping damage BEFORE Gemini sees the text, so its
            # (exact) copy of URLs and DOIs is already clean.
            numbered = "\n".join(f"{i + 1}. {_repair(r)}" for i, r in enumerate(chunk))
            try:
                answer = llm.ask_json("parse_references", PARSE_PROMPT.format(refs=numbered))
            except LLMUnavailable as error:
                notes.append(f"Gemini unavailable for references ({error}); used rules.")
                break
            if not isinstance(answer, list):
                notes.append("Gemini returned references in an unexpected shape; used rules.")
                continue
            for offset, item in enumerate(answer[:len(chunk)]):
                raw = chunk[offset]
                if isinstance(item, dict) and _grounded(item, raw):
                    item = {k: item.get(k, "") for k in
                            ("authors", "title", "venue", "year", "volume", "issue", "pages",
                             "url", "doi")}
                    item["authors"] = [a for a in item["authors"] if isinstance(a, dict)]
                    item.update(raw=raw, source="gemini")
                    _tidy_fields(item)
                    records[start + offset] = item

    rejected = 0
    for i, raw in enumerate(raws):
        if records[i] is None:
            if llm.available:
                rejected += 1
            records[i] = parse_reference_rules(raw)
    if llm.available and rejected and not notes:
        notes.append(f"{rejected} reference(s) parsed by rules after Gemini's answer "
                     "failed the grounding check.")
    return [r for r in records if r is not None], notes


# ---------------------------------------------------------------------------
# IEEE formatting (always plain Python)
# ---------------------------------------------------------------------------

CONFERENCE_WORDS = re.compile(
    r"\b(proc\.?|proceedings|conference|conf\.|workshop|symposium|meeting|"
    r"neurips|nips|icml|iclr|cvpr|iccv|eccv|acl|emnlp|naacl|aaai|ijcai|kdd|miccai|"
    r"computing and computer assisted intervention|advances in neural information processing)\b",
    re.I)


def reference_type(record: Record) -> str:
    venue = record.get("venue", "")
    if re.search(r"arxiv", venue, re.I) or (not venue and ARXIV_ID.search(
            record.get("raw", "") + " " + record.get("url", ""))):
        return "arxiv"
    if CONFERENCE_WORDS.search(venue):
        return "conference"
    if record.get("volume"):
        return "journal"
    if not venue and record.get("url"):
        return "web"
    return "other"


_ABBREVIATABLE = re.compile(r"\b(" + "|".join(sorted(R.VENUE_ABBREVIATIONS, key=len, reverse=True))
                            + r")\b")


def abbreviate_venue(venue: str) -> str:
    """REF-06: IEEE word abbreviations for journal and conference names.

    "IEEE Transactions on Medical Imaging" -> "IEEE Trans. Med. Imag."
    "Journal of Machine Learning Research" -> "J. Mach. Learn. Res."
    A one-word name ("Nature") is never abbreviated, and the small joining
    words ("of", "on", "the", "and") are dropped once a name is abbreviated.
    """
    words = venue.split()
    if len(words) < 2 or not _ABBREVIATABLE.search(venue):
        return venue
    out = []
    for i, word in enumerate(words):
        core = word.rstrip(",:;")
        tail = word[len(core):]
        if core in R.VENUE_ABBREVIATIONS:
            out.append(R.VENUE_ABBREVIATIONS[core] + tail)
        elif core.lower() in R.VENUE_DROP_WORDS and i > 0:
            continue
        else:
            out.append(word)
    return " ".join(out)


def unabbreviated_words(venue: str) -> List[str]:
    """Words in a venue name that IEEE would abbreviate (for the compliance check)."""
    if len(venue.split()) < 2:
        return []
    return _ABBREVIATABLE.findall(venue)


def format_ieee(record: Record) -> Segments:
    """Write one reference in IEEE style, as (text, italic) pieces."""
    pieces: Segments = []
    authors = ieee_author_list(record.get("authors", []))
    title = _clean(record.get("title", ""))
    venue = _clean(record.get("venue", ""))
    year = str(record.get("year", "")).strip() or R.NO_DATE      # REF-10
    pages = (record.get("pages") or "").replace("-", "–").replace(" ", "")
    kind = reference_type(record)
    if kind in ("journal", "conference"):
        venue = abbreviate_venue(re.sub(r"^(In\s+)?Proceedings of (the )?", "Proc. ", venue))

    if authors:
        pieces.append((f"{authors}, ", False))
    if title:
        # a title ending in "?" or "!" takes no extra comma: "...a 5th grader?" in
        comma = "" if title.endswith(("?", "!")) else ","
        pieces.append((f"“{title}{comma}” ", False))

    if kind == "journal":
        pieces.append((venue, True))
        tail = []
        if record.get("volume"):
            tail.append(f"vol. {record['volume']}")
        if record.get("issue"):
            tail.append(f"no. {record['issue']}")
        if pages:
            tail.append(f"pp. {pages}" if "–" in pages else f"p. {pages}")
        if year:
            tail.append(year)
        pieces.append(((", " + ", ".join(tail)) if tail else "", False))
    elif kind == "conference":
        # REF-07: "in Proc. ..." ("Proceedings of the" was shortened above)
        pieces.append(("in ", False))
        pieces.append((venue, True))
        tail = [f"vol. {record['volume']}"] if record.get("volume") else []
        if year:
            tail.append(year)
        if pages:
            tail.append(f"pp. {pages}" if "–" in pages else f"p. {pages}")
        pieces.append(((", " + ", ".join(tail)) if tail else "", False))
    elif kind == "arxiv":
        found = ARXIV_ID.search(venue) or ARXIV_ID.search(
            record.get("raw", "") + " " + record.get("url", ""))
        label = f"arXiv:{found.group(1)}" if found else (venue or "arXiv preprint")
        # REF-09: J. K. Author, "Title," year, arXiv:xxxx.
        pieces.append(((f"{year}, " if year else "") + label, False))
    else:
        if venue:
            pieces.append((venue, True))
            pieces.append((f", {year}" if year else "", False))
        elif year:
            pieces.append((year, False))

    # finish with a full stop, then any online link
    text_so_far = "".join(t for t, _ in pieces).rstrip(" ,")
    pieces = _trim_trailing(pieces)
    if not text_so_far.endswith("."):
        pieces.append((".", False))
    if record.get("doi"):
        pieces.append((f" doi: {record['doi']}.", False))
    elif record.get("url") and kind in ("web", "other"):
        pieces.append((f" [Online]. Available: {record['url']}", False))
    return [(t, i) for t, i in pieces if t]


def _trim_trailing(pieces: Segments) -> Segments:
    """Remove trailing spaces/commas from the last non-empty piece."""
    out = [p for p in pieces if p[0]]
    if out:
        text, italic = out[-1]
        out[-1] = (text.rstrip(" ,"), italic)
    return out


def segments_to_text(segments: Segments) -> str:
    return "".join(t for t, _ in segments)
