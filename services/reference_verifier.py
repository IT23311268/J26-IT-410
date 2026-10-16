import re
import sys
import os
import logging
import unicodedata
import urllib.parse
import time
import requests
import difflib
import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ── Utility ────────────────────────────────────────────────────────────────────

def safe_print(text):
    try:
        sys.stdout.write(str(text) + "\n")
        sys.stdout.flush()
    except UnicodeEncodeError:
        sys.stdout.write(str(text).encode('ascii', 'replace').decode('ascii') + "\n")
        sys.stdout.flush()

# ── Reference Extraction & Parsing (UNCHANGED) ─────────────────────────────────

def parse_references_from_text(ref_text: str) -> dict:
    references_by_number = {}

    # Check for bracketed markers like [1], [2], ...
    bracket_matches = list(re.finditer(r"^\s*\[(\d+)\]\s*", ref_text, re.MULTILINE))
    if bracket_matches:
        for i, match in enumerate(bracket_matches):
            num_str = match.group(1)
            start_pos = match.end()
            end_pos = bracket_matches[i+1].start() if i + 1 < len(bracket_matches) else len(ref_text)
            content = ref_text[start_pos:end_pos].strip()
            content = re.sub(r"\s+", " ", content).strip()
            if content:
                references_by_number[num_str] = content

    elif re.search(r"^\s*1\.\s+", ref_text, re.MULTILINE):
        dotted_matches = list(re.finditer(r"^\s*(\d{1,3})\.\s+", ref_text, re.MULTILINE))
        if dotted_matches:
            for i, match in enumerate(dotted_matches):
                num_str = match.group(1)
                start_pos = match.end()
                end_pos = dotted_matches[i+1].start() if i + 1 < len(dotted_matches) else len(ref_text)
                content = ref_text[start_pos:end_pos].strip()
                content = re.sub(r"\s+", " ", content).strip()
                if content:
                    references_by_number[num_str] = content
    else:
        parts = re.split(r"\n\s*\n", ref_text)
        idx = 1
        for p in parts:
            p_clean = re.sub(r"\s+", " ", p).strip()
            if len(p_clean) > 20:
                references_by_number[str(idx)] = p_clean
                idx += 1

    sorted_keys = sorted(references_by_number.keys(), key=lambda x: int(x) if x.isdigit() else 9999)
    detected_numbers = [int(k) for k in sorted_keys if k.isdigit()]

    ref_11 = references_by_number.get("11", "")
    contains_online = "YES" if "[online]" in ref_11.lower() else "NO"
    contains_url = "YES" if ("http://" in ref_11.lower() or "https://" in ref_11.lower()) else "NO"
    standalone_url = "YES" if any(r.strip().startswith("[Online]") or r.strip().startswith("http") for r in references_by_number.values()) else "NO"

    safe_print("\n[REFERENCE PARSER]")
    safe_print(f"detected_numbers: {detected_numbers}")
    safe_print(f"parsed_count: {len(detected_numbers)}")
    if "11" in references_by_number:
        safe_print("")
        safe_print("For reference 11:")
        safe_print("[11]")
        safe_print(f"contains_online_line: {contains_online}")
        safe_print(f"contains_url: {contains_url}")
        safe_print(f"standalone_url_entry_created: {standalone_url}")

    return references_by_number

def parse_bibliography_text(ref_text: str) -> list:
    refs_by_number = parse_references_from_text(ref_text)
    sorted_keys = sorted(refs_by_number.keys(), key=lambda x: int(x) if x.isdigit() else 9999)
    return [refs_by_number[k] for k in sorted_keys]

def extract_references_from_pdf(pdf_path: str) -> list:
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"

        lines = full_text.split("\n")
        ref_start_idx = -1

        for i in range(len(lines)-1, -1, -1):
            line = lines[i].strip().lower()
            if line in ["references", "bibliography", "references.", "bibliography.", "references:", "bibliography:"]:
                ref_start_idx = i
                break

        if ref_start_idx == -1:
            for i in range(len(lines)-1, -1, -1):
                line = lines[i].strip().lower()
                if re.match(r"^\d*\.?\s*(references|bibliography)$", line):
                    ref_start_idx = i
                    break

        if ref_start_idx == -1:
            logger.warning(f"Could not find references section in {pdf_path}")
            return []

        ref_text = "\n".join(lines[ref_start_idx+1:])

        refs_by_number = parse_references_from_text(ref_text)
        sorted_keys = sorted(refs_by_number.keys(), key=lambda x: int(x) if x.isdigit() else 9999)
        refs = [refs_by_number[k] for k in sorted_keys]
        return refs
    except Exception as e:
        logger.error(f"Error extracting references from {pdf_path}: {str(e)}")
        return []

# ── Normalization Utilities (Step 1) ───────────────────────────────────────────

def clean_unicode(text: str) -> str:
    """Normalize Unicode characters, handle ligatures, and fix column hyphenation breaks."""
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text)
    ligatures = {
        '\ufb00': 'ff', '\ufb01': 'fi', '\ufb02': 'fl',
        '\ufb03': 'ffi', '\ufb04': 'ffl', '\ufb05': 'ft', '\ufb06': 'st'
    }
    for k, v in ligatures.items():
        text = text.replace(k, v)
    # Fix hyphenated line breaks in words:
    # "Path- ways" -> "Pathways", "Transform- ers" -> "Transformers", "Detec- tion" -> "Detection"
    text = re.sub(r'(\b[a-zA-Z]{2,})-\s+([a-z]{2,}\b)', r'\1\2', text)
    # "Knowledge- Intensive" -> "Knowledge-Intensive"
    text = re.sub(r'(\b[a-zA-Z]{2,})-\s+([A-Z][a-zA-Z]*\b)', r'\1-\2', text)
    text = re.sub(r'[\u2010\u2011\u2012\u2013\u2014\u2015]', '-', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text(text: str) -> str:
    """Lowercase, strip edition boilerplate, punctuation, and extra whitespace."""
    if not text:
        return ""
    text = clean_unicode(text)
    text = text.lower()
    text = re.sub(r'\(\s*\d+(?:st|nd|rd|th)?\s+edition\s*\)', '', text)
    text = re.sub(r'\b\d+(?:st|nd|rd|th)\s+ed(?:ition)?\.?\b', '', text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_approx_title(ref_text: str) -> str:
    """Extract approximate paper/book title from reference text."""
    text = clean_unicode(ref_text)
    clean_no_url = re.sub(r"https?://\S+", "", text)
    clean_no_url = re.sub(r"arxiv:\s*\d+\.\d+(?:v\d+)?", "", clean_no_url, flags=re.I)
    clean_no_url = re.sub(r"doi:\s*10\.\S+", "", clean_no_url, flags=re.I)

    # Priority 1: Quoted titles (handles standard, curved, and guillemet quotes)
    quoted = re.search(r'["\u201c\u00ab\u2018]([^"\u201c\u201d\u00ab\u00bb\u2018\u2019]{4,}?)["\u201d\u00bb\u2019]', clean_no_url)
    if quoted:
        title = quoted.group(1).strip().rstrip('.,;:')
        if len(title.split()) >= 2:
            return title

    # Priority 2: Author list ending with colon: "Author1, Author2: Title"
    colon_m = re.match(r'^\s*([A-Z][^:]{3,80}?):\s+([A-Z0-9].+)', clean_no_url)
    if colon_m:
        author_part = colon_m.group(1).strip()
        rest = colon_m.group(2).strip()
        if ("," in author_part or "et al" in author_part.lower() or re.search(r'\b[A-Z]\.', author_part)) and not author_part.lower().startswith("in"):
            title_candidate = re.split(r'\.\s+(?:In:|In\s+[A-Z]|Prentice|Proc|Conference|Journal|Vol|IEEE|ACM|Springer|Published|\(\d{4}\))', rest, flags=re.I)[0]
            parts = re.split(r'\.\s+', title_candidate)
            if parts:
                title = parts[0].strip().rstrip('.,;:')
                if len(parts) > 1 and len(title.split()) < 4:
                    title = (parts[0] + ". " + parts[1]).strip().rstrip('.,;:')
                title = re.sub(r'\s*\(\d{4}\).*$', '', title).strip().rstrip('.,;:')
                if len(title.split()) >= 2:
                    return title

    # Priority 3: Author list ending with period: "Author, A. Title. Journal..."
    author_dot_m = re.match(r'^\s*([A-Z][a-zA-Z\s,.\'-]+?(?:et al\.|[A-Z]\.))\s+([A-Z0-9].+)', clean_no_url)
    if author_dot_m:
        rest = author_dot_m.group(2).strip()
        title = re.split(r'\.\s+(?:In:|In\s+[A-Z]|Journal|Proc|Conference|Vol|IEEE|ACM|\d{4})', rest, flags=re.I)[0]
        parts = re.split(r'\.\s+', title)
        if parts:
            t = parts[0].strip().rstrip('.,;:')
            t = re.sub(r'\s*\(\d{4}\).*$', '', t).strip().rstrip('.,;:')
            if len(t.split()) >= 2:
                return t

    # Priority 4: Organization report "OpenAI: GPT-4V(ision) System Card (2023)..."
    org_m = re.match(r'^\s*([A-Za-z0-9\s]+):\s+([A-Za-z0-9].+)', clean_no_url)
    if org_m:
        rest = org_m.group(2).strip()
        title = re.split(r'\s*\(\d{4}\)', rest)[0].strip().rstrip('.,;:')
        if len(title.split()) >= 2:
            return title

    # Fallback split
    parts = re.split(r'(?<=[a-z0-9])\.\s+', clean_no_url)
    if len(parts) >= 2:
        return parts[1].strip()[:100]
    return clean_no_url[:100]

def extract_author_surnames_from_ref(ref_text: str) -> tuple:
    """Extract author surnames from the citation header."""
    text = clean_unicode(ref_text)
    has_et_al = bool(re.search(r'\bet al\b', text, re.I))

    quote_m = re.search(r'["\u201c\u00ab\u2018]', text)
    if quote_m:
        author_block = text[:quote_m.start()].strip()
    elif ":" in text[:80] and not text[:80].lower().startswith("http"):
        author_block = text[:text.index(":")].strip()
    else:
        author_block = text[:80].strip()

    author_block = re.sub(r'\bet al\.?', '', author_block, flags=re.I)
    author_block = re.sub(r'\band\b|&', ',', author_block, flags=re.I)

    surnames = []
    chunks = re.split(r'[,;]', author_block)
    for chunk in chunks:
        words = chunk.strip().split()
        for w in words:
            clean_w = re.sub(r'^[^\w]+|[^\w]+$', '', w)
            if len(clean_w) >= 2 and not clean_w.isupper() and clean_w.isalpha():
                surnames.append(clean_w.lower())
            elif len(clean_w) >= 2 and clean_w.isalpha() and not re.match(r'^[A-Z]\.?$', w):
                surnames.append(clean_w.lower())

    stop = {"in", "proc", "ieee", "acm", "proceedings", "conference", "journal", "vol", "no", "pp", "at", "by", "on"}
    surnames = [s for s in surnames if s not in stop]
    return surnames, has_et_al

def extract_surnames_from_author_list(authors: list) -> list:
    """Extract normalized surnames from candidate author representations."""
    surnames = []
    for a in authors:
        if isinstance(a, dict):
            fam = a.get("family") or a.get("name") or ""
            if fam:
                surnames.append(normalize_text(fam))
        elif isinstance(a, str):
            clean_a = clean_unicode(a).strip()
            if "," in clean_a:
                last = clean_a.split(",")[0].strip()
                surnames.append(normalize_text(last))
            else:
                parts = clean_a.split()
                if parts:
                    surnames.append(normalize_text(parts[-1]))
    return [s for s in surnames if s]

def get_source_type(ref_text: str) -> str:
    ref_lower = ref_text.lower()
    if "openreview" in ref_lower or "iclr" in ref_lower or "international conference on learning representations" in ref_lower:
        return "openreview_conference"
    elif any(k in ref_lower for k in ["system card", "technical report", "white paper", "research report", "openai", "google research", "deepmind", "meta ai", "microsoft research", "facebook ai"]):
        return "official_report"
    elif "edition" in ref_lower or "press" in ref_lower or "publisher" in ref_lower or "isbn" in ref_lower or "textbook" in ref_lower:
        return "book"
    elif "arxiv" in ref_lower:
        return "arxiv"
    return "journal_conference"

def parse_reference(ref_text: str) -> dict:
    doi_match = re.search(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", ref_text, re.I)
    doi = doi_match.group(1).rstrip('.') if doi_match else None

    arxiv_match = re.search(r"arxiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)", ref_text, re.I)
    if not arxiv_match:
        arxiv_match = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)", ref_text, re.I)
    arxiv = arxiv_match.group(1) if arxiv_match else None

    # Year extraction: prioritize year directly following quoted title or parentheses
    year = None
    m_after_quote = re.search(r'["\u201d\u00bb\u2019]\s*,?\s*\(?(19\d{2}|20\d{2})\)?', ref_text)
    if m_after_quote:
        year = m_after_quote.group(1)
    else:
        clean_ref = re.split(r'\bPublished by\b|\bPage \d+\b|\bISSN:\b', ref_text)[0]
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", clean_ref)
        if years:
            paren_year = re.search(r"\((19\d{2}|20\d{2})\)", clean_ref)
            year = paren_year.group(1) if paren_year else years[0]

    url_match = re.search(r"https?://[^\s)]+", ref_text)
    url = url_match.group(0).rstrip('.') if url_match else None

    surnames, has_et_al = extract_author_surnames_from_ref(ref_text)
    approx_title = extract_approx_title(ref_text)

    venue_hints = []
    for v in ["NeurIPS", "ICML", "ICLR", "ACL", "NAACL", "EMNLP", "KDD", "TACL", "JMLR", "IEEE", "ACM", "Prentice-Hall", "Prentice Hall"]:
        if re.search(r'\b' + re.escape(v) + r'\b', ref_text, re.I):
            venue_hints.append(v)

    return {
        "doi": doi,
        "arxiv": arxiv,
        "year": year,
        "url": url,
        "source_type": get_source_type(ref_text),
        "approx_title": approx_title,
        "author_surnames": surnames,
        "has_et_al": has_et_al,
        "venue_hints": venue_hints,
        "full_text": ref_text
    }

def clean_year(y):
    if not y:
        return None
    m = re.search(r"(19\d{2}|20\d{2})", str(y))
    return int(m.group(1)) if m else None

# ── Multi-Source API Clients (Step 2) ──────────────────────────────────────────

def query_doi(doi: str) -> Optional[dict]:
    """Lookup exact DOI via Crossref, with DataCite fallback."""
    url = f"https://api.crossref.org/works/{doi}"
    headers = {"User-Agent": "HallucinationDetector/1.0 (mailto:admin@example.com)"}
    try:
        resp = requests.get(url, headers=headers, timeout=6)
        if resp.status_code == 200:
            item = resp.json().get("message", {})
            title = item.get("title", [""])[0] if item.get("title") else ""
            authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item.get("author", [])]
            year = ""
            if item.get("issued") and item.get("issued").get("date-parts"):
                year = str(item["issued"]["date-parts"][0][0])
            return {"source": "Crossref (DOI)", "title": title, "authors": authors, "year": year, "doi": doi, "arxiv": None, "raw": item}
        elif resp.status_code == 404:
            dc_url = f"https://api.datacite.org/dois/{doi}"
            dc_resp = requests.get(dc_url, timeout=6)
            if dc_resp.status_code == 200:
                attrs = dc_resp.json().get("data", {}).get("attributes", {})
                title = attrs.get("titles", [{}])[0].get("title", "") if attrs.get("titles") else ""
                authors = [c.get("name", "") for c in attrs.get("creators", [])]
                year = str(attrs.get("publicationYear") or "")
                return {"source": "DataCite (DOI)", "title": title, "authors": authors, "year": year, "doi": doi, "arxiv": None, "raw": attrs}
    except Exception:
        pass
    return None

def query_arxiv(arxiv_id: str) -> Optional[dict]:
    """Lookup exact arXiv ID via arXiv Atom API, with DataCite fallback."""
    clean_id = arxiv_id.lower().replace("arxiv:", "").strip()
    headers = {"User-Agent": "HallucinationDetector/1.0 (mailto:admin@example.com)"}
    try:
        url = f"https://export.arxiv.org/api/query?id_list={clean_id}"
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            root = ET.fromstring(resp.text)
            entry = root.find("{http://www.w3.org/2005/Atom}entry")
            if entry is not None:
                title_el = entry.find("{http://www.w3.org/2005/Atom}title")
                if title_el is not None and title_el.text:
                    title = title_el.text.replace("\n", " ").strip()
                    authors = [a.find("{http://www.w3.org/2005/Atom}name").text for a in entry.findall("{http://www.w3.org/2005/Atom}author")]
                    published = entry.find("{http://www.w3.org/2005/Atom}published").text
                    year = published[:4] if published else ""
                    return {"source": "arXiv (ID)", "title": title, "authors": authors, "year": year, "doi": None, "arxiv": clean_id, "raw": {"title": title, "authors": authors}}
    except Exception:
        pass

    try:
        dc_doi = f"10.48550/arXiv.{clean_id}"
        dc_url = f"https://api.datacite.org/dois/{dc_doi}"
        resp = requests.get(dc_url, timeout=5)
        if resp.status_code == 200:
            attrs = resp.json().get("data", {}).get("attributes", {})
            title = attrs.get("titles", [{}])[0].get("title", "") if attrs.get("titles") else ""
            authors = [c.get("name", "") for c in attrs.get("creators", [])]
            year = str(attrs.get("publicationYear") or "")
            return {"source": "arXiv (DataCite)", "title": title, "authors": authors, "year": year, "doi": dc_doi, "arxiv": clean_id, "raw": attrs}
    except Exception:
        pass
    return None

def query_arxiv_search(approx_title: str) -> list:
    """Search arXiv via Atom API for papers matching title / subtitle."""
    results = []
    if not approx_title or len(approx_title.strip()) < 3:
        return results
    headers = {"User-Agent": "HallucinationDetector/1.0 (mailto:admin@example.com)"}

    clean = re.sub(r'[^\w\s]', ' ', approx_title)
    clean = re.sub(r'\s+', ' ', clean).strip()

    search_queries = [f'ti:"{clean}"']
    if ":" in approx_title:
        parts = approx_title.split(":", 1)
        sub = re.sub(r'[^\w\s]', ' ', parts[1]).strip()
        if len(sub.split()) >= 3:
            search_queries.append(f'ti:"{sub}"')
            sub_short = " ".join(sub.split()[:5])
            search_queries.append(f'ti:"{sub_short}"')
        prefix = re.sub(r'[^\w\s]', ' ', parts[0]).strip()
        search_queries.append(f'ti:"{prefix}" AND ti:"{sub.split()[0]}"')
    else:
        words = clean.split()
        if len(words) > 6:
            search_queries.append(f'ti:"{" ".join(words[:6])}"')

    for q in search_queries:
        try:
            url = f"https://export.arxiv.org/api/query?search_query={urllib.parse.quote(q)}&max_results=2"
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                    title_el = entry.find("{http://www.w3.org/2005/Atom}title")
                    if title_el is not None and title_el.text:
                        title = title_el.text.replace("\n", " ").strip()
                        authors = [a.find("{http://www.w3.org/2005/Atom}name").text for a in entry.findall("{http://www.w3.org/2005/Atom}author")]
                        published = entry.find("{http://www.w3.org/2005/Atom}published").text
                        year = published[:4] if published else ""
                        id_el = entry.find("{http://www.w3.org/2005/Atom}id")
                        arxiv_id = id_el.text.split('/')[-1] if id_el is not None else None
                        results.append({"source": "arXiv (Search)", "title": title, "authors": authors, "year": year, "doi": None, "arxiv": arxiv_id, "raw": {"title": title, "authors": authors}})
                if results:
                    break
        except Exception:
            pass
    return results

def query_crossref_biblio(ref_text: str, approx_title: str = None) -> list:
    """Bibliographic and title search on Crossref."""
    results = []
    headers = {"User-Agent": "HallucinationDetector/1.0 (mailto:admin@example.com)"}

    try:
        url = "https://api.crossref.org/works"
        params = {"query.bibliographic": ref_text[:200], "rows": 3, "select": "title,author,DOI,issued"}
        resp = requests.get(url, params=params, headers=headers, timeout=6)
        if resp.status_code == 200:
            for item in resp.json().get("message", {}).get("items", []):
                title = item.get("title", [""])[0] if item.get("title") else ""
                authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item.get("author", [])]
                year = ""
                if item.get("issued") and item.get("issued").get("date-parts"):
                    year = str(item["issued"]["date-parts"][0][0])
                doi = item.get("DOI")
                results.append({"source": "Crossref (Search)", "title": title, "authors": authors, "year": year, "doi": doi, "arxiv": None, "raw": item})
    except Exception:
        pass

    if approx_title and len(approx_title.split()) >= 2:
        try:
            url = "https://api.crossref.org/works"
            params = {"query.title": approx_title, "rows": 3, "select": "title,author,DOI,issued"}
            resp = requests.get(url, params=params, headers=headers, timeout=6)
            if resp.status_code == 200:
                for item in resp.json().get("message", {}).get("items", []):
                    title = item.get("title", [""])[0] if item.get("title") else ""
                    authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item.get("author", [])]
                    year = ""
                    if item.get("issued") and item.get("issued").get("date-parts"):
                        year = str(item["issued"]["date-parts"][0][0])
                    doi = item.get("DOI")
                    results.append({"source": "Crossref (Title)", "title": title, "authors": authors, "year": year, "doi": doi, "arxiv": None, "raw": item})
        except Exception:
            pass

    return results

def query_openalex(query: str, source_label: str = "OpenAlex") -> list:
    """Search OpenAlex for academic works."""
    results = []
    if not query or len(query.strip()) < 3:
        return results
    url = "https://api.openalex.org/works"
    params = {"search": query, "per-page": 3, "mailto": "admin@example.com"}
    try:
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            for item in resp.json().get("results", []):
                title = item.get("title") or ""
                authors = [a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])]
                year = str(item.get("publication_year")) if item.get("publication_year") else ""
                doi_url = item.get("doi")
                doi = doi_url.replace("https://doi.org/", "") if doi_url else None
                results.append({"source": source_label, "title": title, "authors": authors, "year": year, "doi": doi, "arxiv": None, "raw": item})
    except Exception:
        pass
    return results

def query_openlibrary(query: str) -> list:
    """Search OpenLibrary for book citations."""
    results = []
    if not query:
        return results
    clean_q = re.sub(r'\s*\([^)]*\)', '', query).strip()
    url = "https://openlibrary.org/search.json"
    params = {"q": clean_q, "limit": 2}
    try:
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            for doc in resp.json().get("docs", []):
                title = doc.get("title", "")
                authors = doc.get("author_name", [])
                year = str(doc.get("first_publish_year")) if doc.get("first_publish_year") else ""
                results.append({"source": "OpenLibrary", "title": title, "authors": authors, "year": year, "doi": None, "arxiv": None, "raw": doc})
    except Exception:
        pass
    return results

def query_semantic_scholar(query: str) -> list:
    """
    Search Semantic Scholar — covers ACL Anthology, OpenAI/Google/Meta technical
    reports, and most academic papers not indexed by Crossref or arXiv directly.
    Retries with a shorter query on 429 or empty results.
    """
    results = []
    if not query or len(query.strip()) < 3:
        return results

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"User-Agent": "HallucinationDetector/1.0 (mailto:admin@example.com)"}

    # Build query list: try full title first, then first-6-word prefix
    queries_to_try = [query]
    words = query.split()
    if len(words) > 6:
        queries_to_try.append(" ".join(words[:6]))
    elif len(words) > 4:
        queries_to_try.append(" ".join(words[:4]))

    for attempt, q in enumerate(queries_to_try):
        params = {
            "query": q,
            "fields": "title,authors,year,externalIds",
            "limit": 3
        }
        try:
            if attempt > 0:
                time.sleep(1.0)   # brief pause before retry
            resp = requests.get(url, params=params, headers=headers, timeout=6)
            if resp.status_code == 429:
                # Rate-limited — try shorter query on next iteration
                continue
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if not data:
                    continue   # empty — try shorter query
                for paper in data:
                    title = paper.get("title") or ""
                    authors = [a.get("name", "") for a in paper.get("authors", [])]
                    year = str(paper.get("year")) if paper.get("year") else ""
                    ext_ids = paper.get("externalIds") or {}
                    doi = ext_ids.get("DOI")
                    arxiv_id = ext_ids.get("ArXiv")
                    results.append({
                        "source": "Semantic Scholar",
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "doi": doi,
                        "arxiv": arxiv_id,
                        "raw": paper
                    })
                break   # got data — stop retrying
        except Exception:
            pass
    return results

def query_dblp(approx_title: str) -> list:
    """
    Search DBLP — covers conference papers, workshop papers, and informal/org
    publications (OpenAI, Google Brain, DeepMind blog posts, etc.) via a stable
    public JSON API without strict rate limits.
    """
    results = []
    if not approx_title or len(approx_title.strip()) < 3:
        return results
    url = "https://dblp.org/search/publ/api"
    params = {"q": approx_title, "h": 3, "format": "json"}
    headers = {"User-Agent": "HallucinationDetector/1.0 (mailto:admin@example.com)"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=6)
        if resp.status_code == 200:
            hits = resp.json().get("result", {}).get("hits", {}).get("hit", [])
            for hit in hits:
                info = hit.get("info", {})
                title = info.get("title", "")
                # DBLP authors: can be a single dict or a list of dicts
                authors_raw = info.get("authors", {}).get("author", [])
                if isinstance(authors_raw, dict):
                    authors_raw = [authors_raw]
                authors = [a.get("text", "") for a in authors_raw if isinstance(a, dict)]
                year = str(info.get("year", ""))
                doi = info.get("doi")
                results.append({
                    "source": "DBLP",
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "doi": doi,
                    "arxiv": None,
                    "raw": info
                })
    except Exception:
        pass
    return results

def query_official_report(approx_title: str) -> list:
    """
    Lookup official organization reports (OpenAI, Google, Meta, Microsoft, DeepMind, etc.)
    and institutional preprints/whitepapers via Wikidata's scholarly entity index.
    """
    results = []
    if not approx_title or len(approx_title.strip()) < 3:
        return results
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": approx_title,
        "language": "en",
        "format": "json",
        "limit": 3
    }
    headers = {"User-Agent": "HallucinationDetector/1.0 (mailto:admin@example.com)"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=5)
        if r.status_code == 200:
            for item in r.json().get("search", []):
                desc = item.get("description", "").lower()
                label = item.get("label", "")
                if any(w in desc for w in ["article", "paper", "publication", "report", "preprint", "research"]):
                    m_year = re.search(r"\b(19\d{2}|20\d{2})\b", desc)
                    year = m_year.group(1) if m_year else ""
                    ent_id = item.get("id")
                    authors = []
                    try:
                        r_ent = requests.get(
                            url,
                            params={"action": "wbgetentities", "ids": ent_id, "props": "claims", "format": "json"},
                            headers=headers,
                            timeout=4
                        )
                        claims = r_ent.json().get("entities", {}).get(ent_id, {}).get("claims", {})
                        a_ids = [
                            p.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
                            for p in claims.get("P50", [])
                            if p.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
                        ][:5]
                        if a_ids:
                            r_a = requests.get(
                                url,
                                params={"action": "wbgetentities", "ids": "|".join(a_ids), "props": "labels", "languages": "en", "format": "json"},
                                headers=headers,
                                timeout=4
                            )
                            authors = [v.get("labels", {}).get("en", {}).get("value", "") for v in r_a.json().get("entities", {}).values()]
                    except Exception:
                        pass
                    work_url = None
                    try:
                        p953 = claims.get("P953", [])
                        if p953:
                            v = p953[0].get("mainsnak", {}).get("datavalue", {}).get("value")
                            if isinstance(v, str) and v.startswith("http"):
                                work_url = v
                    except Exception:
                        pass
                    results.append({
                        "source": "Official Report",
                        "title": label,
                        "authors": authors,
                        "year": year,
                        "doi": None,
                        "arxiv": None,
                        "url": work_url,
                        "raw": item
                    })
    except Exception:
        pass
    return results

# ── Separate Metadata Scoring (Step 3) ─────────────────────────────────────────

def compute_title_score(cand_title: str, ref_title: str, ref_full_text: str) -> float:
    """
    Score title similarity using exact match, prefix match, colon-split match,
    and token Jaccard with SequenceMatcher blend.
    """
    t_cand = normalize_text(cand_title)
    t_ref = normalize_text(ref_title)
    if not t_cand or not t_ref:
        return 0.0

    if t_cand == t_ref:
        return 1.0

    # Colon split match (e.g. "XGBoost: A Scalable..." vs "XGBoost", "BERT: ...")
    cand_main = t_cand.split(':')[0].strip()
    ref_main = t_ref.split(':')[0].strip()
    if cand_main == t_ref or ref_main == t_cand or cand_main == ref_main:
        return 0.95

    # Prefix match
    if t_cand.startswith(t_ref) or t_ref.startswith(t_cand):
        shorter_len = min(len(t_cand.split()), len(t_ref.split()))
        longer_len = max(len(t_cand.split()), len(t_ref.split()))
        if shorter_len / longer_len >= 0.60:
            return 0.90
        elif shorter_len >= 4 and shorter_len / longer_len >= 0.40:
            return 0.75

    cand_tokens = set(t_cand.split())
    ref_tokens = set(t_ref.split())
    if not cand_tokens or not ref_tokens:
        return 0.0

    inter = cand_tokens & ref_tokens
    union = cand_tokens | ref_tokens
    jaccard = len(inter) / len(union) if union else 0.0
    seq_ratio = difflib.SequenceMatcher(None, t_cand, t_ref).ratio()

    # Contiguous substring check with coverage gating
    if f" {t_ref} " in f" {t_cand} " or f" {t_cand} " in f" {t_ref} ":
        shorter_len = min(len(t_cand.split()), len(t_ref.split()))
        longer_len = max(len(t_cand.split()), len(t_ref.split()))
        if shorter_len / longer_len >= 0.70:
            return 0.85
        else:
            token_score = max(jaccard, 0.40 + 0.30 * (shorter_len / longer_len))
            return min(1.0, round(token_score, 4))

    token_score = max(jaccard, 0.70 * jaccard + 0.30 * seq_ratio)

    t_full = normalize_text(ref_full_text)
    if f" {t_cand} " in f" {t_full} " and len(t_cand.split()) >= 3:
        token_score = max(token_score, 0.92)

    return min(1.0, round(token_score, 4))

def compute_author_score(cand_authors: list, ref_surnames: list, ref_has_et_al: bool, ref_full_text: str) -> float:
    """Score author surname agreement, tolerating et al., initials, and missing middle names."""
    cand_surnames = extract_surnames_from_author_list(cand_authors)

    if not cand_surnames:
        return 0.5  # Neutral when candidate does not specify authors

    if ref_surnames:
        matched = [s for s in ref_surnames if s in cand_surnames]
        if ref_has_et_al:
            if ref_surnames[0] == cand_surnames[0]:
                return 1.0
            elif ref_surnames[0] in cand_surnames:
                return 0.85
            elif matched:
                return 0.70
            else:
                return 0.0
        else:
            score = len(matched) / len(ref_surnames)
            return score

    t_full = normalize_text(ref_full_text)
    hits = sum(1 for s in cand_surnames if s and s in t_full)
    return min(1.0, hits / max(1, len(cand_surnames[:3])))

def compute_year_score(cand_year: str, ref_year: str, is_book: bool = False) -> tuple:
    """Compare publication years, tolerating editions and reprint/proceedings dates."""
    y_c = clean_year(cand_year)
    y_r = clean_year(ref_year)
    if y_c and y_r:
        if y_c == y_r:
            return 1.0, True
        elif abs(y_c - y_r) == 1:
            return 0.8, True
        elif abs(y_c - y_r) == 2:
            return 0.5, True
        elif is_book:
            return 0.6, True
        elif y_c > y_r and (y_c - y_r) <= 8:
            # Publisher reprint or proceedings volume date (e.g. Attention Is All You Need 2025 vs 2017)
            return 0.5, True
        else:
            return 0.0, False
    return 0.5, False

def compute_venue_score(cand: dict, venue_hints: list) -> float:
    """Score publication venue agreement."""
    if not venue_hints:
        return 0.5
    cand_str = clean_unicode(str(cand.get("raw") or "")).lower()
    for v in venue_hints:
        if v.lower() in cand_str:
            return 1.0
    return 0.3

def compute_identifier_score(cand: dict, parsed: dict) -> float:
    """Score exact DOI or arXiv ID match."""
    if parsed.get("doi") and cand.get("doi"):
        if parsed["doi"].lower().strip() == cand["doi"].lower().strip():
            return 1.0
    if parsed.get("arxiv") and cand.get("arxiv"):
        if parsed["arxiv"].lower().split('v')[0] == cand["arxiv"].lower().split('v')[0]:
            return 1.0
    return 0.0

def score_candidate(cand: dict, parsed: dict) -> dict:
    """Compute individual signal scores and combined weighted confidence."""
    ts = compute_title_score(cand.get("title", ""), parsed["approx_title"], parsed["full_text"])
    as_ = compute_author_score(cand.get("authors", []), parsed["author_surnames"], parsed["has_et_al"], parsed["full_text"])
    is_book = (parsed.get("source_type") == "book" or "OpenLibrary" in cand.get("source", ""))
    ys, ym = compute_year_score(cand.get("year"), parsed.get("year"), is_book=is_book)
    vs = compute_venue_score(cand, parsed.get("venue_hints", []))
    ids = compute_identifier_score(cand, parsed)

    if ids == 1.0:
        confidence = 0.95 + 0.05 * ts
    elif "Official URL" in cand.get("source", ""):
        confidence = 0.95
    else:
        # Weighted score: title 50%, authors 25%, year 10%, identifiers 10%, venue 5%
        confidence = 0.50 * ts + 0.25 * as_ + 0.10 * ys + 0.10 * ids + 0.05 * vs
        if ts >= 0.85 and as_ >= 0.70:
            confidence = max(confidence, 0.85 + 0.10 * ys)

    return {
        "title_score": ts,
        "author_score": as_,
        "year_score": ys,
        "year_match": ym,
        "venue_score": vs,
        "identifier_score": ids,
        "confidence": min(1.0, round(confidence, 4))
    }

def derive_source_urls(best_cand: dict, parsed: dict, internal_status: str) -> dict:
    """
    Derive authoritative source_url and verified_source label
    from the best matched candidate. Only called for VERIFIED_REAL results.

    URL priority (highest first):
      1. DOI URL / ACL Anthology DOI
      2. ACL Anthology
      3. arXiv
      4. Official publisher page / official report
      5. OpenAlex landing page (last resort)

    Returns a dict with keys: verified_source, source_url
    """
    if internal_status != "VERIFIED_REAL" or best_cand is None:
        return {"verified_source": None, "source_url": None}

    source_label = best_cand.get("source", "")
    doi   = best_cand.get("doi")
    arxiv = best_cand.get("arxiv")
    raw   = best_cand.get("raw") or {}

    # 1. Official URL already in reference text (e.g. OpenAI CDN PDF, publisher URL)
    if "Official URL" in source_label:
        official_url = (raw.get("url") or parsed.get("url") or "").strip()
        if official_url:
            domain_m = re.search(r"https?://([^/]+)", official_url)
            domain = domain_m.group(1) if domain_m else "Official Page"
            return {
                "verified_source": f"Official URL ({domain})",
                "source_url":      official_url,
            }

    # 2. DOI handling (including ACL Anthology and arXiv DOIs)
    if doi:
        doi_clean = doi.strip()
        doi_url = f"https://doi.org/{doi_clean}"

        # 2a. ACL Anthology DOI (10.18653/...)
        if doi_clean.startswith("10.18653"):
            suffix = doi_clean.split("/", 2)[-1]  # e.g. "v1/2023.emnlp-main.557"
            acl_id = suffix.split("/", 1)[-1] if "/" in suffix else suffix  # "2023.emnlp-main.557"
            return {
                "verified_source": "ACL Anthology",
                "source_url":      f"https://aclanthology.org/{acl_id}/",
            }

        # 2b. arXiv DOI (10.48550/arXiv....)
        if "10.48550/arxiv." in doi_clean.lower():
            clean_id = re.sub(r"^.*10\.48550/arXiv\.", "", doi_clean, flags=re.I).strip()
            return {
                "verified_source": "arXiv",
                "source_url":      f"https://arxiv.org/abs/{clean_id}",
            }

        # Check raw for ACL Anthology hint
        raw_str = str(raw)
        acl_m = re.search(r"aclanthology\.org/([A-Za-z0-9.\-]+)", raw_str)
        if acl_m:
            acl_id = acl_m.group(1).rstrip("/")
            return {
                "verified_source": "ACL Anthology",
                "source_url":      f"https://aclanthology.org/{acl_id}/",
            }

        # Generic DOI: source_url is DOI URL; no PDF fabricated
        return {
            "verified_source": "DOI",
            "source_url":      doi_url,
        }

    # 3. arXiv ID
    if arxiv:
        arxiv_clean = arxiv.lower().replace("arxiv:", "").strip()
        return {
            "verified_source": "arXiv",
            "source_url":      f"https://arxiv.org/abs/{arxiv_clean}",
        }

    # 4. Official Report / Tech report
    if "Official Report" in source_label or "DBLP" in source_label:
        report_url = best_cand.get("url") or (raw.get("url") if isinstance(raw, dict) else None) or parsed.get("url")
        if report_url and isinstance(report_url, str) and report_url.startswith("http"):
            domain_m = re.search(r"https?://([^/]+)", report_url)
            domain = domain_m.group(1) if domain_m else "Official Page"
            return {
                "verified_source": f"Official Report ({domain})",
                "source_url":      report_url,
            }
        return {"verified_source": source_label, "source_url": None}

    # 5. OpenAlex fallback
    if "OpenAlex" in source_label or "OpenReview" in source_label:
        landing = raw.get("id") or raw.get("landing_page_url") or raw.get("url")
        if landing and str(landing).startswith("http"):
            return {
                "verified_source": source_label,
                "source_url":      str(landing),
            }

    # 6. Check if raw has DOI
    if isinstance(raw, dict):
        nested_doi = raw.get("DOI") or raw.get("doi")
        if nested_doi and isinstance(nested_doi, str):
            doi_clean = nested_doi.strip()
            doi_url = f"https://doi.org/{doi_clean}"
            if doi_clean.startswith("10.18653"):
                suffix  = doi_clean.split("/", 2)[-1]
                acl_id  = suffix.split("/", 1)[-1] if "/" in suffix else suffix
                return {
                    "verified_source": "ACL Anthology",
                    "source_url":      f"https://aclanthology.org/{acl_id}/",
                }
            return {
                "verified_source": "DOI",
                "source_url":      doi_url,
            }

    return {"verified_source": source_label or None, "source_url": None}

# ── Decision Hierarchy & Classification ────────────────────────────────────────

def classify_reference(best_scores: dict, best_cand: dict, parsed: dict) -> tuple:
    """
    Apply decision hierarchy to determine internal_status, frontend_status, and reason.

    Statuses:
      VERIFIED_REAL          -> Frontend: REAL
      REAL_METADATA_MISMATCH -> Frontend: UNCERTAIN
      UNCERTAIN              -> Frontend: UNCERTAIN
      LIKELY_FABRICATED      -> Frontend: FAKE
    """
    if best_cand is None or best_scores is None:
        return "LIKELY_FABRICATED", "FAKE", "No credible scholarly record found across configured sources."

    ts = best_scores["title_score"]
    as_ = best_scores["author_score"]
    ys = best_scores["year_score"]
    ym = best_scores["year_match"]
    ids = best_scores["identifier_score"]
    src = best_cand["source"]
    is_book = (parsed.get("source_type") == "book" or "OpenLibrary" in src)
    citation_has_authors = len(parsed.get("author_surnames", [])) > 0

    # Rule 1: Exact identifier match
    if ids == 1.0:
        if ts >= 0.40:
            if "arxiv" in src.lower():
                return "VERIFIED_REAL", "REAL", "Verified using exact arXiv identifier."
            return "VERIFIED_REAL", "REAL", "Verified using exact DOI identifier."
        else:
            return "REAL_METADATA_MISMATCH", "UNCERTAIN", "Real publication found, but citation metadata conflicts with authoritative record."

    # Rule 2: Official publisher / institutional URL
    if "Official URL" in src:
        return "VERIFIED_REAL", "REAL", f"Verified via official publisher URL ({src})."

    # Rule 3: Very strong / strong title match (>= 0.80)
    if ts >= 0.80:
        # Contradictory authors
        if citation_has_authors and as_ == 0.0:
            return "REAL_METADATA_MISMATCH", "UNCERTAIN", "Real publication found, but author metadata differs from the citation."

        # Inconsistent year (e.g. publication date conflicts with citation date)
        y_c = clean_year(best_cand.get("year"))
        y_r = clean_year(parsed.get("year"))
        if y_c and y_r and (y_r - y_c) > 3 and not is_book:
            return "REAL_METADATA_MISMATCH", "UNCERTAIN", "Real publication found, but citation metadata conflicts with authoritative record."

        # Compatible authors and year
        if as_ >= 0.40 or parsed.get("has_et_al") or not citation_has_authors or is_book:
            if ym or ys >= 0.8:
                return "VERIFIED_REAL", "REAL", f"Strong title, author, and year match found on {src}."
            return "VERIFIED_REAL", "REAL", f"Strong title and author match found on {src}."

        if as_ >= 0.30:
            return "VERIFIED_REAL", "REAL", f"Authoritative record found on {src} with compatible metadata."

        return "REAL_METADATA_MISMATCH", "UNCERTAIN", "Real publication found, but citation metadata conflicts with authoritative record."

    # Rule 4: Moderate title match (0.65 <= ts < 0.80)
    if ts >= 0.65:
        if as_ >= 0.50 and ys >= 0.50:
            return "VERIFIED_REAL", "REAL", f"Authoritative record found on {src} with compatible metadata."
        return "UNCERTAIN", "UNCERTAIN", "Partial scholarly metadata match; manual review recommended."

    # Rule 5: Low title match (0.40 <= ts < 0.65)
    if ts >= 0.40:
        if citation_has_authors and as_ == 0.0 and ids == 0.0:
            # Official reports (OpenAI, Google, Meta, etc.) are commonly absent from
            # Crossref. A weak unrelated Crossref result must NOT be treated as evidence
            # of fabrication — return UNCERTAIN so Semantic Scholar / manual review can
            # decide later.
            if parsed.get("source_type") == "official_report":
                return "UNCERTAIN", "UNCERTAIN", "Official organizational report — no credible scholarly record found; manual review recommended."
            return "LIKELY_FABRICATED", "FAKE", "No credible scholarly record found across configured sources."
        # Rule 4b: Moderate title but very strong author + exact year confirmation.
        # Handles ACL/conference papers where canonical title differs slightly from
        # how the citation phrased it (e.g. "Natural Language Inference" vs
        # "Sentence Understanding through Inference") but all authors and year agree.
        if ts >= 0.55 and as_ >= 0.70 and ym and citation_has_authors:
            return "VERIFIED_REAL", "REAL", f"Authoritative record found on {src} with strong author and year confirmation."
        return "UNCERTAIN", "UNCERTAIN", "Partial scholarly metadata match; manual review recommended."

    # Rule 6: No credible match across configured sources
    return "LIKELY_FABRICATED", "FAKE", "No credible scholarly record found across configured sources."

# ── Helper Backward Compatibility ─────────────────────────────────────────────

def calculate_similarity(candidate_title: str, ref_full_text: str) -> float:
    return compute_title_score(candidate_title, ref_full_text, ref_full_text)

def title_similarity(candidate_title: str, reference_title: str) -> float:
    return compute_title_score(candidate_title, reference_title, reference_title)

def determine_status(best_scores: dict, best_cand: dict, parsed: dict) -> tuple:
    return classify_reference(best_scores, best_cand, parsed)

# ── Main Single-Reference Verification ─────────────────────────────────────────

def verify_reference(ref_text: str, ref_index: int = 1) -> dict:
    """Verify a single reference string against multi-source scholarly databases."""
    parsed = parse_reference(ref_text)
    approx_title = parsed["approx_title"]

    safe_print(f"\n[BIB VERIFY] R{ref_index}")
    safe_print(f"type: {parsed['source_type']}")
    safe_print(f"parsed title: {approx_title}")
    safe_print(f"year: {parsed['year']}")
    safe_print(f"DOI: {parsed['doi']}")
    safe_print(f"arXiv: {parsed['arxiv']}")
    safe_print(f"URL: {parsed['url']}")

    candidates = []

    # Source 1: Official URL
    if parsed["source_type"] == "official_report" and parsed["url"]:
        domain_m = re.search(r"https?://([^/]+)", parsed["url"])
        domain = domain_m.group(1) if domain_m else "official"
        candidates.append({
            "source": f"Official URL ({domain})",
            "title": approx_title or ref_text,
            "authors": [],
            "year": parsed["year"],
            "doi": None,
            "arxiv": None,
            "raw": {"url": parsed["url"]}
        })

    # Source 2: Exact DOI lookup
    if parsed["doi"]:
        c = query_doi(parsed["doi"])
        if c:
            candidates.append(c)

    # Source 3: Exact arXiv ID lookup
    if parsed["arxiv"]:
        c = query_arxiv(parsed["arxiv"])
        if c:
            candidates.append(c)

    # Source 4: OpenLibrary (books)
    if parsed["source_type"] == "book":
        for c in query_openlibrary(approx_title):
            candidates.append(c)

    # Source 5: OpenReview (via OpenAlex)
    if parsed["source_type"] == "openreview_conference":
        for c in query_openalex(approx_title, source_label="OpenReview (via OpenAlex)"):
            candidates.append(c)

    # Source 5b: Official reports & technical whitepapers (OpenAI, Google, Meta, Microsoft, DeepMind, etc.)
    # Example: GPT-2 "Language Models are Unsupervised Multitask Learners"
    if parsed["source_type"] == "official_report":
        for c in query_official_report(approx_title):
            candidates.append(c)
        for c in query_dblp(approx_title):
            candidates.append(c)
        for c in query_semantic_scholar(approx_title):
            candidates.append(c)

    # Source 6: Crossref bibliographic search
    for c in query_crossref_biblio(ref_text, approx_title):
        candidates.append(c)

    # Check if a strong candidate already exists before calling fallbacks
    has_strong_cand = any(
        score_candidate(c, parsed)["title_score"] >= 0.85 and score_candidate(c, parsed)["author_score"] >= 0.40
        for c in candidates
    )

    # Source 7: arXiv title search fallback
    if not has_strong_cand:
        for c in query_arxiv_search(approx_title):
            candidates.append(c)

    # Source 8: OpenAlex fallback
    has_good_cand = any(score_candidate(c, parsed)["title_score"] >= 0.80 for c in candidates)
    if not has_good_cand:
        for c in query_openalex(approx_title):
            candidates.append(c)

    # Source 9: Official reports, DBLP + Semantic Scholar broad fallback.
    # Catches ACL Anthology papers, org-published reports, and grey literature missed
    # by the above sources when the best title score is still below 0.75.
    has_good_cand_75 = any(score_candidate(c, parsed)["title_score"] >= 0.75 for c in candidates)
    if not has_good_cand_75:
        for c in query_official_report(approx_title):
            candidates.append(c)
        for c in query_dblp(approx_title):
            candidates.append(c)
        for c in query_semantic_scholar(approx_title):
            candidates.append(c)

    # Score all candidates
    scored = []
    for c in candidates:
        scores = score_candidate(c, parsed)
        scored.append((c, scores))

    best_cand = None
    best_scores = None
    best_conf = -1.0
    for c, s in scored:
        if s["confidence"] > best_conf:
            best_conf = s["confidence"]
            best_cand = c
            best_scores = s

    internal_status, frontend_status, reason = classify_reference(best_scores, best_cand, parsed)

    # Print debug log
    safe_print("\n[BIB DEBUG]")
    safe_print(f"ref_number: {ref_index}")
    safe_print(f"raw_reference: {ref_text[:200]}")
    safe_print(f"extracted_title: {approx_title}")
    safe_print(f"extracted_authors: {parsed['author_surnames']}")
    safe_print(f"extracted_year: {parsed['year']}")
    safe_print(f"best_source: {best_cand['source'] if best_cand else 'None'}")
    safe_print(f"best_candidate: {best_cand['title'][:80] if best_cand else 'None'}")
    _ts_dbg = best_scores["title_score"] if best_scores else 0.0
    _as_dbg = best_scores["author_score"] if best_scores else 0.0
    safe_print(f"best_title_score: {_ts_dbg:.3f}")
    safe_print(f"best_author_score: {_as_dbg:.3f}")
    safe_print(f"best_year_match: {best_scores['year_match'] if best_scores else False}")
    safe_print(f"internal_status: {internal_status}")
    safe_print(f"frontend_status: {frontend_status}")
    safe_print(f"final_reason: {reason}")

    url_info = derive_source_urls(best_cand, parsed, internal_status)

    return {
        "status": frontend_status,
        "reason": reason,
        "score": round(best_conf, 4) if best_conf > 0 else 0.0,
        "match": best_cand["raw"] if best_cand else None,
        "verification_detail": internal_status,
        "matched_source": best_cand["source"] if best_cand else None,
        "matched_title": best_cand["title"] if best_cand else None,
        "matched_authors": ", ".join(best_cand["authors"]) if best_cand and isinstance(best_cand.get("authors"), list) else None,
        "matched_year": str(best_cand["year"]) if best_cand and best_cand.get("year") else None,
        "matched_doi": best_cand["doi"] if best_cand else None,
        "matched_arxiv": best_cand.get("arxiv") if best_cand else None,
        "confidence": round(best_conf, 4) if best_conf > 0 else 0.0,
        "verified_source": url_info.get("verified_source"),
        "source_url": url_info.get("source_url"),
    }

# ── Document-Level Verification ────────────────────────────────────────────────

def verify_document_references(pdf_path: str) -> dict:
    """Extract and verify all references in a PDF paper."""
    refs = extract_references_from_pdf(pdf_path)
    results = []

    real_count = 0
    fake_count = 0
    uncertain_count = 0

    for i, ref in enumerate(refs[:100], 1):
        verification = verify_reference(ref, ref_index=i)
        results.append({
            "reference_text": ref,
            "verification": verification
        })

        status = verification["status"]
        if status == "REAL":
            real_count += 1
        elif status == "FAKE":
            fake_count += 1
        elif status == "UNCERTAIN":
            uncertain_count += 1

    return {
        "total_extracted": len(refs),
        "total_analyzed": len(results),
        "summary": {
            "real": real_count,
            "fake": fake_count,
            "uncertain": uncertain_count
        },
        "details": results
    }
