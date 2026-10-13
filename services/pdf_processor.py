"""
PDF Processing Service.

Extracts text from uploaded PDF files using PyMuPDF,
extracts approximate paper metadata from the first pages,
and splits text into overlapping sentence-based chunks while
retaining source-paper and page-number metadata.
"""

import os
import re
from typing import List, Dict, Tuple, Optional


# -------------------------------------------------------------------
# PyMuPDF import
# -------------------------------------------------------------------

# New PyMuPDF versions prefer "pymupdf".
# Fall back to "fitz" for compatibility with older installations.
try:
    import pymupdf
except ImportError:
    import fitz as pymupdf


# -------------------------------------------------------------------
# Sentence splitting
# -------------------------------------------------------------------

# We intentionally avoid variable-width look-behind because Python's
# regex engine rejects patterns such as (?<!et al)(?<!Dr)... when
# the alternatives have different lengths.

_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])"
)

# Protect common academic abbreviations before sentence splitting.
_ABBREVIATIONS = [
    "et al.",
    "e.g.",
    "i.e.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Prof.",
    "Fig.",
    "Eq.",
    "Vol.",
    "No.",
    "vs.",
]


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences without using variable-width look-behind.

    Common abbreviations are temporarily protected so that periods
    inside expressions such as "et al." do not incorrectly split a sentence.
    """
    if not text:
        return []

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text.strip())

    if not text:
        return []

    protected = text

    # Replace abbreviation periods temporarily.
    abbreviation_map = {}

    for index, abbreviation in enumerate(_ABBREVIATIONS):
        token = f"__ABBR_{index}__"
        protected_abbreviation = abbreviation.replace(".", token)
        abbreviation_map[token] = "."
        protected = protected.replace(abbreviation, protected_abbreviation)

    parts = _SENTENCE_SPLIT.split(protected)

    sentences: List[str] = []

    for part in parts:
        # Restore protected periods.
        for token, replacement in abbreviation_map.items():
            part = part.replace(token, replacement)

        part = part.strip()

        if len(part) > 10:
            sentences.append(part)

    return sentences


# -------------------------------------------------------------------
# Metadata extraction
# -------------------------------------------------------------------

def _clean_line(line: str) -> str:
    """Normalize one extracted PDF line."""
    return re.sub(r"\s+", " ", line).strip()


def _extract_title(first_page_text: str) -> Optional[str]:
    """
    Approximate the paper title from the first page.

    We select the first substantial non-empty line that does not look
    like a page number, URL, DOI, journal footer, or author-address line.
    """
    lines = [_clean_line(line) for line in first_page_text.splitlines()]
    lines = [line for line in lines if line]

    for line in lines[:20]:
        lower = line.lower()

        if len(line) < 12:
            continue

        if re.fullmatch(r"\d+", line):
            continue

        if lower.startswith("http"):
            continue

        if "doi.org" in lower:
            continue

        if "publication date" in lower:
            continue

        if "authors' address" in lower or "authors’ address" in lower:
            continue

        if "arxiv:" in lower:
            continue

        return line

    return None


def _extract_authors(
    first_page_text: str,
    title: Optional[str]
) -> List[str]:
    """
    Approximate author names from the beginning of the first page.

    This is intentionally heuristic because academic PDFs have many
    different layouts.
    """
    lines = [_clean_line(line) for line in first_page_text.splitlines()]
    lines = [line for line in lines if line]

    if not lines:
        return []

    start_index = 0

    if title:
        for i, line in enumerate(lines[:20]):
            if line == title:
                start_index = i + 1
                break

    candidate_lines = []

    # Usually author names occur immediately after the title.
    for line in lines[start_index:start_index + 8]:
        lower = line.lower()

        if any(
            marker in lower
            for marker in [
                "abstract",
                "university",
                "institute",
                "department",
                "center for",
                "centre for",
                "laboratory",
                "publication date",
                "arxiv:",
                "doi.org",
            ]
        ):
            if candidate_lines:
                break
            continue

        candidate_lines.append(line)

    author_text = " ".join(candidate_lines)

    if not author_text:
        return []

    # Remove obvious affiliation fragments after institutional keywords.
    author_text = re.split(
        r"\b(?:University|Institute|Department|Center|Centre|Laboratory)\b",
        author_text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    # Split common author separators.
    raw_authors = re.split(
        r"\s*(?:,|;|\band\b|&)\s*",
        author_text
    )

    authors = []

    for author in raw_authors:
        author = author.strip(" ,.;")

        # Keep only plausible names.
        if len(author) < 3:
            continue

        if re.search(r"[A-Za-z]", author):
            authors.append(author)

    return authors


def _extract_years(first_pages_text: str) -> List[int]:
    """Return all plausible publication years found in the first pages."""
    years = re.findall(
        r"\b(?:19\d{2}|20\d{2})\b",
        first_pages_text
    )

    unique_years = []

    for year_text in years:
        year = int(year_text)

        if year not in unique_years:
            unique_years.append(year)

    return unique_years


def _extract_metadata_from_text(
    filename: str,
    first_page_text: str,
    first_pages_text: str,
) -> dict:
    """
    Extract approximate paper metadata.

    The raw first-pages text is retained because citation matching can
    use it as a robust fallback when structured metadata extraction is
    imperfect.
    """
    title = _extract_title(first_page_text)
    authors = _extract_authors(first_page_text, title)
    years = _extract_years(first_pages_text)

    author_surnames = []

    for author in authors:
        words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ'’-]+", author)

        if words:
            surname = words[-1]

            if surname.lower() not in {
                "et",
                "al",
                "university",
                "institute",
                "center",
                "centre",
            }:
                author_surnames.append(surname)

    # Keep a single year for backward compatibility.
    # Citation matching should preferably inspect "years" as well.
    year = years[0] if years else None

    # Extract DOI and arXiv
    doi_match = re.search(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", first_pages_text, re.I)
    doi = doi_match.group(1).rstrip('.') if doi_match else None
    
    arxiv_match = re.search(r"arxiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)", first_pages_text, re.I)
    if not arxiv_match:
        arxiv_match = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)", first_pages_text, re.I)
    arxiv = arxiv_match.group(1) if arxiv_match else None

    return {
        "filename": filename,
        "title": title,
        "authors": authors,
        "author_surnames": author_surnames,
        "year": year,
        "years": years,
        "doi": doi,
        "arxiv": arxiv,
        "first_pages_text": first_pages_text[:5000],
    }


# -------------------------------------------------------------------
# PDF text extraction
# -------------------------------------------------------------------

def extract_text_from_pdf(
    pdf_path: str
) -> Tuple[List[dict], dict]:
    """
    Extract text page-by-page from one PDF.

    Returns:
        pages:
            [
                {
                    "paper": "...",
                    "page": 1,
                    "text": "..."
                }
            ]

        metadata:
            {
                "filename": "...",
                "title": "...",
                "authors": [...],
                "author_surnames": [...],
                "year": ...,
                "years": [...],
                "first_pages_text": "..."
            }
    """
    filename = os.path.basename(pdf_path)

    pages: List[dict] = []

    first_page_text = ""
    first_pages_text = ""

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open PDF '{filename}': {exc}"
        ) from exc

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]

            try:
                text = page.get_text("text")
            except Exception:
                continue

            if not text:
                continue

            # Repair hyphenated line breaks (e.g. "develop- \n ment" -> "development")
            text = re.sub(r"([A-Za-z]+)-\s*\n\s*([A-Za-z]+)", r"\1\2", text)

            # Normalize horizontal whitespace.
            text = re.sub(r"[ \t]+", " ", text)

            # Keep useful paragraph breaks while removing excessive ones.
            text = re.sub(r"\n{3,}", "\n\n", text)

            text = text.strip()

            if page_num == 0:
                first_page_text = text

            if page_num < 2:
                first_pages_text += text + "\n"

            if len(text) < 20:
                continue

            pages.append(
                {
                    "paper": filename,
                    "page": page_num + 1,
                    "text": text,
                }
            )

    finally:
        doc.close()

    metadata = _extract_metadata_from_text(
        filename=filename,
        first_page_text=first_page_text,
        first_pages_text=first_pages_text,
    )

    return pages, metadata


def extract_text_from_all_pdfs(
    uploads_dir: str
) -> Tuple[List[dict], Dict[str, dict]]:
    """
    Extract text from every PDF in the uploads directory.

    Returns:
        all_pages:
            Flat list of page-level text dictionaries.

        metadata_dict:
            {
                "paper.pdf": {...metadata...}
            }
    """
    all_pages: List[dict] = []
    metadata_dict: Dict[str, dict] = {}

    if not os.path.isdir(uploads_dir):
        return all_pages, metadata_dict

    for filename in sorted(os.listdir(uploads_dir)):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(
            uploads_dir,
            filename
        )

        try:
            pages, metadata = extract_text_from_pdf(pdf_path)

            all_pages.extend(pages)
            metadata_dict[filename] = metadata

        except RuntimeError as exc:
            # Skip unreadable PDFs without crashing the entire analysis.
            print(f"[PDF WARNING] {exc}")
            continue

    return all_pages, metadata_dict


# -------------------------------------------------------------------
# Text chunking
# -------------------------------------------------------------------

def chunk_text(
    pages: List[dict],
    sentences_per_chunk: int = 5,
    overlap_sentences: int = 2,
) -> List[dict]:
    """
    Split page-level text into overlapping sentence-based chunks.

    By default:
        5 sentences per chunk
        2 sentence overlap

    Each chunk retains:
        chunk_id
        paper
        page
        text
    """
    if sentences_per_chunk < 1:
        sentences_per_chunk = 1

    if overlap_sentences < 0:
        overlap_sentences = 0

    # Prevent overlap from being equal to/larger than chunk size,
    # which would otherwise create an infinite loop.
    if overlap_sentences >= sentences_per_chunk:
        overlap_sentences = sentences_per_chunk - 1

    step = max(
        1,
        sentences_per_chunk - overlap_sentences
    )

    all_chunks: List[dict] = []
    chunk_counter: Dict[str, int] = {}

    for page_info in pages:
        paper = page_info.get("paper")
        page = page_info.get("page")
        text = page_info.get("text", "")

        if not paper or not text:
            continue

        sentences = _split_sentences(text)

        if not sentences:
            continue

        if paper not in chunk_counter:
            chunk_counter[paper] = 0

        start = 0

        while start < len(sentences):
            end = min(
                start + sentences_per_chunk,
                len(sentences)
            )

            chunk_sentences = sentences[start:end]

            chunk_text_str = " ".join(
                chunk_sentences
            ).strip()

            if len(chunk_text_str) > 30:
                chunk_counter[paper] += 1

                chunk_id = (
                    f"{paper}_p{page}_"
                    f"c{chunk_counter[paper]}"
                )

                all_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "paper": paper,
                        "page": page,
                        "text": chunk_text_str,
                    }
                )

            if end >= len(sentences):
                break

            start += step

    return all_chunks


# -------------------------------------------------------------------
# Optional local test
# -------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "Natural language generation has advanced rapidly. "
        "Ji et al. (2023) discuss hallucination in NLG. "
        "Deep learning systems can generate information that is not "
        "supported by the source. "
        "This creates problems for factual reliability. "
        "Several mitigation methods have therefore been proposed."
    )

    print("Sentence splitter test:")

    for i, sentence in enumerate(
        _split_sentences(sample),
        start=1
    ):
        print(f"{i}. {sentence}")