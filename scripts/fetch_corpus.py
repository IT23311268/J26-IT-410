"""
Fetch a small corpus of real papers from arXiv.

Every threshold in `ingestion/layout.py` was measured on one or two
pages of one paper. That is the honest way to pick a number and it is
not evidence that the number holds anywhere else. `check_corpus.py`
is how that gets checked; this is how it gets something to check.

    python scripts/fetch_corpus.py              # 12 papers into data/raw/
    python scripts/fetch_corpus.py --count 20
    python scripts/fetch_corpus.py --query "retrieval augmented generation"

Why these papers: the corpus has to look like what the real pipeline
will be fed — two-column NLP and IR papers with figures, ruled tables
and equations. A corpus of single-column preprints would pass every
check and prove nothing.

Only the standard library is used, so this runs on a fresh clone with
nothing installed but the project's own requirements.

arXiv asks for no more than one request every three seconds and for a
descriptive User-Agent. Both are honoured below. Please do not lower
the delay — it is a shared, free service run by a university.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API = "http://export.arxiv.org/api/query"

#: arXiv's stated rate limit. Do not lower this.
REQUEST_DELAY_SECONDS = 3.0

#: Sent on every request so arXiv can see who is calling and block us
#: specifically rather than the whole university if something misbehaves.
USER_AGENT = "J26-IT-410-ingestion-research/1.0 (SLIIT student project)"

#: cs.CL is computational linguistics, cs.IR information retrieval.
#: Both are dense two-column papers with the figures, ruled tables and
#: equations that Box 3's rules were written for.
DEFAULT_QUERY = "cat:cs.CL OR cat:cs.IR"

ATOM = {"a": "http://www.w3.org/2005/Atom"}


def _safe_name(arxiv_id: str, title: str) -> str:
    """A filename that is readable in a directory listing and legal on
    Windows, which forbids \\ / : * ? " < > | in a name."""
    stem = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")[:60]
    return f"{arxiv_id}_{stem}.pdf" if stem else f"{arxiv_id}.pdf"


def _get(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def search(query: str, count: int) -> list[tuple[str, str, str]]:
    """Ask arXiv for `count` recent papers. Returns (id, title, pdf_url)."""
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": count,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    feed = ET.fromstring(_get(f"{API}?{params}"))

    found: list[tuple[str, str, str]] = []
    for entry in feed.findall("a:entry", ATOM):
        raw_id = entry.findtext("a:id", default="", namespaces=ATOM)
        title = " ".join(
            entry.findtext("a:title", default="", namespaces=ATOM).split()
        )
        pdf_url = next(
            (
                link.attrib["href"]
                for link in entry.findall("a:link", ATOM)
                if link.attrib.get("title") == "pdf"
            ),
            "",
        )
        arxiv_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""
        if arxiv_id and pdf_url:
            found.append((arxiv_id, title, pdf_url))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=12, help="how many papers")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="arXiv search query")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
        help="where to put the PDFs (default: data/raw/)",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"searching arXiv: {args.query}")
    try:
        papers = search(args.query, args.count)
    except (urllib.error.URLError, ET.ParseError) as exc:
        print(f"  !! could not reach arXiv: {exc}")
        print("     Check the connection, or download a few PDFs by hand")
        print(f"     into {args.out} — the checker does not care how they got there.")
        return 1

    if not papers:
        print("  no results. Try a broader --query.")
        return 1

    print(f"  {len(papers)} found\n")

    saved = skipped = failed = 0
    for n, (arxiv_id, title, pdf_url) in enumerate(papers, start=1):
        target = args.out / _safe_name(arxiv_id, title)
        label = title[:58] + ("…" if len(title) > 58 else "")

        if target.exists():
            print(f"  [{n:>2}/{len(papers)}] have it   {label}")
            skipped += 1
            continue

        try:
            # Never write a partial file: a truncated PDF would look
            # like a Box 3 bug when the checker chokes on it.
            body = _get(pdf_url)
            if not body.startswith(b"%PDF"):
                raise ValueError("response was not a PDF")
            target.write_bytes(body)
            print(f"  [{n:>2}/{len(papers)}] saved     {label}  ({len(body)//1024} KB)")
            saved += 1
        except Exception as exc:  # noqa: BLE001 — one bad paper must not stop the run
            print(f"  [{n:>2}/{len(papers)}] FAILED    {label}\n              {exc}")
            failed += 1

        if n < len(papers):
            time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\n{saved} saved, {skipped} already there, {failed} failed")
    print(f"in {args.out}")
    print("\nNow run the checker over them:")
    print(f"  python scripts/check_corpus.py {args.out}")
    return 0 if saved or skipped else 1


if __name__ == "__main__":
    sys.exit(main())
