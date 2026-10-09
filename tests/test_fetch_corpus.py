"""Tests for the corpus fetcher.

The download itself is not tested — that would mean hitting arXiv from
CI, which is rude and flaky. What is tested is everything around it,
because those are the parts that fail silently:

  the filename, which has to be legal on Windows. The demo machine is
  a Windows laptop and a title containing a colon would otherwise
  raise OSError halfway through a download;

  the parsing of arXiv's reply, which is the part most likely to be
  quietly wrong. A namespace mistake returns an empty list and looks
  exactly like "no results for that query".
"""

import xml.etree.ElementTree as ET

import pytest

from scripts.fetch_corpus import _safe_name, search

# A real arXiv reply, trimmed to two entries. The namespace, the
# `title="pdf"` link and the versioned id are all exactly as served.
FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <title>Attention Is All You
      Need</title>
    <link href="http://arxiv.org/abs/1706.03762v7" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/1706.03762v7" rel="related"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2005.11401v4</id>
    <title>Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks</title>
    <link title="pdf" href="http://arxiv.org/pdf/2005.11401v4" rel="related"/>
  </entry>
</feed>
"""


@pytest.fixture
def papers(monkeypatch):
    monkeypatch.setattr("scripts.fetch_corpus._get", lambda url, **kw: FEED.encode())
    return search("cat:cs.CL", 2)


# --------------------------------------------------------------------------
# reading arXiv's reply
# --------------------------------------------------------------------------


def test_both_entries_are_found(papers):
    assert len(papers) == 2


def test_the_id_keeps_its_version(papers):
    """1706.03762 and 1706.03762v7 are different files. Dropping the
    version makes a re-run download a different paper under the same
    name."""
    assert papers[0][0] == "1706.03762v7"


def test_a_title_wrapped_across_lines_comes_back_as_one(papers):
    """arXiv hard-wraps titles in the XML."""
    assert papers[0][1] == "Attention Is All You Need"


def test_the_pdf_link_is_taken_not_the_html_one(papers):
    """Both links sit in the same entry and only one is the PDF."""
    assert papers[0][2] == "http://arxiv.org/pdf/1706.03762v7"


def test_an_entry_with_no_pdf_link_is_skipped(monkeypatch):
    feed = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/1234.5678v1</id>
        <title>Abstract only, no PDF</title>
      </entry>
    </feed>"""
    monkeypatch.setattr("scripts.fetch_corpus._get", lambda url, **kw: feed.encode())
    assert search("cat:cs.CL", 1) == []


def test_an_empty_feed_is_not_an_error(monkeypatch):
    feed = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"/>'
    monkeypatch.setattr("scripts.fetch_corpus._get", lambda url, **kw: feed.encode())
    assert search("cat:nonsense", 5) == []


def test_a_broken_reply_raises_rather_than_returning_nothing(monkeypatch):
    """Silently returning [] on malformed XML would read as 'no
    results' and send someone off rewriting their query."""
    monkeypatch.setattr("scripts.fetch_corpus._get", lambda url, **kw: b"<not xml")
    with pytest.raises(ET.ParseError):
        search("cat:cs.CL", 1)


# --------------------------------------------------------------------------
# the filename
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arxiv_id,title",
    [
        ("1706.03762v7", "Attention Is All You Need"),
        ("2005.11401v4", "RAG: Knowledge-Intensive NLP Tasks"),
        ("2303.08774v6", 'A "quoted" title with <angles> and |pipes|'),
        ("2401.00001v1", "Ratio a/b and c\\d and a*b and q?"),
    ],
)
def test_the_filename_is_legal_on_windows(arxiv_id: str, title: str):
    r"""Windows forbids \ / : * ? " < > | in a filename, and the demo
    runs on a Windows laptop."""
    name = _safe_name(arxiv_id, title)
    assert not set(name) & set('\\/:*?"<>|')
    assert name.endswith(".pdf")


def test_the_filename_carries_the_id(papers):
    assert _safe_name(*papers[0][:2]).startswith("1706.03762v7_")


def test_a_very_long_title_is_cut(papers):
    """Windows' total path limit is 260 characters and data/raw/ is
    already deep inside a user's Documents folder."""
    assert len(_safe_name("2401.00001v1", "word " * 80)) < 90


def test_a_title_of_pure_punctuation_still_gives_a_name():
    assert _safe_name("2401.00001v1", "!!! ???") == "2401.00001v1.pdf"
