"""
Section-aware chunking — pipeline Box 5.

Turns Box 4's labelled regions into the Chunk objects Members 2 and 3
retrieve from.

What makes it "section-aware"
-----------------------------
The baseline cuts the paper into fixed windows of characters. On a real
paper that lands mid-word — "…interleaves the column | s and shreds
every sentence…" — and it puts the end of the Method and the start of
the Results in one chunk, so nothing downstream can say which section a
retrieved passage came from.

This stage cuts at two kinds of boundary and nowhere else:

  a section boundary, always — a chunk is never half Method and half
  Results;

  a region boundary, when a section is longer than one chunk should be.
  A region is a paragraph, so the cut falls between paragraphs and never
  inside a sentence.

The result is a chunk that is a whole number of paragraphs, from exactly
one section, carrying the heading that opened it.
"""

from __future__ import annotations

from schema.ingestion_schema_v1 import BoundingBox, Chunk, LayoutRegion, RegionType

# Regions that carry prose. A FIGURE, TABLE or EQUATION is an Artifact
# (Box 6) and is bound back to the chunks that discuss it in Box 7 — it
# is not prose and must not be retrieved as if it were.
#
# A HEADING is kept, and kept at the *front* of its chunk: "2. Method"
# ahead of the paragraphs under it is a real retrieval signal, and it
# costs four words.
PROSE = (RegionType.HEADING, RegionType.BODY)

# A caption is prose about an artifact, not prose about the paper. It
# becomes its own chunk so that retrieving "Figure 7 shows accuracy on
# TruthfulQA" does not drag a page of unrelated Method text with it.
STANDALONE = (RegionType.CAPTION,)

# How long a chunk should be, in characters.
#
# Characters rather than tokens because tokenising here would tie this
# stage to one model's tokeniser, and Members 2 and 3 may not use the
# same one. Roughly four characters to a token, so 1200 is about 300
# tokens — comfortable inside every embedding window in use, and long
# enough to hold an argument rather than a sentence.
#
# It is a target, not a limit: the cut only ever falls on a region
# boundary, so a single long paragraph comes out whole and over size.
# Splitting it would put the cut inside a sentence, which is the thing
# this stage exists to stop.
TARGET_CHUNK_CHARS = 1200

# A chunk shorter than this is not worth retrieving on its own — a
# one-line paragraph, a stray heading. It is folded into the chunk
# before it, provided that chunk is in the same section.
MIN_CHUNK_CHARS = 120


def _token_estimate(text: str) -> int:
    """A rough token count, about four characters each.

    Deliberately an estimate: a real tokeniser would tie this stage to
    one model. Members 2 and 3 are told to treat it as approximate.
    """
    return max(1, len(text) // 4)


def _build_chunk(
    regions: list[LayoutRegion], paper_id: str, order_index: int
) -> Chunk:
    """Make one Chunk out of consecutive regions from one section."""
    text = "\n".join(r.text for r in regions if r.text.strip())

    pages = [r.bbox.page for r in regions]
    first_page = min(pages)

    # The box is only meaningful when the chunk sits on one page: a
    # chunk spanning a page break has no single rectangle, and inventing
    # one would put a wrong highlight on a reader's screen.
    bbox = None
    if first_page == max(pages):
        on_page = [r.bbox for r in regions]
        bbox = BoundingBox(
            page=first_page,
            x0=min(b.x0 for b in on_page),
            y0=min(b.y0 for b in on_page),
            x1=max(b.x1 for b in on_page),
            y1=max(b.y1 for b in on_page),
        )

    return Chunk(
        chunk_id=f"{paper_id}::chunk::{order_index}",
        paper_id=paper_id,
        section=regions[0].section,
        section_title_raw=regions[0].section_title_raw,
        text=text,
        page_start=first_page,
        page_end=max(pages),
        bbox=bbox,
        order_index=order_index,
        token_count=_token_estimate(text),
    )


def chunk_regions(regions: list[LayoutRegion], paper_id: str) -> list[Chunk]:
    """Turn labelled regions into chunks, in reading order.

    Captions become chunks of their own. Figures, tables and equations
    are skipped — they are Artifacts, and Box 7 links them back to the
    chunks that discuss them.
    """
    chunks: list[Chunk] = []
    # Captions are short by nature, so the runt rule below would fold
    # every one of them into the paragraph above. These indices are how
    # it knows to leave them alone.
    standalone: set[int] = set()
    buffer: list[LayoutRegion] = []
    buffered_chars = 0

    def flush() -> None:
        nonlocal buffered_chars
        if not buffer:
            return
        chunks.append(_build_chunk(buffer, paper_id, len(chunks)))
        buffer.clear()
        buffered_chars = 0

    for region in regions:
        if region.region_type in STANDALONE:
            flush()
            standalone.add(len(chunks))
            chunks.append(_build_chunk([region], paper_id, len(chunks)))
            continue

        if region.region_type not in PROSE:
            continue
        if not region.text.strip():
            continue

        # A section boundary always cuts.
        if buffer and region.section != buffer[0].section:
            flush()

        # A heading opens its own chunk, so the chunk reads as a unit
        # rather than trailing the end of the section before it.
        elif region.region_type == RegionType.HEADING and buffer:
            flush()

        # Over the target: cut here, between two paragraphs.
        elif buffered_chars + len(region.text) > TARGET_CHUNK_CHARS and buffer:
            flush()

        buffer.append(region)
        buffered_chars += len(region.text)

    flush()
    return _merge_runts(chunks, standalone, paper_id)


def _merge_runts(
    chunks: list[Chunk], standalone: set[int], paper_id: str
) -> list[Chunk]:
    """Fold a too-short chunk into the one before it, same section only.

    A heading alone, or a one-line paragraph, is not worth retrieving:
    it matches nothing and it dilutes the corpus. Merging keeps it
    attached to the text it introduces. A runt that opens a section has
    nothing to merge into and is left as it is.

    `standalone` holds the indices of chunks that must survive however
    short they are. Captions are the reason: they are short by nature,
    and an earlier version of this rule swallowed every one of them into
    the paragraph above — which is exactly the coupling the caption
    chunk exists to avoid.
    """
    out: list[Chunk] = []

    for index, chunk in enumerate(chunks):
        if (
            out
            and index not in standalone
            and len(chunk.text) < MIN_CHUNK_CHARS
            and out[-1].section == chunk.section
        ):
            previous = out[-1]
            merged_text = f"{previous.text}\n{chunk.text}"
            out[-1] = previous.model_copy(
                update={
                    "text": merged_text,
                    "page_end": chunk.page_end,
                    "token_count": _token_estimate(merged_text),
                    # the box no longer describes the merged text
                    "bbox": previous.bbox if previous.page_end == chunk.page_end else None,
                }
            )
            continue
        out.append(chunk)

    # ids and order_index must stay dense after merging
    return [
        c.model_copy(
            update={"chunk_id": f"{paper_id}::chunk::{i}", "order_index": i}
        )
        for i, c in enumerate(out)
    ]
