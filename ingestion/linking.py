"""Artifact-to-chunk binding — pipeline Box 7.

Box 6 cut every figure, table and equation out of its page image. Box 5
cut the prose into chunks. This stage puts the two back together: each
artifact learns which chunks discuss it, and each chunk learns which
artifacts it refers to.

Why it is needed
----------------
A figure retrieved on its own is a picture with a caption, and a chunk
retrieved on its own is a paragraph that says "as shown in Figure 1"
with no Figure 1 attached. Member 2 verifies claims against the text
and needs the figure a claim rests on; Member 3 builds a graph and
needs the edge between them. Neither can recover that edge from the
JSON unless this stage writes it down.

The evidence
------------
No model and no heuristic about distance: the author already stated the
link. A paper prints "Figure 1: Overview of our approach." under the
drawing and then writes "as shown in Figure 1" in the Method. Both
mention the same label, and that label is what this stage matches on.

The cost of that choice is that a figure the text never refers to comes
out with no chunks, and that is deliberate. The alternative — binding
it to whatever chunk happens to sit nearest — would hand Member 2
something that looks like evidence and is not. An empty list is a true
answer; a guessed one is worse than no answer at all.
"""

from __future__ import annotations

import re

from schema.ingestion_schema_v1 import Artifact, Chunk

#: How a cross-reference is written, on either side of the link.
#:
#: The abbreviations are the ones papers actually use: "Fig. 3", "Fig 3",
#: "fig.3" and "Figure 3" are one reference written four ways, and
#: "Tab. 2" is "Table 2". The longer spelling comes first in each
#: alternation because a regex alternation takes the first branch that
#: matches — with "tab" ahead of "table", "Table 2" would match as the
#: abbreviation and leave "le 2" behind.
#:
#: The number is deliberately matched greedily. That is what keeps
#: Figure 1 out of every sentence about Figure 10: both sides are parsed
#: into (kind, number) before anything is compared, so "Figure 10"
#: becomes ("figure", 10) and simply never equals ("figure", 1). A
#: substring search for "Figure 1" would have linked it to Figure 10, 11
#: and 12 — on a paper with twelve figures, to most of them.
#: The plural forms are here because papers write "Figures 1 and 2" far
#: more often than they write the two references out separately, and
#: reading only the first number drops the second figure in silence.
#: A list continues only across a comma, an "and" or an "&" standing
#: immediately between two numbers, so "Figure 1 is the design. We
#: trained for 9 epochs." stops at 1 rather than reaching into the next
#: sentence for a 9.
_REFERENCE = re.compile(
    r"\b(figures|figure|figs|fig|tables|table|tabs|tab|equations|equation|eqs|eq)"
    r"\.?\s*([0-9]+(?:\s*(?:,|and|&)\s*[0-9]+)*)",
    re.IGNORECASE,
)

#: Pulls the numbers back out of a matched list: "1 and 2" -> [1, 2].
_NUMBER = re.compile(r"[0-9]+")

#: "Fig." and "Figure" are the same thing to a reader and must be the
#: same thing here.
_CANONICAL = {
    "figure": "figure", "figures": "figure", "fig": "figure", "figs": "figure",
    "table": "table", "tables": "table", "tab": "table", "tabs": "table",
    "equation": "equation", "equations": "equation", "eq": "equation", "eqs": "equation",
}

#: A reference, reduced to the two things that identify it.
Label = tuple[str, int]


def _labels(kind: str, numbers: str) -> list[Label]:
    """('Figures', '1 and 2') -> [('figure', 1), ('figure', 2)]."""
    canonical = _CANONICAL[kind.lower()]
    return [(canonical, int(n)) for n in _NUMBER.findall(numbers)]


def caption_label(caption: str) -> Label | None:
    """The label a caption opens with: 'Figure 7. Performance…' -> ('figure', 7).

    It must be at the START of the caption, because that is where the
    author put it. A caption reading "…compared against Figure 3" is
    describing a different figure, and matching anywhere in the string
    would file this crop under that one.

    Returns None when there is nothing to match on — an empty caption,
    or one the layout stage picked up as a stray line of body text. The
    artifact then stays unlinked.
    """
    found = _REFERENCE.match(caption.strip())
    if found is None:
        return None
    # a caption names one artifact: its own, which is the first number
    return _labels(found.group(1), found.group(2))[0]


def references_in(text: str) -> set[Label]:
    """Every artifact a passage refers to, as labels.

    A set, because a chunk that says "Figure 1 … Figure 1 again … Fig. 1
    once more" refers to one figure three times, not to three figures.
    """
    return {
        label
        for kind, numbers in _REFERENCE.findall(text)
        for label in _labels(kind, numbers)
    }


def reference_span(text: str, label: Label | None) -> tuple[int, int] | None:
    """Where in `text` the given artifact is named: 'Figure 3' -> (23, 31).

    The walkthrough uses this to mark the phrase a link was made from,
    so a panel can read the evidence rather than take the link on
    trust. It returns the first mention of THIS artifact, not the first
    reference in the passage — a paragraph that discusses Figure 1 and
    then Figure 3 would otherwise have Figure 1 marked under Figure 3's
    crop.

    Returns None when the label is None or the passage never names it.
    """
    if label is None:
        return None
    for found in _REFERENCE.finditer(text):
        if label in _labels(found.group(1), found.group(2)):
            return found.span()
    return None


def link_artifacts(
    artifacts: list[Artifact], chunks: list[Chunk]
) -> tuple[list[Artifact], list[Chunk]]:
    """Bind each artifact to the chunks that cite it, and the reverse.

    Returns new lists in the order they were given; the objects handed
    in are not modified, so Box 5's and Box 6's output stay intact and a
    failure here cannot corrupt what came before:

        artifacts = extract_artifacts(...)              # Box 6
        artifacts, chunks = link_artifacts(artifacts, chunks)   # Box 7

    Within each list the ids come back in reading order, so a reader
    following them is reading forwards through the paper.
    """
    by_label: dict[Label, list[str]] = {}
    for artifact in artifacts:
        label = caption_label(artifact.caption)
        if label is not None:
            by_label.setdefault(label, []).append(artifact.artifact_id)

    chunk_ids: dict[str, list[str]] = {a.artifact_id: [] for a in artifacts}
    linked_chunks: list[Chunk] = []

    for chunk in chunks:
        found: list[str] = []
        for label in references_in(chunk.text):
            for artifact_id in by_label.get(label, []):
                found.append(artifact_id)
                chunk_ids[artifact_id].append(chunk.chunk_id)
        # the artifacts in the order Box 6 produced them, not the order
        # the sentence happened to name them in
        order = {a.artifact_id: i for i, a in enumerate(artifacts)}
        linked_chunks.append(
            chunk.model_copy(update={"artifact_ids": sorted(found, key=order.get)})
        )

    linked_artifacts = [
        artifact.model_copy(update={"linked_chunk_ids": chunk_ids[artifact.artifact_id]})
        for artifact in artifacts
    ]
    return linked_artifacts, linked_chunks
