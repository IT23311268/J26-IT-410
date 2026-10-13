"""Tests for pipeline Box 7 — binding each artifact to the chunks that
cite it.

The evidence this stage works from is the author's own cross-reference.
A paper that prints "Figure 1: Overview of our approach." under a
drawing and then writes "as shown in Figure 1, we combine..." in the
Method has stated the link itself; nothing has to be inferred. So these
tests are mostly about the ways that reference can be written, and the
ways a naive match on it goes wrong.
"""

import pytest

from ingestion.linking import caption_label, link_artifacts, reference_span
from schema.ingestion_schema_v1 import (
    Artifact,
    ArtifactType,
    BoundingBox,
    Chunk,
    SectionType,
)


def _artifact(index: int, caption: str, kind: ArtifactType = ArtifactType.FIGURE) -> Artifact:
    return Artifact(
        artifact_id=f"p::artifact::{index}",
        paper_id="p",
        artifact_type=kind,
        caption=caption,
        bbox=BoundingBox(page=0, x0=0, y0=0, x1=10, y1=10),
    )


def _chunk(index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"p::chunk::{index}",
        paper_id="p",
        section=SectionType.METHOD,
        text=text,
        page_start=0,
        page_end=0,
        order_index=index,
    )


# --------------------------------------------------------------------------
# the link itself
# --------------------------------------------------------------------------


def test_a_chunk_that_cites_a_figure_is_linked_to_it():
    artifacts, chunks = link_artifacts(
        [_artifact(0, "Figure 1: Overview of our approach.")],
        [_chunk(0, "As shown in Figure 1, we combine a retriever and a generator.")],
    )
    assert artifacts[0].linked_chunk_ids == ["p::chunk::0"]


def test_the_link_is_made_in_both_directions():
    """Member 2 arrives holding a chunk and wants its figures; Member 3
    arrives holding a figure and wants its text. Neither should have to
    scan the other list."""
    artifacts, chunks = link_artifacts(
        [_artifact(0, "Figure 1: Overview.")],
        [_chunk(0, "See Figure 1.")],
    )
    assert artifacts[0].linked_chunk_ids == ["p::chunk::0"]
    assert chunks[0].artifact_ids == ["p::artifact::0"]


def test_a_chunk_citing_two_figures_gets_both():
    artifacts, chunks = link_artifacts(
        [_artifact(0, "Figure 1: Overview."), _artifact(1, "Figure 2: Results.")],
        [_chunk(0, "Figure 1 shows the design and Figure 2 the outcome.")],
    )
    assert chunks[0].artifact_ids == ["p::artifact::0", "p::artifact::1"]


def test_a_figure_cited_in_three_places_collects_all_three():
    artifacts, chunks = link_artifacts(
        [_artifact(0, "Figure 1: Overview.")],
        [
            _chunk(0, "Figure 1 gives the design."),
            _chunk(1, "Nothing relevant here."),
            _chunk(2, "Returning to Figure 1, the retriever is frozen."),
            _chunk(3, "Finally, Figure 1 again."),
        ],
    )
    assert artifacts[0].linked_chunk_ids == ["p::chunk::0", "p::chunk::2", "p::chunk::3"]


# --------------------------------------------------------------------------
# the ways a reference is written
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reference",
    ["Figure 3", "figure 3", "FIGURE 3", "Fig. 3", "Fig 3", "fig.3"],
)
def test_one_figure_written_six_ways_is_still_that_figure(reference):
    artifacts, _ = link_artifacts(
        [_artifact(0, "Figure 3: The gutter.")],
        [_chunk(0, f"This is discussed in {reference} above.")],
    )
    assert artifacts[0].linked_chunk_ids == ["p::chunk::0"]


@pytest.mark.parametrize("label", ["Table 2", "Tab. 2", "table 2"])
def test_tables_are_recognised_too(label):
    artifacts, _ = link_artifacts(
        [_artifact(0, "Table 2: Accuracy.", ArtifactType.TABLE)],
        [_chunk(0, f"The numbers are in {label}.")],
    )
    assert artifacts[0].linked_chunk_ids == ["p::chunk::0"]


@pytest.mark.parametrize(
    "sentence",
    [
        "Figures 1 and 2 show the two halves.",
        "Figures 1, 2 show the two halves.",
        "See figs. 1 and 2.",
    ],
)
def test_a_plural_reference_reaches_every_figure_it_names(sentence):
    """Papers write "Figures 1 and 2" far more often than they write
    the two references out separately. Reading only the first number
    silently drops the second figure."""
    artifacts, _ = link_artifacts(
        [_artifact(0, "Figure 1: The first half."), _artifact(1, "Figure 2: The second.")],
        [_chunk(0, sentence)],
    )
    assert artifacts[0].linked_chunk_ids == ["p::chunk::0"]
    assert artifacts[1].linked_chunk_ids == ["p::chunk::0"]


def test_a_plural_reference_does_not_swallow_the_next_sentence():
    """"Figures 1 and 2" is a list. "Figure 1 and Section 2" is not,
    and neither is a number that merely follows the sentence."""
    artifacts, _ = link_artifacts(
        [_artifact(0, "Figure 1: Overview."), _artifact(1, "Figure 9: Unrelated.")],
        [_chunk(0, "Figure 1 is the design. We trained for 9 epochs.")],
    )
    assert artifacts[0].linked_chunk_ids == ["p::chunk::0"]
    assert artifacts[1].linked_chunk_ids == []


def test_a_caption_that_uses_a_full_stop_still_yields_its_number():
    """'Figure 7. Performance of GPT-4' — not every paper uses a colon."""
    artifacts, _ = link_artifacts(
        [_artifact(0, "Figure 7. Performance of GPT-4 on TruthfulQA.")],
        [_chunk(0, "Figure 7 reports the headline number.")],
    )
    assert artifacts[0].linked_chunk_ids == ["p::chunk::0"]


# --------------------------------------------------------------------------
# the ways a naive match goes wrong
# --------------------------------------------------------------------------


def test_figure_1_is_not_figure_10():
    """A substring match links Figure 1 to every sentence about Figure
    10, 11 and 12 — on a paper with twelve figures that is most of
    them."""
    artifacts, _ = link_artifacts(
        [_artifact(0, "Figure 1: Overview.")],
        [_chunk(0, "Figure 10 shows the ablation.")],
    )
    assert artifacts[0].linked_chunk_ids == []


def test_a_table_and_a_figure_with_the_same_number_are_different_things():
    artifacts, _ = link_artifacts(
        [
            _artifact(0, "Figure 2: The architecture."),
            _artifact(1, "Table 2: Accuracy.", ArtifactType.TABLE),
        ],
        [_chunk(0, "Table 2 reports what Figure 2 describes.")],
    )
    assert artifacts[0].linked_chunk_ids == ["p::chunk::0"]
    assert artifacts[1].linked_chunk_ids == ["p::chunk::0"]


def test_the_word_figure_without_a_number_links_nothing():
    artifacts, _ = link_artifacts(
        [_artifact(0, "Figure 1: Overview.")],
        [_chunk(0, "The figure above is hard to read.")],
    )
    assert artifacts[0].linked_chunk_ids == []


# --------------------------------------------------------------------------
# when there is nothing to link
# --------------------------------------------------------------------------


def test_an_artifact_whose_caption_has_no_label_stays_unlinked():
    """A crop whose caption the layout stage never found, or found as
    a stray line of body text. There is no number to match on, so there
    is nothing honest to say."""
    artifacts, chunks = link_artifacts(
        [_artifact(0, "")],
        [_chunk(0, "Figure 1 shows the design.")],
    )
    assert artifacts[0].linked_chunk_ids == []
    assert chunks[0].artifact_ids == []


def test_a_figure_nobody_mentions_stays_unlinked():
    """Papers do print figures the text never refers to. An empty list
    is the true answer, and a nearest-chunk guess would read as
    evidence to Member 2 without being any."""
    artifacts, _ = link_artifacts(
        [_artifact(0, "Figure 4: An unreferenced plot.")],
        [_chunk(0, "This paragraph is about something else entirely.")],
    )
    assert artifacts[0].linked_chunk_ids == []


def test_no_artifacts_and_no_chunks_is_not_an_error():
    artifacts, chunks = link_artifacts([], [])
    assert artifacts == []
    assert chunks == []


# --------------------------------------------------------------------------
# what the stage must not do
# --------------------------------------------------------------------------


def test_the_objects_handed_in_are_left_alone():
    """Box 6 and Box 5 keep their own output, so the stages stay
    separable and a failure here cannot corrupt what came before."""
    original_artifact = _artifact(0, "Figure 1: Overview.")
    original_chunk = _chunk(0, "Figure 1 shows the design.")

    link_artifacts([original_artifact], [original_chunk])

    assert original_artifact.linked_chunk_ids == []
    assert original_chunk.artifact_ids == []


def test_order_is_reading_order_not_discovery_order():
    """The ids come back in the order the chunks appear in the paper,
    so a reader following them is reading forwards."""
    artifacts, _ = link_artifacts(
        [_artifact(0, "Figure 1: Overview.")],
        [
            _chunk(0, "Nothing here."),
            _chunk(1, "Figure 1 is introduced."),
            _chunk(2, "Figure 1 is revisited."),
        ],
    )
    assert artifacts[0].linked_chunk_ids == ["p::chunk::1", "p::chunk::2"]


# --------------------------------------------------------------------------
# showing the link rather than asserting it
# --------------------------------------------------------------------------


def test_the_citing_words_can_be_found_again():
    """A link the panel has to take on trust is worth less than one
    they can see. The walkthrough marks the phrase the link was made
    from, and this is how it finds it."""
    text = "The design is given in Figure 3 and evaluated later."
    start, end = reference_span(text, ("figure", 3))
    assert text[start:end] == "Figure 3"


def test_the_span_is_the_matching_reference_not_the_first_one():
    text = "Figure 1 is the design; Figure 3 is the ablation."
    start, end = reference_span(text, ("figure", 3))
    assert text[start:end] == "Figure 3"


def test_an_abbreviated_citation_is_found_as_written():
    text = "See Fig. 3 for the gutter."
    start, end = reference_span(text, ("figure", 3))
    assert text[start:end] == "Fig. 3"


def test_a_phrase_that_is_not_there_has_no_span():
    assert reference_span("No references at all here.", ("figure", 3)) is None


def test_a_caption_label_round_trips_into_a_span():
    """The two helpers have to agree, or the walkthrough marks a phrase
    that is not the one the link was made from."""
    label = caption_label("Figure 7. Performance of GPT-4 on TruthfulQA.")
    text = "Accuracy is reported in Figure 7."
    start, end = reference_span(text, label)
    assert text[start:end] == "Figure 7"


def test_a_chunk_is_never_linked_to_the_same_artifact_twice():
    artifacts, chunks = link_artifacts(
        [_artifact(0, "Figure 1: Overview.")],
        [_chunk(0, "Figure 1 is here, and Figure 1 again, and Fig. 1 once more.")],
    )
    assert chunks[0].artifact_ids == ["p::artifact::0"]
    assert artifacts[0].linked_chunk_ids == ["p::chunk::0"]
