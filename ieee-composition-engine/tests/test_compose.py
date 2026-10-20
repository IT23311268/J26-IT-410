"""
Tests for stage 5 (composition).

Gemini is NEVER called here. A FakeLLM supplies prepared answers - including
deliberately wrong ones - so we can prove the checks reject bad AI output.

Run with:   pytest -q
"""

from __future__ import annotations

import json
import os
import sys

import docx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compose import FakeLLM, NoLLM, compose                         # noqa: E402
from src.compose.citations import convert_citations                     # noqa: E402
from src.compose.front_matter import extract_terms, ieee_case, tidy_terms  # noqa: E402
from src.compose.grounding import numbers_supported, same_citations     # noqa: E402
from src.compose.ieee_text import apply_text_rules                      # noqa: E402
from src.compose.llm import make_llm, parse_json_reply                  # noqa: E402
from src.compose.references import (format_ieee, ieee_author_list, initials,   # noqa: E402
                                    parse_reference_rules, segments_to_text)
from src.ingest import PackageError, load_document, read_package        # noqa: E402
from src.models import Element, Section, StructuredPaper                # noqa: E402
from src.render import render_docx                                      # noqa: E402
from src.structure import map_to_ieee, structure_document               # noqa: E402

SAMPLE_PDF = os.path.join("data", "input", "sample_paper.pdf")
SAMPLE_PACKAGE = os.path.join("data", "input", "sample_verified_content.json")


def ieee(raw: str) -> str:
    return segments_to_text(format_ieee(parse_reference_rules(raw)))


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def test_initials():
    assert initials("Marc-Alexandre") == "M.-A."
    assert initials("Russ R") == "R. R."
    assert initials("P. A.") == "P. A."


def test_author_list_rules():
    a = [{"given": "Qi", "family": "Dou"}, {"given": "Hao", "family": "Chen"}]
    assert ieee_author_list(a[:1]) == "Q. Dou"
    assert ieee_author_list(a) == "Q. Dou and H. Chen"
    assert ieee_author_list(a + [{"given": "Pheng Ann", "family": "Heng"}]) == \
        "Q. Dou, H. Chen, and P. A. Heng"
    seven = [{"given": "A", "family": f"Name{i}"} for i in range(7)]
    assert ieee_author_list(seven) == "A. Name0 et al."            # more than six


def test_apa_journal_reference():
    out = ieee("Setio, A. A. A., Ciompi, F., & van Riel, S. J. (2016). Pulmonary nodule "
               "detection in CT images. IEEE Transactions on Medical Imaging, 35(5), 1160-1169.")
    assert out == ("A. A. A. Setio, F. Ciompi, and S. J. van Riel, “Pulmonary nodule "
                   "detection in CT images,” IEEE Trans. Med. Imag., vol. 35, "
                   "no. 5, pp. 1160–1169, 2016.")


def test_venue_is_italic():
    segments = format_ieee(parse_reference_rules(
        "Sung, H., & Jemal, A. (2021). Global cancer statistics 2020. "
        "CA: A Cancer Journal for Clinicians, 71(3), 209-249."))
    italic = [t for t, i in segments if i]
    assert italic == ["CA: A Cancer J. Clinicians"]


def test_natbib_arxiv_reference():
    out = ieee("Victor Barres, Honghua Dong, and Karthik Narasimhan. Tau-bench: Evaluating "
               "agents. arXiv preprint arXiv:2506.07982, 2025. URL https://arxiv.org/abs/ 2506.07982.")
    assert out == ("V. Barres, H. Dong, and K. Narasimhan, “Tau-bench: Evaluating agents,” "
                   "2025, arXiv:2506.07982.")


def test_natbib_conference_with_volume():
    out = ieee("Lili Chen, Kevin Lu, and Igor Mordatch. Decision transformer: Reinforcement "
               "learning via sequence modeling. In Advances in Neural Information Processing "
               "Systems, volume 34, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ x.html.")
    assert "in Adv. Neural Inf. Process. Syst., vol. 34, 2021." in out


def test_proceedings_abbreviated_and_publisher_dropped():
    out = ieee("Fabian Gloeckle and Gabriel Synnaeve. Better language models. In Proceedings "
               "of the 41st International Conference on Machine Learning, volume 235 of Proceedings "
               "of Machine Learning Research, pp. 15706–15734. PMLR, 2024.")
    assert "in Proc. 41st Int. Conf. Mach. Learn., vol. 235, 2024, " \
           "pp. 15706–15734." in out
    assert "PMLR" not in out


def test_web_reference_with_broken_url():
    out = ieee("OpenAI. GPT-5 system card, 2025. URL https://openai.com/index/ gpt-5-system-card/.")
    assert out == ("OpenAI, “GPT-5 system card,” 2025. [Online]. Available: "
                   "https://openai.com/index/gpt-5-system-card/")


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

def _paper_with(text: str, refs) -> tuple:
    paper = StructuredPaper(source_path="x")
    s = Section(heading="Intro", number="1", level=1)
    s.elements.append(Element(kind="paragraph", text=text))
    paper.sections.append(s)
    records = [parse_reference_rules(r) for r in refs]
    return paper, records


REFS = [
    "Adams, A. (2019). Alpha. Journal A, 1(1), 1-2.",
    "Brown, B., & Clark, C. (2020). Beta. Journal B, 2(1), 3-4.",
    "Liu, T., & van der Schaar, M. (2025). Gamma. Journal C, 3(1), 5-6.",
    "Liu, Z., Chen, C., & Li, W. (2025). Delta. Journal D, 4(1), 7-8.",
]


def test_citations_numbered_in_order_of_first_appearance():
    paper, records = _paper_with(
        "First (Brown & Clark, 2020). Then Adams (2019) and again (Brown & Clark, 2020).", REFS)
    order, stats = convert_citations(paper, records)
    assert paper.sections[0].elements[0].text == "First [1]. Then Adams [2] and again [1]."
    assert order[:2] == [1, 0]                       # Brown first, then Adams
    assert stats["converted"] == 3


def test_group_citation_and_same_surname_same_year():
    paper, records = _paper_with(
        "See (Liu et al., 2025; Liu & van der Schaar, 2025).", REFS)
    convert_citations(paper, records)
    assert paper.sections[0].elements[0].text == "See [1], [2]."


def test_unknown_citation_is_left_untouched():
    paper, records = _paper_with("As shown (Nobody et al., 2001).", REFS)
    _, stats = convert_citations(paper, records)
    assert paper.sections[0].elements[0].text == "As shown (Nobody et al., 2001)."
    assert stats["unresolved"] == 1


def test_maths_brackets_are_not_citations_in_author_year_papers():
    paper, records = _paper_with("Rewards lie in [0, 1] and (Adams, 2019) agrees.", REFS)
    convert_citations(paper, records)
    assert paper.sections[0].elements[0].text == "Rewards lie in [0, 1] and [1] agrees."


# ---------------------------------------------------------------------------
# IEEE wording rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("before, after", [
    ("As shown in Figure 3.", "As shown in Fig. 3."),
    ("Results are in Table 2.", "Results are in Table II."),
    ("We minimise Eq. 4 here.", "We minimise (4) here."),
    ("Equation (1) defines it.", "Equation (1) defines it."),        # sentence start keeps the word
    ("We don't know.", "We do not know."),
    ("Fig. 2 shows it.", "Fig. 2 shows it."),                       # already correct
])
def test_text_rules(before, after):
    assert apply_text_rules(before)[0] == after


# ---------------------------------------------------------------------------
# Grounding checks
# ---------------------------------------------------------------------------

def test_invented_numbers_are_detected():
    ok, invented = numbers_supported("Accuracy was 99.7% on 888 scans.", "We used 888 scans.")
    assert not ok and invented == ["99.7"]


def test_citation_check():
    assert same_citations("as in [1] and [2]", "see [2] and [1]")
    assert not same_citations("as in [1]", "see [1] and [2]")


# ---------------------------------------------------------------------------
# Index terms and abstract
# ---------------------------------------------------------------------------

def test_index_terms_ieee_form():
    assert ieee_case("Computed Tomography") == "computed tomography"
    assert ieee_case("CNN Models") == "CNN models"
    assert tidy_terms(["Lung cancer", "Deep Learning", "lung cancer", "CT"]) == \
        ["CT", "deep learning", "lung cancer"]


@pytest.fixture
def sample_paper():
    if not os.path.exists(SAMPLE_PDF):
        pytest.skip("run tools/make_sample_paper.py first")
    return map_to_ieee(structure_document(load_document(SAMPLE_PDF)))


def test_rule_based_term_extraction(sample_paper):
    sample_paper.keywords = []
    terms = extract_terms(sample_paper)
    assert "pulmonary nodules" in terms and "lung cancer" in terms


def test_gemini_terms_are_grounded(sample_paper):
    sample_paper.keywords = []
    fake = FakeLLM({"index_terms": {"terms": ["lung cancer", "quantum entanglement",
                                              "computed tomography", "nodule detection"]}})
    compose(sample_paper, fake)
    assert "quantum entanglement" not in sample_paper.keywords
    assert sample_paper.keywords == ["computed tomography", "lung cancer", "nodule detection"]


ABSTRACT = ("Lung cancer is a leading cause of death. We present a three dimensional detector "
            "for pulmonary nodules trained on 888 scans. It reached 94.2 percent sensitivity, "
            "compared with 90.1 percent and 92.0 percent for two baselines. ")


def test_honest_abstract_draft_is_accepted(sample_paper):
    sample_paper.abstract = ""
    compose(sample_paper, FakeLLM({"draft_abstract": {"abstract": ABSTRACT * 2}}))
    assert sample_paper.abstract.startswith("Lung cancer is a leading cause")
    assert any("DRAFTED" in line for line in sample_paper.ai_log)


def test_abstract_with_invented_result_is_rejected(sample_paper):
    sample_paper.abstract = ""
    fake = FakeLLM({"draft_abstract": {"abstract": ABSTRACT + "Accuracy was 99.7 percent."}})
    compose(sample_paper, fake)
    assert sample_paper.abstract == ""
    assert any("99.7" in line for line in sample_paper.compose_report)


def test_short_abstract_is_never_padded(sample_paper):
    before = sample_paper.abstract
    compose(sample_paper, FakeLLM({}))
    assert sample_paper.abstract == before


def test_invented_reference_details_fall_back_to_rules(sample_paper):
    def answer(prompt):
        return [{"authors": [{"given": "X", "family": "Fake"}], "title": "Totally invented title",
                 "venue": "Nowhere", "year": "1999"}] * 4
    compose(sample_paper, FakeLLM({"parse_references": answer}))
    assert all(r["source"] == "rules" for r in sample_paper.reference_records)
    assert "Fake" not in " ".join(sample_paper.references)


# ---------------------------------------------------------------------------
# Polish
# ---------------------------------------------------------------------------

def test_polish_accepts_faithful_and_rejects_changed_numbers():
    paper = StructuredPaper(source_path="x")
    s = Section(heading="Results", number="", level=1)
    good = "The model reached 94.2 percent sensitivity on the test fold, which we think is a " \
           "really good and quite strong result compared to the baseline methods we tried [1]."
    bad = "The detector was trained for 50 epochs with a learning rate of 0.001 and a batch " \
          "size of 16, which took about two days on one graphics card in our lab setup."
    s.elements += [Element("paragraph", good), Element("paragraph", bad)]
    paper.sections.append(s)
    fake = FakeLLM({"polish": {"paragraphs": [
        "The model achieved 94.2 percent sensitivity on the test fold, a strong result "
        "relative to the baseline methods evaluated [1].",
        "The detector was trained for 60 epochs with a learning rate of 0.001 and a batch "
        "size of 16, requiring approximately two days on a single graphics card.",
    ]}})
    compose(paper, fake, polish=True)
    assert paper.sections[0].elements[0].text.startswith("The model achieved")   # accepted
    assert paper.sections[0].elements[1].text == bad                            # 50 -> 60 rejected
    assert any("1/2 paragraphs accepted" in line for line in paper.compose_report)


# ---------------------------------------------------------------------------
# LLM plumbing
# ---------------------------------------------------------------------------

def test_json_reply_with_fences():
    assert parse_json_reply('Sure!\n```json\n{"terms": ["a"]}\n```') == {"terms": ["a"]}


def test_no_api_key_means_rules_only(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setattr("src.compose.llm.PROJECT_ROOT", "/nonexistent")
    assert not make_llm().available
    assert not make_llm(offline=True).available


def test_offline_compose_still_converts_everything(sample_paper):
    compose(sample_paper, NoLLM())
    assert sample_paper.references[0].startswith("H. Sung")          # first cited, first listed
    assert "[1]" in sample_paper.sections[0].paragraphs[0]


# ---------------------------------------------------------------------------
# Verified content package (TAF task 1)
# ---------------------------------------------------------------------------

def test_package_is_read_and_unverified_items_held_back():
    if not os.path.exists(SAMPLE_PACKAGE):
        pytest.skip("example package missing")
    paper = read_package(SAMPLE_PACKAGE)
    assert paper.title.startswith("Automated Detection")
    text = " ".join(e.text for s in paper.sections for e in s.elements)
    assert "guaranteed to detect every nodule" not in text
    assert any("NOT verified" in w for w in paper.warnings)


def test_bad_package_gives_a_clear_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"title": "T"}))
    with pytest.raises(PackageError, match="sections"):
        read_package(str(path))


def test_package_figure_image_is_inserted(tmp_path):
    import pymupdf
    img = tmp_path / "fig1.png"
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 30), False)
    pix.set_rect(pix.irect, (200, 30, 30))
    pix.save(str(img))
    package = {"title": "T", "sections": [{"heading": "Introduction", "level": 1, "content": [
        {"type": "paragraph", "text": "See Figure 1."},
        {"type": "figure", "number": 1, "caption": "A red box.", "image": "fig1.png"}]}]}
    path = tmp_path / "pkg.json"
    path.write_text(json.dumps(package))
    paper = map_to_ieee(read_package(str(path)))
    out = tmp_path / "out.docx"
    render_docx(paper, str(out))
    assert len(docx.Document(str(out)).inline_shapes) == 1


# ---------------------------------------------------------------------------
# The real Gemini client, against a fake Google server (no internet needed)
# ---------------------------------------------------------------------------

class _Reply:
    def __init__(self, text):
        self.text = text


def _gemini_with(responses, tmp_path, monkeypatch):
    """A real GeminiLLM whose network call is replaced by a script of replies."""
    from google.genai import errors
    from src.compose import llm as llm_module
    monkeypatch.setattr(llm_module, "CACHE_DIR", str(tmp_path / "cache"))
    client = llm_module.GeminiLLM("fake-key", models=["model-a", "model-b"], sleep=lambda s: None)
    calls = []

    def generate_content(model, contents, config):
        calls.append(model)
        item = responses.pop(0)
        if isinstance(item, int):
            raise errors.ClientError(item, {"error": {"message": "test", "status": "TEST"}})
        return _Reply(item)

    client.client.models.generate_content = generate_content
    return client, calls


def test_gemini_retries_after_rate_limit_and_switches_retired_model(tmp_path, monkeypatch):
    client, calls = _gemini_with([429, 404, '{"terms": ["a"]}'], tmp_path, monkeypatch)
    assert client.ask_json("index_terms", "prompt") == {"terms": ["a"]}
    assert calls == ["model-a", "model-a", "model-b"]      # waited, retried, then switched model


def test_gemini_answers_are_cached(tmp_path, monkeypatch):
    client, calls = _gemini_with(['{"x": 1}'], tmp_path, monkeypatch)
    assert client.ask_json("t", "same prompt") == {"x": 1}
    assert client.ask_json("t", "same prompt") == {"x": 1}   # second time: from cache
    assert len(calls) == 1 and client.cache_hits == 1


def test_bad_api_key_becomes_a_clear_message(tmp_path, monkeypatch):
    from src.compose.llm import LLMUnavailable
    client, _ = _gemini_with([400], tmp_path, monkeypatch)
    with pytest.raises(LLMUnavailable, match="GEMINI_API_KEY"):
        client.ask_json("t", "prompt")
