"""
Claim Extraction Service.

Splits a literature review into individual sentences and extracts
academic citations attached to each sentence.

Supported citation examples:
- (Smith et al., 2024)
- Smith et al. (2024)
- (Smith, 2024; Jones, 2023)
- (Costa-jussà et al., 2022)
- (García et al., 2023)
- (Müller et al., 2024)
- [1]
- [2, 3]
- [1-3]

This version supports Unicode author surnames.
"""

import re
from typing import List, Optional


# -------------------------------------------------------------------
# Unicode-aware author name patterns
# -------------------------------------------------------------------

# [^\W\d_] means:
#   any Unicode letter, excluding digits and underscore.
#
# This supports names such as:
#   Costa-jussà
#   García
#   Müller
#   François
#   O'Connor
#   Smith-Jones
#
# Python's re module is Unicode-aware by default.

UNICODE_LETTER = r"[^\W\d_]"
AUTHOR_SURNAME = rf"{UNICODE_LETTER}(?:[^\W\d_]|['’\-])*"

# One author expression:
# Smith
# Smith et al.
# Smith and Jones
# Smith & Jones
AUTHOR_EXPRESSION = (
    rf"{AUTHOR_SURNAME}"
    rf"(?:\s+et\s+al\.)?"
    rf"(?:\s+(?:and|&)\s+{AUTHOR_SURNAME})?"
)


# -------------------------------------------------------------------
# Citation regex patterns
# -------------------------------------------------------------------

# Parenthetical citations:
#
# (Smith, 2024)
# (Smith et al., 2024)
# (Costa-jussà et al., 2022)
# (Smith, 2024; Jones, 2023)
# (García et al., 2023; Müller et al., 2024)

PAREN_CITE = re.compile(
    rf"\("
    rf"(?:"
    rf"{AUTHOR_EXPRESSION}"
    rf",?\s*\d{{4}}"
    rf"(?:"
    rf"\s*;\s*"
    rf"{AUTHOR_EXPRESSION}"
    rf",?\s*\d{{4}}"
    rf")*"
    rf")"
    rf"\)",
    flags=re.UNICODE,
)


# Narrative citations:
#
# Smith et al. (2024)
# Smith and Jones (2024)
# Costa-jussà et al. (2022)
# García et al. (2023)

NARRATIVE_CITE = re.compile(
    rf"(?<!\w)"
    rf"{AUTHOR_EXPRESSION}"
    rf"\s*\(\s*\d{{4}}\s*\)",
    flags=re.UNICODE,
)


# Numeric citations:
#
# [1]
# [2, 3]
# [1-3]
# [1–3]
# [1, 2, 3]

NUMERIC_CITE = re.compile(
    r"\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]"
)


# -------------------------------------------------------------------
# Citation extraction
# -------------------------------------------------------------------

def _extract_citations(text: str) -> List[str]:
    """
    Extract all citation strings from a sentence.

    Returns unique citations while preserving their original order.
    """

    if not text:
        return []

    citations: List[str] = []

    matches = []

    for pattern in (
        PAREN_CITE,
        NARRATIVE_CITE,
        NUMERIC_CITE,
    ):
        for match in pattern.finditer(text):
            matches.append(
                (
                    match.start(),
                    match.group().strip(),
                )
            )

    # Preserve citation order as it appeared in the sentence.
    matches.sort(key=lambda item: item[0])

    for _, citation in matches:
        if citation not in citations:
            citations.append(citation)

    return citations


# -------------------------------------------------------------------
# Sentence splitting
# -------------------------------------------------------------------

# Instead of a variable-width negative look-behind, abbreviations are
# temporarily protected before sentence splitting.
#
# This avoids problems such as:
#
#   "Ji et al. (2023) discuss..."
#
# accidentally splitting after "al."

_PROTECTED_ABBREVIATIONS = {
    "et al.": "et al<prd>",
    "e.g.": "e<prd>g<prd>",
    "i.e.": "i<prd>e<prd>",
    "Dr.": "Dr<prd>",
    "Mr.": "Mr<prd>",
    "Mrs.": "Mrs<prd>",
    "Ms.": "Ms<prd>",
    "Fig.": "Fig<prd>",
    "Eq.": "Eq<prd>",
    "Vol.": "Vol<prd>",
    "No.": "No<prd>",
    "vs.": "vs<prd>",
}


SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[^\s])"
)


def _protect_abbreviations(text: str) -> str:
    """
    Temporarily protect common abbreviations from sentence splitting.
    """

    protected = text

    for original, replacement in _PROTECTED_ABBREVIATIONS.items():
        protected = protected.replace(
            original,
            replacement,
        )

    return protected


def _restore_abbreviations(text: str) -> str:
    """
    Restore abbreviation periods after sentence splitting.
    """

    return text.replace(
        "<prd>",
        ".",
    )


def _split_sentences(text: str) -> List[str]:
    """
    Split literature review text into sentences using safe regex
    heuristics while preserving citation expressions.
    """

    if not text:
        return []

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text.strip(),
    )

    if not text:
        return []

    protected_text = _protect_abbreviations(
        text
    )

    parts = SENTENCE_SPLIT.split(
        protected_text
    )

    sentences: List[str] = []

    for part in parts:

        part = _restore_abbreviations(
            part
        ).strip()

        # Ignore empty or extremely short fragments.
        if len(part) > 10:
            sentences.append(part)

    return sentences


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def extract_claims(
    literature_review: str,
) -> List[dict]:
    """
    Extract claim-like sentences and their citations.

    Returns:

    [
        {
            "claim_id": "C1",
            "claim_text": "...",
            "citation_text": "(Costa-jussà et al., 2022)"
        }
    ]
    """

    sentences = _split_sentences(
        literature_review
    )

    claims: List[dict] = []

    claim_counter = 0

    for sentence in sentences:

        citations = _extract_citations(
            sentence
        )

        # ── Only cited sentences become claims ──
        if not citations:
            continue

        claim_counter += 1

        citation_text = "; ".join(
            citations
        )

        claims.append(
            {
                "claim_id": f"C{claim_counter}",
                "claim_text": sentence,
                "citation_text": citation_text,
            }
        )

    return claims


# -------------------------------------------------------------------
# Optional local test
# -------------------------------------------------------------------

if __name__ == "__main__":

    sample_text = (
        "Deep learning based natural language generation systems "
        "can produce hallucinated content "
        "(Ji et al., 2023). "
        "Multilingual NMT has emerged as a vital paradigm for "
        "building translation systems capable of handling numerous "
        "languages (Costa-jussà et al., 2022). "
        "García et al. (2023) investigated multilingual systems. "
        "Müller et al. (2024) reported similar findings. "
        "François et al. (2021) studied another approach. "
        "Transformer models are widely used "
        "(Smith, 2024; Jones, 2023). "
        "Several studies also report similar findings [1, 2]."
    )

    results = extract_claims(
        sample_text
    )

    print("\nEXTRACTED CLAIMS")
    print("=" * 70)

    for claim in results:
        print(
            f"ID       : {claim['claim_id']}"
        )
        print(
            f"Claim    : {claim['claim_text']}"
        )
        print(
            f"Citation : {claim['citation_text']}"
        )
        print("-" * 70)