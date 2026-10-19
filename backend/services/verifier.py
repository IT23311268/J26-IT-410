"""
Claim-Evidence Verification Service.

Uses:
    cross-encoder/nli-deberta-v3-small

Hallucination statuses:
- SUPPORTED
- PARTIALLY_SUPPORTED
- CONTRADICTED
- INSUFFICIENT_EVIDENCE

Citation statuses:
- VALID
- PARTIALLY_SUPPORTED
- NOT_SUPPORTED
- CITATION_NOT_FOUND

Safeguards:
1. Unicode-aware citations.
2. Citation years removed before NLI/numeric checks.
3. Publication years ignored in numeric comparison.
4. Evidence verified sentence-by-sentence.
5. Contradiction requires:
   - strong NLI contradiction,
   - semantic relevance,
   - lexical relevance,
   - AND a direct proposition conflict.
6. Quantifier/cardinality conflicts are detected:
   - only one vs hundreds
   - single vs multiple
   - none vs many
7. Final decision priority:
   - strong support
   - reliable direct contradiction
   - partial support
   - insufficient evidence
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ================================================================
# NLI MODEL
# ================================================================

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

_nli_model = None
_nli_tokenizer = None

_LABEL_CONTRADICTION = 0
_LABEL_ENTAILMENT = 1
_LABEL_NEUTRAL = 2

# Cached label indices — resolved once after model load.
_cached_contradiction_idx: Optional[int] = None
_cached_entailment_idx: Optional[int] = None
_cached_neutral_idx: Optional[int] = None


def _get_nli_model():
    """Lazy-load the NLI model and tokenizer."""
    global _nli_model, _nli_tokenizer
    global _cached_contradiction_idx, _cached_entailment_idx, _cached_neutral_idx

    if _nli_model is None:
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            logger.info("Loading NLI model: %s", NLI_MODEL)

            _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
            _nli_model = AutoModelForSequenceClassification.from_pretrained(
                NLI_MODEL
            )
            _nli_model.eval()

            logger.info("NLI model loaded: %s", NLI_MODEL)
            logger.info("NLI id2label: %s", _nli_model.config.id2label)

            # Cache label indices once — avoids resolving on every call.
            (
                _cached_contradiction_idx,
                _cached_entailment_idx,
                _cached_neutral_idx,
            ) = _resolve_label_indices(_nli_model)

        except Exception as exc:
            logger.exception(
                "Could not load NLI model '%s': %s",
                NLI_MODEL,
                exc,
            )
            _nli_model = "FAILED"
            _nli_tokenizer = None

    return _nli_model, _nli_tokenizer


def _get_nli_pipeline():
    """Compatibility function used by main.py."""
    return _get_nli_model()


# ================================================================
# CITATION HANDLING
# ================================================================

UNICODE_LETTER = r"[^\W\d_]"

AUTHOR_SURNAME = (
    rf"{UNICODE_LETTER}"
    rf"(?:[^\W\d_]|['’\-])*"
)

AUTHOR_EXPRESSION = (
    rf"{AUTHOR_SURNAME}"
    rf"(?:\s+et\s+al\.)?"
    rf"(?:\s+(?:and|&)\s+{AUTHOR_SURNAME})?"
)

_PAREN_CITATION_PATTERN = re.compile(
    rf"\("
    rf"(?:"
    rf"{AUTHOR_EXPRESSION}"
    rf",?\s*\d{{4}}[a-z]?"
    rf"(?:"
    rf"\s*;\s*"
    rf"{AUTHOR_EXPRESSION}"
    rf",?\s*\d{{4}}[a-z]?"
    rf")*"
    rf")"
    rf"\)",
    flags=re.UNICODE,
)

_BRACKET_AUTHOR_CITATION_PATTERN = re.compile(
    rf"\["
    rf"{AUTHOR_EXPRESSION}"
    rf",?\s*\d{{4}}[a-z]?"
    rf"\]",
    flags=re.UNICODE,
)

_NUMERIC_CITATION_PATTERN = re.compile(
    r"\[\s*\d+(?:\s*[-–,]\s*\d+)*\s*\]"
)


def _remove_all_citation_markers(text: str) -> str:
    """Remove common citation markers from text."""
    if not text:
        return ""

    cleaned = _PAREN_CITATION_PATTERN.sub("", text)
    cleaned = _BRACKET_AUTHOR_CITATION_PATTERN.sub("", cleaned)
    cleaned = _NUMERIC_CITATION_PATTERN.sub("", cleaned)

    # Narrative citation:
    # Costa-jussà et al. (2022)
    # Keep author expression but remove publication year.
    cleaned = re.sub(
        rf"(?<!\w)"
        rf"({AUTHOR_EXPRESSION})"
        rf"\s*\(\s*\d{{4}}[a-z]?\s*\)",
        r"\1",
        cleaned,
        flags=re.UNICODE,
    )

    return cleaned


def _strip_citation_from_claim(
    claim_text: str,
    citation_text: Optional[str],
) -> str:
    """Remove citation metadata before NLI verification."""
    cleaned = claim_text or ""

    if citation_text:
        cleaned = cleaned.replace(citation_text, "")

    cleaned = _remove_all_citation_markers(cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r"\s+;", ";", cleaned)
    cleaned = cleaned.strip()

    return cleaned if cleaned else claim_text


# Public compatibility alias.
strip_citation_from_claim = _strip_citation_from_claim


# ================================================================
# CLAIM DECOMPOSITION
# ================================================================

def decompose_claim(claim_text: str) -> List[str]:
    """
    Compatibility function.

    Current PP1 verifies the complete claim directly.
    """
    if not claim_text:
        return []

    return [claim_text.strip()]


# ================================================================
# SENTENCE SPLITTING
# ================================================================

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


def _split_evidence_sentences(text: str) -> List[str]:
    """Split a retrieved evidence chunk into individual sentences."""
    if not text:
        return []

    cleaned = re.sub(r"\s+", " ", text.strip())

    if not cleaned:
        return []

    for original, protected in _PROTECTED_ABBREVIATIONS.items():
        cleaned = cleaned.replace(original, protected)

    parts = re.split(
        r'(?<=[.!?])\s+(?=[A-Z0-9"\'“‘\[])',
        cleaned,
    )

    sentences = []

    for part in parts:
        part = part.replace("<prd>", ".").strip()

        if len(part) >= 25:
            sentences.append(part)

    if not sentences and cleaned:
        sentences.append(cleaned.replace("<prd>", "."))

    return sentences


# ================================================================
# NUMERICAL CONSISTENCY
# ================================================================

_NUM_PATTERN = re.compile(
    r"(?<![\w.])"
    r"[-+]?"
    r"\d+(?:\.\d+)?"
    r"\s*%?"
)


def _is_publication_year(value: float) -> bool:
    """Return True for likely publication years."""
    if not float(value).is_integer():
        return False

    return 1900 <= int(value) <= 2099


def _prepare_text_for_number_check(text: str) -> str:
    """Remove bibliographic numbers before numeric comparison."""
    if not text:
        return ""

    cleaned = _remove_all_citation_markers(text)

    cleaned = re.sub(
        r"\(\s*(?:19|20)\d{2}[a-z]?\s*\)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    return cleaned


def _extract_numbers(
    text: str,
    ignore_publication_years: bool = True,
) -> List[float]:
    """Extract meaningful factual numeric values."""
    cleaned = _prepare_text_for_number_check(text)
    values: List[float] = []

    for match in _NUM_PATTERN.finditer(cleaned):
        raw = match.group().replace("%", "").strip()

        try:
            value = float(raw)
        except ValueError:
            continue

        if (
            ignore_publication_years
            and _is_publication_year(value)
        ):
            continue

        values.append(value)

    return values


def _numerical_mismatch(
    claim: str,
    evidence: str,
) -> Optional[str]:
    """Detect explicit factual numerical disagreements."""
    claim_nums = _extract_numbers(
        claim,
        ignore_publication_years=True,
    )

    evidence_nums = _extract_numbers(
        evidence,
        ignore_publication_years=True,
    )

    if not claim_nums or not evidence_nums:
        return None

    mismatches = []

    for claim_num in claim_nums:
        if claim_num == 0:
            continue

        closest = min(
            evidence_nums,
            key=lambda value: abs(value - claim_num),
        )

        ratio = (
            abs(claim_num - closest)
            / max(abs(claim_num), 1.0)
        )

        if ratio > 0.20:
            mismatches.append(
                f"claim states {claim_num:g}, "
                f"evidence shows {closest:g}"
            )

    if not mismatches:
        return None

    return (
        "Numerical mismatch detected: "
        + "; ".join(mismatches)
        + "."
    )


# ================================================================
# LEXICAL RELEVANCE
# ================================================================

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but",
    "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was",
    "were", "be", "been", "being", "has",
    "have", "had", "do", "does", "did",
    "can", "could", "may", "might", "will",
    "would", "shall", "should", "of", "to",
    "for", "from", "in", "on", "at", "by",
    "with", "as", "into", "through", "about",
    "over", "under", "between", "after",
    "before", "during", "without", "within",
    "it", "its", "their", "them", "they",
    "we", "our", "using", "used", "use",
}


def _normalize_token(token: str) -> str:
    """Apply lightweight token normalization."""
    token = token.lower().strip()

    if (
        len(token) > 4
        and token.endswith("s")
        and not token.endswith("ss")
    ):
        token = token[:-1]

    return token


def _content_tokens(text: str) -> set:
    """Extract content terms from text."""
    if not text:
        return set()

    raw_tokens = re.findall(
        r"[^\W\d_]+(?:[-’'][^\W\d_]+)*",
        text.lower(),
        flags=re.UNICODE,
    )

    tokens = set()

    for token in raw_tokens:
        normalized = _normalize_token(token)

        if len(normalized) < 3:
            continue

        if normalized in _STOPWORDS:
            continue

        tokens.add(normalized)

    return tokens


def _lexical_relevance(
    claim: str,
    evidence: str,
) -> Tuple[float, int]:
    """Calculate claim-content coverage in evidence."""
    claim_tokens = _content_tokens(claim)
    evidence_tokens = _content_tokens(evidence)

    if not claim_tokens:
        return 0.0, 0

    overlap = claim_tokens & evidence_tokens
    coverage = len(overlap) / len(claim_tokens)

    return coverage, len(overlap)


# ================================================================
# DIRECT CONTRADICTION GATE
# ================================================================

_NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "neither",
    "cannot",
    "can't",
    "doesn't",
    "doesnt",
    "didn't",
    "didnt",
    "isn't",
    "isnt",
    "aren't",
    "arent",
    "without",
    "lack",
    "lacks",
    "lacking",
    "fail",
    "fails",
    "failed",
    "unable",
}

_ANTONYM_PAIRS = [
    ("increase", "decrease"),
    ("improve", "worsen"),
    ("improved", "worsened"),
    ("higher", "lower"),
    ("support", "refute"),
    ("supported", "refuted"),
    ("effective", "ineffective"),
    ("accurate", "inaccurate"),
    ("possible", "impossible"),
    ("enable", "prevent"),
    ("enabled", "prevented"),
    ("present", "absent"),
    ("true", "false"),
]


def _negated_content_terms(text: str) -> set:
    """
    Return content terms close to explicit negation.

    This helps prevent unrelated negation from being treated
    as contradiction of the claim.
    """
    words = re.findall(
        r"[^\W\d_]+(?:[-’'][^\W\d_]+)*",
        text.lower(),
        flags=re.UNICODE,
    )

    normalized_words = [
        _normalize_token(word)
        for word in words
    ]

    result = set()

    for index, word in enumerate(normalized_words):
        if word not in _NEGATION_WORDS:
            continue

        start = index + 1
        end = min(len(normalized_words), index + 6)

        for candidate in normalized_words[start:end]:
            if len(candidate) < 3:
                continue

            if candidate in _STOPWORDS:
                continue

            result.add(candidate)

    return result


def _has_antonym_conflict(
    claim: str,
    evidence: str,
) -> bool:
    """Detect simple opposite-direction wording."""
    claim_lower = claim.lower()
    evidence_lower = evidence.lower()

    for left, right in _ANTONYM_PAIRS:
        if left in claim_lower and right in evidence_lower:
            return True

        if right in claim_lower and left in evidence_lower:
            return True

    return False


# ================================================================
# QUANTIFIER / CARDINALITY CONFLICT
# ================================================================

def _has_quantifier_conflict(
    claim: str,
    evidence: str,
) -> Tuple[bool, str]:
    """
    Detect explicit cardinality conflicts.

    Examples:
        claim:
            "models for only one language"

        evidence:
            "models for hundreds of languages"

        -> direct conflict
    """
    claim_lower = claim.lower()
    evidence_lower = evidence.lower()

    claim_terms = _content_tokens(claim)
    evidence_terms = _content_tokens(evidence)
    shared_terms = claim_terms & evidence_terms

    # Require meaningful proposition overlap.
    if len(shared_terms) < 2:
        return False, "no_quantifier_conflict"

    claim_single = (
        bool(re.search(r"\bonly\s+(?:one|1)\b", claim_lower))
        or bool(re.search(r"\ba\s+single\b", claim_lower))
        or bool(re.search(r"\bsingle\b", claim_lower))
    )

    evidence_multi = (
        bool(
            re.search(
                r"\b(?:many|multiple|numerous|several|"
                r"hundreds|thousands|dozens)\b",
                evidence_lower,
            )
        )
        or bool(re.search(r"\b\d{2,}\b", evidence_lower))
    )

    if claim_single and evidence_multi:
        return True, "single_vs_multiple"

    claim_multi = bool(
        re.search(
            r"\b(?:many|multiple|numerous|several|"
            r"hundreds|thousands|dozens)\b",
            claim_lower,
        )
    )

    evidence_single = (
        bool(re.search(r"\bonly\s+(?:one|1)\b", evidence_lower))
        or bool(re.search(r"\ba\s+single\b", evidence_lower))
        or bool(re.search(r"\bsingle\b", evidence_lower))
    )

    if claim_multi and evidence_single:
        return True, "multiple_vs_single"

    claim_none = bool(
        re.search(r"\b(?:none|no)\b", claim_lower)
    )

    evidence_positive_quantity = bool(
        re.search(
            r"\b(?:some|many|multiple|numerous|several|"
            r"hundreds|thousands|dozens)\b",
            evidence_lower,
        )
    )

    if claim_none and evidence_positive_quantity:
        return True, "none_vs_positive_quantity"

    evidence_none = bool(
        re.search(r"\b(?:none|no)\b", evidence_lower)
    )

    claim_positive_quantity = bool(
        re.search(
            r"\b(?:some|many|multiple|numerous|several|"
            r"hundreds|thousands|dozens)\b",
            claim_lower,
        )
    )

    if evidence_none and claim_positive_quantity:
        return True, "positive_quantity_vs_none"

    return False, "no_quantifier_conflict"


def _direct_conflict(
    claim: str,
    evidence: str,
    num_mismatch: Optional[str],
) -> Tuple[bool, str]:
    """
    Determine whether evidence directly conflicts with
    the same proposition expressed in the claim.

    Strong NLI contradiction alone is NOT enough.
    """
    # 1. Explicit numerical disagreement.
    if num_mismatch:
        return True, "numerical_conflict"

    claim_terms = _content_tokens(claim)
    evidence_terms = _content_tokens(evidence)
    shared_terms = claim_terms & evidence_terms

    # 2. Quantifier/cardinality disagreement.
    quantifier_conflict, quantifier_reason = (
        _has_quantifier_conflict(
            claim,
            evidence,
        )
    )

    if quantifier_conflict:
        return True, quantifier_reason

    # 3. Antonym/opposite-direction disagreement.
    if (
        len(shared_terms) >= 2
        and _has_antonym_conflict(
            claim,
            evidence,
        )
    ):
        return True, "antonym_conflict"

    # 4. Relevant negation.
    negated_terms = _negated_content_terms(evidence)
    negated_claim_terms = negated_terms & claim_terms

    if len(negated_claim_terms) >= 1:
        return True, "claim_term_negated"

    return False, "no_direct_conflict"


# ================================================================
# NLI LABEL RESOLUTION
# ================================================================

def _resolve_label_indices(
    model,
) -> Tuple[int, int, int]:
    """Resolve contradiction / entailment / neutral indices."""
    contradiction_idx = _LABEL_CONTRADICTION
    entailment_idx = _LABEL_ENTAILMENT
    neutral_idx = _LABEL_NEUTRAL

    try:
        id2label = model.config.id2label or {}

        for raw_idx, raw_label in id2label.items():
            idx = int(raw_idx)
            label = str(raw_label).lower()

            if "contrad" in label:
                contradiction_idx = idx
            elif "entail" in label:
                entailment_idx = idx
            elif "neutral" in label:
                neutral_idx = idx

    except Exception:
        pass

    return (
        contradiction_idx,
        entailment_idx,
        neutral_idx,
    )


# ================================================================
# NLI INFERENCE
# ================================================================

def _run_nli(
    premise: str,
    hypothesis: str,
    evidence_page: str = "",
) -> Dict[str, float]:
    """Run sentence-level NLI (single pair)."""
    import torch
    import torch.nn.functional as F

    model, tokenizer = _get_nli_model()

    if (
        model == "FAILED"
        or model is None
        or tokenizer is None
    ):
        logger.warning("NLI model unavailable.")
        return {
            "entailment": 0.34,
            "neutral": 0.33,
            "contradiction": 0.33,
        }

    if not premise or not hypothesis:
        return {
            "entailment": 0.0,
            "neutral": 1.0,
            "contradiction": 0.0,
        }

    try:
        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            logits = model(**inputs).logits[0]

        probabilities = F.softmax(logits, dim=0)

        # Use cached indices when available.
        c_idx = (
            _cached_contradiction_idx
            if _cached_contradiction_idx is not None
            else _resolve_label_indices(model)[0]
        )
        e_idx = (
            _cached_entailment_idx
            if _cached_entailment_idx is not None
            else _resolve_label_indices(model)[1]
        )
        n_idx = (
            _cached_neutral_idx
            if _cached_neutral_idx is not None
            else _resolve_label_indices(model)[2]
        )

        contradiction = float(probabilities[c_idx].item())
        entailment = float(probabilities[e_idx].item())
        neutral = float(probabilities[n_idx].item())

        logger.info(
            "NLI SENTENCE | %s | C=%.4f E=%.4f N=%.4f",
            evidence_page or "unknown",
            contradiction,
            entailment,
            neutral,
        )

        return {
            "entailment": entailment,
            "contradiction": contradiction,
            "neutral": neutral,
        }

    except Exception as exc:
        logger.exception("NLI inference failed: %s", exc)

        return {
            "entailment": 0.34,
            "neutral": 0.33,
            "contradiction": 0.33,
        }


def _run_nli_batch(
    pairs: List[Tuple[str, str]],
) -> List[Dict[str, float]]:
    """
    Run batched NLI inference on a list of (premise, hypothesis) pairs.

    Processes all pairs in a single model forward pass, which is
    significantly faster than calling _run_nli() one pair at a time on CPU.

    Returns a list of result dicts in the same order as the input pairs.
    """
    import torch
    import torch.nn.functional as F

    if not pairs:
        return []

    model, tokenizer = _get_nli_model()

    if (
        model == "FAILED"
        or model is None
        or tokenizer is None
    ):
        return [
            {"entailment": 0.34, "neutral": 0.33, "contradiction": 0.33}
            for _ in pairs
        ]

    # Use cached label indices.
    c_idx = (
        _cached_contradiction_idx
        if _cached_contradiction_idx is not None
        else _resolve_label_indices(model)[0]
    )
    e_idx = (
        _cached_entailment_idx
        if _cached_entailment_idx is not None
        else _resolve_label_indices(model)[1]
    )
    n_idx = (
        _cached_neutral_idx
        if _cached_neutral_idx is not None
        else _resolve_label_indices(model)[2]
    )

    premises = [p for p, _ in pairs]
    hypotheses = [h for _, h in pairs]

    results = []
    sub_batch_size = 32

    try:
        with torch.inference_mode():
            for i in range(0, len(pairs), sub_batch_size):
                chunk_pairs = pairs[i : i + sub_batch_size]
                p_sub = [p for p, _ in chunk_pairs]
                h_sub = [h for _, h in chunk_pairs]

                inputs = tokenizer(
                    p_sub,
                    h_sub,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                    padding=True,
                )

                logits = model(**inputs).logits
                probabilities = F.softmax(logits, dim=1)

                for j in range(len(chunk_pairs)):
                    probs = probabilities[j]
                    results.append({
                        "entailment": float(probs[e_idx].item()),
                        "contradiction": float(probs[c_idx].item()),
                        "neutral": float(probs[n_idx].item()),
                    })

        return results

    except Exception as exc:
        logger.exception("Batch NLI inference failed: %s", exc)
        # Fall back to individual calls.
        fallback_results = []
        for premise, hypothesis in pairs:
            fallback_results.append(
                _run_nli(premise, hypothesis)
            )
        return fallback_results


# ================================================================
# SENTENCE-LEVEL VERIFICATION
# ================================================================

def evaluate_and_rerank_evidence(
    subclaim_text: str,
    raw_evidence: List[dict],
    target_paper: Optional[str],
    top_k: int = 1,
) -> Tuple[dict, List[dict]]:
    """
    Evaluate retrieved evidence sentence-by-sentence.

    Reliable contradiction requires:
    - contradiction >= 0.80
    - adequate semantic similarity
    - adequate lexical overlap
    - direct conflict with the same proposition

    Final decision priority:
    1. Strong support
    2. Reliable direct contradiction
    3. Partial support
    4. Insufficient evidence
    """
    if not raw_evidence:
        return (
            {
                "hallucination_status":
                    "INSUFFICIENT_EVIDENCE",
                "hallucination_confidence":
                    1.0,
                "hallucination_reason":
                    "No evidence chunks were retrieved.",
            },
            [],
        )

    hypothesis = subclaim_text.strip()
    sentence_results = []

    logger.info(
        "Sentence-level verification of %d chunks.",
        len(raw_evidence),
    )

    # ------------------------------------------------------------
    # Phase 1 — collect all candidate sentences (no NLI yet)
    # ------------------------------------------------------------

    # Each entry: (chunk, sentence, relevance, overlap, num_mismatch,
    #              direct_conflict, conflict_type)
    candidate_records = []

    for chunk in raw_evidence:
        chunk_text = (
            chunk.get("chunk_text")
            or chunk.get("text")
            or ""
        )

        chunk_similarity = float(
            chunk.get("similarity_score", 0.0)
        )

        sentences = _split_evidence_sentences(
            chunk_text
        )

        if not sentences:
            continue

        ranked_sentences = []

        for sentence in sentences:
            relevance, overlap = _lexical_relevance(
                hypothesis,
                sentence,
            )

            ranked_sentences.append(
                (
                    relevance,
                    overlap,
                    sentence,
                )
            )

        ranked_sentences.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        # Take the top 3 lexically relevant sentences from each chunk.
        for relevance, overlap, sentence in ranked_sentences[:3]:
            num_mismatch = _numerical_mismatch(
                hypothesis,
                sentence,
            )

            (
                direct_conflict,
                conflict_type,
            ) = _direct_conflict(
                hypothesis,
                sentence,
                num_mismatch,
            )

            candidate_records.append(
                (
                    chunk,
                    chunk_similarity,
                    sentence,
                    relevance,
                    overlap,
                    num_mismatch,
                    direct_conflict,
                    conflict_type,
                )
            )

    # ------------------------------------------------------------
    # Phase 2 — batch NLI on all candidate sentences at once
    # ------------------------------------------------------------

    if not candidate_records:
        return (
            {
                "hallucination_status":
                    "INSUFFICIENT_EVIDENCE",
                "hallucination_confidence":
                    1.0,
                "hallucination_reason":
                    "No usable evidence sentences were found.",
            },
            raw_evidence[:max(1, top_k)],
        )

    nli_pairs = [
        (rec[2], hypothesis)   # (sentence, hypothesis)
        for rec in candidate_records
    ]

    nli_results = _run_nli_batch(nli_pairs)

    # Assemble sentence_results with NLI scores.
    for rec, nli in zip(candidate_records, nli_results):
        (
            chunk,
            chunk_similarity,
            sentence,
            relevance,
            overlap,
            num_mismatch,
            direct_conflict,
            conflict_type,
        ) = rec

        sentence_results.append(
            {
                "chunk": chunk,
                "sentence": sentence,
                "similarity_score":
                    chunk_similarity,
                "lexical_relevance":
                    relevance,
                "overlap_count":
                    overlap,
                "num_mismatch":
                    num_mismatch,
                "direct_conflict":
                    direct_conflict,
                "conflict_type":
                    conflict_type,
                "nli":
                    nli,
            }
        )

    if not sentence_results:
        return (
            {
                "hallucination_status":
                    "INSUFFICIENT_EVIDENCE",
                "hallucination_confidence":
                    1.0,
                "hallucination_reason":
                    "No usable evidence sentences were found.",
            },
            raw_evidence[:max(1, top_k)],
        )



    # ------------------------------------------------------------
    # BEST SUPPORT
    # ------------------------------------------------------------

    best_support = max(
        sentence_results,
        key=lambda item: (
            item["nli"]["entailment"],
            item["lexical_relevance"],
            item["similarity_score"],
        ),
    )

    support_score = float(
        best_support["nli"]["entailment"]
    )

    # ------------------------------------------------------------
    # RELIABLE CONTRADICTIONS
    # ------------------------------------------------------------

    reliable_contradictions = []

    for item in sentence_results:
        contradiction = float(
            item["nli"]["contradiction"]
        )

        relevance = float(
            item["lexical_relevance"]
        )

        overlap = int(
            item["overlap_count"]
        )

        similarity = float(
            item["similarity_score"]
        )

        similarity_ok = (
            similarity >= 0.45
            if similarity <= 1.0
            else similarity >= 45.0
        )

        # Explicit quantifier/numerical conflicts are allowed to use a
        # slightly lower lexical-overlap threshold because short claims
        # such as "models for only one language" may share only two
        # strong content terms ("model", "language") with the opposing
        # evidence sentence.
        strong_explicit_conflict = (
            item["conflict_type"]
            in {
                "single_vs_multiple",
                "multiple_vs_single",
                "none_vs_positive_quantity",
                "positive_quantity_vs_none",
                "numerical_conflict",
            }
        )

        if strong_explicit_conflict:
            lexical_ok = (
                relevance >= 0.20
                and overlap >= 2
            )
        else:
            lexical_ok = (
                relevance >= 0.25
                and overlap >= 3
            )

        direct_conflict_ok = bool(
            item["direct_conflict"]
        )

        if (
            contradiction >= 0.80
            and similarity_ok
            and lexical_ok
            and direct_conflict_ok
        ):
            reliable_contradictions.append(
                item
            )

    best_contradiction = None

    if reliable_contradictions:
        best_contradiction = max(
            reliable_contradictions,
            key=lambda item: (
                item["nli"]["contradiction"],
                item["lexical_relevance"],
            ),
        )

    # ------------------------------------------------------------
    # FINAL DECISION
    #
    # IMPORTANT FIX:
    # A reliable direct contradiction now has priority over
    # weak/partial entailment.
    #
    # Order:
    # 1. Strong support
    # 2. Reliable contradiction
    # 3. Partial support
    # 4. Insufficient evidence
    # ------------------------------------------------------------

    if support_score >= 0.60:
        selected = best_support
        final_status = "SUPPORTED"
        final_confidence = round(
            support_score,
            4,
        )
        final_reason = (
            "A retrieved evidence sentence "
            "directly supports the claim."
        )
        selection_mode = "SUPPORT"

    elif best_contradiction is not None:
        selected = best_contradiction

        contradiction_score = float(
            best_contradiction[
                "nli"
            ]["contradiction"]
        )

        final_status = "CONTRADICTED"
        final_confidence = round(
            contradiction_score,
            4,
        )

        conflict_type = (
            best_contradiction[
                "conflict_type"
            ]
        )

        if best_contradiction[
            "num_mismatch"
        ]:
            final_reason = (
                best_contradiction[
                    "num_mismatch"
                ]
            )

        elif conflict_type == "single_vs_multiple":
            final_reason = (
                "The claim limits the system to one "
                "language, while the retrieved evidence "
                "describes support for many languages."
            )

        elif conflict_type == "multiple_vs_single":
            final_reason = (
                "The claim describes multiple languages, "
                "while the retrieved evidence limits the "
                "system to a single language."
            )

        elif conflict_type == "none_vs_positive_quantity":
            final_reason = (
                "The claim states that none are supported, "
                "while the evidence describes a positive "
                "number or quantity."
            )

        elif conflict_type == "positive_quantity_vs_none":
            final_reason = (
                "The claim describes supported items, "
                "while the evidence explicitly states none."
            )

        elif conflict_type == "numerical_conflict":
            final_reason = (
                "The retrieved evidence contains a "
                "numerical value that directly conflicts "
                "with the claim."
            )

        elif conflict_type == "antonym_conflict":
            final_reason = (
                "The retrieved evidence expresses an "
                "opposite factual relationship to the claim."
            )

        elif conflict_type == "claim_term_negated":
            final_reason = (
                "The retrieved evidence directly negates "
                "a key proposition in the claim."
            )

        else:
            final_reason = (
                "A relevant evidence sentence directly "
                "conflicts with the same proposition "
                "in the claim."
            )

        selection_mode = "CONTRADICTION"

    elif support_score >= 0.35:
        selected = best_support
        final_status = "PARTIALLY_SUPPORTED"
        final_confidence = round(
            support_score,
            4,
        )
        final_reason = (
            "A retrieved evidence sentence "
            "partially supports the claim."
        )
        selection_mode = "PARTIAL_SUPPORT"

    else:
        selected = best_support
        final_status = "INSUFFICIENT_EVIDENCE"

        neutral_score = float(
            best_support["nli"]["neutral"]
        )

        final_confidence = round(
            max(
                neutral_score,
                0.50,
            ),
            4,
        )

        final_reason = (
            "No sufficiently strong supporting "
            "or directly conflicting evidence "
            "sentence was found."
        )

        selection_mode = "INSUFFICIENT"

    selected_chunk = selected["chunk"]
    selected_sentence = selected["sentence"]

    # ------------------------------------------------------------
    # DEBUG OUTPUT
    # ------------------------------------------------------------

    print("\n[FINAL VERIFICATION SELECTION]")

    print(
        f"  Best support entailment : "
        f"{support_score:.4f}"
    )

    if best_contradiction:
        print(
            f"  Reliable contradiction  : "
            f"{best_contradiction['nli']['contradiction']:.4f}"
        )

        print(
            f"  Conflict type           : "
            f"{best_contradiction['conflict_type']}"
        )

        print(
            f"  Conflict sentence       : "
            f"{best_contradiction['sentence']}"
        )

    else:
        strongest_raw = max(
            sentence_results,
            key=lambda item: (
                item["nli"]["contradiction"]
            ),
        )

        print(
            "  Reliable contradiction  : NONE"
        )

        print(
            f"  Strongest raw contradiction: "
            f"{strongest_raw['nli']['contradiction']:.4f}"
        )

        print(
            f"  Raw direct conflict     : "
            f"{strongest_raw['direct_conflict']}"
        )

        print(
            f"  Raw conflict type       : "
            f"{strongest_raw['conflict_type']}"
        )

    print(
        f"  Selection mode          : "
        f"{selection_mode}"
    )

    print(
        f"  Selected page           : "
        f"{selected_chunk.get('page', '?')}"
    )

    print(
        f"  Selected sentence       : "
        f"{selected_sentence}"
    )

    print(
        f"  Final status            : "
        f"{final_status}"
    )

    # ------------------------------------------------------------
    # RESULT
    # ------------------------------------------------------------

    result_dict = {
        "hallucination_status":
            final_status,
        "hallucination_confidence":
            final_confidence,
        "hallucination_reason":
            final_reason,
    }

    # ------------------------------------------------------------
    # FRONTEND EVIDENCE
    # ------------------------------------------------------------

    requested_top_k = max(
        1,
        int(top_k),
    )

    chunk_scores = {}

    for result in sentence_results:
        chunk = result["chunk"]

        chunk_id = (
            chunk.get("chunk_id")
            or (
                f"{chunk.get('paper', '')}"
                f"_p{chunk.get('page', '')}"
                f"_{id(chunk)}"
            )
        )

        score = float(
            result["nli"]["entailment"]
        )

        previous = chunk_scores.get(
            chunk_id
        )

        if (
            previous is None
            or score > previous["score"]
        ):
            chunk_scores[
                chunk_id
            ] = {
                "score": score,
                "chunk": chunk,
            }

    ranked_chunks = sorted(
        chunk_scores.values(),
        key=lambda item: item["score"],
        reverse=True,
    )

    frontend_chunks = [
        selected_chunk
    ]

    for item in ranked_chunks:
        chunk = item["chunk"]

        if chunk is selected_chunk:
            continue

        if (
            len(frontend_chunks)
            >= requested_top_k
        ):
            break

        frontend_chunks.append(
            chunk
        )

    return (
        result_dict,
        frontend_chunks,
    )


# ================================================================
# CITATION VERIFICATION
# ================================================================

def verify_citation(
    claim_text: str,
    citation_text: Optional[str],
    evidence_chunks: List[dict],
    target_paper: Optional[str],
    hallucination_status: str,
    reference_entry_found: bool = False,
) -> dict:
    """Verify citation status using the final hallucination result.

    Distinction:
      - CITATION_NOT_FOUND : citation marker cannot be resolved to any
                             bibliography entry in the main paper.
      - NOT_SUPPORTED      : bibliography entry exists and/or the cited
                             PDF was matched, but no supporting evidence
                             was found.
      - INSUFFICIENT_EVIDENCE : bibliography entry was found but the
                             cited PDF was not uploaded / available.
    """
    if not citation_text:
        return {
            "citation_status":
                "CITATION_NOT_FOUND",
            "citation_reason":
                "No citation was detected for this claim.",
        }

    if not target_paper:
        # Stage A resolved a bib entry, but Stage B found no matching PDF
        if reference_entry_found:
            return {
                "citation_status":
                    "NOT_SUPPORTED",
                "citation_reason": (
                    f"Citation '{citation_text}' was resolved to a "
                    "bibliography entry, but the cited reference paper "
                    "has not been uploaded for verification."
                ),
            }
        # Bib entry itself was not found
        return {
            "citation_status":
                "CITATION_NOT_FOUND",
            "citation_reason": (
                f"Citation '{citation_text}' could not be resolved to "
                "any entry in the References section of the main paper."
            ),
        }

    cited_evidence = [
        evidence
        for evidence in evidence_chunks
        if (
            evidence.get(
                "paper",
                "",
            ).lower()
            ==
            target_paper.lower()
        )
    ]

    if not cited_evidence:
        cited_evidence = [
            evidence
            for evidence in evidence_chunks
            if (
                target_paper.lower()
                in evidence.get(
                    "paper",
                    "",
                ).lower()
            )
        ]

    if not cited_evidence:
        return {
            "citation_status":
                "NOT_SUPPORTED",
            "citation_reason": (
                f"The cited paper '{target_paper}' "
                "was found, but no relevant evidence "
                "was selected."
            ),
        }

    if hallucination_status == "SUPPORTED":
        return {
            "citation_status":
                "VALID",
            "citation_reason": (
                f"The cited paper '{target_paper}' "
                "contains evidence supporting the claim."
            ),
        }

    if hallucination_status == "PARTIALLY_SUPPORTED":
        return {
            "citation_status":
                "PARTIALLY_SUPPORTED",
            "citation_reason": (
                f"The cited paper '{target_paper}' "
                "supports only part of the claim."
            ),
        }

    if hallucination_status == "CONTRADICTED":
        return {
            "citation_status":
                "NOT_SUPPORTED",
            "citation_reason": (
                f"The cited paper '{target_paper}' "
                "contains directly conflicting evidence "
                "for the claim."
            ),
        }

    return {
        "citation_status":
            "NOT_SUPPORTED",
        "citation_reason": (
            f"The cited paper '{target_paper}' "
            "was found, but the retrieved evidence "
            "is insufficient to verify this claim."
        ),
    }


# ================================================================
# LOCAL TESTS
# ================================================================

if __name__ == "__main__":
    print("\nVERIFIER LOCAL TESTS")
    print("=" * 75)

    # Test 1: Previous false contradiction must remain FALSE.
    claim_1 = (
        "Multilingual NMT has emerged as a vital "
        "paradigm for building translation systems "
        "capable of handling numerous languages."
    )

    unrelated_negation = (
        "We do not attempt to optimize the architecture "
        "and parameters of the bilingual NMT systems "
        "to the characteristics of each language pair, "
        "but use the same architecture for all."
    )

    conflict, reason = _direct_conflict(
        claim_1,
        unrelated_negation,
        None,
    )

    print("\nTEST 1 - FALSE CONTRADICTION")
    print(f"Direct conflict: {conflict}")
    print(f"Reason: {reason}")

    # Test 2: Explicit negation.
    real_negation = (
        "Multilingual NMT cannot handle numerous "
        "languages effectively."
    )

    conflict, reason = _direct_conflict(
        claim_1,
        real_negation,
        None,
    )

    print("\nTEST 2 - REAL NEGATION")
    print(f"Direct conflict: {conflict}")
    print(f"Reason: {reason}")

    # Test 3: Quantifier contradiction.
    claim_3 = (
        "The proposed approach enables models "
        "for only one language."
    )

    evidence_3 = (
        "The creation of monolingual and mined "
        "bilingual data enables the creation of "
        "models for hundreds of languages."
    )

    conflict, reason = _direct_conflict(
        claim_3,
        evidence_3,
        None,
    )

    print("\nTEST 3 - QUANTIFIER CONFLICT")
    print(f"Direct conflict: {conflict}")
    print(f"Reason: {reason}")

    print("\nExpected:")
    print("TEST 1 -> False / no_direct_conflict")
    print("TEST 2 -> True / claim_term_negated")
    print("TEST 3 -> True / single_vs_multiple")

    print("=" * 75)
