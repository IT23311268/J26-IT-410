"""
Analysis Pipeline Orchestrator — PP1 Final Stage.

Pipeline:

Literature Review
    -> Claim Extraction
    -> Citation Extraction
    -> PDF Text Extraction & Metadata Parsing
    -> Sentence-based Text Chunking
    -> Embeddings + FAISS Evidence Retrieval
    -> Sentence-level NLI Verification
    -> Citation Verification
    -> Final Results

Important PP1 decisions:

1. The complete citation-stripped claim is verified directly.
2. Automatic claim decomposition is intentionally disabled.
3. Evidence retrieval is restricted to the cited paper when a
   matching paper is available.
4. Retrieval uses:
      - the original clean claim,
      - a quantifier/negation-neutralized query,
      - a short keyword-focused query,
      - a contradiction-focused query when the claim contains
        a restrictive quantifier or explicit polarity cue,
   so contradictory evidence is less likely to be missed.
5. A larger candidate pool is passed to the verifier for
   sentence-level reranking.
"""

import logging

def safe_print(*args, **kwargs):
    try:
        import sys
        if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
            msg = " ".join(str(a) for a in args)
            sys.stdout.buffer.write((msg + "\n").encode("utf-8", "replace"))
        else:
            # We must use builtins.print to avoid recursion if we overrode print
            import builtins
            builtins.print(*args, **kwargs)
    except Exception:
        pass

import re
import time
from typing import Dict, List, Optional

from services.claim_extractor import extract_claims
from services.pdf_processor import (
    extract_text_from_all_pdfs,
    chunk_text,
)
from services.embedding_service import FAISSIndex
from services.verifier import (
    evaluate_and_rerank_evidence,
    _strip_citation_from_claim,
)


logger = logging.getLogger(__name__)


# ================================================================
# RETRIEVAL SETTINGS
# ================================================================

# Broad dense-retrieval candidate pools.
PRIMARY_RETRIEVAL_TOP_K = 25
EXPANDED_RETRIEVAL_TOP_K = 25
KEYWORD_RETRIEVAL_TOP_K = 30
CONTRADICTION_RETRIEVAL_TOP_K = 35

# Maximum number of unique candidates sent to sentence-level verifier.
MAX_VERIFICATION_CANDIDATES = 100


# ================================================================
# CITATION -> PAPER MATCHING
# ================================================================

# Unicode-aware surname pattern.
#
# Supports:
# Ji
# Smith
# Costa-jussà
# García
# Müller
# François

UNICODE_SURNAME = (
    r"[^\W\d_]"
    r"(?:[^\W\d_]|['’\-])*"
)


def extract_literature_review_section(text: str) -> str:
    lines = text.split('\n')
    lr_started = False
    lr_lines = []
    
    lr_headings = [
        r"^\s*(?:\d+\.?\s*)?literature\s+review\s*$",
        r"^\s*(?:\d+\.?\s*)?related\s+work\s*$",
        r"^\s*(?:\d+\.?\s*)?background\s*$",
        r"^\s*(?:\d+\.?\s*)?background\s+and\s+related\s+work\s*$"
    ]
    
    end_headings = [
        r"^\s*(?:\d+\.?\s*)?methodology\s*$",
        r"^\s*(?:\d+\.?\s*)?materials\s+and\s+methods\s*$",
        r"^\s*(?:\d+\.?\s*)?experiment[s]?\s*$",
        r"^\s*(?:\d+\.?\s*)?experimental\s+setup\s*$",
        r"^\s*(?:\d+\.?\s*)?results\s*$",
        r"^\s*(?:\d+\.?\s*)?discussion\s*$",
        r"^\s*(?:\d+\.?\s*)?conclusion[s]?\s*$",
        r"^\s*(?:\d+\.?\s*)?proposed\s+method(?:ology)?\s*$",
        r"^\s*(?:\d+\.?\s*)?method\s*$",
        r"^\s*(?:\d+\.?\s*)?references\s*$",
        r"^\s*(?:\d+\.?\s*)?bibliography\s*$"
    ]
    
    for line in lines:
        if not lr_started:
            for pattern in lr_headings:
                if re.match(pattern, line, re.IGNORECASE):
                    lr_started = True
                    break
        else:
            is_end = False
            for pattern in end_headings:
                if re.match(pattern, line, re.IGNORECASE):
                    is_end = True
                    break
            if is_end:
                break
            lr_lines.append(line)
            
    if not lr_started:
        return text
    return '\n'.join(lr_lines)

def extract_bibliography_section(text: str) -> str:
    lines = text.split('\n')
    bib_started = False
    bib_lines = []
    
    bib_headings = [
        r"^\s*(?:\d+\.?\s*)?references\s*$",
        r"^\s*(?:\d+\.?\s*)?bibliography\s*$"
    ]
    
    for line in lines:
        if not bib_started:
            for pattern in bib_headings:
                if re.match(pattern, line, re.IGNORECASE):
                    bib_started = True
                    break
        else:
            bib_lines.append(line)
            
    if not bib_started:
        return ""
    return '\n'.join(bib_lines)

def parse_bibliography_entries(bib_text: str) -> List[Dict]:
    if re.search(r"^\s*\[\d+\]\s", bib_text, re.MULTILINE):
        entries = []
        matches = list(re.finditer(r"^\s*\[(\d+)\]\s", bib_text, re.MULTILINE))
        for i, match in enumerate(matches):
            ref_num = match.group(1)
            start_pos = match.end()
            end_pos = matches[i+1].start() if i + 1 < len(matches) else len(bib_text)
            ref_content = bib_text[start_pos:end_pos].replace('\n', ' ').strip()
            if ref_content:
                entries.append({
                    "ref_num": ref_num,
                    "ref_text": ref_content
                })
        return entries
        
    if re.search(r"^\s*\d+\.\s", bib_text, re.MULTILINE):
        entries = []
        matches = list(re.finditer(r"^\s*(\d+)\.\s", bib_text, re.MULTILINE))
        for i, match in enumerate(matches):
            ref_num = match.group(1)
            start_pos = match.end()
            end_pos = matches[i+1].start() if i + 1 < len(matches) else len(bib_text)
            ref_content = bib_text[start_pos:end_pos].replace('\n', ' ').strip()
            if ref_content:
                entries.append({
                    "ref_num": ref_num,
                    "ref_text": ref_content
                })
        return entries
        
    refs = re.split(r"\n\s*\n", bib_text)
    entries = []
    for r in refs:
        r = r.replace('\n', ' ').strip()
        if len(r) > 20:
            entries.append({
                "ref_num": None,
                "ref_text": r
            })
    return entries

def _expand_numeric_citation(citation_text: str) -> List[str]:
    """
    Extract all reference numbers from a citation marker.

    Handles:
        [5]       -> ['5']
        [5, 7]    -> ['5', '7']
        [5, 6, 9] -> ['5', '6', '9']
        [5-7]     -> ['5', '6', '7']
        [5–7]     -> ['5', '6', '7']  (en-dash)
    """
    # Find everything inside the outer brackets
    bracket_match = re.search(r"\[([^\]]+)\]", citation_text)
    if not bracket_match:
        return []
    inner = bracket_match.group(1)

    numbers: List[str] = []
    # Split by comma first
    parts = [p.strip() for p in inner.split(",")]
    for part in parts:
        # Range: e.g. "5-7" or "5–7"
        range_match = re.match(r"^(\d+)\s*[-\u2013]\s*(\d+)$", part)
        if range_match:
            lo = int(range_match.group(1))
            hi = int(range_match.group(2))
            numbers.extend(str(n) for n in range(lo, hi + 1))
        elif re.match(r"^\d+$", part):
            numbers.append(part)
    return numbers


def resolve_citation_to_reference(
    citation_text: str,
    refs: List[Dict],
) -> Optional[Dict]:
    """
    Resolve a citation marker to the primary bibliography entry.

    For numeric citations ([5], [5,7], [5-7]) the lookup is a direct
    dictionary lookup by reference number — no fuzzy matching.

    For author-year citations ((Smith et al., 2024)) fuzzy author/year
    matching is used as a fallback.

    Returns the *primary* (first) resolved entry, with an additional key
    ``all_resolved`` containing every resolved entry for multi-number
    citations.
    """
    if not citation_text:
        return None

    # Build a fast lookup dict keyed by ref_num string
    refs_by_num: Dict[str, Dict] = {
        r["ref_num"]: r
        for r in refs
        if r.get("ref_num") is not None
    }

    # ── Numeric citation path ──────────────────────────────────────
    numbers = _expand_numeric_citation(citation_text)
    if numbers:
        resolved_all = []
        for num in numbers:
            if num in refs_by_num:
                resolved_all.append(refs_by_num[num])
                logger.info(
                    "[REFERENCE RESOLUTION] citation='%s' number=%s "
                    "reference_found=YES text=%s",
                    citation_text, num,
                    refs_by_num[num]["ref_text"][:80],
                )
            else:
                logger.info(
                    "[REFERENCE RESOLUTION] citation='%s' number=%s "
                    "reference_found=NO  reason='number not in parsed map'",
                    citation_text, num,
                )
        if resolved_all:
            primary = dict(resolved_all[0])
            primary["all_resolved"] = resolved_all
            return primary
        # All numbers missing from map
        return None

    # ── Author-year citation fallback ─────────────────────────────
    author_match = re.search(UNICODE_SURNAME, citation_text, flags=re.UNICODE)
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", citation_text)

    author = author_match.group().lower() if author_match else None
    year = year_match.group(1) if year_match else None

    if not author:
        logger.info(
            "[REFERENCE RESOLUTION] citation='%s' reference_found=NO "
            "reason='no numeric or author pattern found'",
            citation_text,
        )
        return None

    for r in refs:
        ref_text_lower = r["ref_text"].lower()
        if author in ref_text_lower:
            if year and year in ref_text_lower:
                logger.info(
                    "[REFERENCE RESOLUTION] citation='%s' "
                    "reference_found=YES via author-year",
                    citation_text,
                )
                result = dict(r)
                result["all_resolved"] = [r]
                return result
            elif not year:
                logger.info(
                    "[REFERENCE RESOLUTION] citation='%s' "
                    "reference_found=YES via author only",
                    citation_text,
                )
                result = dict(r)
                result["all_resolved"] = [r]
                return result

    logger.info(
        "[REFERENCE RESOLUTION] citation='%s' reference_found=NO "
        "reason='author %s not matched in any reference entry'",
        citation_text, author,
    )
    return None

def _match_reference_to_paper(ref: Dict, metadata_dict: Dict[str, dict]) -> Optional[str]:
    ref_text = ref["ref_text"]
    doi_match = re.search(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", ref_text, re.I)
    doi = doi_match.group(1).rstrip('.') if doi_match else None
    
    arxiv_match = re.search(r"arxiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)", ref_text, re.I)
    if not arxiv_match:
        arxiv_match = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)", ref_text, re.I)
    arxiv = arxiv_match.group(1) if arxiv_match else None
    
    author_match = re.search(UNICODE_SURNAME, ref_text, flags=re.UNICODE)
    author = author_match.group().lower() if author_match else None
    
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", ref_text)
    year = int(years[-1]) if years else None
    
    if doi:
        for filename, metadata in metadata_dict.items():
            if metadata.get("doi") == doi:
                logger.info(f"Matched reference to paper (Priority 1 - DOI): {filename}")
                return filename
                
    if arxiv:
        for filename, metadata in metadata_dict.items():
            if metadata.get("arxiv") == arxiv:
                logger.info(f"Matched reference to paper (Priority 2 - arXiv): {filename}")
                return filename
                
    if not author:
        return None
        
    for filename, metadata in metadata_dict.items():
        author_surnames = [s.lower() for s in metadata.get("author_surnames", [])]
        metadata_year = metadata.get("year")
        metadata_years = metadata.get("years", [])
        
        author_matches = any((author in s or s in author) for s in author_surnames)
        year_matches = (year is None or metadata_year == year or year in metadata_years)
        
        if author_matches and year_matches:
            logger.info(f"Matched reference to paper (Priority 3 - Author+Year): {filename}")
            return filename
            
    for filename, metadata in metadata_dict.items():
        author_surnames = [s.lower() for s in metadata.get("author_surnames", [])]
        author_matches = any((author in s or s in author) for s in author_surnames)
        if author_matches:
            logger.info(f"Matched reference to paper (Priority 4 - Author only): {filename}")
            return filename
            
    logger.info("Matched reference to paper: None")
    return None


# ================================================================
# RETRIEVAL QUERY EXPANSION
# ================================================================

_QUERY_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "that",
    "this",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "may",
    "might",
    "will",
    "would",
    "shall",
    "should",
    "of",
    "to",
    "for",
    "from",
    "in",
    "on",
    "at",
    "by",
    "with",
    "as",
    "into",
    "through",
    "about",
    "over",
    "under",
    "between",
    "after",
    "before",
    "during",
    "without",
    "within",
    "it",
    "its",
    "their",
    "them",
    "they",
    "we",
    "our",
    "using",
    "used",
    "use",
    "only",
    "one",
    "single",
    "none",
    "no",
    "not",
    "never",
    "cannot",
    "can't",
    "unable",
}


def _build_keyword_query(
    clean_claim: str,
) -> Optional[str]:
    """
    Build a short topic-focused retrieval query.

    Example:

        "The proposed approach enables models for only one language."

    becomes approximately:

        "proposed approach enables models language"

    This query is used only for retrieval. The original claim is still
    used for NLI verification.
    """

    if not clean_claim:
        return None

    tokens = re.findall(
        r"[^\W\d_]+(?:[-’'][^\W\d_]+)*",
        clean_claim.lower(),
        flags=re.UNICODE,
    )

    content_tokens = []

    for token in tokens:
        if len(token) < 3:
            continue

        if token in _QUERY_STOPWORDS:
            continue

        if token not in content_tokens:
            content_tokens.append(token)

    # Prefer a short query so broad topical evidence is not pushed down
    # by restrictive wording from the original claim.
    keyword_tokens = content_tokens[:8]

    if len(keyword_tokens) < 2:
        return None

    return " ".join(keyword_tokens)



def _build_contradiction_query(
    clean_claim: str,
) -> Optional[str]:
    """
    Build a retrieval-only query that searches for an opposing
    quantifier/polarity while preserving the claim topic.

    Examples:

        "models for only one language"
            -> "models for many languages"

        "models for numerous languages"
            -> "models for only one language"

    IMPORTANT:
    This query is used only to retrieve candidate evidence.
    The original claim is still used for NLI verification.
    """

    if not clean_claim:
        return None

    query = re.sub(
        r"\s+",
        " ",
        clean_claim.strip(),
    )

    original = query

    # ------------------------------------------------------------
    # Restrictive singular -> plural/many
    # ------------------------------------------------------------

    if re.search(
        r"\bonly\s+(?:one|1)\b",
        query,
        flags=re.IGNORECASE,
    ):
        query = re.sub(
            r"\bonly\s+(?:one|1)\b",
            "many",
            query,
            count=1,
            flags=re.IGNORECASE,
        )

    elif re.search(
        r"\ba\s+single\b",
        query,
        flags=re.IGNORECASE,
    ):
        query = re.sub(
            r"\ba\s+single\b",
            "many",
            query,
            count=1,
            flags=re.IGNORECASE,
        )

    elif re.search(
        r"\bsingle\b",
        query,
        flags=re.IGNORECASE,
    ):
        query = re.sub(
            r"\bsingle\b",
            "multiple",
            query,
            count=1,
            flags=re.IGNORECASE,
        )

    # ------------------------------------------------------------
    # Positive multi-quantity -> restrictive singular
    # ------------------------------------------------------------

    elif re.search(
        r"\b(?:many|multiple|numerous|several|"
        r"hundreds|thousands|dozens)\b",
        query,
        flags=re.IGNORECASE,
    ):
        query = re.sub(
            r"\b(?:many|multiple|numerous|several|"
            r"hundreds|thousands|dozens)\b",
            "only one",
            query,
            count=1,
            flags=re.IGNORECASE,
        )

    # ------------------------------------------------------------
    # None/no -> some/many
    # ------------------------------------------------------------

    elif re.search(
        r"\b(?:none|no)\b",
        query,
        flags=re.IGNORECASE,
    ):
        query = re.sub(
            r"\b(?:none|no)\b",
            "many",
            query,
            count=1,
            flags=re.IGNORECASE,
        )

    # ------------------------------------------------------------
    # Explicit negation -> positive form for retrieval.
    # This does not decide contradiction; verifier.py still does.
    # ------------------------------------------------------------

    elif re.search(
        r"\b(?:not|never|cannot|can't|unable)\b",
        query,
        flags=re.IGNORECASE,
    ):
        query = re.sub(
            r"\b(?:not|never|cannot|can't|unable)\b",
            "",
            query,
            count=1,
            flags=re.IGNORECASE,
        )

    else:
        return None

    query = re.sub(
        r"\s+",
        " ",
        query,
    ).strip()

    query = re.sub(
        r"\s+([.,;:!?])",
        r"\1",
        query,
    )

    if (
        not query
        or query.lower() == original.lower()
        or len(query.split()) < 3
    ):
        return None

    return query


def _build_retrieval_queries(
    clean_claim: str,
) -> List[str]:
    """
    Build multiple retrieval queries for evidence search.

    Searches:
    1. Original claim
    2. Quantifier/negation-neutralized claim
    3. Keyword-focused topic query
    4. Contradiction-focused query when applicable

    Query expansion affects retrieval only.
    The original complete claim is still used for NLI verification.
    """

    if not clean_claim:
        return []

    original = re.sub(
        r"\s+",
        " ",
        clean_claim.strip(),
    )

    queries = [original]

    # ------------------------------------------------------------
    # Query 2 — neutralize restrictive quantities / negation
    # ------------------------------------------------------------

    expanded = original

    expanded = re.sub(
        r"\bonly\s+(?:one|1)\b",
        "",
        expanded,
        flags=re.IGNORECASE,
    )

    expanded = re.sub(
        r"\ba\s+single\b",
        "",
        expanded,
        flags=re.IGNORECASE,
    )

    expanded = re.sub(
        r"\bsingle\b",
        "",
        expanded,
        flags=re.IGNORECASE,
    )

    expanded = re.sub(
        r"\b(?:none|numerous|many|multiple|"
        r"several|hundreds|thousands|dozens)\b",
        "",
        expanded,
        flags=re.IGNORECASE,
    )

    expanded = re.sub(
        r"\b(?:not|never|cannot|can't|unable)\b",
        "",
        expanded,
        flags=re.IGNORECASE,
    )

    expanded = re.sub(
        r"\s+",
        " ",
        expanded,
    ).strip()

    expanded = re.sub(
        r"\s+([.,;:!?])",
        r"\1",
        expanded,
    )

    if (
        expanded
        and expanded.lower() != original.lower()
        and len(expanded.split()) >= 3
    ):
        queries.append(expanded)

    # ------------------------------------------------------------
    # Query 3 — short keyword/topic query
    # ------------------------------------------------------------

    keyword_query = _build_keyword_query(
        original
    )

    if keyword_query:
        existing_lower = {
            query.lower()
            for query in queries
        }

        if keyword_query.lower() not in existing_lower:
            queries.append(keyword_query)

    # ------------------------------------------------------------
    # Query 4 — contradiction-focused retrieval
    # ------------------------------------------------------------

    contradiction_query = _build_contradiction_query(
        original
    )

    if contradiction_query:
        existing_lower = {
            query.lower()
            for query in queries
        }

        if (
            contradiction_query.lower()
            not in existing_lower
        ):
            queries.append(
                contradiction_query
            )

    return queries


# ================================================================
# EVIDENCE DEDUPLICATION
# ================================================================

def _deduplicate_evidence(
    evidence_chunks: List[dict],
) -> List[dict]:
    """
    Remove duplicate evidence chunks while retaining order.
    """

    unique_evidence = []
    seen_ids = set()

    for evidence in evidence_chunks:
        chunk_id = evidence.get(
            "chunk_id"
        )

        if chunk_id:
            if chunk_id in seen_ids:
                continue

            seen_ids.add(
                chunk_id
            )

        else:
            fallback_key = (
                evidence.get("paper"),
                evidence.get("page"),
                evidence.get(
                    "chunk_text",
                    evidence.get(
                        "text",
                        "",
                    ),
                ),
            )

            if fallback_key in seen_ids:
                continue

            seen_ids.add(
                fallback_key
            )

        unique_evidence.append(
            evidence
        )

    return unique_evidence


def _retrieve_evidence(
    faiss_index: FAISSIndex,
    verification_claim: str,
    target_paper: Optional[str],
) -> List[dict]:
    """
    Retrieve a broad but controlled evidence candidate set.

    Search order:
    1. Original query
    2. Expanded/neutralized query
    3. Keyword-focused query
    4. Contradiction-focused query when applicable

    All searches remain restricted to the matched cited paper when
    target_paper is available.
    """

    queries = _build_retrieval_queries(
        verification_claim
    )

    all_candidates: List[dict] = []

    for query_index, query in enumerate(
        queries,
        start=1,
    ):

        # Identify query type by comparing against generated forms.
        original_query = queries[0]

        expanded_query = None
        keyword_query = _build_keyword_query(
            verification_claim
        )
        contradiction_query = _build_contradiction_query(
            verification_claim
        )

        if query == original_query:
            retrieval_top_k = (
                PRIMARY_RETRIEVAL_TOP_K
            )
            query_type = "original"

        elif (
            contradiction_query
            and query.lower()
            == contradiction_query.lower()
        ):
            retrieval_top_k = (
                CONTRADICTION_RETRIEVAL_TOP_K
            )
            query_type = "contradiction"

        elif (
            keyword_query
            and query.lower()
            == keyword_query.lower()
        ):
            retrieval_top_k = (
                KEYWORD_RETRIEVAL_TOP_K
            )
            query_type = "keyword"

        else:
            retrieval_top_k = (
                EXPANDED_RETRIEVAL_TOP_K
            )
            query_type = "expanded"

        logger.info(
            "Evidence retrieval query %d/%d "
            "(%s): %s",
            query_index,
            len(queries),
            query_type,
            query,
        )

        search_results = faiss_index.search(
            query_text=query,
            top_k=retrieval_top_k,
            paper_filter=target_paper,
        )

        logger.info(
            "Query %d (%s) returned %d chunks.",
            query_index,
            query_type,
            len(search_results),
        )

        # --------------------------------------------------------
        # RETRIEVAL DEBUG
        #
        # Temporary diagnostic output:
        # shows exactly which pages/chunks each retrieval query
        # returns, including contradiction-focused retrieval,
        # so we can confirm whether relevant evidence reaches
        # the verifier.
        # --------------------------------------------------------

        safe_print(
            f"\n[RETRIEVAL DEBUG] "
            f"Query {query_index}/{len(queries)} "
            f"({query_type}): {query}"
        )

        for rank, item in enumerate(
            search_results,
            start=1,
        ):
            item_text = str(
                item.get("text")
                or item.get("chunk_text")
                or ""
            ).replace("\n", " ")

            similarity = float(
                item.get(
                    "similarity_score",
                    0.0,
                )
            )
            
            try:
                safe_text = item_text[:180].encode('ascii', 'ignore').decode('ascii')
                safe_print(
                    f"  #{rank:02d} | "
                    f"page={item.get('page', '?')} | "
                    f"score={similarity:.4f} | "
                    f"text={safe_text}"
                )
            except Exception:
                pass

        all_candidates.extend(
            search_results
        )

    deduplicated = _deduplicate_evidence(
        all_candidates
    )

    # Keep a broader set for sentence-level reranking.
    deduplicated = deduplicated[
        :MAX_VERIFICATION_CANDIDATES
    ]

    logger.info(
        "Evidence retrieval produced %d "
        "unique verification candidates.",
        len(deduplicated),
    )

    return deduplicated


# ================================================================
# SUMMARY COUNTERS
# ================================================================

def _compute_summary(
    claims: List[dict],
) -> dict:
    """Build dashboard summary counts for Layer 1 and Layer 2."""

    summary = {
        "total_cited_claims":
            len(claims),

        "valid_citation_links":
            0,

        "citation_links_not_found":
            0,

        "supported":
            0,

        "partially_supported":
            0,

        "contradicted":
            0,

        "insufficient_evidence":
            0,

        "not_available":
            0,
    }

    for claim in claims:
        link_status = claim.get("citation_link_status", "LINK_NOT_FOUND")
        content_status = claim.get("content_verification_status", "NOT_AVAILABLE")

        if link_status == "LINK_VALID":
            summary["valid_citation_links"] += 1
        else:
            summary["citation_links_not_found"] += 1

        if content_status == "SUPPORTED":
            summary["supported"] += 1
        elif content_status == "PARTIALLY_SUPPORTED":
            summary["partially_supported"] += 1
        elif content_status == "CONTRADICTED":
            summary["contradicted"] += 1
        elif content_status == "INSUFFICIENT_EVIDENCE":
            summary["insufficient_evidence"] += 1
        elif content_status == "NOT_AVAILABLE":
            summary["not_available"] += 1

    return summary


# ================================================================
# MAIN ANALYSIS PIPELINE
# ================================================================

def run_analysis_pipeline(
    literature_review: str,
    uploads_dir: str,
    top_k: int = 3,
) -> dict:
    """
    Run the complete PP1 verification pipeline.
    """

    start_time = time.time()

    # ============================================================
    # STEP 1 — Extract claims
    # ============================================================

    lr_text = extract_literature_review_section(literature_review)
    bib_text = extract_bibliography_section(literature_review)
    bibliography_entries = parse_bibliography_entries(bib_text)
    
    logger.info(f"Extracted {len(bibliography_entries)} bibliography entries from main paper text.")

    claims = extract_claims(
        lr_text
    )

    if not claims:
        empty_summary = {
            "total_claims": 0,
            "supported": 0,
            "partially_supported": 0,
            "contradicted": 0,
            "insufficient_evidence": 0,
            "valid_citations": 0,
            "partially_supported_citations": 0,
            "not_supported_citations": 0,
            "citation_not_found": 0,
        }

        return {
            "status":
                "completed",

            "total_claims":
                0,

            "total_chunks_indexed":
                0,

            "total_papers_processed":
                0,

            "processing_time_seconds":
                round(
                    time.time()
                    - start_time,
                    2,
                ),

            "summary":
                empty_summary,

            "claims":
                [],
        }

    # ============================================================
    # STEP 2 — Extract PDF text and metadata
    # ============================================================

    pages, metadata_dict = (
        extract_text_from_all_pdfs(
            uploads_dir
        )
    )

    if not pages:
        no_pdf_results = []

        for claim in claims:
            citation_text = claim.get(
                "citation_text"
            )

            resolved_ref = resolve_citation_to_reference(citation_text, bibliography_entries)
            reference_entry_found = resolved_ref is not None

            no_pdf_results.append(
                {
                    "claim_id":
                        claim["claim_id"],

                    "claim_text":
                        claim["claim_text"],

                    "citation_text":
                        citation_text,

                    "resolved_reference":
                        resolved_ref.get("ref_text") if resolved_ref else None,

                    "resolved_reference_number":
                        resolved_ref.get("ref_num") if resolved_ref else None,

                    "reference_entry_found":
                        reference_entry_found,

                    "cited_paper_matched":
                        False,

                    "citation_link_status":
                        "LINK_VALID" if reference_entry_found else "LINK_NOT_FOUND",

                    "content_verification_status":
                        "NOT_AVAILABLE",

                    "hallucination_confidence":
                        0.0,

                    "hallucination_reason":
                        "No readable text could be extracted from the uploaded PDFs.",

                    "evidence":
                        [],
                }
            )

        return {
            "status":
                "completed",

            "total_claims":
                len(no_pdf_results),

            "total_chunks_indexed":
                0,

            "total_papers_processed":
                0,

            "processing_time_seconds":
                round(
                    time.time()
                    - start_time,
                    2,
                ),

            "summary":
                _compute_summary(
                    no_pdf_results
                ),

            "claims":
                no_pdf_results,
        }

    # ============================================================
    # STEP 3 — Sentence-aware chunking
    # ============================================================

    chunks = chunk_text(
        pages,
        sentences_per_chunk=4,
        overlap_sentences=1,
    )

    logger.info(
        "Created %d chunks from %d pages.",
        len(chunks),
        len(pages),
    )

    # ============================================================
    # STEP 4 — Build FAISS index
    # ============================================================

    faiss_index = FAISSIndex()

    total_indexed = (
        faiss_index.build(
            chunks
        )
    )

    logger.info(
        "FAISS index built with %d chunks.",
        total_indexed,
    )

    # ============================================================
    # STEP 5 — Verify every complete claim
    # ============================================================

    results = []

    for claim in claims:
        claim_id = claim.get(
            "claim_id"
        )

        claim_text = claim.get(
            "claim_text",
            "",
        )

        citation_text = claim.get(
            "citation_text"
        )

        # --------------------------------------------------------
        # 5A — Match citation to reference paper
        # --------------------------------------------------------

        resolved_ref = resolve_citation_to_reference(citation_text, bibliography_entries)

        reference_entry_found = resolved_ref is not None
        target_paper = None
        if resolved_ref:
            target_paper = _match_reference_to_paper(resolved_ref, metadata_dict)

        # ── Debug: reference resolution block ─────────────────────
        safe_print(
            f"\n[REFERENCE RESOLUTION]"
            f"\n  claim_id         : {claim_id}"
            f"\n  citation_marker  : {citation_text}"
            f"\n  parsed_numbers   : {_expand_numeric_citation(citation_text or '')}"
            f"\n  reference_found  : {'YES' if reference_entry_found else 'NO'}"
        )
        if resolved_ref:
            safe_print(
                f"  resolved_ref_num : {resolved_ref.get('ref_num')}"
                f"\n  resolved_text    : {resolved_ref.get('ref_text', '')[:100]}"
                f"\n  cited_paper_matched : {'YES — ' + target_paper if target_paper else 'NO (PDF not uploaded)'}"
            )
        else:
            safe_print(
                f"  reason           : citation marker not in parsed bibliography"
            )

        # --------------------------------------------------------
        # 5B — Remove citation before verification
        # --------------------------------------------------------

        clean_claim = (
            _strip_citation_from_claim(
                claim_text,
                citation_text,
            )
        ).strip()

        # Verify the complete clean claim directly.
        verification_claim = clean_claim

        safe_print(
            "\n"
            "[VERIFY] "
            "========================================"
        )

        safe_print(
            f"[VERIFY] Claim ID: "
            f"{claim_id}"
        )

        safe_print(
            f"[VERIFY] Original claim: "
            f"{claim_text}"
        )

        safe_print(
            f"[VERIFY] Citation: "
            f"{citation_text}"
        )

        safe_print(
            f"[VERIFY] Matched paper: "
            f"{target_paper}"
        )

        safe_print(
            "[VERIFY] NLI verification claim: "
            f"{verification_claim}"
        )

        # --------------------------------------------------------
        # 5C — Show retrieval queries
        # --------------------------------------------------------

        retrieval_queries = (
            _build_retrieval_queries(
                verification_claim
            )
        )

        safe_print(
            "[VERIFY] Retrieval queries:"
        )

        for query_index, query in enumerate(
            retrieval_queries,
            start=1,
        ):
            safe_print(
                f"    Q{query_index}: {query}"
            )

        safe_print(
            "[VERIFY] "
            "========================================"
        )

        # --------------------------------------------------------
        # 5D — Retrieve candidate evidence
        # --------------------------------------------------------

        if target_paper:
            raw_evidence = (
                _retrieve_evidence(
                    faiss_index=faiss_index,
                    verification_claim=(
                        verification_claim
                    ),
                    target_paper=(
                        target_paper
                    ),
                )
            )
        else:
            raw_evidence = []
            logger.info("Skipping retrieval: no matched target paper.")

        logger.info(
            "Retrieved %d unique candidate chunks "
            "for claim %s.",
            len(raw_evidence),
            claim_id,
        )

        # --------------------------------------------------------
        # 5E — Sentence-level NLI reranking
        # --------------------------------------------------------

        (
            verification_result,
            selected_evidence,
        ) = evaluate_and_rerank_evidence(
            subclaim_text=(
                verification_claim
            ),

            raw_evidence=(
                raw_evidence
            ),

            target_paper=(
                target_paper
            ),

            top_k=max(
                top_k,
                3,
            ),
        )

        hallucination_status = (
            verification_result.get(
                "hallucination_status",
                "INSUFFICIENT_EVIDENCE",
            )
        )

        hallucination_confidence = (
            verification_result.get(
                "hallucination_confidence",
                0.0,
            )
        )

        hallucination_reason = (
            verification_result.get(
                "hallucination_reason",
                (
                    "Unable to determine claim "
                    "verification status."
                ),
            )
        )

        safe_print(
            "[VERIFY RESULT] "
            f"{claim_id} -> "
            f"{hallucination_status} "
            f"({hallucination_confidence:.4f})"
        )

        # --------------------------------------------------------
        # 5F — Deduplicate selected evidence
        # --------------------------------------------------------

        unique_evidence = (
            _deduplicate_evidence(
                selected_evidence
            )
        )

        # --------------------------------------------------------
        # 5G — Layer 1 and Layer 2 evaluation
        # --------------------------------------------------------

        if reference_entry_found:
            citation_link_status = "LINK_VALID"
        else:
            citation_link_status = "LINK_NOT_FOUND"

        if target_paper:
            content_verification_status = hallucination_status
        else:
            content_verification_status = "NOT_AVAILABLE"

        # --------------------------------------------------------
        # 5H — Build final claim result
        # --------------------------------------------------------

        result = {
            "claim_id":
                claim_id,

            "claim_text":
                claim_text,

            "citation_text":
                citation_text,

            "resolved_reference":
                resolved_ref.get("ref_text") if resolved_ref else None,

            "resolved_reference_number":
                resolved_ref.get("ref_num") if resolved_ref else None,

            "reference_entry_found":
                reference_entry_found,

            "cited_paper_matched":
                target_paper is not None,

            "citation_link_status":
                citation_link_status,

            "content_verification_status":
                content_verification_status,

            "hallucination_confidence":
                hallucination_confidence,

            "hallucination_reason":
                hallucination_reason if target_paper else "The actual cited reference paper was not uploaded.",

            "evidence":
                unique_evidence[
                    :max(
                        top_k,
                        3,
                    )
                ],
        }

        results.append(
            result
        )

    # ============================================================
    # STEP 6 — Final response
    # ============================================================

    elapsed = round(
        time.time()
        - start_time,
        2,
    )

    final_response = {
        "status":
            "completed",

        "total_claims":
            len(results),

        "total_chunks_indexed":
            total_indexed,

        "total_papers_processed":
            len(
                metadata_dict
            ),

        "processing_time_seconds":
            elapsed,

        "summary":
            _compute_summary(
                results
            ),

        "claims":
            results,
    }

    logger.info(
        "Analysis completed: "
        "%d claims, %d papers, %d chunks, %.2fs",
        len(results),
        len(metadata_dict),
        total_indexed,
        elapsed,
    )

    return final_response
