"""
Pydantic models for request/response data validation.
Updated for Stage 3: Hallucination Detection + Citation Verification.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ── Literature Review ──────────────────────────────────────────────────────

class LiteratureReviewRequest(BaseModel):
    """Request model for submitting a literature review."""
    literature_review: str = Field(
        ...,
        min_length=50,
        description="The full text of the literature review to analyze.",
    )


class LiteratureReviewResponse(BaseModel):
    """Response model after a literature review is saved."""
    id: str
    message: str
    literature_review_preview: str
    character_count: int
    submitted_at: str


# ── Reference Papers ───────────────────────────────────────────────────────

class PaperMetadata(BaseModel):
    """Metadata for a single uploaded reference paper."""
    original_filename: str
    saved_filename: str
    file_size_bytes: int
    uploaded_at: str


class PapersUploadResponse(BaseModel):
    """Response model after uploading reference papers."""
    message: str
    total_uploaded: int
    papers: List[PaperMetadata]


# ── Analysis Request ───────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    """Request model to start analysis."""
    literature_review_id: str
    paper_filenames: List[str]


# ── Evidence & Claims (Stage 3) ────────────────────────────────────────────

class EvidenceChunk(BaseModel):
    """A single piece of evidence retrieved from a reference paper."""
    chunk_id: str
    paper: str
    page: int
    chunk_text: str
    similarity_score: float
    match_type: str  # "citation_match" | "similarity_based"


class ClaimResult(BaseModel):
    """Full result for a single extracted claim."""
    claim_id: str
    claim_text: str
    citation_text: Optional[str] = None
    
    resolved_reference: Optional[str] = None
    resolved_reference_number: Optional[str] = None
    reference_entry_found: bool
    cited_paper_matched: bool

    # Layer 1
    citation_link_status: str

    # Layer 2
    content_verification_status: str

    hallucination_confidence: float
    hallucination_reason: str

    evidence: List[EvidenceChunk]

# ── Summary Statistics ─────────────────────────────────────────────────────

class AnalysisSummary(BaseModel):
    """Aggregated counts across all claims."""
    total_cited_claims: int
    valid_citation_links: int
    citation_links_not_found: int
    
    supported: int
    partially_supported: int
    contradicted: int
    insufficient_evidence: int
    not_available: int


# ── Analysis Response ──────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    """Full analysis response: claims, evidence, verification, and summary."""
    analysis_id: str
    status: str
    message: str
    literature_review_id: str
    papers_analyzed: int
    total_claims: int
    total_chunks_indexed: int
    processing_time_seconds: float
    summary: AnalysisSummary
    claims: List[ClaimResult]

# "?"? Reference Verification "?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?"?

class ReferenceVerificationDetail(BaseModel):
    status: str
    reason: str
    match: Optional[dict] = None
    score: float
    verification_detail: Optional[str] = None
    matched_source: Optional[str] = None
    matched_title: Optional[str] = None
    matched_authors: Optional[str] = None
    matched_year: Optional[str] = None
    matched_doi: Optional[str] = None
    matched_arxiv: Optional[str] = None
    confidence: Optional[float] = None
    verified_source: Optional[str] = None
    source_url: Optional[str] = None

class ReferenceVerificationResult(BaseModel):
    reference_text: str
    verification: ReferenceVerificationDetail

class ReferenceVerificationSummary(BaseModel):
    real: int
    fake: int
    uncertain: int

class ReferenceVerificationResponse(BaseModel):
    paper_filename: str
    total_extracted: int
    total_analyzed: int
    summary: ReferenceVerificationSummary
    details: List[ReferenceVerificationResult]
