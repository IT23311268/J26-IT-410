import React, { useState } from 'react';

// ── Helpers ──────────────────────────────────────────────────────────────────

function truncate(text, maxLen = 140) {
  if (!text || text.length <= maxLen) return text;
  return text.slice(0, maxLen) + '…';
}

// ── Badge Components ──────────────────────────────────────────────────────────

const CONTENT_VERIFY_CONFIG = {
  SUPPORTED:             { label: 'Supported',            css: 'hall-supported' },
  PARTIALLY_SUPPORTED:   { label: 'Partially Supported',  css: 'hall-partial' },
  CONTRADICTED:          { label: 'Contradicted',         css: 'hall-contradicted' },
  INSUFFICIENT_EVIDENCE: { label: 'Insufficient Evidence',css: 'hall-insufficient' },
  NOT_AVAILABLE:         { label: 'Not Available',        css: 'hall-insufficient' },
};

const CITE_LINK_CONFIG = {
  LINK_VALID:     { label: 'Valid',     css: 'cite-valid' },
  LINK_NOT_FOUND: { label: 'Not Found', css: 'cite-not-found' },
};

function ContentVerifyBadge({ status }) {
  const cfg = CONTENT_VERIFY_CONFIG[status] || { label: status, css: 'hall-insufficient' };
  return <span className={`status-pill ${cfg.css}`}>{cfg.label}</span>;
}

function CiteLinkBadge({ status }) {
  const cfg = CITE_LINK_CONFIG[status] || { label: status, css: 'cite-not-found' };
  return <span className={`status-pill ${cfg.css}`}>{cfg.label}</span>;
}

function SimilarityBadge({ score }) {
  const pct = (score * 100).toFixed(1);
  let cls = 'sim-low';
  if (score >= 0.7) cls = 'sim-high';
  else if (score >= 0.4) cls = 'sim-mid';
  return <span className={`sim-badge ${cls}`}>{pct}%</span>;
}

function ConfidencePill({ value }) {
  const pct = (value * 100).toFixed(1);
  let cls = 'conf-low';
  if (value >= 0.7) cls = 'conf-high';
  else if (value >= 0.4) cls = 'conf-mid';
  return <span className={`conf-pill ${cls}`}>{pct}%</span>;
}

// ── Summary Cards ─────────────────────────────────────────────────────────────

function SummaryCards({ summary, total_chunks_indexed, papers_analyzed, processing_time_seconds }) {
  const linkCards = [
    { label: 'Total Cited Claims',  value: summary.total_cited_claims,       cls: 'card-neutral' },
    { label: 'Valid Links',         value: summary.valid_citation_links,     cls: 'card-green' },
    { label: 'Links Not Found',     value: summary.citation_links_not_found, cls: 'card-red' },
  ];

  const verifyCards = [
    { label: 'Supported',        value: summary.supported,                 cls: 'card-green' },
    { label: 'Partial',          value: summary.partially_supported,       cls: 'card-yellow' },
    { label: 'Contradicted',     value: summary.contradicted,              cls: 'card-red' },
    { label: 'Insufficient',     value: summary.insufficient_evidence,     cls: 'card-gray' },
    { label: 'Not Available',    value: summary.not_available,             cls: 'card-neutral' },
  ];

  const sysCards = [
    { label: 'Chunks Indexed',   value: total_chunks_indexed,              cls: 'card-neutral' },
    { label: 'Papers',           value: papers_analyzed,                   cls: 'card-neutral' },
    { label: 'Time (s)',         value: processing_time_seconds,           cls: 'card-neutral' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div>
        <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.9rem', color: '#666' }}>Citation Link Summary</h4>
        <div className="summary-grid">
          {linkCards.map((c) => (
            <div key={c.label} className={`summary-card ${c.cls}`}>
              <span className="summary-card-value">{c.value}</span>
              <span className="summary-card-label">{c.label}</span>
            </div>
          ))}
        </div>
      </div>
      <div>
        <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.9rem', color: '#666' }}>Content Verification Summary</h4>
        <div className="summary-grid">
          {verifyCards.map((c) => (
            <div key={c.label} className={`summary-card ${c.cls}`}>
              <span className="summary-card-value">{c.value}</span>
              <span className="summary-card-label">{c.label}</span>
            </div>
          ))}
        </div>
      </div>
      <div>
        <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.9rem', color: '#666' }}>System Status</h4>
        <div className="summary-grid">
          {sysCards.map((c) => (
            <div key={c.label} className={`summary-card ${c.cls}`}>
              <span className="summary-card-value">{c.value}</span>
              <span className="summary-card-label">{c.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Evidence Panel ────────────────────────────────────────────────────────────

function EvidencePanel({ evidence }) {
  if (!evidence || evidence.length === 0) {
    return (
      <div className="evidence-empty">
        No evidence chunks were retrieved for this claim.
      </div>
    );
  }
  return (
    <div className="evidence-panel">
      {evidence.map((ev, idx) => (
        <div key={idx} className="evidence-card">
          <div className="evidence-header">
            <div className="evidence-meta">
              <span className="evidence-paper" title={ev.paper}>📄 {ev.paper}</span>
              <span className="evidence-page">Page {ev.page}</span>
              <span className={`match-badge ${ev.match_type === 'citation_match' ? 'match-citation' : 'match-similarity'}`}>
                {ev.match_type === 'citation_match' ? '📎 Citation Match' : '🔍 Similarity'}
              </span>
            </div>
            <SimilarityBadge score={ev.similarity_score} />
          </div>
          <div className="evidence-text">{ev.chunk_text}</div>
        </div>
      ))}
    </div>
  );
}

// ── Expanded Claim Detail ─────────────────────────────────────────────────────

function ClaimDetail({ claim }) {
  return (
    <div className="claim-detail">
      <div className="detail-grid">
        <div className="detail-section">
          <h4 className="detail-heading">🔗 Layer 1 — Citation Link Verification</h4>
          <div className="detail-row">
            <span className="detail-label">Citation Marker</span>
            <span className="citation-tag-sm">{claim.citation_text || '—'}</span>
          </div>
          <div className="detail-row">
            <span className="detail-label">Ref # Found</span>
            <span className={`status-pill ${claim.reference_entry_found ? 'hall-supported' : 'hall-insufficient'}`}>
              {claim.reference_entry_found
                ? `✓ Found [${claim.resolved_reference_number || '?'}]`
                : '✗ Not in Bibliography'}
            </span>
          </div>
          <div className="detail-row">
            <span className="detail-label">Resolved Reference</span>
            <span className="detail-value-text" style={{fontSize: '0.85em', fontStyle: 'italic'}}>
              {claim.resolved_reference || '—'}
            </span>
          </div>
          <div className="detail-row">
            <span className="detail-label">Citation Link</span>
            <CiteLinkBadge status={claim.citation_link_status} />
          </div>
        </div>

        <div className="detail-section">
          <h4 className="detail-heading">📄 Layer 2 — Content Verification</h4>
          <div className="detail-row">
            <span className="detail-label">Cited PDF Available</span>
            <span className={`status-pill ${claim.cited_paper_matched ? 'hall-supported' : 'hall-insufficient'}`}>
              {claim.cited_paper_matched ? '✓ Uploaded' : '✗ Not Uploaded'}
            </span>
          </div>
          <div className="detail-row">
            <span className="detail-label">Content Verification</span>
            <ContentVerifyBadge status={claim.content_verification_status} />
          </div>
          {claim.content_verification_status !== 'NOT_AVAILABLE' && (
            <div className="detail-row">
              <span className="detail-label">Confidence</span>
              <ConfidencePill value={claim.hallucination_confidence} />
            </div>
          )}
          <div className="detail-row">
            <span className="detail-label">Evidence Source</span>
            <span className="detail-value-text">
              {claim.cited_paper_matched && claim.evidence && claim.evidence.length > 0
                ? claim.evidence[0].paper + ' · p' + claim.evidence[0].page
                : claim.reference_entry_found
                  ? 'Reference found in bibliography — cited PDF not uploaded'
                  : '—'}
            </span>
          </div>
          <div className="detail-row detail-reason">
            <span className="detail-label">Reason</span>
            <span className="detail-value-text">{claim.hallucination_reason}</span>
          </div>
        </div>
      </div>

      <div className="detail-section" style={{ marginTop: '1rem' }}>
        <h4 className="detail-heading">📋 Full Claim Text</h4>
        <p className="claim-full-text">{claim.claim_text}</p>
      </div>

      <div className="detail-section" style={{ marginTop: '1rem' }}>
        <h4 className="detail-heading">📚 Retrieved Evidence</h4>
        <EvidencePanel evidence={claim.evidence} />
      </div>
    </div>
  );
}

// ── Claim Table Row ───────────────────────────────────────────────────────────

function ClaimRow({ claim }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <>
      <tr
        className={`claim-row ${expanded ? 'claim-row-expanded' : ''}`}
        onClick={() => setExpanded(!expanded)}
      >
        <td className="claim-id-cell">
          <span className="claim-id-badge">{claim.claim_id}</span>
        </td>
        <td className="claim-text-cell" title={claim.claim_text}>
          {truncate(claim.claim_text, 150)}
        </td>
        <td className="citation-cell">
          {claim.citation_text
            ? <span className="citation-tag">{truncate(claim.citation_text, 40)}</span>
            : <span className="no-citation">—</span>}
        </td>
        <td className="status-cell">
          <CiteLinkBadge status={claim.citation_link_status} />
        </td>
        <td className="status-cell">
          <ContentVerifyBadge status={claim.content_verification_status} />
        </td>
        <td className="conf-cell">
          {claim.content_verification_status !== 'NOT_AVAILABLE'
            ? <ConfidencePill value={claim.hallucination_confidence} />
            : <span className="no-citation">—</span>}
        </td>
        <td className="source-cell">
          {claim.evidence && claim.evidence.length > 0
            ? <span className="source-tag" title={claim.evidence[0].paper}>
                {truncate(claim.evidence[0].paper, 28)} · p{claim.evidence[0].page}
              </span>
            : <span className="no-citation">—</span>}
        </td>
        <td className="expand-cell">
          <button className="expand-btn">
            {expanded ? '▲' : '▼'}
          </button>
        </td>
      </tr>
      {expanded && (
        <tr className="detail-row-wrapper">
          <td colSpan={8}>
            <ClaimDetail claim={claim} />
          </td>
        </tr>
      )}
    </>
  );
}

// ── Download Report ───────────────────────────────────────────────────────────

function DownloadReportButton({ result }) {
  const handleDownload = () => {
    const report = {
      analysis_id: result.analysis_id,
      generated_at: new Date().toISOString(),
      summary: result.summary,
      claims: result.claims.map((c) => ({
        claim_id: c.claim_id,
        claim_text: c.claim_text,
        citation_text: c.citation_text,
        resolved_reference: c.resolved_reference,
        resolved_reference_number: c.resolved_reference_number,
        reference_entry_found: c.reference_entry_found,
        cited_paper_matched: c.cited_paper_matched,
        citation_link_status: c.citation_link_status,
        content_verification_status: c.content_verification_status,
        hallucination_confidence: c.hallucination_confidence,
        hallucination_reason: c.hallucination_reason,
        top_evidence: c.evidence.slice(0, 1).map((e) => ({
          paper: e.paper,
          page: e.page,
          similarity_score: e.similarity_score,
        })),
      })),
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `hallucination_report_${result.analysis_id?.slice(0, 8)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <button className="download-btn" onClick={handleDownload} title="Download JSON verification report">
      ⬇ Download Report (JSON)
    </button>
  );
}

// ── Main AnalysisResults Component ───────────────────────────────────────────

function AnalysisResults({ result }) {
  return (
    <section className="section">
      <h2 className="section-title">
        <span className="icon">📊</span>
        Verification Dashboard
      </h2>

      {!result ? (
        <div className="results-placeholder">
          <div className="placeholder-icon">🔬</div>
          <h3>No Analysis Yet</h3>
          <p>
            Paste your literature review, upload reference papers, then click
            "Analyze Literature Review" to see hallucination and citation results.
          </p>
        </div>
      ) : (
        <div className="results-content">
          {/* Top Action Bar */}
          <div className="results-action-bar">
            <div className="result-status-msg">
              <span className={`status-badge-sm ${result.status}`}>● {result.status}</span>
              <span className="result-msg-text">{result.message}</span>
            </div>
            <DownloadReportButton result={result} />
          </div>

          {/* Summary Cards */}
          <SummaryCards
            summary={result.summary}
            total_chunks_indexed={result.total_chunks_indexed}
            papers_analyzed={result.papers_analyzed}
            processing_time_seconds={result.processing_time_seconds}
          />

          {/* Claims Table */}
          {result.claims && result.claims.length > 0 ? (
            <div className="claims-table-wrapper">
              <table className="claims-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Claim</th>
                    <th>Citation</th>
                    <th>Citation Link</th>
                    <th>Content Verification</th>
                    <th>Confidence</th>
                    <th>Evidence Source</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {result.claims.map((claim) => (
                    <ClaimRow key={claim.claim_id} claim={claim} />
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="no-claims-msg">
              No claims were extracted from the literature review.
            </p>
          )}
        </div>
      )}
    </section>
  );
}

export default AnalysisResults;
