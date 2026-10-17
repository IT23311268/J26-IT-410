import React, { useState } from 'react';

// ── Badge helpers ─────────────────────────────────────────────────────────────

const REF_STATUS_CONFIG = {
  REAL:      { label: 'Real',      css: 'ref-real' },
  FAKE:      { label: 'Fake',      css: 'ref-fake' },
  UNCERTAIN: { label: 'Uncertain', css: 'ref-uncertain' },
  ERROR:     { label: 'Error',     css: 'ref-error' },
};

function RefStatusBadge({ status }) {
  const cfg = REF_STATUS_CONFIG[status] || { label: status, css: 'ref-error' };
  return <span className={`status-pill ${cfg.css}`}>{cfg.label}</span>;
}

function ScoreBadge({ score }) {
  const pct = (score * 100).toFixed(0);
  let cls = 'sim-low';
  if (score >= 0.7) cls = 'sim-high';
  else if (score >= 0.4) cls = 'sim-mid';
  return <span className={`sim-badge ${cls}`}>{pct}%</span>;
}

// ── Single Reference Row ──────────────────────────────────────────────────────

function ReferenceRow({ ref_item, index }) {
  const [expanded, setExpanded] = useState(false);
  const v = ref_item.verification;

  const matchedTitle = v.matched_title;
  const matchedDOI = v.matched_doi;
  const matchedAuthor = v.matched_authors;
  const matchedArxiv = v.matched_arxiv;
  const matchedYear = v.matched_year;
  const matchedSource = v.matched_source || 'Crossref';
  const verifiedSource = v.verified_source;
  const sourceUrl = v.source_url;
  const isReal = v.status === 'REAL';

  return (
    <>
      <tr
        className={`claim-row ${expanded ? 'claim-row-expanded' : ''}`}
        onClick={() => setExpanded(!expanded)}
      >
        <td className="claim-id-cell">
          <span className="claim-id-badge">R{index + 1}</span>
        </td>
        <td className="claim-text-cell" title={ref_item.reference_text}>
          {ref_item.reference_text.length > 120
            ? ref_item.reference_text.slice(0, 120) + '…'
            : ref_item.reference_text}
        </td>
        <td className="status-cell">
          <RefStatusBadge status={v.status} />
        </td>
        <td className="conf-cell">
          <ScoreBadge score={v.score} />
        </td>
        <td className="actions-cell" onClick={(e) => e.stopPropagation()}>
          {isReal && sourceUrl && (
            <a
              href={sourceUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="paper-link-btn"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                fontSize: '0.75rem',
                padding: '3px 8px',
                borderRadius: '4px',
                background: '#e0e7ff',
                color: '#3730a3',
                textDecoration: 'none',
                fontWeight: 600,
                whiteSpace: 'nowrap',
                width: 'fit-content'
              }}
              title="Open authoritative source page"
            >
              🔗 View Paper
            </a>
          )}
        </td>
        <td className="expand-cell">
          <button className="expand-btn">{expanded ? '▲' : '▼'}</button>
        </td>
      </tr>
      {expanded && (
        <tr className="detail-row-wrapper">
          <td colSpan={6}>
            <div className="claim-detail">
              <div className="detail-section">
                <h4 className="detail-heading">📄 Full Reference Text</h4>
                <p className="claim-full-text">{ref_item.reference_text}</p>
              </div>

              <div className="detail-grid" style={{ marginTop: '1rem' }}>
                <div className="detail-section">
                  <h4 className="detail-heading">🔎 Verification Details</h4>
                  <div className="detail-row">
                    <span className="detail-label">Status</span>
                    <RefStatusBadge status={v.status} />
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Match Score</span>
                    <ScoreBadge score={v.score} />
                  </div>
                  <div className="detail-row detail-reason">
                    <span className="detail-label">Reason</span>
                    <span className="detail-value-text">{v.reason}</span>
                  </div>
                  {isReal && verifiedSource && (
                    <div className="detail-row">
                      <span className="detail-label">Verified Source</span>
                      <span className="detail-value-text" style={{ fontWeight: 600, color: '#166534' }}>
                        {verifiedSource}
                      </span>
                    </div>
                  )}
                  {isReal && sourceUrl && (
                    <div className="detail-row">
                      <span className="detail-label">Source Page</span>
                      <a
                        className="detail-value-text"
                        href={sourceUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        onClick={(e) => e.stopPropagation()}
                        style={{ color: '#1e40af', textDecoration: 'underline', fontWeight: 600 }}
                      >
                        Open ↗
                      </a>
                    </div>
                  )}
                </div>

                                {matchedTitle && (
                  <div className="detail-section">
                    <h4 className="detail-heading">📖 Matched Record ({matchedSource})</h4>
                    <div className="detail-row detail-reason">
                      <span className="detail-label">Title</span>
                      <span className="detail-value-text">{matchedTitle}</span>
                    </div>
                    {matchedAuthor && (
                      <div className="detail-row">
                        <span className="detail-label">Author</span>
                        <span className="detail-value-text">{matchedAuthor}</span>
                      </div>
                    )}
                    {matchedYear && (
                      <div className="detail-row">
                        <span className="detail-label">Year</span>
                        <span className="detail-value-text">{matchedYear}</span>
                      </div>
                    )}
                    {matchedSource && (
                      <div className="detail-row">
                        <span className="detail-label">Source</span>
                        <span className="detail-value-text">{matchedSource}</span>
                      </div>
                    )}
                    {matchedArxiv && (
                      <div className="detail-row">
                        <span className="detail-label">arXiv ID</span>
                        <a
                          className="detail-value-text"
                          href={`https://arxiv.org/abs/${matchedArxiv}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          onClick={(e) => e.stopPropagation()}
                          style={{ color: '#1e40af', textDecoration: 'underline' }}
                        >
                          {matchedArxiv}
                        </a>
                      </div>
                    )}
                    {matchedDOI && (
                      <div className="detail-row">
                        <span className="detail-label">DOI</span>
                        <a
                          className="detail-value-text"
                          href={
                            matchedDOI.startsWith('http')
                              ? matchedDOI
                              : `https://doi.org/${matchedDOI}`
                          }
                          target="_blank"
                          rel="noopener noreferrer"
                          onClick={(e) => e.stopPropagation()}
                          style={{ color: '#1e40af', textDecoration: 'underline' }}
                        >
                          {matchedDOI}
                        </a>
                      </div>
                    )}
                  </div>
                )}

              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

// ── Summary Cards ─────────────────────────────────────────────────────────────

function RefSummaryCards({ summary, totalExtracted, totalAnalyzed }) {
  const cards = [
    { label: 'Total References', value: totalExtracted, cls: 'card-neutral' },
    { label: 'Analyzed',         value: totalAnalyzed,   cls: 'card-neutral' },
    { label: 'Real',             value: summary.real,      cls: 'card-green' },
    { label: 'Fake',             value: summary.fake,      cls: 'card-red' },
    { label: 'Uncertain',        value: summary.uncertain,  cls: 'card-yellow' },
  ];

  return (
    <div className="summary-grid">
      {cards.map((c) => (
        <div key={c.label} className={`summary-card ${c.cls}`}>
          <span className="summary-card-value">{c.value}</span>
          <span className="summary-card-label">{c.label}</span>
        </div>
      ))}
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────

function ReferenceVerification({ paperFilenames }) {
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Dedicated PDF for bibliography audit
  const [auditFile, setAuditFile] = useState(null);
  const [auditFilename, setAuditFilename] = useState(null); // saved filename after upload
  const fileInputRef = React.useRef(null);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Please select a PDF file.');
      return;
    }
    setAuditFile(file);
    setAuditFilename(null);
    setResults(null);
    setError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.name.toLowerCase().endsWith('.pdf')) {
      setAuditFile(file);
      setAuditFilename(null);
      setResults(null);
      setError(null);
    }
  };

  const handleVerify = async () => {
    setError(null);
    setResults(null);
    setIsLoading(true);

    try {
      let filename = auditFilename;

      // Upload the audit PDF if not yet uploaded
      if (!filename && auditFile) {
        const formData = new FormData();
        formData.append('files', auditFile);
        const uploadResp = await fetch('http://localhost:8000/api/papers/upload', {
          method: 'POST',
          body: formData,
        });
        if (!uploadResp.ok) throw new Error('Failed to upload PDF.');
        const uploadData = await uploadResp.json();
        filename = uploadData.papers[0].saved_filename;
        setAuditFilename(filename);
      }

      // Fall back to first uploaded paper from main analysis
      if (!filename) {
        if (!paperFilenames || paperFilenames.length === 0) {
          setError('Please upload a PDF to audit, or run the main analysis first.');
          setIsLoading(false);
          return;
        }
        filename = paperFilenames[0];
      }

      const resp = await fetch(`http://localhost:8000/api/papers/${encodeURIComponent(filename)}/verify-references`);
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${resp.status}`);
      }
      const data = await resp.json();
      setResults(data);
    } catch (err) {
      setError(err.message || 'An error occurred while verifying references.');
    } finally {
      setIsLoading(false);
    }
  };

  const noSource = !auditFile && (!paperFilenames || paperFilenames.length === 0);

  return (
    <section className="section">
      <h2 className="section-title">
        <span className="icon">📖</span>
        Bibliography Audit — Real vs Fake Reference Check
      </h2>
      <p className="section-description">
        Upload the paper whose bibliography you want to audit.
      </p>

      {/* PDF Drop Zone */}
      <div
        className="pdf-drop-zone"
        style={{
          border: '2px dashed #6366f1',
          borderRadius: '10px',
          padding: '1.25rem',
          marginBottom: '1rem',
          textAlign: 'center',
          background: auditFile ? '#f0fdf4' : '#f8f8ff',
          cursor: 'pointer',
          transition: 'background 0.2s',
        }}
        onClick={() => fileInputRef.current && fileInputRef.current.click()}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        {auditFile ? (
          <span style={{ color: '#16a34a', fontWeight: 600 }}>
            📄 {auditFile.name}
            <button
              style={{
                marginLeft: '0.75rem', fontSize: '0.75rem', color: '#dc2626',
                background: 'none', border: 'none', cursor: 'pointer', fontWeight: 700,
              }}
              onClick={(e) => {
                e.stopPropagation();
                setAuditFile(null);
                setAuditFilename(null);
                setResults(null);
              }}
            >✕ Remove</button>
          </span>
        ) : (
          <span style={{ color: '#6366f1' }}>
            📎 Drop your paper PDF here, or <u>click to browse</u>
          </span>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />
      </div>

      {/* Verify Button */}
      <button
        className="ref-verify-btn"
        onClick={handleVerify}
        disabled={isLoading || noSource}
      >
        {isLoading ? (
          <>
            <span className="spinner" />
            Verifying References…
          </>
        ) : (
          <>🔍 Verify References</>
        )}
      </button>

      {/* Error */}
      {error && (
        <div className="validation-message error" style={{ marginTop: '1rem' }}>
          ⚠ {error}
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="ref-results" style={{ marginTop: '1.25rem' }}>
          {/* Summary Cards */}
          <RefSummaryCards
            summary={results.summary}
            totalExtracted={results.total_extracted}
            totalAnalyzed={results.total_analyzed}
          />

          {/* Table */}
          {results.details && results.details.length > 0 ? (
            <div className="claims-table-wrapper">
              <table className="claims-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Reference</th>
                    <th>Status</th>
                    <th>Match Score</th>
                    <th>Actions</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {results.details.map((ref_item, idx) => (
                    <ReferenceRow key={idx} ref_item={ref_item} index={idx} />
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="no-claims-msg">
              No references could be extracted from the document.
            </p>
          )}
        </div>
      )}
    </section>
  );
}

export default ReferenceVerification;

