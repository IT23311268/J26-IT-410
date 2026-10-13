import React from 'react';

function ResultsPlaceholder({ result }) {
  return (
    <section className="section">
      <h2 className="section-title">
        <span className="icon">📊</span>
        Analysis Results
      </h2>

      {!result ? (
        <div className="results-placeholder">
          <div className="placeholder-icon">🔬</div>
          <h3>No Analysis Yet</h3>
          <p>
            Submit your literature review and reference papers above, then click
            "Analyze Literature Review" to see results.
          </p>
        </div>
      ) : (
        <div className="results-content">
          <div className="result-item">
            <h4>Analysis Status</h4>
            <span className={`status-badge ${result.status}`}>
              ● {result.status}
            </span>
          </div>
          <div className="result-item">
            <h4>Analysis ID</h4>
            <p style={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>{result.analysis_id}</p>
          </div>
          <div className="result-item">
            <h4>Papers Analyzed</h4>
            <p>{result.papers_analyzed} paper(s) submitted for verification</p>
          </div>
          <div className="result-item">
            <h4>Message</h4>
            <p>{result.message}</p>
          </div>
        </div>
      )}
    </section>
  );
}

export default ResultsPlaceholder;
