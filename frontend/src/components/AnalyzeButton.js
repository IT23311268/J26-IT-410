import React from 'react';

function AnalyzeButton({ onClick, isLoading, disabled }) {
  return (
    <section className="section">
      <button
        className="analyze-btn"
        onClick={onClick}
        disabled={disabled}
      >
        {isLoading ? (
          <>
            <span className="spinner" />
            Analyzing...
          </>
        ) : (
          <>
            🔍 Analyze Literature Review
          </>
        )}
      </button>
    </section>
  );
}

export default AnalyzeButton;
