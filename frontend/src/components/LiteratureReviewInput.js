import React from 'react';

function LiteratureReviewInput({ value, onChange, disabled }) {
  return (
    <section className="section">
      <h2 className="section-title">
        <span className="icon">📄</span>
        Main Research Paper Text
      </h2>
      <p className="section-description">
        Paste the full text of the main research paper here. The system will automatically detect the 
        Literature Review section to extract claims, and the References section to resolve citations.
      </p>
      <textarea
        className="literature-textarea"
        placeholder="Paste your full research paper text here...&#10;&#10;1. Introduction&#10;...&#10;2. Literature Review&#10;Recent advances in NLP (Smith et al., 2023)...&#10;...&#10;8. References&#10;Smith, J.: A Great Paper (2023)."
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        spellCheck={true}
      />
      <div className="char-count">
        {value.length} characters
        {value.length > 0 && value.length < 50 && (
          <span style={{ color: '#dc2626' }}> — minimum 50 required</span>
        )}
      </div>
    </section>
  );
}

export default LiteratureReviewInput;
