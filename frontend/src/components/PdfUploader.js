import React, { useRef } from 'react';

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function PdfUploader({ files, onAddFiles, onRemoveFile, disabled }) {
  const fileInputRef = useRef(null);

  const handleClick = () => {
    if (!disabled) fileInputRef.current?.click();
  };

  const handleChange = (e) => {
    const selected = Array.from(e.target.files || []);
    const pdfs = selected.filter((f) => f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf'));
    if (pdfs.length > 0) onAddFiles(pdfs);
    // Reset so the same file can be re-selected
    e.target.value = '';
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (disabled) return;
    const dropped = Array.from(e.dataTransfer.files || []);
    const pdfs = dropped.filter((f) => f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf'));
    if (pdfs.length > 0) onAddFiles(pdfs);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  return (
    <section className="section">
      <h2 className="section-title">
        <span className="icon">📄</span>
        Reference Research Papers
      </h2>
      <p className="section-description">
        Upload the reference research papers (PDF) that the literature review cites.
        These will be used to verify claims and citations.
      </p>

      {/* Drop Zone */}
      <div
        className="upload-area"
        onClick={handleClick}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        role="button"
        tabIndex={0}
        aria-label="Upload PDF files"
      >
        <div className="upload-icon">📁</div>
        <p>
          Drag & drop PDF files here, or <span className="browse-link">browse</span>
        </p>
        <p style={{ fontSize: '0.8rem', marginTop: '0.25rem', color: '#9ca3af' }}>
          Only .pdf files are accepted
        </p>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,application/pdf"
        multiple
        className="file-input-hidden"
        onChange={handleChange}
        disabled={disabled}
      />

      {/* File List */}
      {files.length > 0 && (
        <ul className="file-list">
          {files.map((file, index) => (
            <li key={`${file.name}-${index}`} className="file-item">
              <div className="file-item-info">
                <span className="pdf-icon">📕</span>
                <span>{file.name}</span>
                <span className="file-item-size">{formatFileSize(file.size)}</span>
              </div>
              <button
                className="file-remove-btn"
                onClick={() => onRemoveFile(index)}
                disabled={disabled}
                title="Remove file"
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export default PdfUploader;
