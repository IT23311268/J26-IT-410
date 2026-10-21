// PaperPicker.jsx - choose a paper (or upload one), set the options, press Compose.
import { useRef, useState } from "react";
import { IconDoc, IconGear, IconUpload, Spinner } from "./Icons.jsx";

function Toggle({ label, help, checked, disabled, onChange }) {
  return (
    <label className={`toggle ${disabled ? "is-disabled" : ""}`} title={help}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span className="toggle-track"><span className="toggle-thumb" /></span>
      <span className="toggle-label">{label}</span>
    </label>
  );
}

export default function PaperPicker({ settings, uploads, paper, onPaper, onUpload, options, onOptions,
  running, onCompose }) {
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef(null);
  const papers = [...uploads.map((u) => ({ name: u.name, description: "Your uploaded file", label: u.label })),
    ...(settings?.papers || [])];

  function drop(e) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) onUpload(file);
  }

  return (
    <section className="card">
      <div className="card-head">
        <span className="card-icon"><IconDoc /></span>
        <div>
          <h2>Research paper</h2>
          <p className="card-sub">Choose a paper or upload your own PDF, Word file or verified content package.
            The engine maps it to the IEEE skeleton, formats references and citations, places every figure
            after its first mention, and scores the result.</p>
        </div>
      </div>

      <div className="picker">
        <div className="paper-list" role="radiogroup" aria-label="Paper">
          {papers.map((p) => (
            <button
              key={p.name}
              role="radio"
              aria-checked={paper === p.name}
              className={`paper-tile ${paper === p.name ? "is-selected" : ""}`}
              onClick={() => onPaper(p.name)}
              disabled={running}
            >
              <span className="paper-radio" />
              <span className="paper-text">
                <span className="paper-name">{p.label || p.name}</span>
                <span className="paper-desc">{p.description}</span>
              </span>
            </button>
          ))}
          {!settings && <p className="muted">Loading papers…</p>}
        </div>

        <div
          className={`dropzone ${dragging ? "is-dragging" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={drop}
          onClick={() => fileInput.current?.click()}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && fileInput.current?.click()}
          role="button"
          tabIndex={0}
        >
          <IconUpload width={26} height={26} />
          <span className="dz-title">Drop a paper here, or click to browse</span>
          <span className="dz-sub">PDF, Word (.docx) or content package (.json), up to 60 MB</span>
          <input ref={fileInput} type="file" accept=".pdf,.docx,.json" hidden
            onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])} />
        </div>
      </div>

      <div className="options">
        <span className="options-title"><IconGear /> Options</span>
        <Toggle label="Use Gemini" checked={options.gemini} disabled={!settings?.gemini_key || running}
          help="Gemini reads the references and drafts the abstract and index terms. Every answer is checked against the paper."
          onChange={(v) => onOptions({ ...options, gemini: v, polish: v && options.polish })} />
        <Toggle label="Polish the wording" checked={options.polish} disabled={!options.gemini || running}
          help="Gemini rewrites each section; a rewrite is kept only if its numbers and citations are unchanged."
          onChange={(v) => onOptions({ ...options, polish: v })} />
        <div className="segmented" role="radiogroup" aria-label="Paper size">
          {(settings?.paper_sizes || [{ key: "letter", label: "US Letter" }, { key: "a4", label: "A4" }]).map((s) => (
            <button key={s.key} role="radio" aria-checked={options.paper_size === s.key} disabled={running}
              className={options.paper_size === s.key ? "is-on" : ""}
              onClick={() => onOptions({ ...options, paper_size: s.key })}>{s.label}</button>
          ))}
        </div>
        <span className={`key-state ${settings?.gemini_key ? "ok" : "warn"}`}>
          {settings ? (settings.gemini_key ? "Gemini key found in .env" : "No Gemini key in .env: rules only") : ""}
        </span>
      </div>

      <button className="btn btn-primary btn-big" onClick={onCompose} disabled={!paper || running}>
        {running ? <><Spinner /> Composing…</> : "Compose IEEE manuscript"}
      </button>
    </section>
  );
}
