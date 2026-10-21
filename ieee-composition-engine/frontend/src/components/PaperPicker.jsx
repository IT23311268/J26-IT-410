// PaperPicker.jsx - upload a PDF or Word paper, then press Compose.
//
// The engine settings are fixed for now (no Options box):
//   Gemini on when a key is in .env, no polishing, US Letter paper.
import { useRef, useState } from "react";
import { IconCheck, IconDoc, IconUpload, Spinner } from "./Icons.jsx";

const ALLOWED = [".pdf", ".docx"];

export default function PaperPicker({ upload, onUpload, uploading, running, onCompose, onError }) {
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef(null);

  function pick(file) {
    if (!file) return;
    const name = file.name.toLowerCase();
    if (!ALLOWED.some((ext) => name.endsWith(ext))) {
      onError("Please choose a PDF (.pdf) or Word (.docx) file.");
      return;
    }
    onUpload(file);
  }

  function drop(e) {
    e.preventDefault();
    setDragging(false);
    if (!running) pick(e.dataTransfer.files?.[0]);
  }

  const browse = () => !running && fileInput.current?.click();

  return (
    <section className="card">
      <div className="card-head">
        <span className="card-icon"><IconDoc /></span>
        <div>
          <h2>Research paper</h2>
          <p className="card-sub">Upload your research paper as a PDF or Word file. The engine maps it to the IEEE
            skeleton, formats references and citations, places every figure after its first mention, and scores
            the result.</p>
        </div>
      </div>

      <div
        className={`dropzone dropzone-wide ${dragging ? "is-dragging" : ""} ${upload ? "has-file" : ""} ${running ? "is-locked" : ""}`}
        onDragOver={(e) => { e.preventDefault(); if (!running) setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={drop}
        onClick={browse}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && browse()}
        role="button"
        tabIndex={0}
        aria-label="Upload a PDF or Word file"
      >
        {uploading ? (
          <><Spinner /><span className="dz-title">Uploading…</span></>
        ) : upload ? (
          <>
            <span className="dz-ok"><IconCheck width={22} height={22} /></span>
            <span className="dz-title">{upload.label}</span>
            <span className="dz-sub">Ready to compose · drop another file or click to change it</span>
          </>
        ) : (
          <>
            <IconUpload width={28} height={28} />
            <span className="dz-title">Drop your paper here, or click to browse</span>
            <span className="dz-sub">PDF or Word file (.pdf, .docx), up to 60 MB</span>
          </>
        )}
        <input ref={fileInput} type="file" accept=".pdf,.docx" hidden
          onChange={(e) => { pick(e.target.files?.[0]); e.target.value = ""; }} />
      </div>

      <button className="btn btn-primary btn-big" onClick={onCompose} disabled={!upload || uploading || running}>
        {running ? <><Spinner /> Composing…</> : "Compose IEEE manuscript"}
      </button>
    </section>
  );
}
