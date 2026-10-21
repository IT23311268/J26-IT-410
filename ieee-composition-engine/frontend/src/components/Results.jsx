// Results.jsx - the score, the downloads, and the tabs with the details.
import { useState } from "react";
import { IconChart, IconDownload } from "./Icons.jsx";
import RubricReview from "./RubricReview.jsx";
import { Changes, Compliance, Figures, Mapping, References, Rulebook } from "./Panels.jsx";

function ScoreTrack({ before, after }) {
  const b = before ?? null;
  const lo = Math.min(b ?? after, after);
  const hi = Math.max(b ?? after, after);
  return (
    <div className="score">
      <div className="score-row">
        <div>
          <span className="score-num before">{b === null ? "n/a" : `${b}%`}</span>
          <span className="score-cap">The paper as uploaded</span>
        </div>
        <div className="right">
          <span className="score-num after">{after}%</span>
          <span className="score-cap">The IEEE manuscript</span>
        </div>
      </div>
      <div className="track" role="img" aria-label={`IEEE compliance rose from ${b ?? "n/a"}% to ${after}%`}>
        <span className="track-rail" />
        {b !== null && <span className="track-gain" style={{ left: `${lo}%`, width: `${hi - lo}%` }} />}
        {b !== null && <span className="track-dot dot-before" style={{ left: `${b}%` }} />}
        <span className="track-dot dot-after" style={{ left: `${after}%` }} />
      </div>
      <div className="track-scale"><span>0%</span><span>IEEE compliance</span><span>100%</span></div>
    </div>
  );
}

function Download({ href, children, primary }) {
  if (!href) return null;
  return (
    <a className={`btn ${primary ? "btn-primary" : "btn-outline"}`} href={href} download>
      <IconDownload /> {children}
    </a>
  );
}

export default function Results({ result }) {
  const applicable = result.rubric.filter((i) => i.applies).length;
  const tabs = [
    { key: "rubric", label: `IEEE rubric (${applicable})` },
    { key: "compliance", label: "Compliance checks" },
    { key: "mapping", label: "Section mapping" },
    { key: "figures", label: `Figures and tables (${result.visuals.length})` },
    { key: "references", label: `References (${result.references_count})` },
    { key: "changes", label: "What changed" },
    { key: "rulebook", label: `IEEE rulebook (${result.rulebook.rules.length})` },
  ];
  const [tab, setTab] = useState("rubric");
  const p = result.placement || {};
  const exported = Object.entries(result.exported).map(([k, v]) => `${v} ${v === 1 ? k.replace(/s$/, "") : k}`);

  return (
    <>
      <section className="card">
        <div className="card-head">
          <span className="card-icon"><IconChart /></span>
          <div>
            <h2>{result.title}</h2>
            <p className="card-sub">{result.authors.join(", ") || result.input_name}</p>
          </div>
        </div>
        <ScoreTrack before={result.score.before} after={result.score.after} />
        <ul className="facts">
          {p.after?.after_first_mention != null && (
            <li><b>{p.after.after_first_mention}%</b> of figures and tables now come after the paragraph that
              first mentions them (was {p.before.after_first_mention}%).</li>)}
          {result.references_count > 0 && (
            <li><b>{result.references_count}</b> references rewritten in IEEE style, cited in the text as [n].</li>)}
          {exported.length > 0 && <li><b>{exported.join(", ")}</b> copied from the source PDF as images.</li>}
        </ul>
        <div className="downloads">
          <Download href={result.downloads.docx} primary>IEEE Word file</Download>
          <Download href={result.downloads.annotated}>Word file with rule comments</Download>
          <Download href={result.downloads.images}>Figure images (.zip)</Download>
          <Download href={result.downloads.compliance}>Compliance report (.json)</Download>
        </div>
      </section>

      <section className="card card-tabs">
        <div className="tabs" role="tablist">
          {tabs.map((t) => (
            <button key={t.key} role="tab" aria-selected={tab === t.key}
              className={`tab ${tab === t.key ? "is-active" : ""}`} onClick={() => setTab(t.key)}>
              {t.label}
            </button>
          ))}
        </div>
        <div className="tab-body">
          {tab === "rubric" && <RubricReview result={result} />}
          {tab === "compliance" && <Compliance result={result} />}
          {tab === "mapping" && <Mapping result={result} />}
          {tab === "figures" && <Figures result={result} />}
          {tab === "references" && <References result={result} />}
          {tab === "changes" && <Changes result={result} />}
          {tab === "rulebook" && <Rulebook result={result} />}
        </div>
      </section>
    </>
  );
}
