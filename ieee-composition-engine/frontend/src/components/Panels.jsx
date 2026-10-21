// Panels.jsx - the detail tabs: compliance checks, section mapping, figures,
// references, what changed, and the IEEE rulebook.
import { useMemo, useState } from "react";
import { IconExternal } from "./Icons.jsx";

const pct = (v) => (v === null || v === undefined ? "n/a" : `${Math.round(v * 100)}%`);

function Mini({ before, after }) {
  return (
    <span className="mini">
      <span className="mini-rail" />
      {before != null && <span className="mini-dot dot-before" style={{ left: `${before * 100}%` }} />}
      {after != null && <span className="mini-dot dot-after" style={{ left: `${after * 100}%` }} />}
    </span>
  );
}

export function Compliance({ result }) {
  let group = null;
  const rows = [];
  result.compliance.forEach((c) => {
    if (c.group !== group) {
      group = c.group;
      rows.push(<tr key={`g-${group}`} className="group"><th colSpan={4}>{group}</th></tr>);
    }
    const weak = c.after != null && c.after < 1;
    rows.push(
      <tr key={c.key} className={weak ? "weak" : ""}>
        <td>
          <span className="check-name">{c.name}{!c.scored && <span className="tag">not in the score</span>}</span>
          <span className="reason">{c.rules.map((r) => <span key={r} className="rid">{r}</span>)}
            {c.after != null ? c.after_reason : c.before_reason}</span>
        </td>
        <td className="hide-sm"><Mini before={c.before} after={c.after} /></td>
        <td className="num before">{pct(c.before)}</td>
        <td className="num after">{pct(c.after)}</td>
      </tr>
    );
  });
  return (
    <>
      <p className="muted">Each check runs on the paper as uploaded (<span className="c-before">before</span>) and on
        the IEEE manuscript (<span className="c-after">after</span>). Checks that cannot be measured show n/a and are
        left out of the average.</p>
      <table className="table">
        <thead><tr><th>Check</th><th className="hide-sm">0–100%</th><th className="num">Before</th><th className="num">After</th></tr></thead>
        <tbody>
          {rows}
          <tr className="total"><td>Overall</td><td className="hide-sm" />
            <td className="num before">{result.score.before ?? "n/a"}{result.score.before != null && "%"}</td>
            <td className="num after">{result.score.after}%</td></tr>
        </tbody>
      </table>
    </>
  );
}

export function Mapping({ result }) {
  return (
    <>
      <p className="muted">How the author's sections were matched onto the IEEE skeleton, and why.</p>
      <table className="table">
        <thead><tr><th>Author wrote</th><th>IEEE heading</th><th>Confidence</th><th className="hide-sm">Why</th></tr></thead>
        <tbody>
          {result.mapping.map((m, i) => (
            <tr key={i} className={m.level > 1 ? "sub" : ""}>
              <td>{m.original}</td>
              <td className="serif">{m.ieee}</td>
              <td>{m.confidence != null && (
                <span className="conf"><span className="conf-bar"><span style={{ width: `${m.confidence * 100}%` }} /></span>
                  {Math.round(m.confidence * 100)}%</span>)}</td>
              <td className="muted small hide-sm">{m.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {result.missing_sections.length > 0 &&
        <p className="warn-text">Missing IEEE sections: {result.missing_sections.join(", ")}</p>}
    </>
  );
}

export function Figures({ result }) {
  const p = result.placement || {};
  if (!result.visuals.length) return <p className="muted">This paper has no figures or tables to place.</p>;
  return (
    <>
      {p.before?.mean_distance != null && (
        <div className="metrics">
          <div className="metric"><span className="metric-num">{p.after.mean_distance}</span>
            <span className="metric-cap">paragraphs to first mention (was {p.before.mean_distance})</span></div>
          <div className="metric"><span className="metric-num">{p.after.after_first_mention}%</span>
            <span className="metric-cap">after their first mention (was {p.before.after_first_mention}%)</span></div>
          <div className="metric"><span className="metric-num">{p.after.adjacent}%</span>
            <span className="metric-cap">directly after first mention (was {p.before.adjacent}%)</span></div>
        </div>
      )}
      <div className="vis-grid">
        {result.visuals.map((v) => (
          <figure key={v.label} className="vis-card">
            {v.image ? <img src={v.image} alt={v.label} loading="lazy" /> :
              <div className="vis-missing">No image: the visual could not be found on the page</div>}
            <figcaption>
              <b>{v.label}</b>
              <span className={v.moved ? "c-after" : "muted"}>{v.method.charAt(0).toUpperCase() + v.method.slice(1)}</span>
              <span className="muted small">{v.distance}</span>
            </figcaption>
          </figure>
        ))}
      </div>
      {result.equations.length > 0 && (
        <details className="more"><summary>Equations copied as images ({result.equations.length})</summary>
          <div className="eq-grid">{result.equations.map((u) => <img key={u} src={u} alt="equation" loading="lazy" />)}</div>
        </details>
      )}
    </>
  );
}

export function References({ result }) {
  if (!result.references.length) return <p className="muted">No references were found.</p>;
  const byGemini = result.references.filter((r) => r.source === "gemini").length;
  return (
    <>
      <p className="muted">{byGemini} read by Gemini and {result.references.length - byGemini} by the rules, then
        written in IEEE style. Journal and conference names are in italics.</p>
      <ol className="refs">
        {result.references.map((r) => (
          <li key={r.n}><span className="refno">[{r.n}]</span>
            <span>{r.segments.map(([text, italic], i) => italic ? <i key={i}>{text}</i> : <span key={i}>{text}</span>)}</span></li>
        ))}
      </ol>
    </>
  );
}

export function Changes({ result }) {
  const llm = result.llm;
  return (
    <div className="changes">
      <h3>Composition</h3>
      <ul>{result.compose_report.map((l, i) => <li key={i}>{l}</li>)}</ul>
      <p className="muted">{llm.calls || llm.cache_hits ? `Gemini requests: ${llm.calls} new, ${llm.cache_hits} reused from the cache.` :
        !llm.available ? `Rules only: ${llm.reason}.` : ""}</p>
      {result.ai_log.length > 0 && (<><h3>Written or changed by AI or automatic extraction</h3>
        <ul>{result.ai_log.map((l, i) => <li key={i}>{l}</li>)}</ul></>)}
      {result.notes.length > 0 && (<><h3>Notes</h3><ul>{result.notes.map((l, i) => <li key={i}>{l}</li>)}</ul></>)}
    </div>
  );
}

const STATUS = { enforced: "Applied", checked: "Checked", author: "Author" };

export function Rulebook({ result }) {
  const { rules, summary } = result.rulebook;
  const [query, setQuery] = useState("");
  const shown = useMemo(() => rules.filter((r) =>
    `${r.id} ${r.group} ${r.rule}`.toLowerCase().includes(query.toLowerCase())), [rules, query]);
  let group = null;
  return (
    <>
      <p className="muted">Every IEEE rule the engine follows, from <code>src/rules/ieee_rules.py</code>.
        <b> {summary.enforced}</b> are applied automatically, <b>{summary.checked}</b> are checked and reported,
        and <b>{summary.author}</b> need the author's judgement.</p>
      <input className="search" placeholder="Search the rules, e.g. caption, margin, REF-06" value={query}
        onChange={(e) => setQuery(e.target.value)} />
      <table className="table">
        <thead><tr><th>ID</th><th>Rule</th><th>Status</th><th className="hide-sm">Source</th></tr></thead>
        <tbody>
          {shown.map((r) => {
            const head = r.group !== group ? (group = r.group, <tr key={`g-${r.group}`} className="group"><th colSpan={4}>{r.group}</th></tr>) : null;
            return [head,
              <tr key={r.id}>
                <td><span className="rid">{r.id}</span></td>
                <td>{r.rule}{r.value && <span className="reason">{r.value}</span>}</td>
                <td><span className={`badge st-${r.status}`}>{STATUS[r.status]}</span></td>
                <td className="small hide-sm"><a href={r.source_url} target="_blank" rel="noreferrer">{r.source} <IconExternal /></a>
                  {r.section && <span className="reason">{r.section}</span>}</td>
              </tr>];
          })}
        </tbody>
      </table>
    </>
  );
}
