// RubricReview.jsx - the guided, rule-by-rule IEEE rubric review.
//
// For every rule that applies to the paper, one at a time:
//   left   the rule, what the engine did, HOW the text changed (before -> after),
//          and "Place 3 of 19" with buttons to step through every place
//   right  the finished IEEE manuscript with that rule's places highlighted,
//          scrolled to the current one (built by the engine: src/ui/preview.py)
//   then   Yes, applied / No - and on to the next rule
// At the end: the IEEE rubric-based evaluation score = Yes / rules that apply.
//
// Keyboard: Y = yes, N = no, ← → = previous / next place, [ ] = previous / next rule.
import { useEffect, useMemo, useState } from "react";
import { downloadRubric, previewUrl } from "../api.js";
import { IconArrowLeft, IconArrowRight, IconCheck, IconDownload, IconExternal, IconX, Spinner } from "./Icons.jsx";

const STATE_CLASS = { applied: "st-applied", met: "st-met", partly: "st-partly", not_met: "st-partly",
  review: "st-review", na: "st-na" };

export default function RubricReview({ result }) {
  const items = useMemo(() => result.rubric.filter((i) => i.applies), [result]);
  const storageKey = `rubric-${result.job_id}`;
  const [decisions, setDecisions] = useState(() => {
    try { return JSON.parse(sessionStorage.getItem(storageKey)) || {}; } catch { return {}; }
  });
  const [idx, setIdx] = useState(0);
  const [place, setPlace] = useState(0);
  const [loading, setLoading] = useState(true);
  const finished = idx >= items.length;
  const item = items[Math.min(idx, items.length - 1)];
  const count = item?.evidence.length || 0;

  useEffect(() => {
    try { sessionStorage.setItem(storageKey, JSON.stringify(decisions)); } catch { /* private window */ }
  }, [decisions, storageKey]);

  const goRule = (n) => { setIdx(Math.max(0, Math.min(n, items.length))); setPlace(0); setLoading(true); };
  const goPlace = (d) => { if (count > 1) { setPlace((p) => (p + d + count) % count); setLoading(true); } };
  const decide = (value) => { setDecisions((d) => ({ ...d, [item.id]: value })); goRule(idx + 1); };

  useEffect(() => {
    if (finished) return undefined;
    const onKey = (e) => {
      if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
      if (e.key === "y" || e.key === "Y") decide(true);
      else if (e.key === "n" || e.key === "N") decide(false);
      else if (e.key === "ArrowRight") goPlace(1);
      else if (e.key === "ArrowLeft") goPlace(-1);
      else if (e.key === "]") goRule(idx + 1);
      else if (e.key === "[") goRule(idx - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const yes = Object.values(decisions).filter(Boolean).length;
  const reviewed = Object.keys(decisions).length;

  if (finished) {
    return <Final result={result} items={items} decisions={decisions} onRestart={() => goRule(0)} />;
  }

  const e = item.evidence[place];
  const answered = decisions[item.id];

  return (
    <div className="review">
      <div className="review-top">
        <div className="progress small"><span style={{ width: `${(reviewed / items.length) * 100}%` }} /></div>
        <span className="muted small">Rule {idx + 1} of {items.length} · {reviewed} reviewed · {yes} confirmed</span>
      </div>

      <div className="review-grid">
        <aside className="rule-panel">
          <div className="rule-card">
            <div className="rule-top">
              <span className="rid big">{item.id}</span>
              <span className="muted small">{item.group}</span>
              {answered !== undefined && (
                <span className={`badge ${answered ? "st-applied" : "st-partly"}`}>You said: {answered ? "Yes" : "No"}</span>)}
            </div>
            <p className="rule-text">{item.rule}</p>
            {item.value && <p className="muted small">{item.value}</p>}
            <p className="engine">Engine: <span className={`badge ${STATE_CLASS[item.state]}`}>{item.state_text}</span>
              {item.reason && <span className="muted small"> {item.reason}</span>}</p>
          </div>

          {e ? (
            <div className="change">
              <p className="place">Place {place + 1} of {count}: <b>{e.where}</b></p>
              {e.before && e.after ? (
                <>
                  <div className="chg"><span className="chg-label">Before</span><span className="chg-before">{e.before}</span></div>
                  <div className="chg"><span className="chg-label">After</span><span className="chg-after">{e.after}</span></div>
                </>
              ) : (
                <div className="chg"><span className="chg-label">Result</span><span className="chg-after">{e.after || e.before}</span></div>
              )}
              {e.note && <p className="muted small">How the engine decided: {e.note}</p>}
              {count > 1 && (
                <div className="btn-row">
                  <button className="btn btn-outline" onClick={() => goPlace(-1)}><IconArrowLeft /> Previous place</button>
                  <button className="btn btn-outline" onClick={() => goPlace(1)}>Next place <IconArrowRight /></button>
                </div>
              )}
            </div>
          ) : (
            <p className="change place">{item.state === "review"
              ? "The engine cannot judge this rule. Look at the highlighted part of the manuscript and decide."
              : "Nothing needed changing: the paper already follows this rule. The related part is highlighted."}</p>
          )}

          <p className="question">Is this rule applied correctly in the IEEE manuscript?</p>
          <div className="btn-row">
            <button className="btn btn-yes" onClick={() => decide(true)}><IconCheck /> Yes, applied <kbd>Y</kbd></button>
            <button className="btn btn-no" onClick={() => decide(false)}><IconX /> No <kbd>N</kbd></button>
          </div>
          <div className="btn-row">
            <button className="btn btn-ghost" onClick={() => goRule(idx - 1)} disabled={idx === 0}><IconArrowLeft /> Previous rule</button>
            <button className="btn btn-ghost" onClick={() => goRule(idx + 1)}>Skip <IconArrowRight /></button>
          </div>

          <label className="jump">Go to rule
            <select value={idx} onChange={(ev) => goRule(Number(ev.target.value))}>
              {items.map((it, i) => (
                <option key={it.id} value={i}>
                  {decisions[it.id] === true ? "✔" : decisions[it.id] === false ? "✘" : "·"} {it.id} {it.rule.slice(0, 60)}
                </option>
              ))}
            </select>
          </label>

          <div className="finish">
            <button className="btn btn-link" onClick={() => {
              const next = { ...decisions };
              items.forEach((it) => { if (it.suggested && next[it.id] === undefined) next[it.id] = true; });
              setDecisions(next);
              goRule(items.length);
            }}>Accept the engine's suggestions for the rest and finish</button>
            <span className="muted small">Rules a person must check stay “Not reviewed”.</span>
          </div>
          <p className="muted small"><a href={item.source_url} target="_blank" rel="noreferrer">
            {item.source} <IconExternal /></a></p>
        </aside>

        <div className="preview">
          {loading && <div className="preview-loading"><Spinner /> Loading the manuscript…</div>}
          <iframe
            key={`${item.id}-${place}`}
            title={`IEEE manuscript with ${item.id} highlighted`}
            src={previewUrl(result.job_id, item.id, place)}
            onLoad={() => setLoading(false)}
          />
          <p className="legend"><span className="sw sw-cur" /> this place <span className="sw sw-all" /> the rule's other
            places <span className="sw sw-rel" /> the paragraph that mentions a moved figure or table</p>
        </div>
      </div>
    </div>
  );
}

function Final({ result, items, decisions, onRestart }) {
  const [busy, setBusy] = useState(false);
  const yes = items.filter((i) => decisions[i.id] === true).length;
  const no = items.filter((i) => decisions[i.id] === false).length;
  const notReviewed = items.length - yes - no;
  const score = items.length ? Math.round((1000 * yes) / items.length) / 10 : null;
  const na = result.rubric.length - items.length;

  return (
    <div className="final">
      <div className="final-score">
        <span className="muted">IEEE rubric-based evaluation score</span>
        <span className="final-num">{score ?? "n/a"}%</span>
        <span className="muted">{yes} of {items.length} rules confirmed as applied correctly
          {no ? ` · ${no} marked No` : ""}{notReviewed ? ` · ${notReviewed} not reviewed (not counted as confirmed)` : ""}.
          {" "}{na} rules do not apply to this paper. Engine compliance score: {result.score.after}%.</span>
      </div>
      <div className="btn-row center">
        <button className="btn btn-primary" disabled={busy} onClick={async () => {
          setBusy(true);
          try { await downloadRubric(result.job_id, decisions, `${result.input_name.replace(/\.[^.]+$/, "")}_IEEE_rubric.xlsx`); }
          finally { setBusy(false); }
        }}>{busy ? <Spinner /> : <IconDownload />} Download rubric (.xlsx)</button>
        {result.downloads.annotated && <a className="btn btn-outline" href={result.downloads.annotated} download>
          <IconDownload /> Word file with rule comments</a>}
        <button className="btn btn-ghost" onClick={onRestart}>Review again from the first rule</button>
      </div>
      <table className="table">
        <thead><tr><th>ID</th><th>Rule</th><th className="hide-sm">Engine</th><th>Your answer</th></tr></thead>
        <tbody>
          {items.map((i) => (
            <tr key={i.id}>
              <td><span className="rid">{i.id}</span></td>
              <td>{i.rule}</td>
              <td className="small hide-sm">{i.state_text}</td>
              <td>{decisions[i.id] === true ? <span className="badge st-applied">Yes</span> :
                decisions[i.id] === false ? <span className="badge st-partly">No</span> :
                  <span className="badge st-na">Not reviewed</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
