// Stages.jsx - the seven engine stages, ticking off live while the paper is composed.
import { IconCheck, IconSteps, IconX } from "./Icons.jsx";

export default function Stages({ stages, status, details }) {
  const done = stages.filter((s) => s.state === "done").length;
  return (
    <section className="card">
      <div className="card-head">
        <span className="card-icon"><IconSteps /></span>
        <div>
          <h2>Engine stages</h2>
          <p className="card-sub">
            {status === "running" ? `Working… ${done} of ${stages.length || 7} stages done` :
              status === "done" ? "All stages finished." : "The engine stopped. See the message above."}
          </p>
        </div>
      </div>
      <div className="progress"><span style={{ width: `${(done / (stages.length || 7)) * 100}%` }} /></div>
      <ol className="stages">
        {stages.map((s, i) => (
          <li key={s.key} className={`stage is-${s.state}`}>
            <span className="stage-mark">
              {s.state === "done" ? <IconCheck width={14} height={14} /> :
                s.state === "failed" ? <IconX width={14} height={14} /> : i + 1}
            </span>
            <span className="stage-body">
              <span className="stage-title">{s.title}</span>
              <span className="stage-note">
                {s.state === "running" ? "Working…" : s.state === "failed" ? "Stopped here" : s.note}
              </span>
            </span>
          </li>
        ))}
      </ol>
      {details && <details className="tech"><summary>Technical details</summary><pre>{details}</pre></details>}
    </section>
  );
}
