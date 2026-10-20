const BASELINE_CHUNK_CHARS = 800;

/**
 * One labelled bar.
 *
 * Lower is better on both measures here, so the bar is literally "how
 * much of this goes wrong": a full bar is a total failure and an empty
 * one is a clean run. The percentage is printed on every bar and the
 * row is labelled, so the chart still reads with the colour removed.
 */
function Bar({ label, n, total, series }) {
  const share = total ? (100 * n) / total : 0;
  return (
    <div className="bar">
      <span className="who">{label}</span>
      <span className="track">
        <i className={series} style={{ width: `${share.toFixed(1)}%` }} />
      </span>
      <span className="val mono" title={`${n} of ${total} chunks`}>
        {total ? `${Math.round(share)}%` : "—"}
      </span>
    </div>
  );
}

/**
 * One chunk from each run, with its opening marked.
 *
 * The bars say what share goes wrong. This says what the share IS: a
 * chunk that opens "lumn paragraphs, even though…" is the tail of a
 * word the chunk before it kept the head of, and no amount of
 * retrieval tuning recovers the sentence it came from.
 *
 * The broken one is the argument, so it is on the left where reading
 * starts.
 */
function Facing({ result, baseline }) {
  const broken = baseline.chunks.find((c) => /^[a-z]/.test(c.text.trimStart()));
  const clean =
    result.chunks.find((c) => c.text.includes("\n") && c.section !== "unknown") ??
    result.chunks[0];
  if (!broken || !clean) return null;

  const trimmed = broken.text.trimStart();
  const space = trimmed.indexOf(" ");
  const tail = space === -1 ? trimmed : trimmed.slice(0, space);
  const rest = space === -1 ? "" : trimmed.slice(space + 1);

  const newline = clean.text.indexOf("\n");
  const head = newline === -1 ? clean.text : clean.text.slice(0, newline);
  const body = newline === -1 ? "" : clean.text.slice(newline + 1);

  return (
    <div className="facing">
      <div className="side">
        <h4>Baseline</h4>
        <p className="text">
          <mark className="cut">{tail}</mark>{" "}
          {rest.split(/\s+/).join(" ").slice(0, 190)} …
        </p>
        <p className="verdict bad">
          Opens inside a word. The sentence it belongs to is in the chunk before
          this one.
        </p>
      </div>
      <div className="side">
        <h4>This pipeline</h4>
        <p className="text">
          <mark className="keep">{head.slice(0, 70)}</mark>{" "}
          {body.split(/\s+/).join(" ").slice(0, 170)} …
        </p>
        <p className="verdict good">
          Opens at the heading that introduces it, and carries{" "}
          <strong>{clean.section.replace(/_/g, " ")}</strong> with it.
        </p>
      </div>
    </div>
  );
}

export default function AgainstTheBaseline({ result, baseline, paperId }) {
  if (!baseline) {
    return (
      <p className="empty">
        The source PDF was not kept for this paper, so the baseline cannot be
        re-run. One more trip through POST /ingest fixes it.
      </p>
    );
  }

  const unknown = (run) => run.chunks.filter((c) => c.section === "unknown").length;
  const midWord = (run) =>
    run.chunks.filter((c) => /^[a-z]/.test(c.text.trimStart())).length;

  const measures = [
    {
      title: "Chunks that do not know which section they came from",
      why: "Member 2 cannot tell a cited finding from this paper's own without it.",
      count: unknown,
    },
    {
      title: "Chunks that begin in the middle of a word",
      why: "The tail of a sentence the chunk before it kept the head of.",
      count: midWord,
    },
  ];

  return (
    <>
      <div className="measures">
        {measures.map((measure) => (
          <div className="measure" key={measure.title}>
            <h4>{measure.title}</h4>
            <p className="why">{measure.why}</p>
            <Bar
              label="Baseline"
              n={measure.count(baseline)}
              total={baseline.chunks.length}
              series="baseline"
            />
            <Bar
              label="This pipeline"
              n={measure.count(result)}
              total={result.chunks.length}
              series="ours"
            />
          </div>
        ))}
      </div>

      <Facing result={result} baseline={baseline} />

      <p className="note">
        The same PDF down both paths. The baseline cuts flat text every{" "}
        {BASELINE_CHUNK_CHARS} characters and this pipeline cuts at section and
        paragraph boundaries, so the two runs produce different numbers of
        chunks — the bars are shares of each run, not counts. Hover a bar for
        the counts.{" "}
        <a href={`/compare/${encodeURIComponent(paperId)}`}>
          Open the two side by side
        </a>
        .
      </p>
    </>
  );
}
