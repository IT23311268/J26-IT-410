import Readout from "../components/Readout";

const PREVIEW = 3;
const CHARS = 300;
const TARGET_CHARS = 1200;

export default function ChunksCut({ result, paperId }) {
  if (!result.chunks.length) return <p className="empty">No chunks.</p>;

  const placed = result.chunks.filter((c) => c.section !== "unknown").length;
  const tokens = result.chunks.reduce((sum, c) => sum + (c.token_count ?? 0), 0);

  return (
    <>
      <div className="scores">
        <Readout value={result.chunks.length} label="chunks" />
        <Readout
          value={`${placed}/${result.chunks.length}`}
          label="carry a section"
          win
        />
        <Readout value={TARGET_CHARS} label="target chars" />
        <Readout value={tokens} label="tokens, estimated" />
      </div>

      <div className="chunks">
        {result.chunks.slice(0, PREVIEW).map((chunk) => {
          // A chunk that opens at a heading has that heading as its
          // first line. Setting it bold is this stage's claim made
          // visible: the cut landed on a boundary, not inside a
          // sentence.
          const newline = chunk.text.indexOf("\n");
          const head = newline === -1 ? chunk.text : chunk.text.slice(0, newline);
          const rest = newline === -1 ? "" : chunk.text.slice(newline + 1);
          const body = rest.split(/\s+/).join(" ").slice(0, CHARS);
          const only = head.split(/\s+/).join(" ").slice(0, CHARS);

          return (
            <div className="chunk" key={chunk.chunk_id}>
              <div className="top">
                <span
                  className={chunk.section === "unknown" ? "sect none" : "sect"}
                >
                  {chunk.section.replace(/_/g, " ")}
                </span>
                <span className="mono">chunk {chunk.order_index}</span>
                <span className="mono">{chunk.text.length} chars</span>
                <span className="mono">p{chunk.page_start + 1}</span>
              </div>
              <p>
                {rest ? (
                  <>
                    <span className="head">{head}</span>{" "}
                    {body}
                    {rest.length > CHARS ? " …" : ""}
                  </>
                ) : (
                  <>
                    {only}
                    {head.length > CHARS ? " …" : ""}
                  </>
                )}
              </p>
            </div>
          );
        })}
      </div>

      <p className="note">
        First {Math.min(PREVIEW, result.chunks.length)} of {result.chunks.length}.
        Each one starts at a heading or a paragraph, never inside a sentence.{" "}
        <a href={`/compare/${encodeURIComponent(paperId)}`}>
          See them against the baseline&rsquo;s cuts
        </a>
        .
      </p>
    </>
  );
}
