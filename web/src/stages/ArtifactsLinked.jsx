import Readout from "../components/Readout";
import { captionLabel, markReference } from "../references";

const PREVIEW = 4;

export default function ArtifactsLinked({ result }) {
  if (!result.artifacts.length) {
    return (
      <p className="empty">
        Nothing to link: this paper produced no figures, tables or equations.
      </p>
    );
  }

  const linked = result.artifacts.filter((a) => a.linked_chunk_ids.length);
  const citing = result.chunks.filter((c) => c.artifact_ids.length).length;
  const mentions = result.artifacts.reduce(
    (sum, a) => sum + a.linked_chunk_ids.length,
    0
  );
  const byId = new Map(result.chunks.map((c) => [c.chunk_id, c]));
  const unlinked = result.artifacts.length - linked.length;

  return (
    <>
      <div className="scores">
        <Readout
          value={`${linked.length}/${result.artifacts.length}`}
          label="artifacts linked"
          win
        />
        <Readout value={citing} label="chunks cite one" />
        <Readout value={mentions} label="links in total" />
        <Readout value="0" label="models used" />
      </div>

      <ul className="links">
        {linked.slice(0, PREVIEW).map((artifact) => {
          const chunk = byId.get(artifact.linked_chunk_ids[0]);
          if (!chunk) return null;
          // A link a panel has to take on trust is worth less than one
          // they can read, and the whole chunk would bury the three
          // words that matter.
          const { before, hit, after } = markReference(
            chunk.text,
            captionLabel(artifact.caption)
          );
          return (
            <li className="link" key={artifact.artifact_id}>
              <div className="from">
                <span
                  className="kind kind-tint"
                  style={{ "--kind": `var(--kind-${artifact.artifact_type})` }}
                >
                  {artifact.artifact_type} · p{artifact.bbox.page + 1}
                </span>
                <b>{artifact.caption.slice(0, 90)}</b>
              </div>
              <div className="to">
                <span className="kind">
                  chunk {chunk.order_index} · {chunk.section.replace(/_/g, " ")}
                </span>
                <p>
                  {before}
                  {hit && <mark className="ref">{hit}</mark>}
                  {after}
                </p>
              </div>
            </li>
          );
        })}
      </ul>

      <p className="note">
        {unlinked
          ? `${unlinked} of ${result.artifacts.length} are left unlinked: the text never names them, and binding one to whichever chunk sits nearest would hand Member 2 something that looks like evidence and is not.`
          : "Every artifact in this paper is named somewhere in the text."}
      </p>
    </>
  );
}
