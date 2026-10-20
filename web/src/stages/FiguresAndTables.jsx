import { artifactImage } from "../api";
import { useZoom } from "../lightbox";

const LIMIT = 6;

export default function FiguresAndTables({ result, paperId }) {
  const zoom = useZoom();

  if (!result.artifacts.length) {
    return (
      <p className="empty">
        No figures, tables or equations were found in this paper.
      </p>
    );
  }

  const kinds = {};
  result.artifacts.forEach((a) => {
    kinds[a.artifact_type] = (kinds[a.artifact_type] ?? 0) + 1;
  });
  const summary = Object.entries(kinds)
    .map(([kind, n]) => `${n} ${kind}${n === 1 ? "" : "s"}`)
    .join(", ");

  return (
    <>
      <div className="crops">
        {result.artifacts.slice(0, LIMIT).map((artifact, index) => {
          const caption = artifact.caption || "No caption printed beside it.";
          return (
            <figure key={artifact.artifact_id}>
              {artifact.image_path ? (
                <button
                  type="button"
                  onClick={() => zoom(artifactImage(paperId, index), caption)}
                >
                  <img
                    src={artifactImage(paperId, index)}
                    alt={caption.slice(0, 80)}
                    loading="lazy"
                  />
                </button>
              ) : (
                <div className="empty">Detected, not cropped.</div>
              )}
              <figcaption>
                {/* The same colour the outline was drawn in on the page
                    image two stages above. */}
                <span
                  className="kind kind-tint"
                  style={{ "--kind": `var(--kind-${artifact.artifact_type})` }}
                >
                  {artifact.artifact_type} · p{artifact.bbox.page + 1}
                </span>
                {caption.slice(0, 120)}
              </figcaption>
            </figure>
          );
        })}
      </div>
      <p className="note">
        {summary.charAt(0).toUpperCase() + summary.slice(1)}, each cut from the
        page image with the caption the author printed beside it.
        {result.artifacts.length > LIMIT &&
          ` Showing ${LIMIT} of ${result.artifacts.length}.`}
      </p>
    </>
  );
}
