import { layoutImage } from "../api";
import { useZoom } from "../lightbox";

// The order region kinds are tallied in, so the list reads the same
// way on every paper regardless of which kind happened to be found
// first.
const KINDS = ["heading", "body", "caption", "figure", "table", "equation"];

export default function RegionsFound({ result, paperId }) {
  const zoom = useZoom();

  if (!result.regions.length) {
    return <p className="empty">No regions. Re-ingest with detect_layout=true.</p>;
  }

  const counts = {};
  const perPage = {};
  result.regions.forEach((region) => {
    counts[region.region_type] = (counts[region.region_type] ?? 0) + 1;
    perPage[region.bbox.page] = (perPage[region.bbox.page] ?? 0) + 1;
  });

  const rendered = new Set(result.pages.map((p) => p.page_index));
  const candidates = Object.keys(perPage)
    .map(Number)
    .filter((p) => rendered.has(p));
  const busiest = candidates.length
    ? candidates.reduce((a, b) => (perPage[b] > perPage[a] ? b : a))
    : null;

  return (
    <>
      {/* The claim goes above the picture. This is the one thing a panel
          can check with their own eyes while the presenter is still
          talking, and a caption below an image is read after it — if
          at all. */}
      <p className="claim">
        The numbered badges are <strong>reading order</strong>. On a two-column
        page they run down the left column and only then cross to the right —
        which is the whole difference between this and reading the page top to
        bottom. The colours are the region kinds, and the tally beside the page
        is their legend.
      </p>
      <div className="overlay">
        <div>
          {busiest === null ? (
            <p className="empty">
              Regions were detected, but there is no page image to draw them on.
            </p>
          ) : (
            <>
              <button
                type="button"
                className="sheet"
                onClick={() =>
                  zoom(
                    layoutImage(paperId, busiest),
                    `Page ${busiest + 1} with its detected regions outlined and numbered in reading order`
                  )
                }
              >
                <img
                  src={layoutImage(paperId, busiest)}
                  alt={`Page ${busiest + 1} with its detected regions outlined and numbered in reading order`}
                  loading="lazy"
                />
              </button>
              <p className="note">
                Page {busiest + 1}, the densest page in the paper. Click it for
                full size.
              </p>
            </>
          )}
        </div>
        <ul className="tally">
          {KINDS.filter((kind) => counts[kind]).map((kind) => (
            <li key={kind} style={{ "--kind": `var(--kind-${kind})` }}>
              <i aria-hidden="true" />
              <span>{kind}</span>
              <b>{counts[kind]}</b>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}
