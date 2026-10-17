import { pageImage } from "../api";
import { useZoom } from "../lightbox";

const LIMIT = 8;

export default function PagesRendered({ result, paperId }) {
  const zoom = useZoom();

  if (!result.pages.length) {
    return (
      <p className="empty">
        No page images. This paper was ingested with rasterise=false, so Boxes 6
        and 8 have nothing to work on.
      </p>
    );
  }

  const shown = result.pages.slice(0, LIMIT);
  const first = result.pages[0];

  return (
    <>
      <div className="strip">
        {shown.map((page) => (
          <button
            key={page.page_index}
            type="button"
            onClick={() =>
              zoom(pageImage(paperId, page.page_index), `Page ${page.page_index + 1}`)
            }
          >
            <img
              src={pageImage(paperId, page.page_index)}
              alt={`Page ${page.page_index + 1}`}
              loading="lazy"
            />
            <em className="mono">{page.page_index + 1}</em>
          </button>
        ))}
      </div>
      {/* `PageImage.scale` is a Python property, so it is computed, not
          stored, and does not appear in the JSON. It is printed here as
          a label and nothing else — every real conversion between
          points and pixels happens server-side, where that property is
          the single definition of the factor. Do not reach for this
          value to convert a bounding box. */}
      <p className="note mono">
        {first.dpi} dpi · {first.width_px}×{first.height_px} px ·{" "}
        {(first.dpi / 72).toFixed(3)} px per point
        {result.pages.length > LIMIT && (
          <>
            {" · "}
            <a href={`/gallery/${encodeURIComponent(paperId)}`}>
              all {result.pages.length} pages
            </a>
          </>
        )}
      </p>
    </>
  );
}
