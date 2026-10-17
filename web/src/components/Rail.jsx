import { useEffect, useState } from "react";

/** How far down the viewport the reading line sits. */
const BAND = 0.3;

/**
 * Track which stage the reader is actually looking at.
 *
 * This is the one thing the server-rendered page could not do, and the
 * reason the walkthrough is worth having as a client app at all: on a
 * page this long, "where am I" is a real question, and a rail that
 * only lists the stages does not answer it.
 *
 * The rule is "the last stage that has passed the reading line", not
 * "the stage most visible". An IntersectionObserver band was tried
 * first and got the bottom of the page wrong: scrolled to the end, the
 * final stage filled the screen while the rail still pointed at the
 * one above it, because that stage's foot was the only thing inside
 * the band. Reading the tops directly has no such corner.
 */
function useCurrentStage(stages) {
  const [current, setCurrent] = useState(stages[0].id);

  // `stages` is a module constant, so this effect runs once. Deriving
  // the ids inside it keeps it that way — mapping them in the caller
  // would build a fresh array every render and re-bind the listeners
  // on each one.
  useEffect(() => {
    const ids = stages.map((stage) => stage.id);
    let frame = 0;

    const measure = () => {
      frame = 0;
      const line = window.innerHeight * BAND;
      let found = ids[0];
      ids.forEach((id) => {
        const node = document.getElementById(id);
        if (node && node.getBoundingClientRect().top <= line) found = id;
      });
      // The foot of the document can be short of a full screen, so the
      // last stage may never reach the line. Landing at the bottom
      // means the reader is on it whatever the arithmetic says.
      const atBottom =
        window.innerHeight + window.scrollY >= document.body.scrollHeight - 2;
      setCurrent(atBottom ? ids[ids.length - 1] : found);
    };

    // Scroll fires far faster than the screen repaints; one measure
    // per frame is all that can be seen.
    const onScroll = () => {
      if (!frame) frame = window.requestAnimationFrame(measure);
    };

    measure();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [stages]);

  return current;
}

export default function Rail({ stages, paperId }) {
  const current = useCurrentStage(stages);

  return (
    <nav className="rail" aria-label="Pipeline stages">
      <div className="mark">
        <i aria-hidden="true">J26</i>
        Ingestion engine
      </div>
      <h2>The pipeline</h2>
      <ol>
        {stages.map((stage) => (
          <li key={stage.id}>
            <a href={`#${stage.id}`} aria-current={current === stage.id}>
              <em>{stage.number || "·"}</em>
              <span>{stage.rail}</span>
            </a>
          </li>
        ))}
      </ol>
      <div className="foot">
        <a href={`/pipeline/${encodeURIComponent(paperId)}`}>Server view</a>
        <a href="/docs">API reference</a>
      </div>
    </nav>
  );
}
