// The whole client. Every call is a GET against the FastAPI service
// Members 2, 3 and 4 already write against — nothing here is a private
// endpoint built for the demo.

/** The list of papers already ingested. */
export async function listPapers() {
  const response = await fetch("/papers");
  if (!response.ok) throw new Error(`/papers returned ${response.status}`);
  const body = await response.json();
  return body.paper_ids;
}

/**
 * One paper's result, its baseline, and the size of the source PDF.
 *
 * `baseline` comes back null when the source PDF was not kept. That is
 * a state to render, not an error to throw: seven stages still have
 * something to show and only the comparison is missing.
 */
export async function fetchWalkthrough(paperId) {
  const response = await fetch(`/paper/${encodeURIComponent(paperId)}/walkthrough`);
  if (response.status === 404) {
    throw new Error(`No ingested paper with id "${paperId}".`);
  }
  if (!response.ok) {
    throw new Error(`The service returned ${response.status}.`);
  }
  return response.json();
}

// Image URLs. These are routes, not files — the page images live
// outside the bundle and are served by FastAPI.
export const pageImage = (pid, n) => `/paper/${encodeURIComponent(pid)}/page/${n}`;
export const layoutImage = (pid, n) => `${pageImage(pid, n)}/layout`;
export const artifactImage = (pid, i) =>
  `/paper/${encodeURIComponent(pid)}/artifact/${i}`;
