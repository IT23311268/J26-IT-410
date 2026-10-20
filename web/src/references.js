/**
 * Where in a passage an artifact is named, so the walkthrough can mark
 * the words a link was made from.
 *
 * This is a port of the grammar in `ingestion/linking.py`, and that
 * file is the source of truth. Keeping a second copy is a real cost,
 * so it is worth being exact about what this copy decides and what it
 * does not: **the links themselves come from the server.** Box 7 has
 * already matched caption labels to chunk references and written
 * `artifact_ids` and `linked_chunk_ids` into the JSON. All this does
 * is find the phrase to highlight inside text that is already linked.
 *
 * So if the two ever drift, the worst case is an unhighlighted
 * passage, not a link that should not be there. A wrong link cannot
 * originate here.
 *
 * The longer spelling comes first in each alternation for the same
 * reason as in Python: an alternation takes the first branch that
 * matches, so with "tab" ahead of "table", "Table 2" would match as
 * the abbreviation and leave "le 2" behind.
 */
const REFERENCE =
  /\b(figures|figure|figs|fig|tables|table|tabs|tab|equations|equation|eqs|eq)\.?\s*([0-9]+(?:\s*(?:,|and|&)\s*[0-9]+)*)/gi;

const CANONICAL = {
  figure: "figure",
  figures: "figure",
  fig: "figure",
  figs: "figure",
  table: "table",
  tables: "table",
  tab: "table",
  tabs: "table",
  equation: "equation",
  equations: "equation",
  eq: "equation",
  eqs: "equation",
};

const labels = (kind, numbers) =>
  (numbers.match(/[0-9]+/g) ?? []).map(
    (n) => `${CANONICAL[kind.toLowerCase()]}:${Number(n)}`
  );

/** The label a caption opens with: "Figure 7. Performance…" -> "figure:7". */
export function captionLabel(caption) {
  const match = new RegExp(REFERENCE.source, "i").exec((caption ?? "").trim());
  // It has to be at the START, because that is where the author put
  // it. A caption reading "…compared against Figure 3" describes a
  // different figure, and matching anywhere would file this crop
  // under that one.
  if (!match || match.index !== 0) return null;
  return labels(match[1], match[2])[0] ?? null;
}

/**
 * Split a passage into the words around the first mention of `label`,
 * with the mention itself separated out for marking.
 *
 * Returns `{ before, hit, after }`, or the passage truncated with
 * `hit` null when it never names that artifact. A chunk that discusses
 * Figure 1 and then Figure 3 must not have Figure 1 marked under
 * Figure 3's crop, which is why this looks for THIS label rather than
 * the first reference in the text.
 */
export function markReference(text, label, { pad = [62, 74], max = 150 } = {}) {
  const flat = (text ?? "").split(/\s+/).join(" ");
  if (!label) {
    return { before: flat.slice(0, max) + (flat.length > max ? " …" : ""), hit: null, after: "" };
  }

  const scan = new RegExp(REFERENCE.source, "gi");
  let match;
  while ((match = scan.exec(flat)) !== null) {
    if (labels(match[1], match[2]).includes(label)) {
      const start = match.index;
      const end = start + match[0].length;
      const left = Math.max(0, start - pad[0]);
      const right = Math.min(flat.length, end + pad[1]);
      return {
        before: (left ? "… " : "") + flat.slice(left, start),
        hit: flat.slice(start, end),
        after: flat.slice(end, right) + (right < flat.length ? " …" : ""),
      };
    }
  }
  return { before: flat.slice(0, max) + (flat.length > max ? " …" : ""), hit: null, after: "" };
}
