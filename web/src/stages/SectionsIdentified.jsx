export default function SectionsIdentified({ result }) {
  if (!result.regions.length) {
    return <p className="empty">Nothing to label — Box 3 did not run.</p>;
  }

  // In the order they appear in the paper, not alphabetically: the run
  // of labels is itself the reading order, and sorting it would hide
  // that the paper goes title, abstract, introduction and so on.
  const order = [];
  const counts = {};
  const titles = {};
  result.regions.forEach((region) => {
    if (!(region.section in counts)) order.push(region.section);
    counts[region.section] = (counts[region.section] ?? 0) + 1;
    if (region.section_title_raw && !(region.section in titles)) {
      titles[region.section] = region.section_title_raw;
    }
  });

  const unknown = counts.unknown ?? 0;
  const verdict = unknown
    ? `${unknown} of ${result.regions.length} regions could not be placed in a section.`
    : "Every region landed in a section.";

  return (
    <>
      <ul className="run">
        {order.map((section) => (
          <li key={section}>
            <span
              className={section === "unknown" ? "miss" : undefined}
              title={titles[section] ?? ""}
            >
              {section.replace(/_/g, " ")}
              <em>{counts[section]}</em>
            </span>
          </li>
        ))}
      </ul>
      <p className="note">
        In the order they appear, with the number of regions in each. {verdict}{" "}
        Hover a label to see the heading that opened it.
      </p>
    </>
  );
}
