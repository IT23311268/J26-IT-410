/**
 * The result, above the walkthrough that explains it.
 *
 * A panel reads the top of a page and then listens to the presenter,
 * so whatever is up here is what they take away. The stages below
 * carry the argument; this carries the answer.
 */
export default function Headline({ result, baseline }) {
  const placed = result.chunks.filter((c) => c.section !== "unknown").length;
  const sections = new Set(
    result.regions.filter((r) => r.section !== "unknown").map((r) => r.section)
  );

  const cells = [
    { value: result.regions.length, label: "regions found" },
    { value: sections.size, label: "sections identified" },
    {
      value: `${placed}/${result.chunks.length}`,
      label: "chunks carry their section",
      tone: "good",
    },
    { value: result.artifacts.length, label: "figures and tables lifted out" },
  ];

  if (baseline && baseline.chunks.length && result.chunks.length) {
    const broken = (run) =>
      (100 * run.chunks.filter((c) => /^[a-z]/.test(c.text.trimStart())).length) /
      run.chunks.length;
    cells.push({
      value: `${Math.round(broken(result))}%`,
      label: "start mid-word",
      tone: "good",
    });
    cells.push({
      value: `${Math.round(broken(baseline))}%`,
      label: "without this pipeline",
      tone: "plain",
    });
  }

  return (
    <div className="headline">
      {cells.map((cell) => (
        <div key={cell.label} className={cell.tone ?? ""}>
          <b className="mono">{cell.value}</b>
          <span>{cell.label}</span>
        </div>
      ))}
    </div>
  );
}
