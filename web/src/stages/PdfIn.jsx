const humanSize = (bytes) => {
  if (bytes == null) return "size unknown";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
};

/** Box 1 is the upload and nothing else. One quiet line says it. */
export default function PdfIn({ result, sourceBytes }) {
  const { paper } = result;
  return (
    <div className="filerow">
      <b className="mono">{paper.source_filename}</b>
      <span>
        {paper.page_count} page{paper.page_count === 1 ? "" : "s"}
      </span>
      <span>{humanSize(sourceBytes)}</span>
      <span>kept, so the baseline can be re-run from it</span>
    </div>
  );
}
