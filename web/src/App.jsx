import Headline from "./components/Headline";
import Rail from "./components/Rail";
import Stage from "./components/Stage";
import { LightboxProvider } from "./lightbox";
import { useWalkthrough } from "./useWalkthrough";

import PdfIn from "./stages/PdfIn";
import PagesRendered from "./stages/PagesRendered";
import RegionsFound from "./stages/RegionsFound";
import SectionsIdentified from "./stages/SectionsIdentified";
import ChunksCut from "./stages/ChunksCut";
import FiguresAndTables from "./stages/FiguresAndTables";
import ArtifactsLinked from "./stages/ArtifactsLinked";
import AgainstTheBaseline from "./stages/AgainstTheBaseline";

// The eight boxes of the component diagram, in order, with the one
// line each that says why the box exists. `rail` is shorter than
// `title` on purpose: the rail is read at a glance from across a room
// and must not wrap.
//
// This list is the single source of the order. The rail, the scroll
// spy and the page body all read it, so a stage cannot appear in one
// and be missing from another.
const STAGES = [
  {
    id: "s1",
    number: "1",
    rail: "PDF in",
    title: "PDF in",
    lede: "The upload, untouched. Everything below is derived from it.",
    Body: PdfIn,
  },
  {
    id: "s2",
    number: "2",
    rail: "Pages rendered",
    title: "Pages rendered",
    lede:
      "Each page drawn once to a bitmap, then reused by every stage that needs pixels rather than re-rendered per stage.",
    Body: PagesRendered,
  },
  {
    id: "s3",
    number: "3",
    rail: "Regions found",
    title: "Regions found",
    lede:
      "Headings, body text, captions, figures, tables and equations, located by geometry and font names, and put back into reading order. No model, no GPU.",
    Body: RegionsFound,
  },
  {
    id: "s4",
    number: "4",
    rail: "Sections identified",
    title: "Sections identified",
    lede:
      "Which part of the paper each region belongs to. The same sentence means different things in Related Work and in Results.",
    Body: SectionsIdentified,
  },
  {
    id: "s5",
    number: "5",
    rail: "Chunks cut",
    title: "Chunks cut",
    lede:
      "The retrievable units, cut at section and paragraph boundaries and carrying the section label with them.",
    Body: ChunksCut,
  },
  {
    id: "s6",
    number: "6",
    rail: "Figures and tables",
    title: "Figures and tables lifted out",
    lede:
      "Each figure, table and equation cropped from its page image and kept with its caption, ready to be bound to the chunks that discuss it.",
    Body: FiguresAndTables,
  },
  {
    id: "s7",
    number: "7",
    rail: "Artifacts linked",
    title: "Artifacts bound to the text",
    lede:
      "Each figure, table and equation joined to the chunks that cite it, using the label the author printed rather than any guess about distance.",
    Body: ArtifactsLinked,
  },
  {
    id: "ev",
    number: "",
    rail: "Against the baseline",
    title: "Measured against the baseline",
    lede: "What the same PDF looks like without any of the above.",
    Body: AgainstTheBaseline,
  },
];

/** ?paper=<id>, so a presenter can open the paper they rehearsed on. */
const requestedPaper = () =>
  new URLSearchParams(window.location.search).get("paper");

export default function App() {
  const { loading, error, data } = useWalkthrough(requestedPaper());

  if (loading) {
    return (
      <div className="screen">
        <div>
          <h1>Reading the paper</h1>
          <p>Re-running the baseline so the comparison is never a stale copy.</p>
        </div>
      </div>
    );
  }

  if (error) {
    // An empty screen is an invitation to act, not an apology.
    return (
      <div className="screen">
        <div>
          <h1>Nothing to walk through yet</h1>
          <p>{error}</p>
          <p>
            <a href="/">Open the front page and drop a PDF on it.</a>
          </p>
        </div>
      </div>
    );
  }

  const { result, baseline, source_bytes: sourceBytes, id } = data;
  const title = result.paper.title || result.paper.source_filename;

  return (
    <LightboxProvider>
      <div className="shell">
        <Rail stages={STAGES} paperId={id} />

        <main className="main">
          <header className="masthead">
            <div className="titleband">
              <div className="inner titleblock">
                <div>
                  <h1>{title}</h1>
                  <p className="pid mono">{id}</p>
                </div>
                <div className="particulars mono">
                  {result.paper.page_count} pages · {result.paper.extraction_method}
                  <br />
                  schema {result.schema_version}
                </div>
              </div>
            </div>
            <div className="inner">
              <Headline result={result} baseline={baseline} />
            </div>
          </header>

          <div className="inner stages">
            {STAGES.map(({ id: key, number, title: heading, lede, Body }) => (
              <Stage key={key} id={key} number={number} title={heading} lede={lede}>
                <Body
                  result={result}
                  baseline={baseline}
                  paperId={id}
                  sourceBytes={sourceBytes}
                />
              </Stage>
            ))}
          </div>
        </main>
      </div>
    </LightboxProvider>
  );
}
