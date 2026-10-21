// App.jsx - the whole page: header, choose a paper, live progress, results.
import { useEffect, useRef, useState } from "react";
import { getJob, getResult, getSettings, startJob, uploadPaper } from "./api.js";
import Header from "./components/Header.jsx";
import PaperPicker from "./components/PaperPicker.jsx";
import Stages from "./components/Stages.jsx";
import Results from "./components/Results.jsx";
import { IconAlert, IconCheck } from "./components/Icons.jsx";

export default function App() {
  const [settings, setSettings] = useState(null);
  const [upload, setUpload] = useState(null);           // the uploaded paper: { name: "uploads/x.pdf", label }
  const [uploading, setUploading] = useState(false);
  const [job, setJob] = useState(null);                 // { id, status, stages, error, details }
  const [result, setResult] = useState(null);
  const [notice, setNotice] = useState(null);           // { kind: "ok" | "error", text }
  const resultsRef = useRef(null);

  // what the server knows: papers in data/input, whether a Gemini key is set
  useEffect(() => {
    getSettings()
      .then(setSettings)
      .catch(() => setNotice({ kind: "error", text: "Cannot reach the engine. Is python server.py running?" }));
  }, []);

  // while a paper is being composed, ask the server for progress every 0.4 s
  useEffect(() => {
    if (!job || job.status !== "running") return undefined;
    const timer = setInterval(async () => {
      try {
        const state = await getJob(job.id);
        setJob((j) => ({ ...j, ...state }));
        if (state.status === "done") {
          const data = await getResult(job.id);
          setResult(data);
          setNotice({ kind: "ok", text: `“${data.title}” is now an IEEE manuscript. IEEE compliance went from ${data.score.before ?? "n/a"}% to ${data.score.after}%.` });
          setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 150);
        } else if (state.status === "error") {
          setNotice({ kind: "error", text: state.error });
        }
      } catch (err) {
        setNotice({ kind: "error", text: err.message });
        setJob((j) => ({ ...j, status: "error" }));
      }
    }, 400);
    return () => clearInterval(timer);
  }, [job?.id, job?.status]);

  async function handleUpload(file) {
    setUploading(true);
    setNotice(null);
    try {
      const saved = await uploadPaper(file);
      setUpload({ name: saved.name, label: saved.label });
      setNotice({ kind: "ok", text: `Uploaded ${saved.label}. Press “Compose IEEE manuscript” to start.` });
    } catch (err) {
      setNotice({ kind: "error", text: err.message });
    } finally {
      setUploading(false);
    }
  }

  async function compose() {
    setResult(null);
    setNotice(null);
    try {
      // fixed settings for now: Gemini when a key is in .env, no polishing, US Letter
      const { job_id } = await startJob({
        paper: upload.name, gemini: Boolean(settings?.gemini_key), polish: false, paper_size: "letter",
      });
      setJob({ id: job_id, status: "running", stages: [] });
    } catch (err) {
      setNotice({ kind: "error", text: err.message });
    }
  }

  const running = job?.status === "running";

  return (
    <>
      <Header />
      <main className="page">
        {notice && (
          <div className={`notice notice-${notice.kind}`} role={notice.kind === "error" ? "alert" : "status"}>
            {notice.kind === "ok" ? <IconCheck /> : <IconAlert />}
            <span>{notice.text}</span>
            <button className="notice-close" onClick={() => setNotice(null)} aria-label="Dismiss">×</button>
          </div>
        )}

        <PaperPicker
          upload={upload}
          onUpload={handleUpload}
          uploading={uploading}
          running={running}
          onCompose={compose}
          onError={(text) => setNotice({ kind: "error", text })}
        />

        {job && <Stages stages={job.stages} status={job.status} details={job.details} />}

        {result && (
          <div ref={resultsRef} className="stack">
            <Results result={result} />
          </div>
        )}
      </main>
      <footer className="footer">
        <span>IEEE manuscript composition engine</span>
        <span>J26-IT-410</span>
        <span>Senarath S P J H</span>
      </footer>
    </>
  );
}
