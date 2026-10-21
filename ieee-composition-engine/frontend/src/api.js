// api.js - every call the page makes to server.py (which runs the Python engine).

async function json(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

export const getSettings = () => fetch("/api/settings").then(json);

export const uploadPaper = (file) =>
  fetch(`/api/upload?filename=${encodeURIComponent(file.name)}`, { method: "POST", body: file })
    .then(json);

export const startJob = (options) =>
  fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(options),
  }).then(json);

export const getJob = (id) => fetch(`/api/jobs/${id}`).then(json);

export const getResult = (id) => fetch(`/api/jobs/${id}/result`).then(json);

export const previewUrl = (id, rule, place) =>
  `/api/jobs/${id}/preview?rule=${encodeURIComponent(rule || "")}&place=${place || 0}`;

// The filled rubric (with the reviewer's Yes / No answers) as an Excel file.
export async function downloadRubric(id, decisions, filename) {
  const response = await fetch(`/api/jobs/${id}/rubric.xlsx`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decisions }),
  });
  if (!response.ok) throw new Error("Could not create the rubric file.");
  const url = URL.createObjectURL(await response.blob());
  const link = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
