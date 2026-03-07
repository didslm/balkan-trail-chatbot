const answerEl = document.getElementById("answer");
const uploadResultEl = document.getElementById("uploadResult");
const fileListEl = document.getElementById("fileList");
const questionInput = document.getElementById("question");
const fileInput = document.getElementById("fileInput");
const healthEl = document.getElementById("health");

const setStatus = (text, ok = true) => {
  const dot = healthEl.querySelector(".dot");
  const value = healthEl.querySelector(".value");
  value.textContent = text;
  dot.style.background = ok ? "#7fd7c4" : "#f08080";
  dot.style.boxShadow = ok
    ? "0 0 12px rgba(127, 215, 196, 0.7)"
    : "0 0 12px rgba(240, 128, 128, 0.7)";
};

const prettyJson = (obj) => JSON.stringify(obj, null, 2);

const pollJob = async (jobId, targetEl) => {
  targetEl.textContent = `Job ${jobId} queued...`;
  for (;;) {
    try {
      const res = await fetch(`/jobs/${jobId}`);
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      const job = await res.json();
      if (job.status === "done") {
        targetEl.textContent = prettyJson(job);
        return job;
      }
      if (job.status === "failed") {
        targetEl.textContent = `Job failed: ${job.error || "unknown error"}`;
        return job;
      }
      targetEl.textContent = `Job ${job.status}...`;
    } catch (err) {
      targetEl.textContent = `Error: ${err.message}`;
      return null;
    }
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
};

const askQuestion = async () => {
  const q = questionInput.value.trim();
  if (!q) {
    answerEl.textContent = "Type a question to get started.";
    return;
  }
  answerEl.textContent = "Thinking...";
  try {
    const res = await fetch(`/ask?q=${encodeURIComponent(q)}`);
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    const data = await res.json();
    answerEl.textContent = data.answer || "No answer returned.";
  } catch (err) {
    answerEl.textContent = `Error: ${err.message}`;
  }
};

const uploadFiles = async () => {
  const files = Array.from(fileInput.files || []);
  if (!files.length) {
    uploadResultEl.textContent = "Choose at least one file to upload.";
    return;
  }

  uploadResultEl.textContent = `Uploading ${files.length} file(s)...`;
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file, file.name));

  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(detail || `Upload failed: ${res.status}`);
    }
    const data = await res.json();
    uploadResultEl.textContent = `Upload queued: ${data.job_id}`;
    fileInput.value = "";
    const job = await pollJob(data.job_id, uploadResultEl);
    if (job) {
      await refreshFiles();
    }
  } catch (err) {
    uploadResultEl.textContent = `Error: ${err.message}`;
  }
};

const refreshFiles = async () => {
  fileListEl.textContent = "Loading files...";
  try {
    const res = await fetch("/files");
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    const data = await res.json();
    if (!data.files || data.files.length === 0) {
      fileListEl.textContent = "No uploaded files found.";
      return;
    }
    fileListEl.textContent = data.files
      .map((file) => {
        const parts = [
          `• ${file.name} (${file.size_kb} KB)`,
          `processed: ${file.processed ? "yes" : "no"}`,
        ];
        if (file.category) parts.push(`category: ${file.category}`);
        if (Number.isFinite(file.entries)) parts.push(`entries: ${file.entries}`);
        return parts.join(" | ");
      })
      .join("\n");
  } catch (err) {
    fileListEl.textContent = `Error: ${err.message}`;
  }
};

const reindexAll = async () => {
  fileListEl.textContent = "Reindexing...";
  try {
    const res = await fetch("/reindex", { method: "POST" });
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    const data = await res.json();
    const job = await pollJob(data.job_id, fileListEl);
    if (job) {
      await refreshFiles();
    }
  } catch (err) {
    fileListEl.textContent = `Error: ${err.message}`;
  }
};

const checkHealth = async () => {
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error(`Status ${res.status}`);
    setStatus("Online", true);
  } catch (err) {
    setStatus("Offline", false);
  }
};

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    askQuestion();
  }
});

document.getElementById("askBtn").addEventListener("click", askQuestion);
document.getElementById("uploadBtn").addEventListener("click", uploadFiles);
document.getElementById("refreshFiles").addEventListener("click", refreshFiles);
document.getElementById("reindexBtn").addEventListener("click", reindexAll);

checkHealth();
refreshFiles();
