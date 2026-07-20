const searchForm = document.getElementById("search-form");
const searchInput = document.getElementById("search-input");
const searchError = document.getElementById("search-error");
const resultsTable = document.getElementById("results-table");
const resultsBody = document.getElementById("results-body");
const localList = document.getElementById("local-models-list");

const pollTimers = new Map();

searchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  await runSearch(searchInput.value.trim());
});

async function runSearch(query) {
  if (!query) return;
  searchError.hidden = true;
  try {
    const res = await fetch(`/models/search?q=${encodeURIComponent(query)}&limit=20`);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Search failed (${res.status})`);
    }
    const data = await res.json();
    renderResults(data.results);
  } catch (err) {
    searchError.textContent = err.message;
    searchError.hidden = false;
    resultsTable.hidden = true;
  }
}

function renderResults(results) {
  resultsBody.innerHTML = "";
  if (!results.length) {
    resultsTable.hidden = true;
    searchError.textContent = "No models found.";
    searchError.hidden = false;
    return;
  }
  for (const model of results) {
    const row = document.createElement("tr");

    const nameCell = document.createElement("td");
    nameCell.textContent = model.id;
    if (model.gated) {
      const badge = document.createElement("span");
      badge.className = "badge-gated";
      badge.textContent = "gated";
      nameCell.appendChild(badge);
    }
    row.appendChild(nameCell);

    row.appendChild(textCell(model.author ?? "—"));
    row.appendChild(textCell(formatNumber(model.downloads)));
    row.appendChild(textCell(formatNumber(model.likes)));
    row.appendChild(textCell(model.pipeline_tag ?? "—"));

    const actionCell = document.createElement("td");
    actionCell.className = "download-actions";
    const button = document.createElement("button");
    actionCell.appendChild(button);

    const progress = document.createElement("progress");
    progress.className = "download-progress";
    progress.max = 100;
    progress.hidden = true;
    actionCell.appendChild(progress);

    const progressLabel = document.createElement("span");
    progressLabel.className = "download-progress-label";
    progressLabel.hidden = true;
    actionCell.appendChild(progressLabel);

    const errorSpan = document.createElement("span");
    errorSpan.className = "row-error";
    errorSpan.hidden = true;
    actionCell.appendChild(errorSpan);
    row.appendChild(actionCell);

    setButtonState(button, model.is_downloaded ? "complete" : "idle");
    button.addEventListener("click", () => triggerDownload(model.id, { button, progress, progressLabel, errorSpan }));

    resultsBody.appendChild(row);
  }
  resultsTable.hidden = false;
}

function textCell(value) {
  const td = document.createElement("td");
  td.textContent = value;
  return td;
}

function formatNumber(n) {
  if (n === null || n === undefined) return "—";
  return n.toLocaleString();
}

function formatBytes(n) {
  if (!n && n !== 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = n;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function setButtonState(button, state) {
  button.dataset.state = state;
  if (state === "idle") {
    button.textContent = "Download";
    button.disabled = false;
  } else if (state === "downloading") {
    button.textContent = "Downloading…";
    button.disabled = true;
  } else if (state === "complete") {
    button.textContent = "Downloaded";
    button.disabled = true;
  } else if (state === "error") {
    button.textContent = "Retry download";
    button.disabled = false;
  }
}

function updateProgress(progress, progressLabel, data) {
  if (typeof data.percent === "number") {
    progress.value = data.percent;
    let label = `${data.percent}%`;
    if (typeof data.total_bytes === "number") {
      label = `${formatBytes(data.downloaded_bytes)} / ${formatBytes(data.total_bytes)} (${label})`;
    }
    progressLabel.textContent = label;
  } else {
    progress.removeAttribute("value"); // indeterminate
    progressLabel.textContent =
      typeof data.downloaded_bytes === "number" && data.downloaded_bytes > 0
        ? `${formatBytes(data.downloaded_bytes)} downloaded`
        : "";
  }
}

function hideProgress(progress, progressLabel) {
  progress.hidden = true;
  progressLabel.hidden = true;
  progress.removeAttribute("value");
}

async function triggerDownload(modelId, els) {
  const { button, progress, progressLabel, errorSpan } = els;
  errorSpan.hidden = true;
  setButtonState(button, "downloading");
  progress.hidden = false;
  progressLabel.hidden = false;
  progress.removeAttribute("value"); // indeterminate until the first poll reports real numbers

  try {
    const res = await fetch("/models/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Download failed (${res.status})`);
    }
    const data = await res.json();
    if (data.status === "complete") {
      setButtonState(button, "complete");
      hideProgress(progress, progressLabel);
      loadLocalModels();
      return;
    }
    pollStatus(modelId, els);
  } catch (err) {
    setButtonState(button, "error");
    hideProgress(progress, progressLabel);
    errorSpan.textContent = err.message;
    errorSpan.hidden = false;
  }
}

function pollStatus(modelId, els) {
  const { button, progress, progressLabel, errorSpan } = els;
  if (pollTimers.has(modelId)) {
    clearInterval(pollTimers.get(modelId));
  }
  const timer = setInterval(async () => {
    try {
      const res = await fetch(`/models/download/status?model_id=${encodeURIComponent(modelId)}`);
      const data = await res.json();
      if (data.status === "complete") {
        clearInterval(timer);
        pollTimers.delete(modelId);
        setButtonState(button, "complete");
        hideProgress(progress, progressLabel);
        loadLocalModels();
      } else if (data.status === "error") {
        clearInterval(timer);
        pollTimers.delete(modelId);
        setButtonState(button, "error");
        hideProgress(progress, progressLabel);
        errorSpan.textContent = data.error || "Download failed.";
        errorSpan.hidden = false;
      } else {
        updateProgress(progress, progressLabel, data);
      }
    } catch (err) {
      // transient network hiccup while polling; keep trying
    }
  }, 1000);
  pollTimers.set(modelId, timer);
}

async function loadLocalModels() {
  try {
    const res = await fetch("/models/local");
    const data = await res.json();
    renderLocalModels(data.models);
  } catch (err) {
    localList.innerHTML = "";
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "Could not load local models.";
    localList.appendChild(li);
  }
}

function renderLocalModels(models) {
  localList.innerHTML = "";
  if (!models.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No models downloaded yet.";
    localList.appendChild(li);
    return;
  }
  for (const model of models) {
    const li = document.createElement("li");
    li.textContent = model.id;
    localList.appendChild(li);
  }
}

loadLocalModels();
