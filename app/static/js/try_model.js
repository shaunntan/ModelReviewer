const modelSelect = document.getElementById("model-select");
const promptInput = document.getElementById("prompt-input");
const analyzeForm = document.getElementById("analyze-form");
const analyzeButton = document.getElementById("analyze-button");
const analyzeError = document.getElementById("analyze-error");
const truncatedWarning = document.getElementById("truncated-warning");
const emptyState = document.getElementById("empty-state");
const architectureSection = document.getElementById("architecture-section");
const architectureSvg = document.getElementById("architecture-panel");
const playButton = document.getElementById("play-button");
const stopButton = document.getElementById("stop-button");
const detailSection = document.getElementById("detail-section");
const tokenStrip = document.getElementById("token-strip");
const detailContent = document.getElementById("detail-content");
const outputSection = document.getElementById("output-section");
const outputContent = document.getElementById("output-content");

const architecturePanel = createArchitecturePanel(architectureSvg);
let lastResult = null;

architecturePanel.onNodeClick((node) => {
  if (!lastResult) return;
  if (node.kind === "embedding") {
    renderTokenTable();
  } else if (node.kind === "attn") {
    renderAttentionHeatmap(detailContent, {
      tokens: lastResult.tokens.strings,
      headMatrices: lastResult.attentions[node.layerIndex],
      layerIndex: node.layerIndex,
      initialHead: node.headIndex,
    });
  } else if (node.kind === "post-attn-norm") {
    renderPlaceholder("Post-attention residual view coming in a future milestone.");
  } else if (node.kind === "mlp") {
    renderPlaceholder("Feed-forward activation view coming in a future milestone.");
  } else if (node.kind === "post-mlp-norm") {
    const value = lastResult.activations[node.layerIndex + 1];
    renderPlaceholder(
      `Residual stream after block ${node.layerIndex}'s feed-forward + norm — mean L2 norm: ${value.toFixed(2)}. ` +
        `(See the Play animation to compare this across layers.)`
    );
  } else if (node.kind === "final-norm") {
    renderPlaceholder("Hidden-state view coming in a future milestone.");
  } else if (node.kind === "lm-head") {
    renderOutputPredictions(outputContent, lastResult.output);
  }
});

function renderPlaceholder(text) {
  detailContent.innerHTML = "";
  const p = document.createElement("p");
  p.className = "hint";
  p.textContent = text;
  detailContent.appendChild(p);
}

playButton.addEventListener("click", () => {
  playButton.disabled = true;
  playButton.textContent = "Playing…";
  stopButton.disabled = false;
  architecturePanel.play(() => {
    playButton.disabled = false;
    playButton.textContent = "▶ Play";
    stopButton.disabled = true;
    architecturePanel.selectNode("lm-head");
    renderOutputPredictions(outputContent, lastResult.output);
  });
});

stopButton.addEventListener("click", () => {
  architecturePanel.stop();
  playButton.disabled = false;
  playButton.textContent = "▶ Play";
  stopButton.disabled = true;
});

async function loadLocalModels() {
  const res = await fetch("/models/local");
  const data = await res.json();
  const models = data.models || [];

  modelSelect.innerHTML = "";
  if (!models.length) {
    emptyState.hidden = false;
    modelSelect.disabled = true;
    promptInput.disabled = true;
    analyzeButton.disabled = true;
    return;
  }

  emptyState.hidden = true;
  modelSelect.disabled = false;
  promptInput.disabled = false;
  analyzeButton.disabled = false;

  for (const model of models) {
    const opt = document.createElement("option");
    opt.value = model.id;
    opt.textContent = model.id;
    modelSelect.appendChild(opt);
  }
  const preferred = models.find((m) => m.id === "gpt2");
  modelSelect.value = preferred ? preferred.id : models[0].id;
}

function renderTokenStrip() {
  tokenStrip.innerHTML = "";
  lastResult.tokens.strings.forEach((text, i) => {
    const chip = document.createElement("span");
    chip.className = "token-chip";
    chip.textContent = text;
    chip.title = `id ${lastResult.tokens.ids[i]}, position ${i}`;
    tokenStrip.appendChild(chip);
  });
}

function renderTokenTable() {
  detailContent.innerHTML = "";
  const table = document.createElement("table");
  table.className = "token-table";
  const thead = document.createElement("thead");
  thead.innerHTML = "<tr><th>Position</th><th>Token</th><th>Id</th></tr>";
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  lastResult.tokens.strings.forEach((text, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${i}</td><td>${escapeHtml(text)}</td><td>${lastResult.tokens.ids[i]}</td>`;
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  detailContent.appendChild(table);
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

analyzeForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  analyzeError.hidden = true;
  truncatedWarning.hidden = true;
  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyzing…";

  try {
    const res = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelSelect.value, prompt: promptInput.value }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Analyze failed (${res.status})`);
    }
    lastResult = await res.json();

    architectureSection.hidden = false;
    detailSection.hidden = false;
    outputSection.hidden = false;
    truncatedWarning.hidden = !lastResult.tokens.truncated;

    architecturePanel.setArchitecture(lastResult.architecture, lastResult.activations, lastResult.attentions);
    playButton.disabled = false;
    playButton.textContent = "▶ Play";
    stopButton.disabled = true;
    renderTokenStrip();
    detailContent.innerHTML = '<p class="hint">Click a block\'s Attn circle to inspect its attention.</p>';
    renderOutputPredictions(outputContent, lastResult.output);
  } catch (err) {
    analyzeError.textContent = err.message;
    analyzeError.hidden = false;
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Analyze";
  }
});

loadLocalModels();
