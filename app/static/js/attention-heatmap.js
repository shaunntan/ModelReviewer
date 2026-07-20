// A head's own attention matrix is row-stochastic (every row sums to 1), so
// the matrix-wide mean is 1/n for every head regardless of shape - useless
// as a way to tell heads apart. Average row-max instead: how concentrated
// each token's attention is on a single other token, averaged over tokens.
// Ranges from ~1/n (uniform/diffuse head) to 1 (fully focused head).
function headFocusStrength(matrix) {
  const rowMaxes = matrix.map((row) => Math.max(...row));
  return rowMaxes.reduce((a, b) => a + b, 0) / rowMaxes.length;
}

function renderAttentionHeatmap(container, { tokens, headMatrices, layerIndex, initialHead }) {
  container.innerHTML = "";

  const heading = document.createElement("div");
  heading.className = "heatmap-label";
  heading.textContent = `Layer ${layerIndex} — attention heads (click one to inspect)`;
  container.appendChild(heading);

  const layout = document.createElement("div");
  layout.className = "attention-layout";
  container.appendChild(layout);

  const pickerWrap = document.createElement("div");
  pickerWrap.className = "head-picker-wrap";
  layout.appendChild(pickerWrap);

  const svgWrap = document.createElement("div");
  svgWrap.className = "heatmap-svg-wrap";
  layout.appendChild(svgWrap);

  const headStrengths = headMatrices.map(headFocusStrength);
  const headColor = d3.scaleSequential(d3.interpolateBlues).domain([0, d3.max(headStrengths) || 1]);

  const HEAD_RADIUS = 16;
  const HEAD_GAP = 14;
  // Room to the right of each circle for its index label.
  const pickerWidth = HEAD_RADIUS * 2 + 24;
  const pickerHeight = headMatrices.length * (HEAD_RADIUS * 2 + HEAD_GAP) - HEAD_GAP;

  const pickerSvg = d3
    .select(pickerWrap)
    .append("svg")
    .attr("class", "head-picker")
    .attr("width", pickerWidth)
    .attr("height", pickerHeight)
    .attr("viewBox", `0 0 ${pickerWidth} ${pickerHeight}`);

  let selectedHead = Number.isInteger(initialHead) ? initialHead : 0;

  function markSelected() {
    pickerSvg.selectAll("g.head-node").attr("class", (i) => `head-node${i === selectedHead ? " selected" : ""}`);
  }

  const headNode = pickerSvg
    .selectAll("g.head-node")
    .data(d3.range(headMatrices.length))
    .join("g")
    .attr("class", "head-node")
    .attr("transform", (i) => `translate(${HEAD_RADIUS}, ${i * (HEAD_RADIUS * 2 + HEAD_GAP) + HEAD_RADIUS})`)
    .on("click", (event, i) => {
      selectedHead = i;
      markSelected();
      draw(selectedHead);
    });

  headNode
    .append("circle")
    .attr("r", HEAD_RADIUS)
    .attr("fill", (i) => headColor(headStrengths[i]))
    .append("title")
    .text((i) => `Head ${i} — avg. peak attention ${headStrengths[i].toFixed(2)}`);

  headNode
    .append("text")
    .attr("class", "head-label")
    .attr("x", HEAD_RADIUS + 8)
    .attr("dy", "0.32em")
    .attr("text-anchor", "start")
    .text((i) => i);

  markSelected();

  function draw(headIndex) {
    svgWrap.innerHTML = "";
    const matrix = headMatrices[headIndex];
    const n = tokens.length;
    const cell = Math.max(18, Math.min(36, Math.floor(360 / n)));
    const margin = { top: 90, left: 90, right: 20, bottom: 10 };
    const size = cell * n;

    const svgWidth = margin.left + size + margin.right;
    const svgHeight = margin.top + size + margin.bottom;
    const svg = d3
      .select(svgWrap)
      .append("svg")
      // Explicit width/height (not just viewBox) so the SVG renders at its
      // natural size instead of stretching to fill the container width.
      .attr("width", svgWidth)
      .attr("height", svgHeight)
      .attr("viewBox", `0 0 ${svgWidth} ${svgHeight}`)
      .attr("class", "heatmap-svg");

    const color = d3.scaleSequential(d3.interpolateBlues).domain([0, 1]);
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const rows = g
      .selectAll("g.row")
      .data(matrix)
      .join("g")
      .attr("class", "row")
      .attr("transform", (_, i) => `translate(0, ${i * cell})`);

    rows
      .selectAll("rect")
      .data((d) => d)
      .join("rect")
      .attr("x", (_, j) => j * cell)
      .attr("width", cell - 1)
      .attr("height", cell - 1)
      .attr("fill", (v) => color(v))
      .append("title")
      .text((v) => v.toFixed(4));

    g.selectAll("text.row-label")
      .data(tokens)
      .join("text")
      .attr("class", "row-label")
      .attr("x", -6)
      .attr("y", (_, i) => i * cell + cell / 2)
      .attr("dy", "0.32em")
      .attr("text-anchor", "end")
      .text((d) => d);

    g.selectAll("text.col-label")
      .data(tokens)
      .join("text")
      .attr("class", "col-label")
      .attr("text-anchor", "start")
      .attr("transform", (_, i) => `translate(${i * cell + cell / 2}, -6) rotate(-45)`)
      .text((d) => d);
  }

  draw(selectedHead);
}
