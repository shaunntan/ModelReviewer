function renderOutputPredictions(container, output) {
  container.innerHTML = "";

  const heading = document.createElement("div");
  heading.className = "heatmap-label";
  heading.textContent = "Top predicted next token";
  container.appendChild(heading);

  if (output.approximate) {
    const note = document.createElement("p");
    note.className = "hint output-note";
    note.textContent =
      "Reconstructed from the tied input-embedding matrix (no dedicated output head is loaded) — exact for weight-tied causal models, an approximation otherwise.";
    container.appendChild(note);
  }

  const predictions = output.predictions;
  const BAR_HEIGHT = 24;
  const BAR_GAP = 8;
  const margin = { top: 4, right: 56, bottom: 4, left: 90 };
  const chartWidth = 380;
  const height = margin.top + margin.bottom + predictions.length * (BAR_HEIGHT + BAR_GAP) - BAR_GAP;
  const width = margin.left + chartWidth + margin.right;

  const chartWrap = document.createElement("div");
  chartWrap.className = "output-chart-wrap";
  container.appendChild(chartWrap);

  const svg = d3
    .select(chartWrap)
    .append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("class", "output-chart");

  const maxProb = d3.max(predictions, (p) => p.probability) || 1;
  const x = d3.scaleLinear().domain([0, maxProb]).range([0, chartWidth]);
  const color = d3.scaleSequential(d3.interpolateBlues).domain([0, maxProb]);

  const row = svg
    .selectAll("g.output-row")
    .data(predictions)
    .join("g")
    .attr("class", "output-row")
    .attr("transform", (_, i) => `translate(${margin.left}, ${margin.top + i * (BAR_HEIGHT + BAR_GAP)})`);

  row
    .append("text")
    .attr("class", "output-row-label")
    .attr("x", -8)
    .attr("y", BAR_HEIGHT / 2)
    .attr("dy", "0.32em")
    .attr("text-anchor", "end")
    .text((d) => d.token);

  row
    .append("rect")
    .attr("class", "output-row-bar")
    .attr("height", BAR_HEIGHT)
    .attr("width", (d) => x(d.probability))
    .attr("fill", (d) => color(d.probability))
    .append("title")
    .text((d) => `${d.token} — ${(d.probability * 100).toFixed(2)}%`);

  row
    .append("text")
    .attr("class", "output-row-value")
    .attr("x", (d) => x(d.probability) + 8)
    .attr("y", BAR_HEIGHT / 2)
    .attr("dy", "0.32em")
    .text((d) => `${(d.probability * 100).toFixed(1)}%`);
}
