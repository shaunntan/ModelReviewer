const CIRCLE_RADIUS = 18;
const NODE_GAP = 16; // gap between blocks (and around embedding/final-norm/lm-head)
const INNER_GAP = 8; // tighter gap between a single block's own 4 sub-components
const MARGIN = 16;
const LABEL_AREA_WIDTH = 150; // room for a node's text label beside its circle
const NODE_LABEL_OFFSET = 14; // gap from a node's right edge to its label text
const GROUP_LABEL_WIDTH = 16; // reserved column for the rotated "Transformer Blocks" label
const GROUP_LABEL_GAP = 10; // gap between that label and the block boxes
const SEGMENT_MS = 220;
const BLOCK_BOX_PADDING = 8;
const HEAD_CIRCLE_RADIUS = 5; // per-head dot inside a block's Attn node
const HEAD_CIRCLE_GAP = 3;

// A transformer block is drawn as 4 sub-component circles rather than one -
// self-attention, its add & norm, the feed-forward network, and its add &
// norm - so users can select the sub-component they care about directly
// instead of only ever landing on "the whole block."
const BLOCK_SUBCOMPONENT_KINDS = new Set(["attn", "post-attn-norm", "mlp", "post-mlp-norm"]);

function isBlockSubcomponent(n) {
  return BLOCK_SUBCOMPONENT_KINDS.has(n.kind);
}

// The flow runs top-down, so a block's Attn node draws its per-head dots
// (colored by that head's attention-focus strength) as a horizontal row
// instead of vertical stack - using the axis that's *not* the flow direction
// keeps neighboring nodes above/below from needing extra clearance. Other
// node kinds still get a single dot the size of CIRCLE_RADIUS.
function clusterHalfWidth(numHeads) {
  if (numHeads <= 0) return 0;
  return (numHeads * (HEAD_CIRCLE_RADIUS * 2) + Math.max(0, numHeads - 1) * HEAD_CIRCLE_GAP) / 2;
}

function headDotSpec(numHeads) {
  const half = clusterHalfWidth(numHeads);
  const startX = -half + HEAD_CIRCLE_RADIUS;
  const step = HEAD_CIRCLE_RADIUS * 2 + HEAD_CIRCLE_GAP;
  return d3.range(numHeads).map((i) => ({ r: HEAD_CIRCLE_RADIUS, cx: startX + i * step, headIndex: i }));
}

function createArchitecturePanel(svgNode) {
  const svg = d3.select(svgNode);
  svg.selectAll("*").remove();
  const boxLayer = svg.append("g").attr("class", "block-boxes"); // bottommost: boxes sit behind connectors/nodes
  const connectors = svg.append("g").attr("class", "connectors");
  const headerLayer = svg.append("g").attr("class", "header-layer");
  const nodesLayer = svg.append("g").attr("class", "nodes");

  let architecture = null;
  let attentions = null;
  let selectedId = null;
  let nodeClickHandler = () => {};
  let lastPositions = [];
  let lastNodeActivations = [];
  let animating = false;

  function computeNodes(arch) {
    const blockNodes = [];
    for (let i = 0; i < arch.num_layers; i++) {
      blockNodes.push(
        { id: `layer-${i}-attn`, kind: "attn", label: "Attn", layerIndex: i, tooltip: `Block ${i} — Self-Attention` },
        {
          id: `layer-${i}-post-attn-norm`,
          kind: "post-attn-norm",
          label: "Norm",
          layerIndex: i,
          tooltip: `Block ${i} — Add & Norm (post-attention)`,
        },
        { id: `layer-${i}-mlp`, kind: "mlp", label: "FFN", layerIndex: i, tooltip: `Block ${i} — Feed-Forward` },
        {
          id: `layer-${i}-post-mlp-norm`,
          kind: "post-mlp-norm",
          label: "Norm",
          layerIndex: i,
          tooltip: `Block ${i} — Add & Norm (post-feed-forward)`,
        }
      );
    }
    return [
      { id: "embedding", kind: "embedding", label: "Tokenizer / Embedding", tooltip: "Tokenizer / Embedding" },
      ...blockNodes,
      { id: "final-norm", kind: "final-norm", label: "Final Norm", tooltip: "Final Norm" },
      { id: "lm-head", kind: "lm-head", label: "LM Head", tooltip: "LM Head" },
    ];
  }

  // Every node gets the same vertical slot (CIRCLE_RADIUS*2) since labels now
  // sit beside each node rather than below it, so slot height doesn't depend
  // on label length the way the old horizontal layout's slot width did. A
  // block's own 4 sub-components sit close together (INNER_GAP); the wider
  // NODE_GAP only appears between different blocks (and around embedding/
  // final-norm/lm-head), so blocks read as grouped purely through spacing.
  // Manual layout (not scaleBand) so this mixed tight/wide spacing is
  // possible at all. All nodes share the same cx (centerX) - cross-axis
  // extras (head dots, labels) extend sideways from that shared centerline.
  function layoutNodes(nodes, centerX) {
    let y = MARGIN;
    const positions = [];
    nodes.forEach((n, i) => {
      if (i > 0) {
        const prev = nodes[i - 1];
        const sameBlock = isBlockSubcomponent(n) && isBlockSubcomponent(prev) && n.layerIndex === prev.layerIndex;
        y += sameBlock ? INNER_GAP : NODE_GAP;
      }
      positions.push({ ...n, y, cx: centerX, cy: y + CIRCLE_RADIUS });
      y += CIRCLE_RADIUS * 2;
    });
    return { positions, totalHeight: y + MARGIN };
  }

  function topEdge(p) {
    return p.cy - CIRCLE_RADIUS;
  }

  function bottomEdge(p) {
    return p.cy + CIRCLE_RADIUS;
  }

  // One box per block, bounding just that block's 4 sub-component circles -
  // the explicit visual grouping the boxes provide is in addition to (not a
  // replacement for) the tight INNER_GAP spacing already used to cluster them.
  function computeBlockBoxes(positions) {
    const byLayer = new Map();
    positions.forEach((p) => {
      if (!isBlockSubcomponent(p)) return;
      if (!byLayer.has(p.layerIndex)) byLayer.set(p.layerIndex, []);
      byLayer.get(p.layerIndex).push(p);
    });
    return Array.from(byLayer.entries()).map(([layerIndex, nodes]) => {
      const top = Math.min(...nodes.map(topEdge)) - BLOCK_BOX_PADDING;
      const bottom = Math.max(...nodes.map(bottomEdge)) + BLOCK_BOX_PADDING;
      return { id: `block-box-${layerIndex}`, y: top, height: bottom - top };
    });
  }

  // rawActivations: [embedding_output, block_0_output, ..., block_(N-1)_output]
  // (length num_layers + 1) - one value per whole block, not per sub-
  // component, so all 4 of a block's circles shade using that block's value;
  // final-norm/lm-head reuse the last entry - it's the exact residual-stream
  // value the output head receives as input.
  function alignActivations(positions, rawActivations) {
    if (!rawActivations || !rawActivations.length) return positions.map(() => 0);
    const lastIdx = rawActivations.length - 1;
    return positions.map((p) => {
      if (p.kind === "embedding") return rawActivations[0];
      if (isBlockSubcomponent(p)) return rawActivations[Math.min(p.layerIndex + 1, lastIdx)];
      return rawActivations[lastIdx];
    });
  }

  function render() {
    if (!architecture) return;
    const nodes = computeNodes(architecture);

    const numHeads = architecture.num_heads || 1;
    const halfWidth = Math.max(CIRCLE_RADIUS, clusterHalfWidth(numHeads));
    const centerX = MARGIN + GROUP_LABEL_WIDTH + GROUP_LABEL_GAP + halfWidth;
    const boxLeft = centerX - halfWidth - BLOCK_BOX_PADDING;
    const boxWidth = halfWidth * 2 + BLOCK_BOX_PADDING * 2;
    const svgWidth = centerX + halfWidth + BLOCK_BOX_PADDING + NODE_LABEL_OFFSET + LABEL_AREA_WIDTH + MARGIN;

    const { positions, totalHeight } = layoutNodes(nodes, centerX);
    lastPositions = positions;

    svg.attr("width", svgWidth).attr("height", totalHeight).attr("viewBox", `0 0 ${svgWidth} ${totalHeight}`);

    connectors
      .selectAll("line")
      .data(d3.pairs(positions), (d) => d[0].id + "->" + d[1].id)
      .join("line")
      .attr("class", "connector")
      .attr("x1", centerX)
      .attr("x2", centerX)
      .attr("y1", (d) => bottomEdge(d[0]))
      .attr("y2", (d) => topEdge(d[1]));

    const blockBoxes = computeBlockBoxes(positions);
    boxLayer
      .selectAll("rect.block-box")
      .data(blockBoxes, (d) => d.id)
      .join("rect")
      .attr("class", "block-box")
      .attr("x", boxLeft)
      .attr("y", (d) => d.y)
      .attr("width", boxWidth)
      .attr("height", (d) => d.height)
      .attr("rx", 8);

    const label = headerLayer.selectAll("text.group-label").data(blockBoxes.length ? [blockBoxes] : []);
    label.exit().remove();
    const labelEnter = label.enter().append("text").attr("class", "group-label");
    labelEnter
      .merge(label)
      .attr("x", MARGIN + GROUP_LABEL_WIDTH / 2)
      .attr("y", (d) => (d[0].y + d[d.length - 1].y + d[d.length - 1].height) / 2)
      .attr("transform", function () {
        const x = d3.select(this).attr("x");
        const y = d3.select(this).attr("y");
        return `rotate(-90, ${x}, ${y})`;
      })
      .text(`Transformer Blocks × ${architecture.num_layers}`);

    const node = nodesLayer
      .selectAll("g.node")
      .data(positions, (d) => d.id)
      .join(
        (enter) => {
          const g = enter.append("g");
          g.append("text").attr("class", "node-label");
          return g;
        },
        (update) => update,
        (exit) => exit.remove()
      );

    node
      .attr("class", (d) => `node node-${d.kind}${d.id === selectedId ? " selected" : ""}`)
      .attr("transform", (d) => `translate(${d.cx},${d.cy})`)
      .on("click", (event, d) => {
        selectedId = d.id;
        render();
        applyHeadColors();
        nodeClickHandler(d);
      });

    // Each node is drawn as one or more "dots": a single CIRCLE_RADIUS circle
    // for most kinds, or one small circle per attention head for "attn"
    // nodes. Dot fill/tooltip come from attentions data via applyHeadColors,
    // kept separate so a plain selection-change re-render (see the click
    // handler above) doesn't clobber Play's in-progress shading.
    node.each(function (d) {
      const g = d3.select(this);
      const dots = d.kind === "attn" ? headDotSpec(numHeads) : [{ r: CIRCLE_RADIUS, cx: 0, headIndex: null }];
      const dotJoin = g.selectAll("circle.node-dot").data(dots, (dd) => dd.headIndex);
      dotJoin.exit().remove();
      const dotEnter = dotJoin.enter().append("circle").attr("class", "node-dot");
      dotEnter.append("title");
      const dotMerged = dotEnter.merge(dotJoin);
      dotMerged
        .attr("r", (dd) => dd.r)
        .attr("cx", (dd) => dd.cx)
        .on("click", (event, dd) => {
          if (dd.headIndex === null) return;
          event.stopPropagation();
          selectedId = d.id;
          render();
          applyHeadColors();
          nodeClickHandler({ ...d, headIndex: dd.headIndex });
        });
      if (d.kind !== "attn") {
        dotMerged.select("title").text(d.tooltip || d.label);
      }
    });

    node
      .select("text.node-label")
      .attr("x", (d) => (d.kind === "attn" ? halfWidth : CIRCLE_RADIUS) + NODE_LABEL_OFFSET)
      .attr("y", 0)
      .text((d) => d.label);
  }

  // Colors/labels each Attn node's head dots from the latest attentions data.
  // Split out from render() so a selection-only re-render can skip it and
  // leave any in-progress Play shading on those same <circle> elements alone.
  function applyHeadColors() {
    const headStrengthsByLayer = (attentions || []).map((headMatrices) => headMatrices.map(headFocusStrength));
    const allStrengths = headStrengthsByLayer.flat();
    const headColor = d3.scaleSequential(d3.interpolateBlues).domain([0, d3.max(allStrengths) || 1]);

    nodesLayer.selectAll("g.node-attn").each(function (d) {
      const strengths = headStrengthsByLayer[d.layerIndex] || [];
      d3.select(this)
        .selectAll("circle.node-dot")
        .style("fill", (dd) => headColor(strengths[dd.headIndex]))
        .select("title")
        .text(
          (dd) =>
            `Block ${d.layerIndex} — Head ${dd.headIndex} — avg. peak attention ${(strengths[dd.headIndex] ?? 0).toFixed(2)}`
        );
    });
  }

  let pendingTimer = null;

  function clearPendingTimer() {
    if (pendingTimer) {
      clearTimeout(pendingTimer);
      pendingTimer = null;
    }
  }

  // Used before replaying and when the underlying data changes
  // (setArchitecture) - both need a clean slate, unlike a user-initiated
  // Stop click (haltAnimation below), which should freeze in place instead.
  function stopAnimation() {
    clearPendingTimer();
    animating = false;
    // Plain style (not a d3 .transition()) so the CSS `transition: fill`
    // rule drives the fade - sequencing itself uses setTimeout rather than
    // transition .on("end") callbacks, since those depend on
    // requestAnimationFrame, which browsers freeze in backgrounded tabs
    // (an actual user switching away mid-Play would otherwise stall it).
    nodesLayer.selectAll("g.node circle").style("fill", null);
  }

  // Halts playback where it currently is, leaving already-shaded circles
  // shaded, rather than resetting - a genuine "stop and let me look" action
  // distinct from the reset-then-replay behavior `stopAnimation` provides.
  function haltAnimation() {
    clearPendingTimer();
    animating = false;
  }

  function play(onDone) {
    if (animating || lastPositions.length < 2) {
      if (onDone) onDone();
      return;
    }
    stopAnimation();
    animating = true;

    const maxActivation = d3.max(lastNodeActivations) || 1;
    const colorScale = d3.scaleSequential(d3.interpolateBlues).domain([0, maxActivation]);

    function step(i) {
      if (i >= lastPositions.length) {
        animating = false;
        pendingTimer = null;
        if (onDone) onDone();
        return;
      }
      const pos = lastPositions[i];
      const value = lastNodeActivations[i];
      nodesLayer
        .selectAll("g.node")
        .filter((d) => d.id === pos.id)
        .selectAll("circle.node-dot")
        .style("fill", colorScale(value));
      pendingTimer = setTimeout(() => step(i + 1), SEGMENT_MS);
    }
    step(0);
  }

  return {
    setArchitecture(arch, activations, attns) {
      architecture = arch;
      attentions = attns || null;
      selectedId = null;
      stopAnimation();
      render();
      applyHeadColors();
      lastNodeActivations = alignActivations(lastPositions, activations);
    },
    onNodeClick(cb) {
      nodeClickHandler = cb;
    },
    selectNode(id) {
      selectedId = id;
      render();
      applyHeadColors();
    },
    play,
    stop: haltAnimation,
  };
}
