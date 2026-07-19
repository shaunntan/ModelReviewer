# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ModelReviewer is a web app that visualizes how an input prompt flows through a
language model — tokenization, embeddings, attention, per-layer hidden states,
and final output logits — so a user can interactively inspect what the model
is doing to their input, not just its final text output.

**Status: this repository is currently empty (pre-scaffold).** There is no
existing code, build system, or test suite yet. The sections below define the
intended architecture and conventions for the initial implementation — treat
them as the plan to build against, not as documentation of code that already
exists. Update this file once real commands/structure land so it stays
accurate.

## Intended architecture

- **Backend**: Python, FastAPI. Serves both the JSON API and the rendered
  page(s) — no separate frontend build pipeline.
- **Model layer**: Hugging Face `transformers` + PyTorch to load a model and
  tokenizer and run a single forward pass with `output_attentions=True,
  output_hidden_states=True` to capture every layer's internals in one call
  (avoid multiple forward passes for different views of the same prompt).
- **Frontend**: server-rendered template (Jinja2) + vanilla JS. Use D3.js for
  the architecture panel itself (custom SVG diagram with grouping/expand and
  click handling isn't a fit for an off-the-shelf chart library); D3.js or
  Plotly.js is fine for the detail-panel charts (attention heatmaps,
  embedding projections, logit bar charts). Keep it dependency-light; don't
  introduce a separate JS framework/build step unless the visualizations
  genuinely outgrow this.
- **Request flow**: user submits a prompt → `POST /analyze` → backend
  tokenizes, runs the forward pass, extracts per-layer attention weights,
  hidden-state summaries, and final logits/top-k next-token probabilities →
  one JSON response → frontend renders the views from that single payload.

## Frontend design

- **Visual style**: clean, light, product-like — generous whitespace,
  friendly sans-serif, a neutral (white/gray/black) base with a single
  accent color for highlights, active states, and data-viz elements.
- **Audience**: ML practitioners/researchers. Don't hide standard
  terminology (attention heads, residual stream, logit lens) behind
  explanatory copy — surface it directly, and expose real controls
  (layer/head pickers, model selector) rather than simplifying them away.
- **Layout**: single-page dashboard, not a step-by-step wizard. The prompt
  input and every view (architecture panel, attention, hidden states,
  logits) live on one page without navigating between screens.
- **Centerpiece — horizontal model panel**: a panel spanning the full width
  of the page draws the selected model's architecture left-to-right, input
  to output (tokenizer/embedding → transformer blocks → output head).
  - Repeated blocks are grouped by default (e.g. "Transformer Blocks × 12"
    as one expandable segment) rather than drawn as separate nodes;
    expanding a group reveals its individual layers.
  - A model-selector dropdown lets the user switch models (e.g. `gpt2`,
    `distilgpt2`, `bert-base`); the diagram adapts to the selected model's
    actual architecture and layer count rather than assuming a fixed shape.
    This means the backend must expose architecture metadata (layer count,
    block types) alongside the analysis results, not just the numeric
    outputs.
  - Clicking a node (a block group, or an individual layer once expanded)
    opens/updates a detail panel below the architecture panel with that
    node's data — attention heatmap, hidden-state view, etc. Click-to-select
    drives the detail panel; it renders in place rather than as a modal, to
    keep everything within the single-dashboard layout.

## Core concepts to preserve when extending

These are the visualization primitives the app is built around — keep new
features aligned with them rather than inventing parallel data shapes:

- **Tokenization view**: raw text → token strings/ids exactly as the
  tokenizer produced them. Every other view is keyed back to these token
  positions, so token indices must stay stable across views.
- **Attention view**: per-layer, per-head attention matrices (from
  `output_attentions`), rendered as selectable heatmaps.
- **Hidden state / residual stream view**: per-layer hidden states. Raw
  dimensionality is too high to plot directly, so these need a reduction
  (norm, PCA, or similar) before they reach the frontend.
- **Output / logit lens view**: final-layer logits → softmax → top-k tokens.
  Optionally apply the unembedding matrix to intermediate-layer hidden states
  ("logit lens") to show what the model would predict at each layer, not just
  the last one.

## Model loading

- Load and cache the model/tokenizer once (e.g., at app startup), not per
  request — weights are large and the forward pass is the expensive step.
- Default to a small open model (e.g., `gpt2`) for fast local iteration, but
  keep the model name configurable rather than hardcoded, since users will
  want to compare different models.

## Suggested first milestone

Get the simplest end-to-end slice working before adding more views: a single
`/analyze` endpoint plus one template that shows tokenization and attention
for a prompt against `gpt2`. Add hidden-state and logit-lens views once that
path is solid.
