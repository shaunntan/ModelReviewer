# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ModelReviewer is a web app that visualizes how an input prompt flows through a
language model — tokenization, embeddings, attention, per-layer hidden states,
and final output logits — so a user can interactively inspect what the model
is doing to their input, not just its final text output.

**Status: implemented.** The FastAPI app, model layer, and both pages (Try
Model, Find Model) described below exist and run. The sections below still
describe the intended architecture/conventions; where the actual
implementation has settled on specifics (e.g. exact node kinds in the
architecture panel), see `app/static/js/architecture-panel.js` and
`app/services/inference.py` directly rather than treating this file as a
line-by-line spec. See "Known placeholders" below for what's intentionally
not implemented yet and why.

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
  - Repeated blocks are drawn as individual nodes under a shared "Transformer
    Blocks × 12" label, not as a single collapsed group needing an expand
    click — each block is itself broken into its own sub-component nodes
    (self-attention, add & norm, feed-forward, add & norm), visually boxed
    together so the block reads as one unit.
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

## Known placeholders in the architecture panel

Clicking some nodes shows a "coming in a future milestone" message instead
of real data. This is a deliberate scoping decision, not an oversight — each
one is blocked on something specific:

- **Add & Norm (post-attention)** and **Feed-Forward**: `output_hidden_states`
  (see Model loading below) only exposes the residual stream before/after
  each *whole* transformer block, not the intermediate state between its
  attention and feed-forward sub-layers. Getting that would mean registering
  forward hooks on each block's attention/MLP submodules directly — but
  those submodule names differ per architecture (GPT2's block exposes
  `.attn`/`.mlp`, BERT's exposes `.attention`/`.intermediate`+`.output`,
  etc.), which would mean hardcoding per-model-family submodule paths and
  breaking the architecture-agnostic approach the rest of the app uses (see
  the attribute-fallback-chain pattern in `extract_architecture`,
  `app/services/inference.py`). Nothing currently solves this generically.
- **Final Norm**: this is exactly the "Hidden state / residual stream view"
  concept above, which the Suggested first milestone (below) explicitly
  defers until after tokenization + attention are solid.

**LM Head is no longer a placeholder.** The app still loads models via
generic `AutoModel` (not `AutoModelForCausalLM`/`AutoModelForMaskedLM`), so
there's no real unembedding/LM-head weight loaded — but rather than block on
that, `_output_predictions` in `app/services/inference.py` reconstructs
logits by projecting the final hidden state through the *input* embedding
matrix (`model.get_input_embeddings().weight`). This is exact for models
that tie input/output embeddings (e.g. GPT-2) and an approximation for
models with a separate, untied output head (e.g. BERT's MLM head has its
own transform + bias). The `/analyze` response's `output.approximate` field
is always `true` today as a result — the frontend surfaces that caveat
directly (`renderOutputPredictions` in `app/static/js/output-predictions.js`)
rather than presenting it as an exact result. This same view renders when
Analyze completes, when the LM Head node is clicked, and when the Play
animation reaches the end.

The per-layer activation norm already computed for the Play animation's
shading (`activations` in the `/analyze` response, computed in
`app/services/inference.py` from `output_hidden_states`) is real data, and
answers a coarser version of the hidden-state-view question — clicking a
block's post-feed-forward Add & Norm node surfaces that number directly. The
views above go beyond that single summary value (a fuller per-dimension
view, or real logits) and are what's actually deferred.

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

**Status: done** (tokenization, attention, and the architecture panel are
implemented and working across arbitrary local models, not just `gpt2`).
The final-layer output view is also done (see "Known placeholders" above —
LM Head is no longer blocked). The hidden-state view, and extending output
to a full per-layer logit lens rather than just the final layer, are the
remaining next steps.
