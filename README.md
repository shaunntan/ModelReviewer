# ModelReviewer

A web app for visualizing how an input prompt flows through a language model —
tokenization, embeddings, attention, per-layer hidden states, and output
logits.

Currently implemented:
- **Try Model** (`/`) — run a prompt through a locally-downloaded model and
  inspect its architecture, tokenization, and per-layer attention.
- **Find Model** (`/find-model`) — search the Hugging Face Hub and download
  models for local use.

See [CLAUDE.md](CLAUDE.md) for the full intended architecture (hidden-state
and logit-lens views are not built yet).

## Requirements

- Python 3.10+

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the app

```bash
uvicorn app.main:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

`--reload` restarts the server on code changes; drop it for a plain run.
Use `--port <port>` to run on something other than 8000.

### Gated models

Some models on the Hub require accepting a license before they can be
downloaded. To use those, accept the license on huggingface.co, then set an
access token before starting the server:

```bash
export HF_TOKEN=hf_...
uvicorn app.main:app --reload
```

## Project structure

```
app/
├── main.py                       # FastAPI app entrypoint, page routes
├── routers/
│   ├── models.py                 # /models/* endpoints (search, download, local)
│   └── analyze.py                # /analyze endpoint
├── services/
│   ├── model_registry.py         # Hugging Face Hub search + download logic
│   └── inference.py              # model loading/caching, forward pass, architecture extraction
├── templates/
│   ├── base.html                 # shared nav/layout
│   ├── try_model.html            # main page
│   └── find_model.html           # search/download page
└── static/                       # CSS/JS/vendored D3 for the UI
models/                           # downloaded models land here, one folder per model
```

Downloaded models are stored under `models/<model_id>/` (e.g. `models/gpt2/`
or `models/meta-llama/Llama-2-7b/` for namespaced ids) and are gitignored.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/models/search?q=<query>&limit=<n>` | Free-text search across the Hub |
| POST | `/models/download` | Body `{"model_id": "gpt2"}` — starts a background download |
| GET | `/models/download/status?model_id=<id>` | Poll download progress (`downloading`/`complete`/`error`/`not_found`) |
| GET | `/models/local` | List models already downloaded to `models/` |
| POST | `/analyze` | Body `{"model_id": "gpt2", "prompt": "..."}` — tokenizes, runs a forward pass, returns architecture metadata, tokens, per-layer attention weights, and per-layer activation norms (prompts are truncated to 64 tokens) |
