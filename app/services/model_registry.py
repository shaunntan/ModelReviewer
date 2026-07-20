import os
import re
import threading
from pathlib import Path

from huggingface_hub import HfApi, ModelInfo, RepoFile, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from huggingface_hub.utils import filter_repo_objects
from tqdm import tqdm as _StdTqdm

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Formats we don't need since we only ever load models via PyTorch/transformers.
# Skipping these avoids downloading duplicate weights for repos that ship
# multiple formats (TensorFlow, Flax, ONNX, TorchScript, GGUF-adjacent pb).
_IGNORE_PATTERNS = ["*.h5", "*.msgpack", "*.onnx", "*.tflite", "*.ot", "*.pb"]

# Hub model ids are either "name" or "namespace/name".
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*(?:/[A-Za-z0-9][A-Za-z0-9_.\-]*)?$")

_download_lock = threading.Lock()
_download_status: dict[str, dict] = {}  # model_id -> {"status": ..., "error": ...}


def _validate_model_id(model_id: str) -> None:
    """Reject anything that isn't a plain Hub id. model_id is joined directly
    onto MODELS_DIR to build a filesystem path, so this also blocks traversal."""
    if not model_id or ".." in model_id or not _MODEL_ID_RE.match(model_id):
        raise ValueError(f"Invalid model id: {model_id!r}")


def search_hub_models(query: str, limit: int = 20) -> list[dict]:
    """Free-text search across the entire Hub. filter='transformers' is a soft
    bias toward library-tagged repos, not an allowlist - `search` still matches
    against the full Hub."""
    api = HfApi(token=HF_TOKEN)
    results = api.list_models(
        search=query,
        filter="transformers",
        sort="downloads",
        limit=limit,
        full=True,
    )
    return [_serialize(m) for m in results]


def _serialize(m: ModelInfo) -> dict:
    return {
        "id": m.id,
        "author": m.author,
        "downloads": m.downloads,
        "likes": m.likes,
        "pipeline_tag": m.pipeline_tag,
        "library_name": m.library_name,
        "gated": bool(m.gated),
        "last_modified": m.last_modified.isoformat() if m.last_modified else None,
        "is_downloaded": is_downloaded(m.id),
    }


def local_model_path(model_id: str) -> Path:
    _validate_model_id(model_id)
    # A slash in a namespaced id naturally nests: models/<namespace>/<name>/
    return MODELS_DIR / model_id


def is_downloaded(model_id: str) -> bool:
    try:
        return (local_model_path(model_id) / "config.json").is_file()
    except ValueError:
        return False


def list_local_models() -> list[dict]:
    """Bounded 2-level scan of MODELS_DIR matching the layout this module
    writes: flat `<name>/` or namespaced `<namespace>/<name>/`."""
    if not MODELS_DIR.exists():
        return []
    found = []
    for entry in sorted(MODELS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "config.json").is_file():
            found.append({"id": entry.name, "path": str(entry)})
            continue
        for sub in sorted(entry.iterdir()):
            if sub.is_dir() and (sub / "config.json").is_file():
                found.append({"id": f"{entry.name}/{sub.name}", "path": str(sub)})
    return found


def get_download_status(model_id: str) -> dict:
    with _download_lock:
        state = _download_status.get(model_id)
    if state:
        return state
    return {"status": "complete"} if is_downloaded(model_id) else {"status": "not_found"}


def start_download(model_id: str) -> dict:
    _validate_model_id(model_id)
    with _download_lock:
        current = _download_status.get(model_id)
        if current and current["status"] == "downloading":
            return current
    if is_downloaded(model_id):
        return {"status": "complete"}
    with _download_lock:
        _download_status[model_id] = {
            "status": "downloading",
            "error": None,
            "downloaded_bytes": 0,
            "total_bytes": None,
            "percent": None,
        }
    threading.Thread(target=_run_download, args=(model_id,), daemon=True).start()
    return {"status": "downloading"}


class _SnapshotProgressTqdm(_StdTqdm):
    """tqdm_class for snapshot_download. huggingface_hub instantiates this 3x
    per call (an outer file-count bar, plus "transfer" and "reconstruct" byte
    bars) - per-file progress across however many files/threads are
    downloading is already pre-aggregated by hub into those two shared byte
    bars, so we just intercept updates to one of them rather than summing
    per-file bars ourselves. We track only the "reconstruct" bar: its total
    is real file-size bytes in both the Xet and non-Xet transfer paths,
    whereas the "transfer" bar's total is undefined under Xet (dedup/
    compression). Subclasses plain tqdm.tqdm (not huggingface_hub's wrapper,
    which can silently set disable=True based on logger state).

    `_model_id` is bound per-download by `_make_progress_tqdm_class`.
    """

    _model_id: str | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._track = kwargs.get("unit") == "B" and str(kwargs.get("desc", "")).startswith("Reconstruct")

    def display(self, msg=None, pos=None):
        return  # this runs server-side in a background thread, not a terminal

    def update(self, n=1):
        if self._track and self._model_id is not None and n:
            # tqdm's own counter isn't lock-protected, and up to max_workers
            # download threads call update() concurrently on this shared
            # instance - accumulate under our own lock instead of trusting
            # self.n later.
            with _download_lock:
                state = _download_status.get(self._model_id)
                if state is not None and state.get("status") == "downloading":
                    downloaded = state.get("downloaded_bytes", 0) + int(n)
                    total = state.get("total_bytes")
                    state["downloaded_bytes"] = downloaded
                    state["percent"] = round(100 * downloaded / total, 1) if total else None
        return super().update(n)


def _make_progress_tqdm_class(model_id: str) -> type[_StdTqdm]:
    class _Bound(_SnapshotProgressTqdm):
        _model_id = model_id

    return _Bound


def _estimate_total_bytes(model_id: str) -> int | None:
    """Sum sizes of the files snapshot_download will actually fetch, so
    total_bytes is accurate from the start rather than growing as
    snapshot_download discovers file sizes mid-download."""
    try:
        api = HfApi(token=HF_TOKEN)
        entries = api.list_repo_tree(repo_id=model_id, recursive=True, repo_type="model")
        files = (f for f in entries if isinstance(f, RepoFile))
        matched = filter_repo_objects(files, ignore_patterns=_IGNORE_PATTERNS, key=lambda f: f.path)
        return sum(f.lfs.size if f.lfs is not None else f.size for f in matched)
    except Exception:
        return None  # non-fatal - frontend falls back to an indeterminate bar


def _run_download(model_id: str) -> None:
    target = local_model_path(model_id)
    target.mkdir(parents=True, exist_ok=True)
    total_bytes = _estimate_total_bytes(model_id)
    with _download_lock:
        _download_status[model_id] = {
            "status": "downloading",
            "error": None,
            "downloaded_bytes": 0,
            "total_bytes": total_bytes,
            "percent": 0.0 if total_bytes else None,
        }
    try:
        snapshot_download(
            repo_id=model_id,
            repo_type="model",
            local_dir=str(target),
            token=HF_TOKEN,
            ignore_patterns=_IGNORE_PATTERNS,
            tqdm_class=_make_progress_tqdm_class(model_id),
        )
        with _download_lock:
            _download_status[model_id] = {"status": "complete", "error": None}
    except GatedRepoError:
        with _download_lock:
            _download_status[model_id] = {
                "status": "error",
                "error": (
                    "This model is gated. Accept its license on huggingface.co "
                    "and set the HF_TOKEN environment variable, then retry."
                ),
            }
    except HfHubHTTPError as exc:
        with _download_lock:
            _download_status[model_id] = {"status": "error", "error": str(exc)}
    except Exception as exc:
        with _download_lock:
            _download_status[model_id] = {"status": "error", "error": str(exc)}
