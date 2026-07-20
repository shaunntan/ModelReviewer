import threading
from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from app.services import model_registry

MAX_PROMPT_TOKENS = 64
OUTPUT_TOP_K = 10

# Architecture field names differ by model family (GPT2: n_layer/n_head/n_embd,
# BERT: num_hidden_layers/num_attention_heads/hidden_size, etc.) - resolve
# generically instead of branching on model id.
_NUM_LAYERS_ATTRS = ["num_hidden_layers", "n_layer", "n_layers", "num_layers"]
_NUM_HEADS_ATTRS = ["num_attention_heads", "n_head", "n_heads"]
_HIDDEN_SIZE_ATTRS = ["hidden_size", "n_embd", "dim", "hidden_dim"]
_MAX_POSITION_ATTRS = ["max_position_embeddings", "n_positions", "n_ctx"]


class ModelLoadError(Exception):
    """Raised when a locally-downloaded model directory fails to load."""


@dataclass
class LoadedModel:
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel


_cache: dict[str, LoadedModel] = {}
_cache_lock = threading.Lock()


def _first_attr(config, names: list[str]):
    for name in names:
        val = getattr(config, name, None)
        if val is not None:
            return val
    return None


def _resolve_architecture_config(config):
    """Composite/multimodal configs (vision-language models like Gemma,
    PaliGemma, Llava, Idefics, Qwen2-VL) nest the actual language-model
    fields under `text_config` rather than exposing them on the top-level
    config - fall back to it when the top level doesn't resolve."""
    if _first_attr(config, _NUM_LAYERS_ATTRS) is not None:
        return config
    text_config = getattr(config, "text_config", None)
    if text_config is not None and _first_attr(text_config, _NUM_LAYERS_ATTRS) is not None:
        return text_config
    return config


def extract_architecture(config) -> dict:
    resolved = _resolve_architecture_config(config)
    num_layers = _first_attr(resolved, _NUM_LAYERS_ATTRS)
    num_heads = _first_attr(resolved, _NUM_HEADS_ATTRS)
    hidden_size = _first_attr(resolved, _HIDDEN_SIZE_ATTRS)
    if num_layers is None or num_heads is None or hidden_size is None:
        raise ModelLoadError(
            f"Could not resolve architecture fields on {type(config).__name__}"
        )
    max_position = _first_attr(resolved, _MAX_POSITION_ATTRS)
    return {
        # model_type/is_encoder_decoder stay top-level - "gemma4" is the
        # meaningful type to report, not its nested text sub-config's type.
        "model_type": getattr(config, "model_type", "unknown"),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "hidden_size": int(hidden_size),
        "vocab_size": int(getattr(resolved, "vocab_size", 0)),
        "max_position_embeddings": int(max_position) if max_position is not None else None,
        "is_encoder_decoder": bool(getattr(config, "is_encoder_decoder", False)),
    }


def get_model(model_id: str) -> LoadedModel:
    """Lazy-load and cache a tokenizer/model pair by model_id, so switching
    between locally-downloaded models doesn't reload from disk every request."""
    with _cache_lock:
        if model_id not in _cache:
            _cache[model_id] = _load_model(model_id)
        return _cache[model_id]


def _load_model(model_id: str) -> LoadedModel:
    path = model_registry.local_model_path(model_id)
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        # SDPA (the current default attention backend) does not return attention
        # weights even with output_attentions=True - eager is required to get them.
        model = AutoModel.from_pretrained(path, attn_implementation="eager")
        model.eval()
    except Exception as exc:
        raise ModelLoadError(f"Failed to load model '{model_id}': {exc}") from exc
    return LoadedModel(tokenizer=tokenizer, model=model)


# The app loads models via generic AutoModel (no task-specific LM head), so
# there's no real unembedding matrix to project the final hidden state
# through. Reusing the input embedding matrix as a stand-in is exact for
# weight-tied models (e.g. GPT-2, where the real LM head would be this exact
# matrix transposed) and an approximation for models with a dedicated,
# untied output head (e.g. BERT's MLM head has its own transform + bias).
def _output_predictions(model, tokenizer, last_token_hidden) -> list[dict]:
    embedding_matrix = model.get_input_embeddings().weight
    logits = last_token_hidden @ embedding_matrix.T
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, min(OUTPUT_TOP_K, probs.shape[-1]))
    return [
        {
            "token": tokenizer.convert_ids_to_tokens([token_id.item()])[0],
            "token_id": token_id.item(),
            "probability": round(prob.item(), 4),
        }
        for token_id, prob in zip(top_ids, top_probs)
    ]


def analyze(model_id: str, prompt: str) -> dict:
    loaded = get_model(model_id)
    tokenizer, model = loaded.tokenizer, loaded.model

    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"]
    truncated = input_ids.shape[1] > MAX_PROMPT_TOKENS
    if truncated:
        input_ids = input_ids[:, :MAX_PROMPT_TOKENS]
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # Both requested in one pass per CLAUDE.md's single-forward-pass
            # principle; hidden_states isn't serialized yet (deferred to the
            # future hidden-state view) but costs nothing extra to capture now.
            output_attentions=True,
            output_hidden_states=True,
        )

    token_strings = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    return {
        "model_id": model_id,
        "architecture": extract_architecture(model.config),
        "tokens": {
            "ids": input_ids[0].tolist(),
            "strings": token_strings,
            "truncated": truncated,
        },
        # layer: (1, num_heads, seq, seq) -> strip batch dim, round for payload size
        "attentions": [
            torch.round(layer[0], decimals=4).tolist() for layer in outputs.attentions
        ],
        # mean residual-stream L2 norm per layer (index 0 = embedding output,
        # index i = output of block i-1) - a standard activation-magnitude
        # summary, used to shade the architecture panel during playback.
        "activations": [
            round(hs[0].norm(dim=-1).mean().item(), 4) for hs in outputs.hidden_states
        ],
        "output": {
            "predictions": _output_predictions(model, tokenizer, outputs.hidden_states[-1][0, -1]),
            "approximate": True,
        },
    }
