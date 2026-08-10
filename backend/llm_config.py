# llm_config.py
#
# Generation settings for the chat model, kept in one place so the app and the
# eval harness in tests/ always run with identical values.
#
# num_ctx costs VRAM. 8192 leaves headroom for a long persona plus ten turns of
# history on a 7B model. Lower it if a larger model starts spilling to CPU.

import requests

OLLAMA_HOST = "http://localhost:11434"

CHAT_OPTIONS = {
    "temperature":    0.6,
    "top_p":          0.9,
    "top_k":          40,
    "repeat_penalty": 1.1,
    "num_ctx":        8192,
}

# num_predict is a ceiling, not a target: a model still stops at its own end
# token, so a higher value lengthens nothing by itself, it only stops a reply
# being severed mid-sentence.
CHAT_MAX_TOKENS = 400

# Reasoning models spend this budget on their reasoning before any reply is
# produced, and Ollama counts both against num_predict. Measured on qwen3:14b:
# 367 tokens to answer "say hello", 1007 for an ordinary question. At the normal
# ceiling such a model returns empty content.
CHAT_MAX_TOKENS_THINKING = 2048

# ── Model residency ─────────────────────────────────────────────────────────
# Unloading the chat model after every answer costs about 2s per message
# (measured: 2.20s cold load vs 0.14s warm). That is only worth paying when the
# chat model and Chatterbox cannot share the card. Measured on a 12GB 3060:
# qwen2.5:7b 5.0GB, qwen3:14b 10.0GB, Chatterbox plus its classifier 4.4GB.
# With the 7B resident, TTS measured 6.8s against 7.6s with it unloaded, so
# there is nothing to gain by evicting it. With the 14B resident only 324MB
# remained free and TTS became slow and erratic, so there eviction is correct.

KEEP_ALIVE_RESIDENT = "5m"
KEEP_ALIVE_UNLOAD   = 0

TTS_VRAM_RESERVE_GB = 4.4
VRAM_HEADROOM_GB    = 0.8

_CAPABILITY_CACHE = {}
_SIZE_CACHE = {}


def model_capabilities(model: str) -> list:
    """Ollama's declared capabilities for a model, cached per process."""
    if model in _CAPABILITY_CACHE:
        return _CAPABILITY_CACHE[model]

    caps = []
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/show", json={"model": model}, timeout=10
        )
        response.raise_for_status()
        caps = response.json().get("capabilities") or []
    except Exception:
        caps = []

    _CAPABILITY_CACHE[model] = caps
    return caps


def max_tokens_for(model: str) -> int:
    if "thinking" in model_capabilities(model):
        return CHAT_MAX_TOKENS_THINKING
    return CHAT_MAX_TOKENS


def _total_vram_gb():
    """Total VRAM, or None when there is no usable CUDA device."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.mem_get_info()[1] / 1e9
    except Exception:
        return None


def model_size_gb(model: str):
    """On-disk size Ollama reports, a close proxy for the loaded footprint."""
    if model in _SIZE_CACHE:
        return _SIZE_CACHE[model]

    size = None
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        response.raise_for_status()
        for entry in response.json().get("models", []):
            if entry.get("name") == model:
                size = entry.get("size", 0) / 1e9
                break
    except Exception:
        size = None

    _SIZE_CACHE[model] = size
    return size


def keep_alive_for(model: str, tts_active: bool):
    """
    How long Ollama should hold this model after answering.

    Nothing needs the VRAM when the active persona will not speak, so the model
    stays loaded. When it will speak, the model only stays if it and Chatterbox
    both fit. If the card cannot be inspected at all the model is unloaded, which
    is the old behaviour: a machine we cannot measure is assumed to be tight.
    """
    if not tts_active:
        return KEEP_ALIVE_RESIDENT

    total = _total_vram_gb()
    size = model_size_gb(model)
    if total is None or size is None:
        return KEEP_ALIVE_UNLOAD

    if size + TTS_VRAM_RESERVE_GB + VRAM_HEADROOM_GB <= total:
        return KEEP_ALIVE_RESIDENT
    return KEEP_ALIVE_UNLOAD
