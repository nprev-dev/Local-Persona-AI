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

_CAPABILITY_CACHE = {}


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
