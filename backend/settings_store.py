# settings_store.py
#
# User preferences, persisted to settings.json next to the other backend state.
#
# Before this existed the Settings page saved nothing at all, and the chosen
# model lived in a module global that reset to the default on every restart.
#
# Values here are user overrides. The tuned defaults stay in llm_config, and
# anything the user has not set falls through to those, so a corrupt or missing
# settings file degrades to known-good behaviour rather than breaking.

import json
import os

import llm_config

SETTINGS_FILE = "settings.json"

TEMP_MIN, TEMP_MAX = 0.0, 2.0
TOKENS_MIN, TOKENS_MAX = 64, 8192

_cache = None


def defaults() -> dict:
    return {
        "model":            "qwen2.5:7b",
        "temperature":      llm_config.CHAT_OPTIONS["temperature"],
        # None means "decide per model", which is what picks 2048 for reasoning
        # models and 400 for the rest.
        "max_tokens":       None,
        "compact_messages": False,
        "auto_scroll":      True,
    }


def _clean(raw: dict) -> dict:
    """Coerce whatever is on disk or in a request body into usable settings."""
    out = defaults()
    if not isinstance(raw, dict):
        return out

    model = raw.get("model")
    if isinstance(model, str) and model.strip():
        out["model"] = model.strip()

    try:
        temp = raw.get("temperature", out["temperature"])
        if temp is not None:
            out["temperature"] = min(TEMP_MAX, max(TEMP_MIN, float(temp)))
    except (TypeError, ValueError):
        pass

    tokens = raw.get("max_tokens", out["max_tokens"])
    if tokens in (None, "", "auto"):
        out["max_tokens"] = None
    else:
        try:
            out["max_tokens"] = min(TOKENS_MAX, max(TOKENS_MIN, int(tokens)))
        except (TypeError, ValueError):
            out["max_tokens"] = None

    for key in ("compact_messages", "auto_scroll"):
        if key in raw:
            out[key] = bool(raw[key])

    return out


def load() -> dict:
    global _cache
    if _cache is not None:
        return dict(_cache)

    raw = {}
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        raw = {}

    _cache = _clean(raw)
    return dict(_cache)


def update(patch: dict) -> dict:
    """Merge a partial update over the current settings and persist."""
    global _cache
    merged = load()
    if isinstance(patch, dict):
        merged.update({k: v for k, v in patch.items() if k in defaults()})

    _cache = _clean(merged)
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(_cache, f, indent=2)
    except OSError as e:
        # Report the failure rather than letting the UI claim it saved.
        raise RuntimeError(f"Could not write {SETTINGS_FILE}: {e}")

    return dict(_cache)


def effective_max_tokens(model: str) -> int:
    override = load().get("max_tokens")
    if override:
        return int(override)
    return llm_config.max_tokens_for(model)


def chat_options(max_tokens: int) -> dict:
    options = dict(llm_config.CHAT_OPTIONS)
    options["temperature"] = load()["temperature"]
    options["num_predict"] = max_tokens
    return options
