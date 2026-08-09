# llm_config.py
#
# Generation settings for the chat model, kept in one place so the app and the
# eval harness in tests/ always run with identical values.
#
# num_ctx costs VRAM. 8192 leaves headroom for a long persona plus ten turns of
# history on a 7B model. Lower it if a larger model starts spilling to CPU.

CHAT_OPTIONS = {
    "temperature":    0.6,
    "top_p":          0.9,
    "top_k":          40,
    "repeat_penalty": 1.1,
    "num_ctx":        8192,
}

CHAT_MAX_TOKENS = 150
