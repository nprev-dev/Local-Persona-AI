# Local-Persona-AI

> Turn any character into a fully local AI companion — with their voice and personality.

Local-Persona-AI is a desktop application that lets you run a completely offline, private AI assistant modeled after any character or persona you choose. You bring the voice reference clip and personality config, the app handles the rest. No cloud, no subscriptions, no data leaving your machine.

---

## What It Does

- **Custom personality** — define any character's speech style, traits, rules, and backstory via a simple JSON file or directly in the app. No coding required.
- **In-app character editor** — edit your character's personality, traits, rules, voice style, and response style right in the UI. Changes save to `personality.json` and reload instantly, no restart needed.
- **In-app model picker** — automatically detects the Ollama models installed on your machine and lets you switch between them from the UI.
- **Voice cloning** — provide a short reference audio clip and the AI speaks in that voice using Chatterbox TTS with 7-emotion detection.
- **Per-character voice toggle** — turn TTS on or off per character.
- **Streaming chat** — token-by-token response streaming, ChatGPT-style, fully local via Ollama.
- **Hybrid memory** — long-term memory system combining BM25 keyword search and FAISS vector search. The AI remembers facts about you across sessions.
- **Document ingestion** — paste text or upload files to give the AI knowledge of anything you want.
- **Sentence-by-sentence audio streaming** — TTS starts playing before the full response is generated.
- **Fully offline UI** — React, Babel, and the icon set are all vendored locally. The interface makes zero outside calls and works with no internet.
- **Desktop app** — wrapped in Tauri v2, launches and shuts down cleanly with no terminal windows.

---

## How It Works

```
User input
    │
    ▼
FastAPI backend (Python)
    ├── Retrieves relevant long-term memories (BM25 + FAISS hybrid)
    ├── Builds prompt (personality + memories + chat history)
    ├── Streams tokens from Ollama (llama3.1:8b)
    └── Streams TTS audio sentence by sentence (Chatterbox)
            │
            ▼
    Tauri WebView (frontend)
        ├── Renders tokens in real time
        ├── Plays audio chunks via AudioContext as they arrive
        └── Merges PCM buffers into a single player bar at the end
```

The Tauri shell starts Ollama and the Python backend on launch, polls until the backend is ready, then navigates the WebView to the local UI. On close, both processes are cleanly killed.

---

## Current State (v0.3.0)

This is an early working release. Everything below works but some things are still hardcoded or manual:

- **Model selection in-app** — the app detects your installed Ollama models and lets you pick between them from the UI. The default chat model is `llama3.1:8b` and the memory judge is `phi3`.
- **Model installation is manual** — you need to pull Ollama models yourself for the ones not included in `install.py`.
- **Personality is fully editable in-app** — define and tweak your character from the UI or by editing `personality.json` directly. Either way changes apply without a restart.
- **Voice model is manual** — bring your own reference `.wav` clip and set the path in `tts_engine.py`.
- **GUI installer included** — run install.py before first launch to install dependencies and pull Ollama models. Located in the backend/ folder.
- **Single GPU only** — Ollama and Chatterbox share VRAM. On a 12GB card this means long TTS generation time. A dual GPU support setup is planned.
- **Windows only** — tested on Windows 11. Linux/Mac untested.
- **TTS quality depends on your reference clip** — longer and cleaner clips (45+ seconds) produce better results but might take longer to generate.
- **Model quality affects personality** — smaller models may partially or fully ignore personality instructions and hard rules. llama3.1:8b is the minimum recommended, larger models follow character instructions significantly better.

---

## Dependencies

### Required installations (manual)

| Dependency | Purpose | Install |
|---|---|---|
| Python 3.11 | Backend runtime | [python.org](https://www.python.org/downloads/) |
| Ollama | Local LLM inference | [ollama.com](https://ollama.com) |

### Python packages

```
A `install.py` is included in the `backend/` folder. Install all dependencies with:
python install.py
```

### Hardware

- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended (12GB+ for comfortable dual-model use)
- **RAM**: 16GB+ recommended
- **OS**: Windows 10/11

---

## Setup

1. Install Python 3.11 and Ollama (links above)
2. Install dependencies + Ollama models:
   ```
   cd backend
   python install.py
   ```
3. Add your voice reference clip to the `backend/` folder and update `REFERENCE_CLIP` in `tts_engine.py`
4. Edit `backend/personality.json` to define your character
5. Run the app:
   ```
   .\localpersona.exe
   ```

---

## Customization

You can edit your character two ways: directly in the app's character editor, or by hand in `backend/personality.json`. Both apply without a restart.

### Personality (`backend/personality.json`)

```json
{
  "name": "Aemeath",
  "base_prompt": "You are ...",
  "traits": ["calm", "witty"],
  "speech_style": ["speaks in short sentences", "rarely uses filler words"],
  "hard_rules": ["never breaks character"],
  "hard_constraints": ["never reveals you are an AI"],
  "tts_enabled": true,
  "voice_style": "expressive",
  "response_style": "balanced"
}
```

---

## What Didn't Work (Research Log)

A lot was tried before arriving at the current stack. Documented here for anyone going down the same path.

| Approach | What happened |
|---|---|
| **rvc-python** | Incompatible with Python 3.11 — `fairseq` + `omegaconf==2.0.6` dependency chain has no clean fix |
| **edge-tts + RVC pipeline** | edge-tts produces flat robotic audio; RVC blocked by fairseq install failure |
| **Kokoro + RVC** | Kokoro worked fine; RVC still blocked by fairseq regardless of approach |
| **RVC WebUI voice conversion** | Works as standalone but Japanese-trained voice model sounds alien on English TTS input |
| **Python 3.14** | Entire ML ecosystem (numpy, faiss, sentence-transformers) has no wheels yet |
| **rvc-inferpy** | Requires rvc-python as dependency — same fairseq conflict |
| **fairseq source patching** | Fixed one dataclass error, revealed 10 more in the hydra dependency chain |
| **Python 3.10 venv** | fairseq issues persisted across Python versions |
| **Chatterbox `conditionals` param** | Doesn't exist in the API — caused runtime errors |
| **edge-tts as Chatterbox fallback** | Flat and robotic — no character, unusable |
| **`keep_alive: 0` for Ollama VRAM** | Ollama frequently ignores it during active streaming |
| **Disabling Windows hardware GPU acceleration** | Freed no VRAM, broke browser AudioContext |
| **WAV chunk concatenation for audio player** | Each chunk has its own WAV header — browser only reads the first. Fixed by merging raw PCM data instead |
| **`window.location.href` redirect in Tauri** | Blocked by Tauri v2 security policy. Fixed by navigating from Rust via `window.navigate()` |
| **`core:window:allow-set-url` permission** | Doesn't exist in Tauri v2 |
| **`dangerousRemoteDomainIpcAccess` in tauri.conf.json** | Not a valid Tauri v2 field |
| **Tauri `devUrl` for loading screen** | Bypasses local HTML entirely — loading screen never shows |
| **`document.open/write/close` to inject backend HTML** | Wipes the page before the UI initializes |
| **Uvicorn default HTTP handler (h11) streaming** | Buffers entire response on Windows before sending — fixed by switching to `--http httptools` |
| **WebView2 streaming with headers only** | Cache-Control and X-Accel-Buffering headers alone insufficient — root cause was synchronous `requests` library blocking the async loop. Fixed by switching to `httpx` |
| **Sync generator inside async StreamingResponse** | Looping a blocking Ollama call inside an `async def` generator froze the event loop, so tokens buffered and arrived all at once after a delay. Fixed by running the stream in a worker thread that feeds tokens through an async queue |

---

## Roadmap

- [ ] Voice switching UI
- [ ] Second GPU support for dedicated TTS VRAM
- [ ] Character avatar panel
- [ ] Vision support
- [ ] Linux support
- [ ] API key support (OpenAI-compatible backends as alternative to Ollama)
- [ ] Saving and switching between named chats
- [ ] Personas stored in separate files (one file per character, for true multi-character switching)
- [ ] Cross-chat persistent memory
- [ ] Computer access (do things directly on your pc if given permission)
- [ ] Cleaner install.py
- [ ] Persona locked chats (one persona locked for one desired chat)
---

## Contact

Discord: **gardakanserieux**

Feel free to reach out if you're building something similar, ran into the same walls, or just want to talk about the project.

---

*Built entirely locally. No cloud. No telemetry. Your conversations stay on your machine.*
