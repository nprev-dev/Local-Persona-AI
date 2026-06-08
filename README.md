# Local-Persona-AI

> Turn any character into a fully local AI companion — with their voice and personality.

Local-Persona-AI is a desktop application that lets you run a completely offline, private AI assistant modeled after any character or persona you choose. You bring the voice reference clip and personality config, the app handles the rest. No cloud, no subscriptions, no data leaving your machine.

---

## What It Does

- **Custom personality** — define any character's speech style, traits, rules, and backstory via a simple JSON file. No coding required.
- **Voice cloning** — provide a short reference audio clip and the AI speaks in that voice using Chatterbox TTS with 7-emotion detection.
- **Streaming chat** — token-by-token response streaming, ChatGPT-style, fully local via Ollama.
- **Hybrid memory** — long-term memory system combining BM25 keyword search and FAISS vector search. The AI remembers facts about you across sessions.
- **Document ingestion** — paste text or upload files to give the AI knowledge of anything you want.
- **Sentence-by-sentence audio streaming** — TTS starts playing before the full response is generated.
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

## Current State (v0.2.0)

This is an early working release. Everything below works but some things are still hardcoded or manual:

- **Models are hardcoded** — `llama3.1:8b` for chat, `phi3` for the memory judge. You can change these in `main.py` but there's no UI for it yet.
- **Model installation is manual** — you need to pull Ollama models yourself before running (see setup below).
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
| llama3.1:8b | Chat model | `ollama pull llama3.1:8b` |
| phi3 | Memory judge model | `ollama pull phi3` |

### Python packages

```
A `requirements.txt` is included in the `backend/` folder. Install all dependencies with:
pip install -r requirements.txt
```

### Hardware

- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended (12GB+ for comfortable dual-model use)
- **RAM**: 16GB+ recommended
- **OS**: Windows 10/11

---

## Setup

1. Install Python 3.11 and Ollama (links above)
2. Pull the required models:
   ```
   ollama pull llama3.1:8b
   ollama pull phi3
   ```
3. Install dependencies + Ollama models:
   ```
   cd backend
   python install.py
   ```
4. Add your voice reference clip to the `backend/` folder and update `REFERENCE_CLIP` in `tts_engine.py`
5. Edit `backend/personality.json` to define your character
6. Run the app:
   ```
   .\localpersona.exe
   ```

---

## Customization

### Personality (`backend/personality.json`)

```json
{
  "base_prompt": "You are ...",
  "traits": ["calm", "witty"],
  "speech_style": ["speaks in short sentences", "rarely uses filler words"],
  "hard_rules": ["never breaks character"],
  "hard_constraints": ["never reveals you are an AI"]
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
| **FModel pak extraction** | AES keys loaded correctly but pak files remained inaccessible — unresolved |
| **UnrealPak extraction** | Pak version mismatch (version 12) — tool too old for target game |
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

---

## Roadmap

- [ ] Automated dependency installer
- [ ] Model selection UI
- [ ] Voice switching UI
- [ ] Second GPU support for dedicated TTS VRAM
- [ ] Character avatar panel
- [ ] Vision support
- [ ] Linux support
- [ ] Major UI rework
- [ ] API key support (OpenAI-compatible backends as alternative to Ollama)
- [ ] Multiple named chats / conversation switching
- [ ] Cross-chat persistent memory

---

## Contact

Discord: **gardakanserieux**

Feel free to reach out if you're building something similar, ran into the same walls, or just want to talk about the project.

---

*Built entirely locally. No cloud. No telemetry. Your conversations stay on your machine.*
