# Local-Persona-AI

> Turn any character into a fully local AI companion — with their voice and personality.

Local-Persona-AI is a desktop application that lets you run a completely offline, private AI assistant modeled after any character or persona you choose. You bring the voice reference clip and personality config, the app handles the rest. No cloud, no subscriptions, no data leaving your machine.

> ### ⬇️ Download from [Releases](https://github.com/nprev-dev/Local-Persona-AI/releases) — do not clone
>
> The release `.zip` is the app. It contains the launcher and the default character **Aemeath**, complete with her voice, ready to run.
>
> Cloning gets you the source only: no launcher, no default character, and no voice clip. The app will still run, but it starts with a blank generic assistant instead of Aemeath. Clone only if you intend to build it yourself.

---

## What It Does

- **Multiple personas** — create as many characters as you want, each saved as its own file with its own personality, traits, rules, and voice. Switch between them or delete them right in the app.
- **Custom personality** — define any character's speech style, traits, rules, and backstory via a simple JSON file or directly in the app. No coding required.
- **In-app character editor** — create, edit, and delete characters from the UI. Edit personality, traits, rules, and response style. Changes save to that persona's file and reload instantly, no restart needed.
- **Per-persona voices** — every character can have its own voice. Upload a `.wav` sample in the character editor and that persona speaks in that voice. Switching personas switches the voice automatically.
- **In-app model picker** — automatically detects the Ollama models installed on your machine and lets you switch between them from the UI.
- **Voice cloning** — provide a short reference audio clip and the AI speaks in that voice using Chatterbox TTS with 7-emotion detection.
- **Per-character voice toggle** — turn TTS on or off per character. Characters with no voice sample simply stay silent.
- **Streaming chat** — token-by-token response streaming, ChatGPT-style, fully local via Ollama.
- **Hybrid memory** — long-term memory system combining BM25 keyword search and FAISS vector search. The AI remembers facts about you across sessions.
- **Sentence-by-sentence audio streaming** — TTS starts playing before the full response is generated.
- **Fully offline UI** — React and the icon set are vendored locally, and the interface is precompiled so nothing is transformed in the browser at launch. Zero outside calls, works with no internet.
- **Saved chats** — every conversation is stored separately and reloads when you reopen the app. Switch between them or delete them from the sidebar.
- **Working settings** — adjust reply length and temperature from the Settings tab, or leave the tuned defaults alone. Preferences persist between launches.
- **Sensible character defaults** — new characters start with the rules that keep them in character, so they behave well before you customise anything.
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
    ├── Streams tokens from Ollama (qwen2.5:7b)
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

## Current State (v0.5.0)

This is an early working release. Everything below works but some things are still hardcoded or manual:

- **Multiple personas** — create, switch between, and delete characters in the app. Each persona is stored as its own file in `backend/personas/`, with its own voice clip in `backend/personas/voices/`.
- **Saved separate chats** — each conversation is stored on its own and survives restarts. Starting a new chat gives the AI a clean slate. Long-term memory stays shared across chats, so it still remembers facts about you wherever you are.
- **Model selection in-app** — the app detects your installed Ollama models and lets you pick between them from the UI. The default chat model is `qwen2.5:7b` and the memory judge is `phi3`.
- **Model installation is manual** — you need to pull Ollama models yourself for the ones not included in `install.py`.
- **Personality is fully editable in-app** — define and tweak any character from the UI or by editing its file directly. Either way changes apply without a restart.
- **Per-persona voices, uploaded in-app** — give each character its own voice by uploading a `.wav` in the character editor. A persona with no voice sample stays silent.
- **GUI installer included** — run install.py before first launch to install dependencies and pull Ollama models. Located in the backend/ folder.
- **Single GPU only** — Ollama and Chatterbox share VRAM, so TTS speed depends on how much is left once your chat model loads. A 7B leaves comfortable room on a 12GB card; larger models squeeze Chatterbox and slow speech down noticeably. A dual GPU setup is planned.
- **Windows only** — tested on Windows 11. Linux/Mac untested.
- **TTS quality depends on your reference clip** — see the Voice Reference Tips below.
- **Model choice affects personality** — most local models hold character reliably. `mistral` is a known weak spot and drifts out of character noticeably more than the others, so it isn't recommended for persona work. `qwen2.5:7b` is the default and a good starting point.
- **Aemeath ships by default** — the installer includes Aemeath from Wuthering Waves as a default persona, complete with her voice reference clip. Like any other persona, she can be deleted at any time along with her voice clip.

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
3. Launch the app, then create or edit a character in the **Characters** tab — set its personality and upload a `.wav` voice sample right in the UI
4. Run the app:
```
   .\localpersona.exe
```

---

## Voice Reference Tips
 
Getting a good cloned voice depends a lot on the reference clip you provide. A few things learned the hard way:
 
- **Use a neutral reference clip.** The voice clone picks up the *emotion* of your reference, not just the timbre. If your reference clip sounds excited or happy, every response comes out sounding excited — even sad ones. A calm, neutral clip gives the most flexible, natural result, because the emotion system can then shape it per response.
- **Clean audio, no background music or noise.** Chatterbox copies whatever it hears.
- **10–25 seconds is a good length.** Long enough to capture the voice, short enough to stay clean.
- **On voice splitting / streaming speed.** The app generates speech sentence by sentence so audio can start playing sooner — it *feels* faster. The tradeoff is that each sentence is generated independently, so the model can't carry tone and pacing across sentence boundaries the way it would generating the whole reply at once. Slightly less consistent prosody, noticeably faster start. For most use the speed is worth it.

---

## Customization

You can edit your characters two ways: directly in the app's character editor, or by hand in their files. Both apply without a restart. Each persona lives in `backend/personas/<id>.json`, with its voice clip at `backend/personas/voices/<id>.wav`.

### Personality (`backend/personas/<id>.json`)

```json
{
  "name": "Aemeath",
  "base_prompt": "You are ...",
  "traits": ["calm", "witty"],
  "speech_style": ["speaks in short sentences", "rarely uses filler words"],
  "hard_rules": ["never breaks character"],
  "hard_constraints": ["never reveals you are an AI"],
  "tts_enabled": true,
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
| **`keep_alive: 0` after every reply** | Works, but reloads the model on every message (~2s each) for no gain when the model and Chatterbox both fit in VRAM. Now decided per model instead |
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

- [ ] Second GPU support for dedicated TTS VRAM
- [ ] Character avatar panel
- [ ] Vision support
- [ ] Linux support
- [ ] API key support (OpenAI-compatible backends as alternative to Ollama)
- [x] Saving and switching between named chats
- [ ] Cross-chat persistent memory
- [ ] Computer access (do things directly on your pc if given permission)
- [ ] Cleaner install.py
- [ ] Document ingestion (paste text or upload files to give the AI knowledge)
- [ ] Persona-locked chats (one persona locked to one desired chat)
- [x] Stronger personality adherence

---

## Contact

Discord: **gardakanserieux**

Feel free to reach out if you're building something similar, ran into the same walls, or just want to talk about the project.

---

*Built entirely locally. No cloud. No telemetry. Your conversations stay on your machine.*

## License

This project is licensed under the [MIT License](LICENSE) — you're free to use, modify, and distribute it, as long as you keep the copyright notice and give credit to **nprev-dev**.
