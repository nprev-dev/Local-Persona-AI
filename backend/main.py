# Console output has to survive non-ASCII text such as a persona named in
# Japanese or Cyrillic. On Windows stdout defaults to cp1252 and raises
# UnicodeEncodeError on any character that codepage lacks, which kills startup.
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from openai import OpenAI
import requests
import json
import re
import os
from pydantic import BaseModel
from tts_engine import synthesize, synthesize_sentence, set_reference_clip
from llm_config import CHAT_OPTIONS, CHAT_MAX_TOKENS, max_tokens_for

from memory_engine import (
    init_memory_db,
    add_memory,
    search_memory,
    get_all_memories,
    clear_all_memories,
    ingest_text,
    ingest_file,
    memory_count
)

app = FastAPI()
from fastapi.staticfiles import StaticFiles
app.mount("/vendor", StaticFiles(directory="vendor"), name="vendor")


init_memory_db()

client = None #OpenAI()

USE_OPENAI = False

CHAT_MODEL   = "qwen2.5:7b"
MEMORY_MODEL = "phi3"

CURRENT_MODEL = "qwen2.5:7b"

# ── Persona system (multi-persona) ──────────────────────────────────────────────
# Personas live in personas/<id>.json, each with its own voice clip.
# Chat history and long-term memory stay shared/global.

import persona_manager as pm

pm.ensure_setup()  # creates folder / migrates legacy personality.json on first run


def _apply_active_voice():
    """Point the TTS engine at the active persona's voice clip (if it has one)."""
    p = pm.get_active_persona()
    clip = p.get("voice_clip", "")
    if clip and os.path.exists(clip):
        try:
            set_reference_clip(clip)
        except Exception as e:
            print(f"[Personas] Could not set voice clip: {e}")


def reload_active_persona():
    """Refresh the cached active persona + system prompt + voice."""
    global PERSONALITY, PERSONALITY_PROMPT, PERSONALITY_ANCHOR
    PERSONALITY        = pm.get_active_persona()
    PERSONALITY_PROMPT = pm.build_personality_prompt(PERSONALITY)
    PERSONALITY_ANCHOR = pm.build_reanchor(PERSONALITY)
    _apply_active_voice()
    return PERSONALITY


PERSONALITY        = pm.get_active_persona()
PERSONALITY_PROMPT = pm.build_personality_prompt(PERSONALITY)
PERSONALITY_ANCHOR = pm.build_reanchor(PERSONALITY)
_apply_active_voice()
print(f"[Personas] Active persona: '{PERSONALITY.get('name', 'Assistant')}'")


class ChatRequest(BaseModel):
    user_input: str

class IngestRequest(BaseModel):
    text:   str
    source: str = "manual"

class PersonalityUpdate(BaseModel):
    name:             str = "Assistant"
    avatar_color:     str = "#22d3ee"
    avatar_initial:   str = "A"
    persona:          str = ""
    base_prompt:      str = ""
    traits:           list = []
    speech_style:     list = []
    hard_constraints: list = []
    hard_rules:       list = []
    tts_enabled:      bool = True
    voice_style:      str = "neutral"
    response_style:   str = "balanced"
    language:         str = "English"


# ── Thinking tag stripper ──────────────────────────────────────────────────────
# outputs <think>...</think> blocks before its reply.
# We strip these so users only see the actual response.
def strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def _pending_tag_len(buffer: str, tag: str) -> int:
    """Length of the longest suffix of buffer that could still grow into tag."""
    for n in range(min(len(buffer), len(tag) - 1), 0, -1):
        if buffer.endswith(tag[:n]):
            return n
    return 0


def strip_thinking_stream(tokens):
    """
    Yield only the text outside <think>...</think>.

    Ollama 0.32 returns reasoning in a separate 'thinking' field, so this covers
    the other case: models whose reasoning arrives inline in the content stream.
    A tag can be split across token boundaries, so partial tags are held back
    rather than emitted.
    """
    OPEN, CLOSE = "<think>", "</think>"
    buffer, inside = "", False

    for token in tokens:
        buffer += token
        while buffer:
            if not inside:
                idx = buffer.find(OPEN)
                if idx == -1:
                    held = _pending_tag_len(buffer, OPEN)
                    emit, buffer = buffer[:len(buffer) - held], buffer[len(buffer) - held:]
                    if emit:
                        yield emit
                    break
                emit, buffer = buffer[:idx], buffer[idx + len(OPEN):]
                if emit:
                    yield emit
                inside = True
            else:
                idx = buffer.find(CLOSE)
                if idx == -1:
                    held = _pending_tag_len(buffer, CLOSE)
                    buffer = buffer[len(buffer) - held:] if held else ""
                    break
                buffer = buffer[idx + len(CLOSE):]
                inside = False

    if buffer and not inside:
        yield buffer


# ── Ollama API helpers ─────────────────────────────────────────────────────────
# ask_ollama uses /api/chat (messages array) — supports proper system messages.
# _ask_generate uses /api/generate (raw prompt) — used only by the memory judge.

def _chat_options(max_tokens: int) -> dict:
    options = dict(CHAT_OPTIONS)
    options["num_predict"] = max_tokens
    return options


def ask_ollama(messages: list, model: str = CHAT_MODEL, max_tokens: int = None) -> str:
    if model is None:
        model = CURRENT_MODEL
    if max_tokens is None:
        max_tokens = max_tokens_for(model)
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model":    model,
            "messages": messages,
            "stream":   False,
            "options":  _chat_options(max_tokens),
            "keep_alive": 0
        }
    )
    response.raise_for_status()
    raw = response.json()["message"]["content"]
    return strip_thinking(raw)

def ask_ollama_stream(messages: list, model: str = CHAT_MODEL, max_tokens: int = None):
    """Generator that yields tokens as they stream from Ollama."""
    if model is None:
        model = CURRENT_MODEL
    if max_tokens is None:
        max_tokens = max_tokens_for(model)
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model":      model,
            "messages":   messages,
            "stream":     True,
            "options":    _chat_options(max_tokens),
            "keep_alive": 0
        },
        stream=True
    )
    response.raise_for_status()
    for line in response.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                yield token
            if data.get("done"):
                break
        except json.JSONDecodeError:
            continue

def unload_ollama():
    """Force Ollama to release VRAM after responding."""
    try:
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": CURRENT_MODEL, "keep_alive": 0, "prompt": ""},
            timeout=5
        )
    except Exception:
        pass

def _ask_generate(prompt: str, model: str = MEMORY_MODEL, max_tokens: int = 180) -> str:
    """
    Raw prompt → response via /api/generate.
    Used internally by the memory judge which builds its own flat prompt.
    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model":   model,
            "prompt":  prompt,
            "stream":  False,
            "options": {"num_predict": max_tokens}
        }
    )
    response.raise_for_status()
    raw = response.json()["response"]
    return strip_thinking(raw)


def ask_openai(messages: list) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )
    return response.choices[0].message.content


# ── Chat history ───────────────────────────────────────────────────────────────

def load_chat_memory():
    try:
        with open("memory.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []


def save_chat_memory(data):
    with open("memory.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Prompt builders ────────────────────────────────────────────────────────────

def memories_to_text(memories):
    if not memories:
        return "No relevant long-term memories found."
    text = ""
    for memory in memories:
        text += f"- [{memory['category']}] {memory['content']} (importance {memory['importance']}/10)\n"
    return text.strip()


def build_chat_messages(user_input: str, recent_chat: list, relevant_memories: list) -> list:
    """
    Build a proper messages list for /api/chat.
    System message carries the personality + relevant memories.
    Recent chat history is passed as alternating user/assistant turns.
    """
    system_content = (
        f"{PERSONALITY_PROMPT}\n\n"
        "Use the long-term memories only when relevant. "
        "Do not dump memories unless the user asks.\n\n"
        "RELEVANT LONG-TERM MEMORIES:\n"
        f"{memories_to_text(relevant_memories)}"
    )

    messages = [{"role": "system", "content": system_content}]

    # Add the last 10 conversation turns
    for msg in recent_chat[-10:]:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    # The re-anchor rides on the user turn rather than a second system message.
    # Templates disagree about mid-conversation system messages: qwen2.5 honours
    # one, llama3.1 ignores it, and mistral has no system branch at all. Every
    # template renders the final user turn, so this placement reaches all of them.
    # It is never written to chat history, only to the outgoing prompt.
    if user_input:
        messages.append({"role": "user",
                         "content": f"{user_input}\n\n({PERSONALITY_ANCHOR})"})
    else:
        messages.append({"role": "user", "content": PERSONALITY_ANCHOR})
    return messages


def build_memory_judge_prompt(user_input, assistant_reply):
    return f"""
You are a local memory manager for a personal AI assistant.

Decide if the user's message contains information worth saving long-term.

Save only stable useful facts, such as:
- identity
- preferences
- goals
- projects
- hardware/software setup
- recurring interests
- communication style
- important plans

Ignore:
- random temporary details
- one-time emotions
- useless small talk
- things that probably won't matter later

Return ONLY valid JSON.

If worth saving:
{{
  "should_store": true,
  "category": "projects",
  "content": "User is building a local anime AI assistant on Windows.",
  "importance": 9
}}

If not worth saving:
{{
  "should_store": false
}}

User message:
{user_input}

Assistant reply:
{assistant_reply}
""".strip()


def extract_memory(user_input, assistant_reply):
    prompt = build_memory_judge_prompt(user_input, assistant_reply)
    raw    = _ask_generate(prompt, model=MEMORY_MODEL, max_tokens=180)

    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1

        if start == -1 or end == 0:
            return None

        data = json.loads(raw[start:end])

        if not data.get("should_store"):
            return None

        category   = data.get("category", "general")
        content    = data.get("content", "").strip()
        importance = int(data.get("importance", 5))

        if not content:
            return None

        return {"category": category, "content": content, "importance": importance}

    except:
        return None


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>index.html not found</h1>"


@app.post("/chat")
async def chat(data: ChatRequest):
    import asyncio

    user_input = data.user_input.strip()

    chat_memory       = load_chat_memory()
    relevant_memories = search_memory(user_input, limit=6)
    messages          = build_chat_messages(user_input, chat_memory, relevant_memories)

    async def response_stream():
        loop  = asyncio.get_event_loop()
        queue = asyncio.Queue()
        full_reply_box = {"text": ""}

        # Run the blocking Ollama stream in a worker thread.
        def produce():
            try:
                raw = ask_ollama_stream(messages, model=CURRENT_MODEL)
                for token in strip_thinking_stream(raw):
                    full_reply_box["text"] += token
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel = done

        loop.run_in_executor(None, produce)

        # Drain the queue as tokens arrive and flush each to the client.
        while True:
            token = await queue.get()
            if token is None:
                break
            yield json.dumps({"token": token}) + "\n"

        full_reply = full_reply_box["text"]

        # Save chat history
        chat_memory.append({"role": "user",      "content": user_input})
        chat_memory.append({"role": "assistant", "content": full_reply})
        save_chat_memory(chat_memory)
        unload_ollama()

        # Save memory
        new_memory   = extract_memory(user_input, full_reply)
        memory_saved = False
        if new_memory:
            memory_saved = add_memory(
                category   = new_memory["category"],
                content    = new_memory["content"],
                importance = new_memory["importance"],
                source     = "chat"
            )

        counts = memory_count()
        yield json.dumps({
            "done":         True,
            "memory_saved": memory_saved,
            "memory_count": counts
        }) + "\n"

    return StreamingResponse(response_stream(), media_type="application/x-ndjson")

@app.post("/ingest")
async def ingest(data: IngestRequest):
    """
    Load raw text into the document memory.
    The AI will be able to search this content when answering questions.
    """
    text   = data.text.strip()
    source = data.source.strip() or "manual"

    if not text:
        return {"status": "error", "message": "No text provided.", "chunks_stored": 0}

    stored = ingest_text(text, source=source)
    counts = memory_count()

    return {
        "status":        "ok",
        "chunks_stored": stored,
        "source":        source,
        "memory_count":  counts
    }


@app.post("/ingest-file")
async def ingest_file_upload(file: UploadFile = File(...)):
    """
    Upload a .txt or .md file and load its contents into document memory.
    """
    content = await file.read()
    try:
        text = content.decode("utf-8", errors="ignore")
    except Exception:
        return {"status": "error", "message": "Could not decode file."}

    stored = ingest_text(text, source=file.filename or "uploaded_file")
    counts = memory_count()

    return {
        "status":        "ok",
        "filename":      file.filename,
        "chunks_stored": stored,
        "memory_count":  counts
    }


@app.get("/memories")
async def memories():
    return get_all_memories()


@app.get("/memory-count")
async def get_memory_count():
    return memory_count()


@app.get("/settings")
async def get_settings():
    return {"model": CURRENT_MODEL}


@app.post("/settings")
async def update_settings(settings: dict):
    global CURRENT_MODEL
    if "model" in settings:
        CURRENT_MODEL = settings["model"]
    return {"status": "ok", "model": CURRENT_MODEL}


@app.get("/models")
async def list_models():
    """
    Return the list of locally installed Ollama models so the UI's
    model picker only shows models that actually exist on this machine.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        models = []
        for m in data.get("models", []):
            name = m.get("name", "")
            if name:
                models.append({
                    "id":       name,
                    "label":    name,
                    "provider": "Ollama (local)"
                })
        return {"models": models, "current": CURRENT_MODEL}
    except Exception as e:
        # Fall back to at least the current model so the picker isn't empty
        return {
            "models":  [{"id": CURRENT_MODEL, "label": CURRENT_MODEL, "provider": "Ollama (local)"}],
            "current": CURRENT_MODEL,
            "error":   str(e)
        }


@app.get("/personality")
async def get_personality():
    """Return the ACTIVE persona as a character object (back-compat alias)."""
    return PERSONALITY


@app.post("/personality")
async def update_personality(update: PersonalityUpdate):
    """
    Back-compat: save edits to the ACTIVE persona's file and reload in place.
    Preserves the persona's existing voice_clip.
    """
    data = update.model_dump()
    active_id = pm.get_active_id()
    try:
        existing = pm.load_persona(active_id)
        data["voice_clip"] = existing.get("voice_clip", "")
    except Exception:
        data["voice_clip"] = ""

    pm.save_persona(active_id, data)
    reload_active_persona()
    print(f"[Personas] Updated active persona '{PERSONALITY.get('name','?')}'")
    return {"status": "ok", "personality": PERSONALITY}


# ── Multi-persona endpoints ──────────────────────────────────────────────────

@app.get("/personas")
async def personas_list():
    return {"personas": pm.list_personas_summary(), "active": pm.get_active_id()}


@app.get("/personas/{pid}")
async def personas_get(pid: str):
    try:
        return pm.load_persona(pid)
    except Exception:
        return Response(status_code=404)


@app.post("/personas/new")
async def personas_new(payload: dict):
    """Create a fresh persona from a name. Body: { "name": "Jarvis" }

    NOTE: This route MUST be defined before POST /personas/{pid}, otherwise
    FastAPI matches '/personas/new' against '{pid}' (pid='new') and this never runs.
    """
    name = (payload.get("name") or "New Persona").strip()
    base_id = pm.slugify(name)
    pid = base_id
    n = 2
    while pid in pm.list_persona_ids():
        pid = f"{base_id}_{n}"
        n += 1

    persona = pm._default_persona(pid)
    persona["name"] = name
    persona["avatar_initial"] = name[:1].upper()
    pm.save_persona(pid, persona)
    return {"status": "ok", "id": pid, "persona": pm.load_persona(pid)}


@app.post("/personas/{pid}")
async def personas_save(pid: str, update: PersonalityUpdate):
    """Create or update a persona by id. Preserves its voice_clip if it exists."""
    # Guard against bogus ids (e.g. 'undefined' from a frontend race).
    if not pid or pid in ("undefined", "null"):
        return {"status": "error", "message": "Invalid persona id."}

    data = update.model_dump()
    try:
        existing = pm.load_persona(pid)
        data["voice_clip"] = existing.get("voice_clip", "")
    except Exception:
        data["voice_clip"] = ""

    pm.save_persona(pid, data)
    if pid == pm.get_active_id():
        reload_active_persona()
    return {"status": "ok", "persona": pm.load_persona(pid)}


@app.post("/personas/{pid}/activate")
async def personas_activate(pid: str):
    """Switch the active persona (also switches the voice)."""
    if not pm.set_active(pid):
        return Response(status_code=404)
    reload_active_persona()
    print(f"[Personas] Switched to '{PERSONALITY.get('name','?')}'")
    return {"status": "ok", "active": pid, "personality": PERSONALITY}


@app.delete("/personas/{pid}")
async def personas_delete(pid: str):
    """Delete a persona (won't delete the last one)."""
    ok = pm.delete_persona(pid)
    if not ok:
        return {"status": "error", "message": "Cannot delete (not found or last remaining persona)."}
    reload_active_persona()
    return {"status": "ok", "active": pm.get_active_id()}


@app.post("/personas/{pid}/voice")
async def personas_upload_voice(pid: str, file: UploadFile = File(...)):
    """
    Upload a .wav voice sample for a persona. Saves it to
    personas/voices/<id>.wav and sets the persona's voice_clip field.
    """
    if pid not in pm.list_persona_ids():
        return {"status": "error", "message": "Persona not found."}

    fname = (file.filename or "").lower()
    if not fname.endswith(".wav"):
        return {"status": "error", "message": "Only .wav files are accepted."}

    os.makedirs(pm.VOICES_DIR, exist_ok=True)
    dest = os.path.join(pm.VOICES_DIR, f"{pid}.wav")

    try:
        content = await file.read()
        with open(dest, "wb") as f:
            f.write(content)
    except Exception as e:
        return {"status": "error", "message": f"Could not save file: {e}"}

    # Update the persona's voice_clip field.
    persona = pm.load_persona(pid)
    persona["voice_clip"] = dest
    pm.save_persona(pid, persona)

    # If this is the active persona, point the TTS engine at the new clip.
    if pid == pm.get_active_id():
        reload_active_persona()

    print(f"[Personas] Voice sample set for '{persona.get('name', pid)}'")
    return {"status": "ok", "persona": pm.load_persona(pid)}


@app.post("/clear-memory")
async def clear_chat_memory():
    save_chat_memory([])
    return {"status": "ok", "message": "Chat memory cleared."}


@app.post("/clear-long-term-memory")
async def clear_long_term_memory():
    clear_all_memories()
    return {"status": "ok", "message": "Long-term memory cleared."}

class SpeakRequest(BaseModel):
    text: str

@app.post("/speak")
async def speak(data: SpeakRequest):
    import asyncio
    import re as _re

    text = data.text.strip()
    if not text:
        return Response(status_code=204)

    # If the active persona has no voice clip, skip TTS entirely.
    active = pm.get_active_persona()
    clip = active.get("voice_clip", "")
    if not clip or not os.path.exists(clip):
        return Response(status_code=204)

    sentences = _re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if not sentences:
        sentences = [text]

    async def stream_sentences():
        loop = asyncio.get_event_loop()
        for sentence in sentences:
            print(f"[TTS] Generating sentence: {sentence[:50]}")
            audio_bytes = await loop.run_in_executor(
                None, synthesize_sentence, sentence
            )
            if audio_bytes:
                print(f"[TTS] Sending {len(audio_bytes)} bytes")
                length = len(audio_bytes).to_bytes(4, 'big')
                yield length + audio_bytes
            else:
                print(f"[TTS] No audio for sentence")

    return StreamingResponse(stream_sentences(), media_type="audio/octet-stream")