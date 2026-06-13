from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from openai import OpenAI
import requests
import json
import re
from pydantic import BaseModel
from tts_engine import synthesize, synthesize_sentence, set_reference_clip

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

PERSONALITY_FILE = "personality.json"


# ── Personality system ──────────────────────────────────────────────────────────

def load_personality_raw() -> dict:
    """Load the raw personality.json dict. Returns a default if missing/broken."""
    try:
        with open(PERSONALITY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Personality] Could not load {PERSONALITY_FILE}: {e}")
        return {
            "name": "Assistant",
            "avatar_color": "#22d3ee",
            "avatar_initial": "A",
            "persona": "A helpful AI assistant.",
            "base_prompt": "You are a helpful assistant. Be concise, accurate, and friendly.",
            "traits": [],
            "speech_style": [],
            "hard_constraints": [],
            "hard_rules": [],
            "tts_enabled": True,
            "voice_style": "neutral",
            "response_style": "balanced",
            "language": "English"
        }


def save_personality_raw(data: dict):
    with open(PERSONALITY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_personality_prompt(p: dict) -> str:
    """Build the full system prompt text from a personality dict."""
    prompt = p.get("base_prompt", "You are a helpful assistant.")

    traits = p.get("traits", [])
    if traits:
        prompt += "\n\nYour personality traits:\n" + "\n".join(f"- {t}" for t in traits)

    style = p.get("speech_style", [])
    if style:
        prompt += "\n\nHow you speak:\n" + "\n".join(f"- {s}" for s in style)

    rules = p.get("hard_rules", [])
    if rules:
        prompt += "\n\nRules you always follow:\n" + "\n".join(f"- {r}" for r in rules)

    constraints = p.get("hard_constraints", [])
    if constraints:
        prompt += "\n\nABSOLUTE RULES YOU NEVER BREAK:\n" + "\n".join(f"- {c}" for c in constraints)

    return prompt


PERSONALITY        = load_personality_raw()
PERSONALITY_PROMPT = build_personality_prompt(PERSONALITY)
print(f"[Personality] Loaded '{PERSONALITY.get('name', 'Assistant')}'")


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


# ── Ollama API helpers ─────────────────────────────────────────────────────────
# ask_ollama uses /api/chat (messages array) — supports proper system messages.
# _ask_generate uses /api/generate (raw prompt) — used only by the memory judge.

def ask_ollama(messages: list, model: str = CHAT_MODEL, max_tokens: int = 150) -> str:
    if model is None:
        model = CURRENT_MODEL
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model":    model,
            "messages": messages,
            "stream":   False,
            "options":  {"num_predict": max_tokens},
            "keep_alive": 0
        }
    )
    response.raise_for_status()
    raw = response.json()["message"]["content"]
    return strip_thinking(raw)

def ask_ollama_stream(messages: list, model: str = CHAT_MODEL, max_tokens: int = 150):
    """Generator that yields tokens as they stream from Ollama."""
    if model is None:
        model = CURRENT_MODEL
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model":      model,
            "messages":   messages,
            "stream":     True,
            "options":    {"num_predict": max_tokens},
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

    messages.append({"role": "user", "content": user_input})
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
                for token in ask_ollama_stream(messages, model=CURRENT_MODEL):
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
    """Return the current personality as a character object for the UI."""
    return PERSONALITY


@app.post("/personality")
async def update_personality(update: PersonalityUpdate):
    """
    Save edits from the Characters page back to personality.json and
    reload the active personality prompt in place (no restart needed).
    """
    global PERSONALITY, PERSONALITY_PROMPT

    data = update.model_dump()
    save_personality_raw(data)

    PERSONALITY        = data
    PERSONALITY_PROMPT = build_personality_prompt(PERSONALITY)

    print(f"[Personality] Updated '{PERSONALITY.get('name', 'Assistant')}'")
    return {"status": "ok", "personality": PERSONALITY}


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