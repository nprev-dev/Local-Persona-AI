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

def load_personality() -> str:
    try:
        with open("personality.json", "r", encoding="utf-8") as f:
            p = json.load(f)
        prompt = p.get("base_prompt", "You are a helpful AI assistant.")
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
    except Exception as e:
        print(f"[Personality] Could not load personality.json: {e}")
        return "You are a helpful AI assistant."

PERSONALITY_PROMPT = load_personality()
print("[Personality] Loaded successfully")

init_memory_db()

CHAT_MODEL    = "qwen2.5:7b"
MEMORY_MODEL  = "phi3"

CURRENT_MODEL = "qwen2.5:7b"


class ChatRequest(BaseModel):
    user_input: str

class IngestRequest(BaseModel):
    text:   str
    source: str = "manual"


# ── Thinking tag stripper ──────────────────────────────────────────────────────
# outputs <think>...</think> blocks before its reply.
# We strip these so users only see the actual response.
def strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(
        u'[\U0001F600-\U0001F64F'
        u'\U0001F300-\U0001F5FF'
        u'\U0001F680-\U0001F6FF'
        u'\U0001F1E0-\U0001F1FF'
        u'\U00002700-\U000027BF'
        u'\U0001F900-\U0001F9FF'
        u'\U00002600-\U000026FF'
        u'\u2640-\u2642'
        u'\u2194-\u2199'
        u'\u23cf\u23e9\u25aa'
        u'\u231a\u23f0\u23f3'
        u'\ufe0f\u20e3\u200d]+',
        '', cleaned, flags=re.UNICODE
    )
    return cleaned.strip()


# ── Ollama API helpers ─────────────────────────────────────────────────────────

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
    user_input = data.user_input.strip()

    chat_memory       = load_chat_memory()
    relevant_memories = search_memory(user_input, limit=6)
    messages          = build_chat_messages(user_input, chat_memory, relevant_memories)

    async def response_stream():
        full_reply = ""

        for token in ask_ollama_stream(messages, model=CURRENT_MODEL):
            full_reply += token
            clean_token = strip_thinking(token)
            yield json.dumps({"token": clean_token}) + "\n"

        # Save chat history
        full_reply = strip_thinking(full_reply)
        chat_memory.append({"role": "user",      "content": user_input})
        chat_memory.append({"role": "assistant", "content": full_reply})
        if len(chat_memory) > 60:
            chat_memory[-60:]
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
    