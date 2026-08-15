# chat_store.py
#
# One file per conversation at chats/<id>.json, mirroring how personas are kept.
#
# Before this existed the whole app shared a single memory.json, so starting a
# "New chat" in the UI did nothing to what the model remembered: it still saw the
# previous conversation's last ten turns.
#
# Long-term memory stays global and lives in memory_engine. Only the turn-by-turn
# transcript is per conversation.

import json
import os
import re
import uuid
from datetime import datetime

CHATS_DIR = "chats"
LEGACY_HISTORY = "memory.json"

TITLE_MAX = 60


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _safe_id(cid: str) -> str:
    """Conversation ids reach the filesystem, so keep them to a known alphabet."""
    return re.sub(r"[^A-Za-z0-9_-]", "", str(cid or ""))[:64]


def _path(cid: str) -> str:
    return os.path.join(CHATS_DIR, f"{_safe_id(cid)}.json")


def title_from(text: str) -> str:
    text = " ".join(str(text or "").split())
    if not text:
        return "New chat"
    return text[:TITLE_MAX] + ("…" if len(text) > TITLE_MAX else "")


def _blank(cid: str, title: str = "New chat") -> dict:
    return {
        "id": _safe_id(cid),
        "title": title,
        "created_at": _now(),
        "updated_at": _now(),
        "messages": [],
    }


# ── Setup and one-time migration ─────────────────────────────────────────────

def ensure_setup():
    os.makedirs(CHATS_DIR, exist_ok=True)
    if list_ids():
        return

    # Carry an existing single-history install into its own conversation rather
    # than stranding it.
    try:
        with open(LEGACY_HISTORY, "r", encoding="utf-8") as f:
            legacy = json.load(f)
    except Exception:
        legacy = []

    if not isinstance(legacy, list) or not legacy:
        return

    first_user = next(
        (m.get("content", "") for m in legacy
         if isinstance(m, dict) and m.get("role") == "user"),
        "",
    )
    cid = new_id()
    chat = _blank(cid, title_from(first_user) if first_user else "Imported chat")
    chat["messages"] = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in legacy
        if isinstance(m, dict) and m.get("content")
    ]
    save(cid, chat)
    print(f"[Chats] Migrated {len(chat['messages'])} messages from memory.json")


# ── Read / write ─────────────────────────────────────────────────────────────

def list_ids() -> list:
    if not os.path.isdir(CHATS_DIR):
        return []
    return [fn[:-5] for fn in os.listdir(CHATS_DIR) if fn.endswith(".json")]


def exists(cid: str) -> bool:
    return bool(_safe_id(cid)) and os.path.exists(_path(cid))


def load(cid: str) -> dict:
    """Always returns a usable conversation, even if the file is missing or corrupt."""
    cid = _safe_id(cid)
    if not cid:
        return _blank(new_id())
    try:
        with open(_path(cid), "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except Exception:
        return _blank(cid)

    data["id"] = cid
    data.setdefault("title", "New chat")
    data.setdefault("created_at", _now())
    data.setdefault("updated_at", data["created_at"])
    msgs = data.get("messages")
    data["messages"] = [
        m for m in msgs
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")
    ] if isinstance(msgs, list) else []
    return data


def save(cid: str, data: dict):
    os.makedirs(CHATS_DIR, exist_ok=True)
    data = dict(data)
    data["id"] = _safe_id(cid)
    data["updated_at"] = _now()
    with open(_path(cid), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def messages(cid: str) -> list:
    return load(cid)["messages"]


def append_turn(cid: str, user_input: str, reply: str) -> dict:
    """Add one user/assistant exchange, creating the conversation if needed."""
    chat = load(cid)
    if not chat["messages"] and chat["title"] == "New chat":
        chat["title"] = title_from(user_input)
    chat["messages"].append({"role": "user", "content": user_input})
    chat["messages"].append({"role": "assistant", "content": reply})
    save(cid, chat)
    return chat


def rename(cid: str, title: str) -> bool:
    if not exists(cid):
        return False
    chat = load(cid)
    chat["title"] = title_from(title)
    save(cid, chat)
    return True


def delete(cid: str) -> bool:
    if not exists(cid):
        return False
    try:
        os.remove(_path(cid))
    except OSError:
        return False
    return True


def summaries() -> list:
    """Lightweight list for the sidebar, newest activity first."""
    out = []
    for cid in list_ids():
        chat = load(cid)
        out.append({
            "id": cid,
            "title": chat["title"],
            "created_at": chat["created_at"],
            "updated_at": chat["updated_at"],
            "message_count": len(chat["messages"]),
        })
    out.sort(key=lambda c: c["updated_at"], reverse=True)
    return out
