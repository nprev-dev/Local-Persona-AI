# persona_manager.py
#
# Manages multiple persona files in a personas/ folder, tracks which one is
# active, and handles migrating the old single personality.json setup.
#
# Each persona is a JSON file at:   personas/<id>.json
# Each persona's voice clip lives:  personas/voices/<id>.wav
# The active persona id is stored:  active.json   -> {"active": "<id>"}
#
# Chat history and long-term memory remain shared/global (handled in main.py).

import os
import re
import json
import shutil

# ── Paths ────────────────────────────────────────────────────────────────────
PERSONAS_DIR = "personas"
VOICES_DIR   = os.path.join(PERSONAS_DIR, "voices")
ACTIVE_FILE  = "active.json"

# Legacy files (pre-multi-persona) used for one-time migration.
LEGACY_PERSONALITY = "personality.json"
LEGACY_VOICE       = "aemeath_ref.wav"

DEFAULT_VsTYLE = "expressive"


def _default_persona(pid: str = "assistant") -> dict:
    return {
        "id": pid,
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
        "language": "English",
        "voice_clip": "",   # path to this persona's reference clip
    }


def slugify(name: str) -> str:
    """Turn a display name into a safe file id, e.g. 'Iron Man' -> 'iron_man'."""
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "persona"


# ── Folder setup + migration ─────────────────────────────────────────────────

def ensure_setup():
    """
    Make sure the personas/ structure exists. If it's the first run after the
    multi-persona upgrade, migrate the old personality.json into it.
    """
    os.makedirs(PERSONAS_DIR, exist_ok=True)
    os.makedirs(VOICES_DIR, exist_ok=True)

    existing = list_persona_ids()
    if existing:
        # Already have personas; just make sure an active one is set.
        if not _read_active() or _read_active() not in existing:
            _write_active(existing[0])
        return

    # No personas yet — migrate from the legacy single-file setup if present.
    if os.path.exists(LEGACY_PERSONALITY):
        try:
            with open(LEGACY_PERSONALITY, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = _default_persona("aemeath")

        pid = slugify(data.get("name", "aemeath"))
        data["id"] = pid

        # Migrate the voice clip if it exists.
        if os.path.exists(LEGACY_VOICE):
            dest = os.path.join(VOICES_DIR, f"{pid}.wav")
            try:
                shutil.copyfile(LEGACY_VOICE, dest)
                data["voice_clip"] = dest
            except Exception:
                data["voice_clip"] = LEGACY_VOICE
        else:
            data.setdefault("voice_clip", "")

        save_persona(pid, data)
        _write_active(pid)
        print(f"[Personas] Migrated legacy personality.json -> personas/{pid}.json")
    else:
        # Nothing to migrate — create a default persona.
        d = _default_persona("assistant")
        save_persona("assistant", d)
        _write_active("assistant")
        print("[Personas] Created default persona")


# ── Low-level read/write ─────────────────────────────────────────────────────

def _persona_path(pid: str) -> str:
    return os.path.join(PERSONAS_DIR, f"{pid}.json")


def list_persona_ids() -> list:
    if not os.path.isdir(PERSONAS_DIR):
        return []
    ids = []
    for fn in os.listdir(PERSONAS_DIR):
        if fn.endswith(".json"):
            ids.append(fn[:-5])
    return sorted(ids)


def load_persona(pid: str) -> dict:
    path = _persona_path(pid)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["id"] = pid
    data.setdefault("voice_clip", "")
    return data


def save_persona(pid: str, data: dict):
    data = dict(data)
    data["id"] = pid
    with open(_persona_path(pid), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def delete_persona(pid: str) -> bool:
    """Delete a persona file + its voice clip. Won't delete the last remaining one."""
    ids = list_persona_ids()
    if pid not in ids:
        return False
    if len(ids) <= 1:
        return False  # never delete the last persona

    try:
        os.remove(_persona_path(pid))
    except FileNotFoundError:
        pass

    voice = os.path.join(VOICES_DIR, f"{pid}.wav")
    if os.path.exists(voice):
        try:
            os.remove(voice)
        except OSError:
            pass

    # If we deleted the active persona, switch to another one.
    if _read_active() == pid:
        _write_active(list_persona_ids()[0])
    return True


# ── Active persona tracking ──────────────────────────────────────────────────

def _read_active() -> str:
    try:
        with open(ACTIVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("active", "")
    except Exception:
        return ""


def _write_active(pid: str):
    with open(ACTIVE_FILE, "w", encoding="utf-8") as f:
        json.dump({"active": pid}, f, indent=2)


def get_active_id() -> str:
    pid = _read_active()
    ids = list_persona_ids()
    if pid in ids:
        return pid
    return ids[0] if ids else ""


def set_active(pid: str) -> bool:
    if pid not in list_persona_ids():
        return False
    _write_active(pid)
    return True


def get_active_persona() -> dict:
    pid = get_active_id()
    if not pid:
        return _default_persona()
    return load_persona(pid)


# ── Prompt building (same logic as before, now per-persona) ──────────────────

def build_personality_prompt(p: dict) -> str:
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


def list_personas_summary() -> list:
    """Lightweight list for the UI switcher."""
    out = []
    active = get_active_id()
    for pid in list_persona_ids():
        try:
            p = load_persona(pid)
        except Exception:
            continue
        out.append({
            "id": pid,
            "name": p.get("name", pid),
            "avatar_color": p.get("avatar_color", "#22d3ee"),
            "avatar_initial": p.get("avatar_initial", (p.get("name", "A")[:1].upper())),
            "active": pid == active,
        })
    return out
