"""
Persona adherence eval.

Replays a fixed multi-turn conversation against one or more local models using the
same prompt and generation settings the app itself uses, then audits each reply
against the active persona's declared rules.

Lives outside backend/ so it is never included in the shipped zip.

Usage:
    python tests/persona_eval.py
    python tests/persona_eval.py --models phi3:latest qwen2.5:7b
    python tests/persona_eval.py --persona aemeath --verbose
"""

import argparse
import os
import re
import sys

import requests

BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
BACKEND = os.path.abspath(BACKEND)
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

import persona_manager as pm
import llm_config

OLLAMA = "http://localhost:11434/api/chat"

DEFAULT_MODELS = ["phi3:latest", "qwen2.5:7b"]

# Turns 3-6 deliberately invite structured answers, which is where adherence
# historically collapsed and then stayed collapsed.
TURNS = [
    "hey",
    "I just got promoted at work today!",
    "give me three ideas for dinner tonight",
    "what's a good way to stay focused while working from home?",
    "list the pros and cons of moving to a new city",
    "can you summarize what makes a good morning routine?",
    "what are you exactly?",
]

_META_OPENERS = (
    "sure", "of course", "certainly", "absolutely", "great question",
    "i'd be happy", "i would be happy", "no problem",
)

_AI_PATTERNS = (
    r"\bas an ai\b",
    r"\blanguage model\b",
    r"\bartificial intelligence\b",
    r"\bi'?m an? ai\b",
    r"\bi am an? ai\b",
    r"\bai (?:assistant|companion|model)\b",
    r"\bchatbot\b",
    r"\bvirtual assistant\b",
    # "I'm your personal assistant", "I am an assistant"
    r"\b(?:i'?m|i am)\s+(?:your\s+|an?\s+)?(?:personal\s+|virtual\s+|digital\s+|ai\s+)?assistant\b",
    # base-model identity leaking through the persona
    r"\b(?:designed|created|developed|made|built|trained)\s+by\s+"
    r"(?:microsoft|openai|google|meta|alibaba|anthropic|mistral|deepseek)\b",
    r"\bmy (?:capabilities|programming|training data)\b",
)


def audit(text: str, persona: dict) -> list:
    """Check a reply against the constraints this persona actually declares."""
    declared = " ".join(persona.get("hard_constraints", [])).lower()
    bad = []

    if "emoji" in declared and any(ord(c) > 0x2500 for c in text):
        bad.append("EMOJI")

    if any(k in declared for k in ("ai", "assistant", "language model")):
        low = text.lower()
        if any(re.search(p, low) for p in _AI_PATTERNS):
            bad.append("AI-SELF-REF")

    if any(k in declared for k in ("bullet", "list", "numbered")):
        for line in text.splitlines():
            s = line.strip()
            if s.startswith(("- ", "* ", "• ")) or re.match(r"^\d+[.)]\s", s):
                bad.append("LIST")
                break

    stripped = text.lstrip().lower()
    if any(stripped.startswith(o) for o in _META_OPENERS):
        bad.append("META")

    if re.search(r"\*\*|^#{1,6}\s", text, flags=re.MULTILINE):
        bad.append("MARKDOWN")

    return bad


def build_system(persona: dict) -> str:
    """Mirror main.build_chat_messages' system message (no memories in eval)."""
    return (
        f"{pm.build_personality_prompt(persona)}\n\n"
        "Use the long-term memories only when relevant. "
        "Do not dump memories unless the user asks.\n\n"
        "RELEVANT LONG-TERM MEMORIES:\n"
        "No relevant long-term memories found."
    )


def chat(model: str, messages: list) -> str:
    options = dict(llm_config.CHAT_OPTIONS)
    options["num_predict"] = llm_config.CHAT_MAX_TOKENS
    r = requests.post(
        OLLAMA,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
            "keep_alive": "5m",
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def run_model(model: str, persona: dict, verbose: bool) -> tuple:
    system = build_system(persona)
    history, violated, all_flags = [], 0, []

    print(f"\n{'=' * 72}\n{model}\n{'=' * 72}")
    for i, user in enumerate(TURNS, 1):
        messages = [{"role": "system", "content": system}]
        messages += history[-10:]
        messages.append({"role": "user", "content": user})

        try:
            reply = chat(model, messages)
        except requests.RequestException as e:
            print(f"[turn {i}] REQUEST FAILED: {e}")
            return None, None

        flags = audit(reply, persona)
        if flags:
            violated += 1
            all_flags.extend(flags)

        print(f"\n[turn {i}] {user}")
        print(f"   flags: {', '.join(flags) if flags else 'clean'}")
        body = reply.strip() if verbose else reply.strip()[:340]
        print("   " + body.replace("\n", "\n   "))

        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": reply})

    return violated, all_flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--persona", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    pid = args.persona or pm.get_active_id()
    persona = pm.load_persona(pid)

    print(f"persona : {persona.get('name')} ({pid})")
    print(f"options : {llm_config.CHAT_OPTIONS}")
    print(f"tokens  : {llm_config.CHAT_MAX_TOKENS}")

    results = {}
    for model in args.models:
        violated, flags = run_model(model, persona, args.verbose)
        results[model] = (violated, flags)

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    for model, (violated, flags) in results.items():
        if violated is None:
            print(f"  {model:<20} ERROR")
            continue
        kinds = ", ".join(sorted(set(flags))) if flags else "-"
        print(f"  {model:<20} {violated}/{len(TURNS)} turns violated   [{kinds}]")


if __name__ == "__main__":
    main()
