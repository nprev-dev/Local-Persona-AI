"""
Copies backend/ into src-tauri/target/release/backend ready for zipping.

Doing this by hand is how the last release shipped the developer's own data: the
release folder contained memory.json with 58 real chat messages and
memory_store.db with personal facts extracted from them. Anyone who downloaded
it got that history, and saw it in their sidebar on first run.

This copies only what should ship. Everything the app generates for a user is
excluded, and a first run recreates it: personality.json seeds the default
character, aemeath_ref.wav seeds its voice, and both are verified to migrate on
a clean start.

    python tools/prepare_release.py
"""

import os
import shutil
import subprocess
import sys

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
RELEASE = os.path.join(ROOT, "src-tauri", "target", "release", "backend")

# Generated per user. Shipping any of it leaks private data or hands a new user
# somebody else's state.
EXCLUDE_FILES = {
    "memory.json",        # chat history (pre-conversation-split installs)
    "memory_store.db",    # long-term memory: facts extracted about the user
    "settings.json",      # user preferences
    "active.json",        # which persona this user last selected
}
EXCLUDE_DIRS = {
    "chats",              # every conversation the user has had
    "personas",           # user-created characters; the default is reseeded
    "__pycache__",
}

# Must exist in the shipped folder or a first run cannot build the default
# character.
REQUIRED = ["main.py", "personality.json", "aemeath_ref.wav", "index.html", "vendor"]


def build_ui() -> bool:
    script = os.path.join(ROOT, "tools", "build_ui.cjs")
    try:
        subprocess.run(["node", script], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return True
    except Exception as e:
        print(f"  UI build skipped ({e}). The shipped page will fall back to the")
        print("  source and load the 3.14 MB transformer on every launch.")
        return False


def should_skip(rel_path: str) -> bool:
    parts = rel_path.replace("\\", "/").split("/")
    if any(p in EXCLUDE_DIRS for p in parts[:-1]):
        return True
    if parts[0] in EXCLUDE_DIRS:
        return True
    return parts[-1] in EXCLUDE_FILES


def main():
    if not os.path.isdir(BACKEND):
        sys.exit(f"No backend folder at {BACKEND}")

    print("building UI...")
    build_ui()

    if os.path.isdir(RELEASE):
        shutil.rmtree(RELEASE)
    os.makedirs(RELEASE, exist_ok=True)

    copied, skipped = 0, []
    for dirpath, dirnames, filenames in os.walk(BACKEND):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for name in filenames:
            src = os.path.join(dirpath, name)
            rel = os.path.relpath(src, BACKEND)
            if should_skip(rel):
                skipped.append(rel)
                continue
            dst = os.path.join(RELEASE, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    missing = [r for r in REQUIRED if not os.path.exists(os.path.join(RELEASE, r))]
    leaked = [f for f in EXCLUDE_FILES if os.path.exists(os.path.join(RELEASE, f))]
    leaked += [d for d in EXCLUDE_DIRS if os.path.isdir(os.path.join(RELEASE, d))]

    print(f"\ncopied {copied} files to {os.path.relpath(RELEASE, ROOT)}")
    if skipped:
        print("held back (user data, recreated on first run):")
        for rel in sorted(set(skipped))[:12]:
            print(f"  {rel}")

    if missing:
        sys.exit(f"\nFAILED: shipped folder is missing {missing}")
    if leaked:
        sys.exit(f"\nFAILED: user data reached the release folder: {leaked}")

    print("\nclean. zip this folder plus localpersona.exe.")


if __name__ == "__main__":
    main()
