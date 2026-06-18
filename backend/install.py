import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import threading
import sys
import shutil
import os
import urllib.request

# ── Models ─────────────────────────────────────────────────────────────────────

CHAT_MODELS = [
    {
        "id":          "llama3.1:8b",
        "label":       "llama3.1:8b",
        "description": "Recommended — best personality following",
        "vram":        "8GB VRAM",
        "default":     True,
    },
    {
        "id":          "qwen2.5:7b",
        "label":       "qwen2.5:7b",
        "description": "Better reasoning and instruction following",
        "vram":        "8GB VRAM",
        "default":     False,
    },
    {
        "id":          "llama3.2:3b",
        "label":       "llama3.2:3b",
        "description": "Lightweight — lower VRAM, weaker personality",
        "vram":        "4GB VRAM",
        "default":     False,
    },
]

MEMORY_MODEL = "phi3"

# ── PyTorch (CUDA build) ────────────────────────────────────────────────────────
# The CUDA builds of torch are NOT on PyPI — they only resolve from PyTorch's own
# index. They must be installed separately BEFORE requirements.txt, otherwise a
# plain `pip install -r requirements.txt` fails to resolve torch and aborts the
# entire install (which is why uvicorn and everything else ends up missing).

TORCH_PACKAGES   = ["torch==2.6.0", "torchvision", "torchaudio"]
TORCH_INDEX_URL  = "https://download.pytorch.org/whl/cu124"

# ── Vendored front-end libraries (served locally so the UI works offline) ───────
# Downloaded into backend/vendor/ . Only fetched if missing, so this is a no-op
# when the repo already shipped them and a repair when they're absent.

VENDOR_FILES = [
    {
        "path": "vendor/react.js",
        "url":  "https://unpkg.com/react@18/umd/react.production.min.js",
    },
    {
        "path": "vendor/react-dom.js",
        "url":  "https://unpkg.com/react-dom@18/umd/react-dom.production.min.js",
    },
    {
        "path": "vendor/babel.min.js",
        "url":  "https://unpkg.com/@babel/standalone/babel.min.js",
    },
    {
        "path": "vendor/tabler-icons.min.css",
        "url":  "https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.19.0/dist/tabler-icons.min.css",
    },
    {
        "path": "vendor/fonts/tabler-icons.woff2",
        "url":  "https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.19.0/dist/fonts/tabler-icons.woff2",
    },
    {
        "path": "vendor/fonts/tabler-icons.woff",
        "url":  "https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.19.0/dist/fonts/tabler-icons.woff",
    },
    {
        "path": "vendor/fonts/tabler-icons.ttf",
        "url":  "https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.19.0/dist/fonts/tabler-icons.ttf",
    },
]

# ── App ────────────────────────────────────────────────────────────────────────

class InstallerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Local Persona AI — Setup")
        self.root.geometry("600x720")
        self.root.resizable(False, False)
        self.root.configure(bg="#0f1115")

        self.selected_model = tk.StringVar(value=CHAT_MODELS[0]["id"])
        self.install_deps   = tk.BooleanVar(value=True)
        self.pull_memory    = tk.BooleanVar(value=True)
        self.fetch_vendor   = tk.BooleanVar(value=True)
        self.installing     = False

        self._build_ui()
        self._check_requirements()

    def _build_ui(self):
        BG       = "#0f1115"
        CARD     = "#1a1d24"
        ACCENT   = "#7c6ff7"
        TEXT     = "#e8e8e8"
        SUBTEXT  = "#888"
        SUCCESS  = "#4caf74"
        WARNING  = "#e8a838"
        ERROR    = "#e85454"
        FONT     = ("Segoe UI", 10)
        FONT_SM  = ("Segoe UI", 9)
        FONT_LG  = ("Segoe UI", 13, "bold")

        self.colors = {
            "bg": BG, "card": CARD, "accent": ACCENT,
            "text": TEXT, "subtext": SUBTEXT,
            "success": SUCCESS, "warning": WARNING, "error": ERROR
        }

        # ── Header ──
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=24, pady=(20, 4))
        tk.Label(header, text="Local Persona AI", font=("Segoe UI", 16, "bold"),
                 bg=BG, fg=TEXT).pack(anchor="w")
        tk.Label(header, text="Setup & Dependency Installer",
                 font=FONT_SM, bg=BG, fg=SUBTEXT).pack(anchor="w")

        ttk.Separator(self.root).pack(fill="x", padx=24, pady=(10, 0))

        # ── Requirements check ──
        req_frame = tk.Frame(self.root, bg=CARD, bd=0)
        req_frame.pack(fill="x", padx=24, pady=(12, 0))
        tk.Label(req_frame, text="System Requirements",
                 font=("Segoe UI", 10, "bold"), bg=CARD, fg=TEXT).pack(anchor="w", padx=14, pady=(10, 4))

        self.req_python = self._req_row(req_frame, "Python 3.11")
        self.req_ollama = self._req_row(req_frame, "Ollama")
        tk.Frame(req_frame, bg=CARD, height=8).pack()

        # ── Model selection ──
        model_frame = tk.Frame(self.root, bg=CARD)
        model_frame.pack(fill="x", padx=24, pady=(10, 0))
        tk.Label(model_frame, text="Chat Model",
                 font=("Segoe UI", 10, "bold"), bg=CARD, fg=TEXT).pack(anchor="w", padx=14, pady=(10, 4))

        for m in CHAT_MODELS:
            row = tk.Frame(model_frame, bg=CARD)
            row.pack(fill="x", padx=14, pady=2)
            rb = tk.Radiobutton(
                row, variable=self.selected_model, value=m["id"],
                bg=CARD, fg=TEXT, selectcolor=CARD,
                activebackground=CARD, activeforeground=TEXT,
                font=("Segoe UI", 10, "bold"), text=m["label"],
                highlightthickness=0
            )
            rb.pack(side="left")
            tk.Label(row, text=f"  {m['description']}",
                     font=FONT_SM, bg=CARD, fg=SUBTEXT).pack(side="left")
            tk.Label(row, text=m["vram"],
                     font=FONT_SM, bg=CARD, fg=ACCENT).pack(side="right", padx=4)

        tk.Frame(model_frame, bg=CARD, height=8).pack()

        # ── Options ──
        opt_frame = tk.Frame(self.root, bg=CARD)
        opt_frame.pack(fill="x", padx=24, pady=(10, 0))
        tk.Label(opt_frame, text="Install Options",
                 font=("Segoe UI", 10, "bold"), bg=CARD, fg=TEXT).pack(anchor="w", padx=14, pady=(10, 4))

        self._check_row(opt_frame, "Install Python dependencies (requirements.txt)", self.install_deps)
        self._check_row(opt_frame, f"Pull memory judge model ({MEMORY_MODEL})", self.pull_memory)
        self._check_row(opt_frame, "Download UI libraries for offline use (repairs vendor/)", self.fetch_vendor)
        tk.Frame(opt_frame, bg=CARD, height=8).pack()

        # ── Log ──
        log_frame = tk.Frame(self.root, bg=BG)
        log_frame.pack(fill="both", expand=True, padx=24, pady=(12, 0))
        tk.Label(log_frame, text="Log", font=("Segoe UI", 9, "bold"),
                 bg=BG, fg=SUBTEXT).pack(anchor="w")
        self.log = scrolledtext.ScrolledText(
            log_frame, height=7, bg="#13151c", fg="#c8c8c8",
            font=("Consolas", 9), bd=0, relief="flat",
            insertbackground=TEXT, wrap="word"
        )
        self.log.pack(fill="both", expand=True, pady=(4, 0))
        self.log.configure(state="disabled")

        # ── Install button ──
        btn_frame = tk.Frame(self.root, bg=BG)
        btn_frame.pack(fill="x", padx=24, pady=(12, 20))

        self.progress = ttk.Progressbar(btn_frame, mode="indeterminate", length=400)
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 12))

        self.btn = tk.Button(
            btn_frame, text="Install", font=("Segoe UI", 10, "bold"),
            bg=ACCENT, fg="white", activebackground="#6a5fe0",
            activeforeground="white", bd=0, padx=20, pady=8,
            cursor="hand2", command=self._start_install
        )
        self.btn.pack(side="right")

    def _req_row(self, parent, label):
        row = tk.Frame(parent, bg=parent["bg"])
        row.pack(fill="x", padx=14, pady=1)
        tk.Label(row, text=label, font=("Segoe UI", 9),
                 bg=parent["bg"], fg=self.colors["text"]).pack(side="left")
        status = tk.Label(row, text="Checking...", font=("Segoe UI", 9),
                          bg=parent["bg"], fg=self.colors["subtext"])
        status.pack(side="right")
        return status

    def _check_row(self, parent, label, var):
        row = tk.Frame(parent, bg=parent["bg"])
        row.pack(fill="x", padx=14, pady=2)
        cb = tk.Checkbutton(
            row, variable=var, text=label,
            bg=parent["bg"], fg=self.colors["text"],
            selectcolor=parent["bg"], activebackground=parent["bg"],
            activeforeground=self.colors["text"], font=("Segoe UI", 9),
            highlightthickness=0
        )
        cb.pack(side="left")

    def _check_requirements(self):
        # Python version
        v = sys.version_info
        if v.major == 3 and v.minor == 11:
            self.req_python.config(text=f"Python {v.major}.{v.minor}.{v.micro} ✓",
                                   fg=self.colors["success"])
        elif v.major == 3 and v.minor > 11:
            self.req_python.config(text=f"Python {v.major}.{v.minor} (untested)",
                                   fg=self.colors["warning"])
        else:
            self.req_python.config(text=f"Python {v.major}.{v.minor} — need 3.11",
                                   fg=self.colors["error"])

        # Ollama
        if shutil.which("ollama"):
            self.req_ollama.config(text="Found ✓", fg=self.colors["success"])
        else:
            self.req_ollama.config(text="Not found — install from ollama.com",
                                   fg=self.colors["error"])

    def _log(self, msg, color=None):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _start_install(self):
        if self.installing:
            return
        self.installing = True
        self.btn.config(state="disabled", text="Installing...")
        self.progress.start(12)
        threading.Thread(target=self._run_install, daemon=True).start()

    def _run_install(self):
        success = True

        # 1. PyTorch (CUDA build) — must come BEFORE requirements.txt
        if self.install_deps.get():
            self._log("→ Installing PyTorch (CUDA build)...")
            self._log("  (large download, ~2.5GB — this takes a while)")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 *TORCH_PACKAGES, "--index-url", TORCH_INDEX_URL],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace"
            )
            if result.returncode == 0:
                self._log("  ✓ PyTorch installed")
            else:
                self._log(f"  ✗ Failed:\n{(result.stderr or '')[-400:]}")
                success = False

        # 2. Remaining Python deps
        if self.install_deps.get():
            self._log("→ Installing Python dependencies...")
            req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", req_path],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace"
            )
            if result.returncode == 0:
                self._log("  ✓ Dependencies installed")
            else:
                self._log(f"  ✗ Failed:\n{(result.stderr or '')[-400:]}")
                success = False

        # 3. Chat model
        model = self.selected_model.get()
        self._log(f"→ Pulling chat model: {model}")
        self._log("  (this may take a while depending on your connection)")
        result = subprocess.run(
            ["ollama", "pull", model],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace"
        )
        if result.returncode == 0:
            self._log(f"  ✓ {model} ready")
        else:
            self._log(f"  ✗ Failed to pull {model}:\n{(result.stderr or '')[-300:]}")
            success = False

        # 4. Memory judge model
        if self.pull_memory.get():
            self._log(f"→ Pulling memory judge model: {MEMORY_MODEL}")
            result = subprocess.run(
                ["ollama", "pull", MEMORY_MODEL],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace"
            )
            if result.returncode == 0:
                self._log(f"  ✓ {MEMORY_MODEL} ready")
            else:
                self._log(f"  ✗ Failed to pull {MEMORY_MODEL}:\n{(result.stderr or '')[-300:]}")
                success = False

        # 5. Vendor UI libraries (offline support)
        if self.fetch_vendor.get():
            self._log("→ Checking offline UI libraries...")
            base = os.path.dirname(__file__)
            missing = [f for f in VENDOR_FILES
                       if not os.path.exists(os.path.join(base, f["path"]))]

            if not missing:
                self._log("  ✓ All UI libraries already present")
            else:
                self._log(f"  Downloading {len(missing)} missing file(s)...")
                for f in missing:
                    dest = os.path.join(base, f["path"])
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    try:
                        urllib.request.urlretrieve(f["url"], dest)
                        self._log(f"  ✓ {f['path']}")
                    except Exception as e:
                        self._log(f"  ✗ {f['path']} — {e}")
                        success = False

        # Done
        self.root.after(0, self._finish, success)

    def _finish(self, success):
        self.progress.stop()
        self.installing = False
        if success:
            self.btn.config(
                text="Done ✓", bg=self.colors["success"],
                state="disabled"
            )
            self._log("\n✓ Setup complete. You can now launch localpersona.exe")
        else:
            self.btn.config(
                text="Retry", bg=self.colors["error"],
                state="normal", command=self._start_install
            )
            self._log("\n✗ Setup finished with errors. Check the log above.")


# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = InstallerApp(root)
    root.mainloop()