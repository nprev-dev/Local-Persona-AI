# memory_engine.py — OpenJarvis-level memory system
#
# Upgrades the original SQLite/FTS5 system with:
#   Backend 1 — SQLite + FTS5  (persistent, keyword search, zero extra deps)
#   Backend 2 — BM25           (smarter keyword scoring, in-memory)
#   Backend 3 — FAISS          (semantic/vector search — finds meaning not just words)
#   Backend 4 — Hybrid RRF     (BM25 + FAISS fused — best overall quality)
#   Chunker                    (splits big text into overlapping pieces)
#   Document ingestion         (load .txt / .md / code files into memory)
#
# Graceful fallback:
#   faiss-cpu + sentence-transformers + rank-bm25 installed → Hybrid mode (best)
#   rank-bm25 only                                          → BM25 mode
#   nothing extra                                           → SQLite/FTS5 mode
#
# Public API — same function names as before, plus new additions:
#   init_memory_db()            call once at startup
#   add_memory(...)             save an AI-extracted fact
#   search_memory(query, n)     find relevant memories
#   get_all_memories()          return all stored facts
#   clear_all_memories()        wipe everything
#   ingest_text(text, source)   NEW — chunk + store raw text
#   ingest_file(path)           NEW — chunk + store a file
#   memory_count()              NEW — how many entries are stored

import sqlite3
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Optional heavy deps — graceful fallback if not installed ──────────────────
try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH         = "memory_store.db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 22MB model, 384-dim, runs on CPU in ~5ms per query
CHUNK_SIZE      = 200                   # words per chunk
CHUNK_OVERLAP   = 30                    # words to repeat at the start of the next chunk
MIN_CHUNK_SIZE  = 20                    # discard chunks smaller than this
RRF_K           = 60                    # standard Reciprocal Rank Fusion constant


# ── Data type ─────────────────────────────────────────────────────────────────
@dataclass
class RetrievalResult:
    content:    str
    score:      float = 0.0
    source:     str   = ""
    category:   str   = ""
    importance: int   = 5
    metadata:   Dict[str, Any] = field(default_factory=dict)


# ── Abstract base — every backend implements these 4 methods ──────────────────
class MemoryBackend(ABC):
    @abstractmethod
    def store(self, content: str, *, source: str = "", category: str = "",
              importance: int = 5, doc_id: str = None) -> str:
        """Store content. Returns doc_id, or empty string if duplicate/failed."""

    @abstractmethod
    def retrieve(self, query: str, *, top_k: int = 6) -> List[RetrievalResult]:
        """Return top_k most relevant results for this query."""

    @abstractmethod
    def delete(self, doc_id: str) -> bool:
        """Delete by doc_id. Returns True if it existed."""

    @abstractmethod
    def clear(self) -> None:
        """Wipe all stored entries."""


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND 1 — SQLite + FTS5
# The only backend that writes to disk. All others are in-memory and get
# warmed up from SQLite on each startup.
# ══════════════════════════════════════════════════════════════════════════════
class SQLiteBackend(MemoryBackend):
    """
    Persistent storage in memory_store.db.
    Uses SQLite's built-in FTS5 extension for full-text keyword search.
    No extra packages needed — sqlite3 ships with Python.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._conn

    def init(self):
        """Create all tables. Safe to call multiple times."""
        self.conn.executescript("""
            -- AI-extracted facts table
            CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id     TEXT    UNIQUE,
                category   TEXT    DEFAULT '',
                content    TEXT    NOT NULL,
                importance INTEGER DEFAULT 5,
                source     TEXT    DEFAULT 'chat',
                created_at TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_search
            USING fts5(doc_id UNINDEXED, content, category, source);

            -- Ingested document chunks table
            CREATE TABLE IF NOT EXISTS doc_chunks (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id     TEXT    UNIQUE,
                content    TEXT    NOT NULL,
                source     TEXT    DEFAULT '',
                created_at TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS doc_search
            USING fts5(doc_id UNINDEXED, content, source);
        """)

        # Migration: add doc_id column if upgrading from the old version of this file
        try:
            self.conn.execute("ALTER TABLE memories ADD COLUMN doc_id TEXT UNIQUE")
            self.conn.execute("UPDATE memories SET doc_id = CAST(id AS TEXT) WHERE doc_id IS NULL")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists — fine

        self.conn.commit()

    # ── Facts (AI-extracted memories) ─────────────────────────────────────────

    def store(self, content, *, source="", category="", importance=5, doc_id=None) -> str:
        content  = content.strip()
        category = category.strip().lower()
        if not content:
            return ""
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        # Exact-match deduplication
        exists = self.conn.execute(
            "SELECT doc_id FROM memories WHERE LOWER(TRIM(content)) = LOWER(TRIM(?))", (content,)
        ).fetchone()
        if exists:
            return ""

        created_at = datetime.now().isoformat(timespec="seconds")
        try:
            self.conn.execute(
                "INSERT INTO memories (doc_id, category, content, importance, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (doc_id, category, content, importance, source, created_at)
            )
            self.conn.execute(
                "INSERT INTO memory_search (doc_id, content, category, source) VALUES (?, ?, ?, ?)",
                (doc_id, content, category, source)
            )
            self.conn.commit()
            return doc_id
        except sqlite3.IntegrityError:
            return ""

    def retrieve(self, query, *, top_k=6) -> List[RetrievalResult]:
        if not query.strip():
            return []

        # Quote each word so FTS5 doesn't choke on special characters
        safe = " ".join(f'"{w}"' for w in query.split() if w and len(w) > 1)
        if not safe:
            return []

        try:
            rows = self.conn.execute("""
                SELECT m.doc_id, m.content, m.category, m.importance, m.source, ms.rank
                FROM memory_search ms
                JOIN memories m ON m.doc_id = ms.doc_id
                WHERE memory_search MATCH ?
                ORDER BY ms.rank
                LIMIT ?
            """, (safe, top_k)).fetchall()
        except sqlite3.OperationalError:
            return []

        return [
            RetrievalResult(
                content    = r[1],
                score      = max(0.0, -r[5] / 10.0),   # FTS5 rank is negative; flip it
                source     = r[4],
                category   = r[2],
                importance = r[3],
                metadata   = {"doc_id": r[0]}
            )
            for r in rows
        ]

    def delete(self, doc_id) -> bool:
        cur = self.conn.execute("DELETE FROM memories WHERE doc_id = ?", (doc_id,))
        self.conn.execute("DELETE FROM memory_search WHERE doc_id = ?", (doc_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def clear(self):
        self.conn.execute("DELETE FROM memories")
        self.conn.execute("DELETE FROM memory_search")
        self.conn.commit()

    def get_all(self) -> List[Dict]:
        rows = self.conn.execute("""
            SELECT id, category, content, importance, created_at
            FROM memories ORDER BY importance DESC, id DESC
        """).fetchall()
        return [
            {"id": r[0], "category": r[1], "content": r[2],
             "importance": r[3], "created_at": r[4]}
            for r in rows
        ]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    # ── Document chunks ────────────────────────────────────────────────────────

    def store_chunk(self, content: str, source: str = "", doc_id: str = None) -> str:
        content = content.strip()
        if not content:
            return ""
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        exists = self.conn.execute(
            "SELECT doc_id FROM doc_chunks WHERE LOWER(TRIM(content)) = LOWER(TRIM(?))", (content,)
        ).fetchone()
        if exists:
            return ""

        created_at = datetime.now().isoformat(timespec="seconds")
        try:
            self.conn.execute(
                "INSERT INTO doc_chunks (doc_id, content, source, created_at) VALUES (?, ?, ?, ?)",
                (doc_id, content, source, created_at)
            )
            self.conn.execute(
                "INSERT INTO doc_search (doc_id, content, source) VALUES (?, ?, ?)",
                (doc_id, content, source)
            )
            self.conn.commit()
            return doc_id
        except sqlite3.IntegrityError:
            return ""

    def search_chunks(self, query: str, top_k: int = 6) -> List[RetrievalResult]:
        if not query.strip():
            return []
        safe = " ".join(f'"{w}"' for w in query.split() if w and len(w) > 1)
        if not safe:
            return []
        try:
            rows = self.conn.execute("""
                SELECT c.doc_id, c.content, c.source, ds.rank
                FROM doc_search ds
                JOIN doc_chunks c ON c.doc_id = ds.doc_id
                WHERE doc_search MATCH ?
                ORDER BY ds.rank
                LIMIT ?
            """, (safe, top_k)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            RetrievalResult(
                content  = r[1],
                score    = max(0.0, -r[3] / 10.0),
                source   = r[2],
                category = "document",
                metadata = {"doc_id": r[0]}
            )
            for r in rows
        ]

    def get_all_chunks(self) -> List[Dict]:
        return [
            {"doc_id": r[0], "content": r[1], "source": r[2]}
            for r in self.conn.execute(
                "SELECT doc_id, content, source FROM doc_chunks"
            ).fetchall()
        ]

    def clear_chunks(self):
        self.conn.execute("DELETE FROM doc_chunks")
        self.conn.execute("DELETE FROM doc_search")
        self.conn.commit()

    def chunk_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0]


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND 2 — BM25 (Okapi BM25, in-memory)
# ══════════════════════════════════════════════════════════════════════════════
class BM25Backend(MemoryBackend):
    """
    Okapi BM25 keyword scoring. Better than FTS5 for short queries and rare words
    because it accounts for document length and term frequency saturation.
    In-memory only — warmed up from SQLite on startup.
    Requires: pip install rank-bm25
    """

    def __init__(self):
        self._docs: List[Dict] = []
        self._index = None      # rebuilt on every store/delete

    def _rebuild(self):
        if not self._docs or not BM25_AVAILABLE:
            return
        self._index = BM25Okapi([d["content"].lower().split() for d in self._docs])

    def store(self, content, *, source="", category="", importance=5, doc_id=None) -> str:
        content = content.strip()
        if not content:
            return ""
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        if any(d["content"].lower() == content.lower() for d in self._docs):
            return ""

        self._docs.append({
            "doc_id": doc_id, "content": content,
            "source": source, "category": category, "importance": importance
        })
        self._rebuild()
        return doc_id

    def retrieve(self, query, *, top_k=6) -> List[RetrievalResult]:
        if not self._docs or self._index is None:
            return []

        scores = self._index.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            RetrievalResult(
                content    = self._docs[i]["content"],
                score      = float(s),
                source     = self._docs[i]["source"],
                category   = self._docs[i]["category"],
                importance = self._docs[i]["importance"],
                metadata   = {"doc_id": self._docs[i]["doc_id"]}
            )
            for i, s in ranked if s > 0
        ]

    def delete(self, doc_id) -> bool:
        before = len(self._docs)
        self._docs = [d for d in self._docs if d["doc_id"] != doc_id]
        if len(self._docs) < before:
            self._rebuild()
            return True
        return False

    def clear(self):
        self._docs  = []
        self._index = None


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND 3 — FAISS (semantic / dense vector search)
# ══════════════════════════════════════════════════════════════════════════════
class FAISSBackend(MemoryBackend):
    """
    Semantic vector search. Embeds text with sentence-transformers, then searches
    with FAISS (Facebook AI Similarity Search, inner-product index).

    This finds MEANING — "how do I go faster" matches "performance optimization tips"
    even though they share zero words. FTS5 and BM25 would miss this entirely.

    In-memory only — warmed up from SQLite on startup.
    Requires: pip install faiss-cpu sentence-transformers
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        if not FAISS_AVAILABLE:
            raise RuntimeError("Run: pip install faiss-cpu sentence-transformers")
        print(f"[Memory] Loading embedding model '{model_name}'...")
        self._model = SentenceTransformer(model_name)
        self._dim   = self._model.get_sentence_embedding_dimension()
        # IndexFlatIP = inner product. On L2-normalized vectors this equals cosine similarity.
        self._index = faiss.IndexFlatIP(self._dim)
        self._docs: List[Dict] = []     # parallel list to FAISS index rows

    def _embed(self, texts: List[str]) -> "np.ndarray":
        return np.array(
            self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32
        )

    def store(self, content, *, source="", category="", importance=5, doc_id=None) -> str:
        content = content.strip()
        if not content:
            return ""
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        if any(d["content"].lower() == content.lower() for d in self._docs if not d.get("deleted")):
            return ""

        self._index.add(self._embed([content]))
        self._docs.append({
            "doc_id": doc_id, "content": content, "deleted": False,
            "source": source, "category": category, "importance": importance
        })
        return doc_id

    def retrieve(self, query, *, top_k=6) -> List[RetrievalResult]:
        if self._index.ntotal == 0:
            return []

        k = min(top_k * 3, self._index.ntotal)    # over-fetch to account for soft-deleted entries
        scores, indices = self._index.search(self._embed([query]), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._docs):
                continue
            doc = self._docs[idx]
            if doc.get("deleted"):
                continue
            results.append(RetrievalResult(
                content    = doc["content"],
                score      = float(score),
                source     = doc["source"],
                category   = doc["category"],
                importance = doc["importance"],
                metadata   = {"doc_id": doc["doc_id"]}
            ))
            if len(results) >= top_k:
                break

        return results

    def delete(self, doc_id) -> bool:
        # FAISS doesn't support per-element deletion — we soft-delete
        for doc in self._docs:
            if doc["doc_id"] == doc_id:
                doc["deleted"] = True
                return True
        return False

    def clear(self):
        self._index = faiss.IndexFlatIP(self._dim)
        self._docs  = []


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND 4 — Hybrid (BM25 + FAISS fused with Reciprocal Rank Fusion)
# ══════════════════════════════════════════════════════════════════════════════
class HybridBackend(MemoryBackend):
    """
    Combines BM25 and FAISS using Reciprocal Rank Fusion.

    RRF score for each document = sum of  1 / (k + rank_in_backend)
    across all backends.  k=60 is the standard constant from the original RRF paper.

    Why this beats both alone:
    - BM25 wins when you type exact keywords ("qwen3 context window")
    - FAISS wins when you paraphrase ("how much can the model remember")
    - RRF gives each a fair vote and takes the union of their top results
    """

    def __init__(self, sparse: MemoryBackend, dense: MemoryBackend, rrf_k: int = RRF_K):
        self.sparse = sparse    # BM25
        self.dense  = dense     # FAISS
        self.rrf_k  = rrf_k

    def store(self, content, *, source="", category="", importance=5, doc_id=None) -> str:
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        self.sparse.store(content, source=source, category=category,
                          importance=importance, doc_id=doc_id)
        self.dense.store(content,  source=source, category=category,
                         importance=importance,  doc_id=doc_id)
        return doc_id

    def retrieve(self, query, *, top_k=6) -> List[RetrievalResult]:
        fetch_k = top_k * 3

        sparse_results = self.sparse.retrieve(query, top_k=fetch_k)
        dense_results  = self.dense.retrieve(query,  top_k=fetch_k)

        rrf_scores:  Dict[str, float]           = {}
        content_map: Dict[str, RetrievalResult] = {}

        for rank, result in enumerate(sparse_results):
            key = result.metadata.get("doc_id", result.content[:60])
            rrf_scores[key]  = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            content_map[key] = result

        for rank, result in enumerate(dense_results):
            key = result.metadata.get("doc_id", result.content[:60])
            rrf_scores[key]  = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            if key not in content_map:
                content_map[key] = result

        results = []
        for key in sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:top_k]:
            r       = content_map[key]
            r.score = rrf_scores[key]
            results.append(r)

        return results

    def delete(self, doc_id) -> bool:
        return self.sparse.delete(doc_id) or self.dense.delete(doc_id)

    def clear(self):
        self.sparse.clear()
        self.dense.clear()


# ══════════════════════════════════════════════════════════════════════════════
# TEXT CHUNKER
# ══════════════════════════════════════════════════════════════════════════════
def chunk_text(text: str, source: str = "",
               chunk_size: int = CHUNK_SIZE,
               overlap:    int = CHUNK_OVERLAP,
               min_size:   int = MIN_CHUNK_SIZE) -> List[Dict]:
    """
    Split text into overlapping chunks so nothing gets cut off at a boundary.

    Example with chunk_size=5, overlap=2:
      Input:  [A B C D E F G H]
      Chunk1: [A B C D E]
      Chunk2: [D E F G H]   <- D and E repeated for continuity

    Returns: list of {"content": str, "source": str}
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks  = []
    current = []

    for para in paragraphs:
        words = para.split()

        # Single paragraph too big — use sliding window
        if len(words) > chunk_size:
            for i in range(0, len(words), chunk_size - overlap):
                window = words[i: i + chunk_size]
                if len(window) >= min_size:
                    chunks.append({"content": " ".join(window), "source": source})
            current = []
            continue

        # Adding this paragraph would overflow the current chunk — flush first
        if len(current) + len(words) > chunk_size:
            if len(current) >= min_size:
                chunks.append({"content": " ".join(current), "source": source})
            current = current[-overlap:] + words    # keep overlap tail for next chunk
        else:
            current += words

    if len(current) >= min_size:
        chunks.append({"content": " ".join(current), "source": source})

    return chunks


BINARY_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
               ".pdf", ".zip", ".exe", ".bin", ".db", ".pyc",
               ".mp3", ".mp4", ".wav", ".avi", ".mov"}


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL BACKENDS — initialized once by init_memory_db()
# ══════════════════════════════════════════════════════════════════════════════
_sqlite:       Optional[SQLiteBackend]  = None
_fact_backend: Optional[MemoryBackend] = None   # for AI-extracted facts
_doc_backend:  Optional[MemoryBackend] = None   # for document chunks


def _warm_up():
    """
    Load existing SQLite data into in-memory backends on startup.
    Needed because FAISS and BM25 are in-memory only and start empty.
    """
    if _sqlite is None or _fact_backend is _sqlite:
        return

    facts = _sqlite.get_all()
    for f in facts:
        _fact_backend.store(
            f["content"],
            source    = f.get("source",    ""),
            category  = f.get("category",  ""),
            importance= f.get("importance", 5)
        )
    if facts:
        print(f"[Memory] Loaded {len(facts)} facts into search backend")

    if _doc_backend is not _sqlite and _doc_backend is not _fact_backend:
        chunks = _sqlite.get_all_chunks()
        for c in chunks:
            _doc_backend.store(c["content"], source=c.get("source", ""), category="document")
        if chunks:
            print(f"[Memory] Loaded {len(chunks)} document chunks into search backend")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — same names as before + new additions
# ══════════════════════════════════════════════════════════════════════════════

def init_memory_db():
    """
    Initialize all backends. Already called at startup in main.py.
    Auto-selects the best available backend.
    """
    global _sqlite, _fact_backend, _doc_backend

    _sqlite = SQLiteBackend(DB_PATH)
    _sqlite.init()

    if FAISS_AVAILABLE and BM25_AVAILABLE:
        print("[Memory] Mode: Hybrid (BM25 + FAISS semantic search) ✓")
        _fact_backend = HybridBackend(BM25Backend(), FAISSBackend(EMBEDDING_MODEL))
        _doc_backend  = HybridBackend(BM25Backend(), FAISSBackend(EMBEDDING_MODEL))
    elif BM25_AVAILABLE:
        print("[Memory] Mode: BM25 (smart keyword search) ✓")
        _fact_backend = BM25Backend()
        _doc_backend  = BM25Backend()
    else:
        print("[Memory] Mode: SQLite/FTS5 — run 'pip install faiss-cpu sentence-transformers rank-bm25' for better results")
        _fact_backend = _sqlite
        _doc_backend  = _sqlite

    _warm_up()


def add_memory(category: str, content: str, importance: int = 5, source: str = "chat") -> bool:
    """
    Save an AI-extracted fact.
    Always writes to SQLite (persistent) and to the fast search backend.
    Returns True if it was new, False if it was a duplicate.
    """
    content  = content.strip()
    category = category.strip().lower()
    if not content:
        return False

    doc_id = _sqlite.store(content, source=source, category=category, importance=importance)
    if not doc_id:
        return False    # duplicate

    if _fact_backend is not _sqlite:
        _fact_backend.store(content, source=source, category=category,
                            importance=importance, doc_id=doc_id)

    return True


def search_memory(query: str, limit: int = 6) -> List[Dict]:
    """
    Find the most relevant memories for a query.
    Searches both extracted facts and ingested document chunks, then merges.
    """
    query = query.strip()
    if not query:
        return []

    fact_results = _fact_backend.retrieve(query, top_k=limit)

    # Fetch doc results from the appropriate backend
    if _doc_backend is not _fact_backend and _doc_backend is not _sqlite:
        doc_results = _doc_backend.retrieve(query, top_k=limit)
    elif _doc_backend is _sqlite:
        doc_results = _sqlite.search_chunks(query, top_k=limit)
    else:
        doc_results = []

    # Merge: facts first (they carry importance scores), then doc chunks
    seen   = set()
    merged = []

    for r in fact_results:
        key = r.content[:80].lower()
        if key not in seen:
            merged.append(r)
            seen.add(key)

    for r in doc_results:
        key = r.content[:80].lower()
        if key not in seen:
            merged.append(r)
            seen.add(key)

    # Sort by importance-weighted score so high-importance facts bubble up
    merged.sort(key=lambda r: r.score * (1.0 + r.importance * 0.1), reverse=True)

    return [
        {
            "id":         r.metadata.get("doc_id", ""),
            "category":   r.category,
            "content":    r.content,
            "importance": r.importance,
            "created_at": r.metadata.get("created_at", ""),
            "score":      round(r.score, 4)
        }
        for r in merged[:limit]
    ]


def get_all_memories() -> List[Dict]:
    """Return all stored AI-extracted facts from the persistent SQLite store."""
    return _sqlite.get_all()


def clear_all_memories():
    """Wipe all facts and document chunks from all backends."""
    _sqlite.clear()
    _sqlite.clear_chunks()
    if _fact_backend is not _sqlite:
        _fact_backend.clear()
    if _doc_backend not in (_sqlite, _fact_backend):
        _doc_backend.clear()


def memory_count() -> Dict[str, int]:
    """Return counts of stored facts and document chunks."""
    facts  = _sqlite.count()
    chunks = _sqlite.chunk_count()
    return {"facts": facts, "chunks": chunks, "total": facts + chunks}


def ingest_text(text: str, source: str = "manual") -> int:
    """
    Chunk raw text and store all chunks in the document backend.
    Returns the number of new chunks stored (0 if all duplicates).
    """
    chunks = chunk_text(text, source=source)
    stored = 0
    for chunk in chunks:
        doc_id = _sqlite.store_chunk(chunk["content"], source=chunk["source"])
        if doc_id:
            if _doc_backend is not _sqlite:
                _doc_backend.store(chunk["content"], source=chunk["source"],
                                   category="document", doc_id=doc_id)
            stored += 1
    return stored


def ingest_file(path: str) -> int:
    """
    Read a file, chunk it, and store all chunks.
    Skips binary files. Returns number of new chunks stored.
    """
    p = Path(path)
    if not p.exists() or p.suffix.lower() in BINARY_EXTS:
        return 0
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
        return ingest_text(text, source=str(p))
    except Exception:
        return 0
