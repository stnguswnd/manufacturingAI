from __future__ import annotations

from app.storage.sqlite_store import SQLiteStore


class JsonLineStore(SQLiteStore):
    """Backward-compatible alias for the SQLite-backed store."""

