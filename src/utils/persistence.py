"""Durable persistence layer for the app's runtime-learned stores.

The semantic-memory and golden-query stores are small JSON blobs (a list of
dicts). On ephemeral hosts (e.g. Streamlit Community Cloud) the local filesystem
is wiped on every redeploy/restart, so a local JSON file does not persist.

`JSONBlobStore` transparently picks a backend:

- If ``DATABASE_URL`` is set (e.g. a hosted Postgres like Supabase/Neon, or
  ``sqlite:///abs/path.db``), the blob is stored in a ``kv_store`` table keyed by
  name. A network database persists across redeploys → real durable learning.
- Otherwise it falls back to the original local JSON file behavior, so local
  development and unconfigured deploys keep working unchanged.

The store always reads/writes the whole blob (these collections are tiny), which
keeps the calling stores' existing load-all / save-all logic intact.
"""
import os
import json
import time
import threading
from typing import Any
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

# ON CONFLICT(key) upsert is supported by both SQLite and PostgreSQL.
_UPSERT_SQL = (
    "INSERT INTO kv_store (key, value, updated_at) VALUES (:k, :v, :t) "
    "ON CONFLICT (key) DO UPDATE SET value = :v, updated_at = :t"
)


class JSONBlobStore:
    """Loads/saves one JSON-serializable blob under ``key``, with a DB or local-file backend."""

    def __init__(self, key: str, local_path: str):
        self.key = key
        self.local_path = local_path
        self._engine = None
        self._lock = threading.Lock()

        if DATABASE_URL:
            try:
                from sqlalchemy import create_engine, text
                self._engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)
                with self._engine.begin() as conn:
                    conn.execute(text(
                        "CREATE TABLE IF NOT EXISTS kv_store "
                        "(key TEXT PRIMARY KEY, value TEXT, updated_at DOUBLE PRECISION)"
                    ))
                logger.info(f"JSONBlobStore '{key}': using database backend (durable).")
            except Exception as e:
                logger.error(f"JSONBlobStore '{key}': DB backend init failed ({e}); using local file.")
                self._engine = None

        if self._engine is None:
            os.makedirs(os.path.dirname(self.local_path), exist_ok=True)

    @property
    def is_durable(self) -> bool:
        """True when a database backend is attached (survives redeploys);
        False when falling back to the ephemeral local JSON file."""
        return self._engine is not None

    def load(self, default: Any) -> Any:
        """Return the stored blob, or ``default`` if absent/unreadable."""
        if self._engine is not None:
            try:
                from sqlalchemy import text
                with self._engine.begin() as conn:
                    row = conn.execute(
                        text("SELECT value FROM kv_store WHERE key = :k"), {"k": self.key}
                    ).fetchone()
                if row and row[0]:
                    return json.loads(row[0])
                return default
            except Exception as e:
                logger.error(f"JSONBlobStore '{self.key}': DB load failed ({e}).")
                return default

        try:
            if not os.path.exists(self.local_path):
                return default
            with open(self.local_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"JSONBlobStore '{self.key}': local load failed ({e}).")
            return default

    def save(self, data: Any) -> None:
        """Persist the whole blob."""
        if self._engine is not None:
            try:
                from sqlalchemy import text
                payload = json.dumps(data)
                with self._lock, self._engine.begin() as conn:
                    conn.execute(text(_UPSERT_SQL), {"k": self.key, "v": payload, "t": time.time()})
                return
            except Exception as e:
                logger.error(f"JSONBlobStore '{self.key}': DB save failed ({e}); falling back to local file.")

        try:
            with open(self.local_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"JSONBlobStore '{self.key}': local save failed ({e}).")
