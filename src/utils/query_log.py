"""Append-only log of questions asked in the app.

Persistence is delegated to JSONBlobStore: durable in a database when
DATABASE_URL is configured (survives Streamlit Cloud redeploys), local JSON file
otherwise. This is a global log across all sessions/datasets, intended for the
app owner to review and export what users are asking.
"""
import os
import time
from typing import List, Dict, Any
from src.utils.logger import get_logger
from src.utils.persistence import JSONBlobStore

logger = get_logger(__name__)

DEFAULT_STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "query_log.json",
)


class QueryLogStore:
    """Stores every asked question with its dataset, provider, and timestamp."""

    def __init__(self, store_path: str = DEFAULT_STORE_PATH):
        self.store_path = store_path
        self._blob = JSONBlobStore(key="query_log", local_path=store_path)

    def log_question(self, question: str, dataset: str = "", provider: str = "") -> None:
        if not question or not question.strip():
            return
        log = self._blob.load([])
        log.append({
            "question": question.strip(),
            "dataset": dataset,
            "provider": provider,
            "timestamp": time.time(),
        })
        self._blob.save(log)
        logger.info(f"Logged question: '{question[:80]}'")

    def get_all(self) -> List[Dict[str, Any]]:
        return self._blob.load([])


# Global singleton instance
query_log_store = QueryLogStore()
