import os
import time
from typing import List, Dict, Any, Optional
from src.utils.logger import get_logger
from src.utils.config import config
from src.utils.persistence import JSONBlobStore

logger = get_logger(__name__)

# Fallback path if data directory isn't accessible
DEFAULT_STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "golden_queries.json"
)

class GoldenQueryStore:
    """
    Razorpay-inspired Golden Queries Feedback Loop.
    Stores highly successful, high-confidence analysis pipelines (Question -> Pandas Code).
    Acts as a dynamic few-shot example repository to boost future LLM reliability.

    Persistence is delegated to JSONBlobStore, which uses a database when
    DATABASE_URL is configured (durable across redeploys) and the local JSON
    file otherwise.
    """
    def __init__(self, store_path: str = DEFAULT_STORE_PATH):
        self.store_path = store_path
        self._blob = JSONBlobStore(key="golden_queries", local_path=store_path)

    def _load_queries(self) -> List[Dict[str, Any]]:
        return self._blob.load([])

    def _save_queries(self, queries: List[Dict[str, Any]]):
        self._blob.save(queries)

    def save_golden_query(self, user_query: str, pandas_code: str, confidence_score: float, dataset_schema: str = ""):
        """
        Saves an executed query to the Golden Repository if confidence is high.
        """
        if confidence_score < 0.75:
            logger.info(f"Query confidence ({confidence_score}) too low for Golden Store. Skipping.")
            return False
            
        queries = self._load_queries()
        
        # Check for exact duplicates to avoid spam
        for q in queries:
            if q.get("user_query", "").strip().lower() == user_query.strip().lower():
                # Update if new one is better
                if confidence_score > q.get("confidence_score", 0):
                    q["pandas_code"] = pandas_code
                    q["confidence_score"] = confidence_score
                    q["timestamp"] = time.time()
                    self._save_queries(queries)
                    logger.info("Updated existing Golden Query with higher confidence version.")
                return True
                
        # Append new golden query
        queries.append({
            "user_query": user_query,
            "pandas_code": pandas_code,
            "confidence_score": confidence_score,
            "dataset_schema_summary": dataset_schema, # Optional: context for future complex matching
            "timestamp": time.time(),
            "use_count": 0
        })
        
        self._save_queries(queries)
        logger.info(f"Saved new Golden Query: '{user_query}'")
        return True

    # Filler words dropped before similarity so phrasing noise ("what is the...")
    # doesn't dilute the match; content words (metrics, dimensions) decide it.
    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "of", "by", "for", "to",
        "in", "on", "and", "or", "what", "whats", "how", "show", "me", "give",
        "list", "get", "do", "does", "we", "our", "my", "i", "can", "you",
        "please", "tell", "find", "there", "many", "much", "with",
    }

    @classmethod
    def _tokenize(cls, text: str) -> set:
        """Lowercase, strip punctuation, split on whitespace, and drop filler words."""
        import re
        tokens = re.sub(r'[^\w\s]', '', text.lower()).split()
        return {t for t in tokens if t not in cls._STOPWORDS}

    def get_cached_code(self, query: str, threshold: float = 0.85) -> Optional[Dict[str, Any]]:
        """
        Returns a previously-successful query whose phrasing is near-identical to
        `query` (token Jaccard >= threshold), so the SIMPLE path can reuse its
        pandas code instead of regenerating it. Returns None if no close match.

        The threshold is intentionally strict: only near-identical phrasings hit,
        so semantically different questions (e.g. "...by gender" vs "...by age")
        do not collide. The caller still executes the cached code through the
        validator/engine, so a stale match (wrong dataset/columns) fails safely
        and falls back to normal generation.
        """
        target = self._tokenize(query)
        if not target:
            return None

        best, best_sim = None, 0.0
        for q in self._load_queries():
            toks = self._tokenize(q.get("user_query", ""))
            if not toks:
                continue
            sim = len(target & toks) / len(target | toks)
            if sim > best_sim:
                best, best_sim = q, sim

        if best and best_sim >= threshold and best.get("pandas_code"):
            logger.info(f"Golden cache hit (similarity={best_sim:.2f}) for query: '{query}'")
            return best
        return None

    def get_relevant_examples(self, current_query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most similar past successful queries to use as few-shot examples.
        Uses Jaccard token similarity for lightweight, dependency-free matching.
        """
        queries = self._load_queries()
        if not queries:
            return []
            
        def tokenize(text: str) -> set:
            # Very basic tokenization: lowercase, remove punctuation, split by space
            import re
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            return set(text.split())
            
        target_tokens = tokenize(current_query)
        
        scored_queries = []
        for q in queries:
            q_tokens = tokenize(q.get("user_query", ""))
            
            # Jaccard similarity
            intersection = len(target_tokens.intersection(q_tokens))
            union = len(target_tokens.union(q_tokens))
            similarity = intersection / union if union > 0 else 0
            
            # Weight heavily by confidence score
            final_score = similarity * (q.get("confidence_score", 0.8))
            
            scored_queries.append({
                "score": final_score,
                "query": q
            })
            
        # Sort by score descending and take top K
        scored_queries.sort(key=lambda x: x["score"], reverse=True)
        
        # Filter out 0 similarity unless it's just grabbing popular ones
        best_matches = [item["query"] for item in scored_queries if item["score"] > 0][:top_k]
        
        # If no semantic matches, just return the highest confidence ones as generic examples
        if not best_matches:
            queries.sort(key=lambda x: x.get("confidence_score", 0), reverse=True)
            best_matches = queries[:top_k]
            
        return best_matches

# Global singleton instance
golden_store = GoldenQueryStore()
