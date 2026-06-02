import os
import time
from typing import List, Dict, Any
from src.utils.logger import get_logger
from src.utils.persistence import JSONBlobStore

logger = get_logger(__name__)

DEFAULT_STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "semantic_memory.json"
)

class SemanticMemoryStore:
    """
    Persistent Knowledge Base that learns and stores business rules
    and semantic preferences directly from user interactions.

    Persistence is delegated to JSONBlobStore, which uses a database when
    DATABASE_URL is configured (durable across redeploys) and the local JSON
    file otherwise.
    """
    def __init__(self, store_path: str = DEFAULT_STORE_PATH):
        self.store_path = store_path
        self._blob = JSONBlobStore(key="semantic_memory", local_path=store_path)

    def _load_rules(self) -> List[Dict[str, Any]]:
        return self._blob.load([])

    def _save_rules(self, rules: List[Dict[str, Any]]):
        self._blob.save(rules)

    def save_rule(self, rule_text: str):
        """Saves a new business rule to memory."""
        rules = self._load_rules()
        
        # Avoid exact duplicates
        for r in rules:
            if r.get("rule", "").strip().lower() == rule_text.strip().lower():
                logger.info("Rule already exists in semantic memory.")
                return False
                
        rules.append({
            "rule": rule_text.strip(),
            "timestamp": time.time()
        })
        
        self._save_rules(rules)
        logger.info(f"Learned and saved new business rule: '{rule_text}'")
        return True

    def get_all_rules(self) -> List[str]:
        """Retrieves all active business rules."""
        rules = self._load_rules()
        return [r["rule"] for r in rules]

    def delete_rule(self, index: int) -> bool:
        """Deletes the rule at the given 0-based index (same order as get_all_rules)."""
        rules = self._load_rules()
        if 0 <= index < len(rules):
            removed = rules.pop(index)
            self._save_rules(rules)
            logger.info(f"Deleted business rule: '{removed.get('rule', '')}'")
            return True
        logger.warning(f"delete_rule: index {index} out of range (have {len(rules)} rules).")
        return False

    def clear_all_rules(self) -> int:
        """Removes every learned rule. Returns the number of rules deleted."""
        count = len(self._load_rules())
        self._save_rules([])
        logger.info(f"Cleared all {count} business rule(s).")
        return count

# Global singleton instance
semantic_store = SemanticMemoryStore()
