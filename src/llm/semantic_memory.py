import json
import os
import time
from typing import List, Dict, Any
from src.utils.logger import get_logger

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
    """
    def __init__(self, store_path: str = DEFAULT_STORE_PATH):
        self.store_path = store_path
        self._ensure_store_exists()
        
    def _ensure_store_exists(self):
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        if not os.path.exists(self.store_path):
            with open(self.store_path, 'w') as f:
                json.dump([], f)
                
    def _load_rules(self) -> List[Dict[str, Any]]:
        try:
            with open(self.store_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load semantic memory: {e}")
            return []
            
    def _save_rules(self, rules: List[Dict[str, Any]]):
        try:
            with open(self.store_path, 'w') as f:
                json.dump(rules, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save semantic memory: {e}")

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

# Global singleton instance
semantic_store = SemanticMemoryStore()
