import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Standard directory to store uploaded datasets and dictionaries
DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data")
)

class ContextManager:
    """
    Manages session and persistent context including:
    - Saving and loading uploaded CSV datasets
    - Saving and loading optional data dictionaries
    - Initializing and storing metadata profiles
    - Tracking conversation history and active analytics logs
    """
    def __init__(self):
        # Ensure the data directory exists inside the workspace
        os.makedirs(DATA_DIR, exist_ok=True)
        logger.info(f"ContextManager initialized. Data files stored in: {DATA_DIR}")

    def save_uploaded_file(self, file_name: str, file_bytes: bytes) -> str:
        """
        Saves uploaded file bytes to the local data directory.
        
        Args:
            file_name: Name of the uploaded file.
            file_bytes: Byte content of the file.
            
        Returns:
            The absolute path of the saved file.
        """
        target_path = os.path.join(DATA_DIR, file_name)
        logger.info(f"Saving uploaded file '{file_name}' to {target_path}")
        with open(target_path, "wb") as f:
            f.write(file_bytes)
        return target_path

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Loads a CSV dataset from a file path into a pandas DataFrame.
        
        Args:
            file_path: Absolute path to the CSV file.
            
        Returns:
            A pandas DataFrame.
        """
        logger.info(f"Loading dataset from: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        # Standard load
        return pd.read_csv(file_path)

    def delete_all_data(self):
        """Removes all saved uploads in the data directory."""
        logger.info("Clearing context data uploads...")
        for filename in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    logger.info(f"Deleted context file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")

# Instantiated single manager
context_manager = ContextManager()
