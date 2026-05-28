import logging
from src.utils.config import config

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance.
    
    Args:
        name: The name of the logger (typically __name__)
        
    Returns:
        A configured logging.Logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if the logger doesn't already have handlers
    if not logger.handlers:
        logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
        
        # Create console handler and set format
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        logger.addHandler(ch)
        
    return logger
