"""
Dynamic Memory Window Service for token-aware conversation history management.
Optimized for low-resource environments like i5-4200U.
"""

from typing import List, Dict, Any
from app.engine.config import settings
from app.services.token_service import token_counter
import structlog

logger = structlog.get_logger()

class MemoryManager:
    """
    Manages conversation history using a sliding token window.
    Ensures that history fits within the allocated token budget.
    """
    
    def __init__(self, max_tokens: int = None):
        self.max_tokens = max_tokens or settings.MEMORY_WINDOW_TOKENS
        self.logger = logger.bind(service="MemoryManager")

    def get_sliding_window(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Returns a subset of the history that fits within the token budget.
        Keeps the most recent messages.
        """
        if not history:
            return []

        trimmed_history = []
        current_tokens = 0
        
        # Iterate backwards through history to keep most recent messages
        for message in reversed(history):
            content = message.get("content", "")
            tokens = token_counter.count(content)
            
            if current_tokens + tokens <= self.max_tokens:
                trimmed_history.insert(0, message)
                current_tokens += tokens
            else:
                self.logger.info(
                    "Memory window limit reached, truncating history",
                    limit=self.max_tokens,
                    current_tokens=current_tokens,
                    trimmed_count=len(history) - len(trimmed_history)
                )
                break
                
        return trimmed_history

# Global memory manager instance
memory_manager = MemoryManager()
