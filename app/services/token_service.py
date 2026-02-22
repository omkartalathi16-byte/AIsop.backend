"""
Lightweight Token Counter Utility.
Separated from nodes to avoid circular imports.
"""

class TokenCounter:
    """
    CPU-optimized token counter.
    Uses llama tokenizer when available, falls back to approximation.
    """
    
    def __init__(self):
        self._llm = None
        self._tokenizer = None
    
    def set_llm(self, llm):
        """Set LLM instance for tokenizer access"""
        self._llm = llm
        if hasattr(llm, 'tokenize'):
            self._tokenizer = llm.tokenize
    
    def count(self, text: str) -> int:
        """Count tokens in text"""
        if self._tokenizer:
            try:
                return len(self._tokenizer(text.encode()))
            except Exception:
                pass
        
        # Fallback: character-based estimation
        # ~4 chars per token for English
        return len(text) // 4 + 1

# Global token counter
token_counter = TokenCounter()
