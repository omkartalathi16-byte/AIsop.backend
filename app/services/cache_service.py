import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib
import lmdb

logger = logging.getLogger(__name__)

class SemanticCache:
    """
    Lightning Memory-Mapped Database (LMDB) for Semantic Caching.
    Provides ultra-fast, persistent key-value storage directly inside the Python process.
    """
    def __init__(self, db_path: str = "./cache_db", max_db_size: int = 104857600): # 100MB default
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Open the environment. Map size represents the maximum possible file size,
        # but the OS only consumes space for actual data written.
        self.env = lmdb.open(
            str(self.db_path),
            map_size=max_db_size,
            sync=True,      # Sync to disk
            writemap=True,  # Use memory-mapped I/O for faster writes
        )
        logger.info(f"LMDB Cache initialized at {self.db_path}")

    def _hash_query(self, text: str) -> bytes:
        """Create a deterministic hash from the query text."""
        # Normalize: strip whitespace and lowercase
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode('utf-8')).digest()

    def get_cached_response(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Check if the exact query exists in the cache. O(1) lookup.
        """
        try:
            key = self._hash_query(query)
            # Read transaction
            with self.env.begin(write=False) as txn:
                cached_bytes = txn.get(key)
                
            if cached_bytes:
                # Cache hit!
                response_dict = json.loads(cached_bytes.decode('utf-8'))
                logger.debug(f"Cache hit for query: '{query}'")
                return response_dict
            
            # Cache miss
            return None
            
        except Exception as e:
            logger.error(f"Error reading from LMDB cache: {e}")
            return None

    def store_response(self, query: str, response_dict: Dict[str, Any]) -> bool:
        """
        Store a successful response in the cache.
        """
        try:
            key = self._hash_query(query)
            
            # Create a clean cache entry omitting transient data like processing_time
            cache_entry = {
                "response": response_dict.get("response", ""),
                "active_sop": response_dict.get("active_sop", None),
                "intent": response_dict.get("intent", "unknown"),
                "sop_count": response_dict.get("sop_count", 0),
                "sources": response_dict.get("sources", []),
                "metadata": {
                    "is_cached": True,  # Mark as cached for UI debugging
                    "original_tokens": response_dict.get("metadata", {}).get("tokens_used", 0)
                }
            }
            
            value = json.dumps(cache_entry).encode('utf-8')
            
            # Write transaction
            with self.env.begin(write=True) as txn:
                txn.put(key, value)
                
            logger.debug(f"Saved response to cache for query: '{query}'")
            return True
            
        except Exception as e:
            logger.error(f"Error writing to LMDB cache: {e}")
            return False

# Global instance
cache_service = SemanticCache()
