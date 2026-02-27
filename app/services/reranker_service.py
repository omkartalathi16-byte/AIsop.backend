"""
CPU-optimized Cross-Encoder Re-ranking Service.
Uses a lightweight Cross-Encoder model to re-rank retrieved documents
by scoring each (query, document) pair for precise relevance.

Performance on i5-4200U:
- Re-ranking 15 docs: ~200-500ms
- Memory: ~100MB additional
"""

import time
import logging
from typing import List, Dict, Any, Optional
from app.engine.config import settings

logger = logging.getLogger(__name__)


class RerankerService:
    """
    Cross-Encoder re-ranking for improved retrieval precision.
    
    Bi-encoders (used in retrieval) encode query and docs separately,
    which is fast but loses cross-attention. Cross-Encoders process
    (query, doc) pairs jointly, giving much more precise relevance scores.
    """
    
    def __init__(
        self,
        model_name: str = None,
        top_k: int = None
    ):
        self.model_name = model_name or settings.RERANKER_MODEL
        self.top_k = top_k or settings.RERANKER_TOP_K
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Cross-Encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading Cross-Encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Cross-Encoder loaded successfully: {self.model_name}")
        except ImportError:
            logger.error(
                "sentence-transformers is required for re-ranking. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load Cross-Encoder model: {e}")
            self.model = None
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents using Cross-Encoder scoring.
        
        Args:
            query: The user's query string.
            documents: List of retrieved documents with 'text' or 'content' fields.
            top_k: Number of top documents to return (overrides default).
            
        Returns:
            Re-ranked list of documents, sorted by relevance, trimmed to top_k.
        """
        if not self.model or not documents:
            return documents
        
        start_time = time.perf_counter()
        k = top_k or self.top_k
        
        try:
            # Extract text from each document
            pairs = []
            for doc in documents:
                text = doc.get("text", doc.get("content", doc.get("chunk", "")))
                pairs.append((query, text))
            
            # Score all (query, doc) pairs using Cross-Encoder
            scores = self.model.predict(pairs)
            
            # Attach scores to documents
            scored_docs = []
            for doc, score in zip(documents, scores):
                doc_copy = dict(doc)
                doc_copy["reranker_score"] = float(score)
                scored_docs.append(doc_copy)
            
            # Sort by score descending and take top_k
            scored_docs.sort(key=lambda x: x["reranker_score"], reverse=True)
            result = scored_docs[:k]
            
            elapsed = time.perf_counter() - start_time
            logger.info(
                f"Re-ranked {len(documents)} docs -> top {len(result)} "
                f"in {elapsed*1000:.1f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Re-ranking failed, returning original docs: {e}")
            return documents[:k]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model_name": self.model_name,
            "top_k": self.top_k,
            "loaded": self.model is not None,
            "enabled": settings.RERANKER_ENABLED
        }
