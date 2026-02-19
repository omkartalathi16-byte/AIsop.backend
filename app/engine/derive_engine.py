"""
Derive Engine — Hybrid SOP Derivation
Combines embedding similarity search with metadata keyword boosting
to return structured SOP results with confidence scores.

Production Version with comprehensive error handling, caching, and configurability.
"""
import logging
import time
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
from dataclasses import dataclass, field
from contextlib import contextmanager
from cachetools import TTLCache, cached

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class DeriveConfig:
    """Configuration for DeriveEngine with sensible defaults."""
    # Metadata boosting
    metadata_boost: float = 0.15
    max_boost_multiplier: int = 3
    
    # Confidence scoring weights
    confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        'max_score': 0.6,
        'avg_score': 0.4
    })
    
    # Search parameters
    search_multiplier: int = 3
    max_chunks_per_sop: int = 5
    
    # Caching
    embedding_cache_ttl: int = 300  # 5 minutes
    embedding_cache_maxsize: int = 100
    results_cache_ttl: int = 60  # 1 minute
    results_cache_maxsize: int = 50
    
    # Boosting weights
    boost_weights: Dict[str, float] = field(default_factory=lambda: {
        'exact_match': 1.0,
        'partial_match': 0.5,
        'fuzzy_match': 0.3,
        'title_keyword': 0.8
    })
    
    # Fuzzy matching threshold
    fuzzy_threshold: float = 0.8
    
    # Diversity settings
    diversity_penalty: float = 0.9
    enable_diversity: bool = False  # Disabled by default, enable if needed


class DeriveError(Exception):
    """Base exception for DeriveEngine errors."""
    pass


class EmbeddingError(DeriveError):
    """Raised when embedding generation fails."""
    pass


class SearchError(DeriveError):
    """Raised when vector search fails."""
    pass


class DeriveEngine:
    """
    Hybrid derivation engine with comprehensive improvements:
    - Configurable parameters
    - Caching layers
    - Error handling
    - Query expansion
    - Weighted metadata boosting
    - Result diversification
    - Performance monitoring
    - Feedback integration
    """

    def __init__(self, embedding_service, qdrant_service, config: Optional[DeriveConfig] = None):
        """
        Initialize the DeriveEngine with services and configuration.
        
        Args:
            embedding_service: Service for generating embeddings
            qdrant_service: Service for vector similarity search
            config: Optional configuration, uses defaults if not provided
        """
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self.config = config or DeriveConfig()
        
        # Initialize caches
        self.embedding_cache = TTLCache(
            maxsize=self.config.embedding_cache_maxsize,
            ttl=self.config.embedding_cache_ttl
        )
        self.results_cache = TTLCache(
            maxsize=self.config.results_cache_maxsize,
            ttl=self.config.results_cache_ttl
        )
        
        # Domain-specific query expansion mappings
        self.expansion_map = {
            "phishing": ["email fraud", "social engineering", "spoofing", "credential theft"],
            "malware": ["virus", "trojan", "ransomware", "worm", "spyware"],
            "ddos": ["denial of service", "dos attack", "flooding", "amplification attack"],
            "ransomware": ["crypto malware", "data kidnapping", "encryption attack"],
            "data breach": ["data leak", "information exposure", "data exfiltration"],
            "insider threat": ["internal threat", "malicious insider", "employee threat"],
            "zero day": ["0-day", "unpatched vulnerability", "unknown exploit"],
            "social engineering": ["human hacking", "psychological manipulation", "pretexting"],
            "man in the middle": ["mitm", "eavesdropping", "session hijacking"],
            "sql injection": ["sqli", "database attack", "code injection"]
        }

    @contextmanager
    def _timer(self, operation_name: str):
        """Context manager for timing operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            logger.debug(f"TIMING {operation_name}: {elapsed:.4f}s")
            # In production, you'd send this to metrics system (StatsD, Prometheus, etc.)

    def _validate_inputs(self, query: str, top_k: int) -> None:
        """Validate input parameters."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        
        if top_k > 100:  # Reasonable upper limit
            logger.warning(f"Large top_k requested: {top_k}. Consider reducing for better performance.")

    def _preprocess_query(self, query: str) -> str:
        """
        Enhance query with domain-specific expansions.
        
        Args:
            query: Original user query
            
        Returns:
            Expanded query string
        """
        query_lower = query.lower()
        expanded_terms = [query]
        
        # Add domain-specific expansions
        for term, expansions in self.expansion_map.items():
            if term in query_lower:
                expanded_terms.extend(expansions)
                logger.debug(f"Query expanded: '{term}' -> {expansions}")
        
        return " ".join(expanded_terms)

    @cached(cache=TTLCache(maxsize=100, ttl=300))
    def _get_embedding_cached(self, query: str) -> List[float]:
        """
        Get embedding with caching.
        
        Args:
            query: Text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            # Preprocess query before embedding
            expanded_query = self._preprocess_query(query)
            
            with self._timer("embedding_generation"):
                embedding = self.embedding_service.generate_query_embedding(expanded_query)
                
            if not embedding:
                raise EmbeddingError("Empty embedding returned from service")
                
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed for query '{query}': {e}")
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

    def _fuzzy_match(self, str1: str, str2: str, threshold: float = None) -> bool:
        """
        Simple fuzzy matching using character overlap ratio.
        
        Args:
            str1: First string
            str2: Second string
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if similarity exceeds threshold
        """
        if not str1 or not str2:
            return False
            
        threshold = threshold or self.config.fuzzy_threshold
        
        # Convert to sets of character bigrams for simple similarity
        def get_bigrams(s):
            return set(s[i:i+2] for i in range(len(s)-1))
        
        bigrams1 = get_bigrams(str1)
        bigrams2 = get_bigrams(str2)
        
        if not bigrams1 or not bigrams2:
            return False
            
        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)
        
        return (intersection / union) >= threshold

    def _compute_metadata_boost(self, 
                               result: Dict[str, Any], 
                               query: str,
                               threat_type: Optional[str] = None, 
                               category: Optional[str] = None) -> float:
        """
        Enhanced metadata boost with weighted scoring and partial matching.
        
        Args:
            result: Search result from Milvus
            query: Original query
            threat_type: Optional threat type filter
            category: Optional category filter
            
        Returns:
            Boost score between 0 and max_boost
        """
        boost = 0.0
        weights = self.config.boost_weights
        
        # Boost if threat_type matches
        result_threat = (result.get("threat_type") or "").lower()
        if threat_type and result_threat:
            threat_lower = threat_type.lower()
            
            if threat_lower == result_threat:
                boost += self.config.metadata_boost * weights['exact_match']
            elif threat_lower in result_threat or result_threat in threat_lower:
                boost += self.config.metadata_boost * weights['partial_match']
            elif self._fuzzy_match(threat_lower, result_threat):
                boost += self.config.metadata_boost * weights['fuzzy_match']

        # Boost if category filter matches
        result_category = (result.get("category") or "").lower()
        if category and result_category:
            category_lower = category.lower()
            
            if category_lower == result_category:
                boost += self.config.metadata_boost * weights['exact_match']
            elif category_lower in result_category or result_category in category_lower:
                boost += self.config.metadata_boost * weights['partial_match']
            elif self._fuzzy_match(category_lower, result_category):
                boost += self.config.metadata_boost * weights['fuzzy_match']

        # Enhanced title keyword boosting
        title_lower = (result.get("title") or "").lower()
        query_lower = query.lower()
        
        # Boost for exact title match
        if query_lower == title_lower:
            boost += self.config.metadata_boost * 1.5
        else:
            # Boost for keyword matches with length filtering
            query_words = [w for w in query_lower.split() if len(w) > 3]
            if query_words:
                matches = sum(1 for w in query_words if w in title_lower)
                boost += (matches / len(query_words)) * self.config.metadata_boost * weights['title_keyword']

        # Cap the boost
        max_boost = self.config.metadata_boost * self.config.max_boost_multiplier
        return min(boost, max_boost)

    def _search_with_fallback(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Search with fallback strategies.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to retrieve
            
        Returns:
            List of search results
            
        Raises:
            SearchError: If all search attempts fail
        """
        errors = []
        
        try:
            return self.qdrant_service.search_sops(query_embedding, top_k=top_k)
        except Exception as e:
            errors.append(f"Primary search failed: {e}")
            logger.warning(f"Primary search failed, attempting fallback: {e}")
        
        # Fallback 1: Reduce top_k
        try:
            reduced_k = max(top_k // 2, 5)
            logger.info(f"Attempting fallback search with reduced k={reduced_k}")
            return self.qdrant_service.search_sops(query_embedding, top_k=reduced_k)
        except Exception as e:
            errors.append(f"Fallback search failed: {e}")
        
        # Fallback 2: Try with simplified query (if embedding service supports it)
        try:
            logger.info("Attempting fallback with simplified query")
            # This assumes your embedding service has a fallback method
            # simplified_embedding = self.embedding_service.generate_fallback_embedding()
            # return self.milvus_service.search_sops(simplified_embedding, top_k=top_k)
            # Default fallback for now: return empty if not supported
            return []
        except Exception as e:
            errors.append(f"Simplified query search failed: {e}")
        
        # All attempts failed
        error_msg = f"All search attempts failed: {'; '.join(errors)}"
        logger.error(error_msg)
        raise SearchError(error_msg)

    def _diversify_results(self, derive_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Ensure diversity in results to avoid redundancy.
        
        Args:
            derive_results: List of SOP results
            top_k: Number of results to return
            
        Returns:
            Diversified list of results
        """
        if not self.config.enable_diversity or not derive_results:
            return derive_results[:top_k]
        
        diversified = []
        categories_seen = set()
        threat_types_seen = set()
        
        for result in derive_results:
            if len(diversified) >= top_k:
                break
                
            metadata = result.get('metadata', {})
            category = metadata.get('category')
            threat_type = metadata.get('threat_type')
            
            # Apply diversity penalty if similar results exist
            if category in categories_seen or threat_type in threat_types_seen:
                result['confidence_score'] *= self.config.diversity_penalty
                result['diversity_penalty_applied'] = True
            
            diversified.append(result)
            
            # Track seen categories and threat types
            if category:
                categories_seen.add(category)
            if threat_type:
                threat_types_seen.add(threat_type)
        
        # Re-sort after penalties
        diversified.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return diversified[:top_k]

    def derive(self, 
              query: str, 
              top_k: int = 5,
              threat_type: Optional[str] = None, 
              category: Optional[str] = None,
              feedback_sop_ids: Optional[List[str]] = None,
              skip_cache: bool = False) -> Dict[str, Any]:
        """
        Full derivation pipeline with all improvements.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            threat_type: Optional threat type filter
            category: Optional category filter
            feedback_sop_ids: Optional list of SOP IDs that user found relevant
            skip_cache: Force skip cache for fresh results
            
        Returns:
            Structured dictionary with query results
            
        Example:
            {
                "query": "phishing attack",
                "results": [...],
                "total_results": 5,
                "performance": {...},
                "error": None
            }
        """
        # Initialize response structure
        response = {
            "query": query,
            "results": [],
            "total_results": 0,
            "performance": {},
            "error": None
        }
        
        try:
            # 1. Validate inputs
            self._validate_inputs(query, top_k)
            
            # 2. Check cache (if enabled)
            cache_key = None
            if not skip_cache:
                cache_key = f"{query}_{top_k}_{threat_type}_{category}"
                if cache_key in self.results_cache:
                    logger.info(f"Cache hit for query: {query}")
                    return self.results_cache[cache_key]
            
            with self._timer("total_derive") as timer_context:
                
                # 3. Generate query embedding (cached)
                with self._timer("embedding_with_cache"):
                    query_embedding = self._get_embedding_cached(query)
                
                # 4. Search Milvus with fallback
                with self._timer("vector_search"):
                    search_k = top_k * self.config.search_multiplier
                    raw_results = self._search_with_fallback(query_embedding, search_k)
                
                if not raw_results:
                    logger.info(f"No results found for query: {query}")
                    response["total_results"] = 0
                    return response
                
                # 5. Apply metadata boosting and group by SOP title
                with self._timer("grouping_and_boosting"):
                    grouped = self._group_and_boost_results(
                        raw_results, query, threat_type, category
                    )
                
                # 6. Build structured results
                with self._timer("result_construction"):
                    derive_results = self._build_structured_results(grouped)
                
                # 7. Apply feedback boosts if provided
                if feedback_sop_ids:
                    derive_results = self._apply_feedback_boosts(
                        derive_results, feedback_sop_ids
                    )
                
                # 8. Diversify results (if enabled)
                derive_results = self._diversify_results(derive_results, top_k)
                
                # 9. Prepare final response
                response["results"] = derive_results[:top_k]
                response["total_results"] = len(derive_results)
                
                # 10. Cache results
                if cache_key:
                    self.results_cache[cache_key] = response
                
        except ValueError as e:
            logger.warning(f"Input validation error: {e}")
            response["error"] = f"Invalid input: {e}"
            
        except (EmbeddingError, SearchError) as e:
            logger.error(f"Service error during derivation: {e}")
            response["error"] = f"Service unavailable: {e}"
            
        except Exception as e:
            logger.exception(f"Unexpected error during derivation: {e}")
            response["error"] = f"Internal error: {str(e)}"
        
        # Add performance metrics (in production, use proper metrics system)
        if 'performance' in response:
            response["performance"]["timestamp"] = time.time()
        
        return response

    def _group_and_boost_results(self, 
                                raw_results: List[Dict[str, Any]], 
                                query: str,
                                threat_type: Optional[str], 
                                category: Optional[str]) -> Dict[str, Dict]:
        """
        Group search results by SOP title and apply metadata boosting.
        
        Args:
            raw_results: Raw search results from Milvus
            query: Original query
            threat_type: Optional threat type filter
            category: Optional category filter
            
        Returns:
            Grouped results dictionary
        """
        grouped = defaultdict(lambda: {
            "chunks": [],
            "sop_link": "",
            "sop_id": None,
            "threat_type": "",
            "category": "",
            "max_score": 0.0,
            "total_boosted_score": 0.0,
            "chunk_count": 0,
            "first_seen": time.time()
        })

        for result in raw_results:
            title = result.get("title")
            if not title:
                logger.warning(f"Result missing title, skipping: {result.get('id', 'unknown')}")
                continue
                
            base_score = result.get("score", 0.0)
            
            # Compute metadata boost
            boost = self._compute_metadata_boost(result, query, threat_type, category)
            boosted_score = min(base_score + boost, 1.0)
            
            group = grouped[title]
            chunk = {
                "content": result.get("content", ""),
                "similarity_score": round(base_score, 4),
                "boosted_score": round(boosted_score, 4),
                "chunk_id": result.get("id"),
                "metadata": {
                    "threat_type": result.get("threat_type"),
                    "category": result.get("category")
                }
            }
            
            group["chunks"].append(chunk)
            
            # Update group metadata (prioritize non-empty values)
            if not group["sop_link"] and result.get("sop_link"):
                group["sop_link"] = result["sop_link"]
            if not group["sop_id"] and result.get("sop_id"):
                group["sop_id"] = result["sop_id"]
            if not group["threat_type"] and result.get("threat_type"):
                group["threat_type"] = result["threat_type"]
            if not group["category"] and result.get("category"):
                group["category"] = result["category"]
            
            # Update scores
            group["max_score"] = max(group["max_score"], boosted_score)
            group["total_boosted_score"] += boosted_score
            group["chunk_count"] += 1
        
        # Sort chunks within each group by score descending
        for group in grouped.values():
            group["chunks"].sort(key=lambda x: x["similarity_score"], reverse=True)
            
        return grouped

    def _build_structured_results(self, grouped: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """
        Build structured results from grouped data.
        
        Args:
            grouped: Grouped results dictionary
            
        Returns:
            List of structured SOP results
        """
        derive_results = []
        
        for title, group in grouped.items():
            # Calculate confidence score
            avg_score = group["total_boosted_score"] / group["chunk_count"]
            confidence = round(
                group["max_score"] * self.config.confidence_weights['max_score'] +
                avg_score * self.config.confidence_weights['avg_score'],
                4
            )
            
            # Prepare chunks (limit to max configured)
            max_chunks = self.config.max_chunks_per_sop
            chunks_display = [
                {k: v for k, v in chunk.items() if k != 'boosted_score'}  # Remove internal scoring
                for chunk in group["chunks"][:max_chunks]
            ]
            
            derive_results.append({
                "title": title,
                "sop_link": group["sop_link"],
                "sop_id": group["sop_id"],
                "confidence_score": confidence,
                "max_chunk_score": round(group["max_score"], 4),
                "avg_chunk_score": round(avg_score, 4),
                "matched_chunks": chunks_display,
                "metadata": {
                    "threat_type": group["threat_type"],
                    "category": group["category"],
                    "total_chunks_matched": group["chunk_count"],
                    "chunks_returned": len(chunks_display)
                }
            })
        
        # Sort by confidence descending
        derive_results.sort(key=lambda x: x["confidence_score"], reverse=True)
        
        return derive_results

    def _apply_feedback_boosts(self, 
                              derive_results: List[Dict[str, Any]], 
                              feedback_sop_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Apply boosts based on user feedback.
        
        Args:
            derive_results: Current results
            feedback_sop_ids: SOP IDs that user marked as relevant
            
        Returns:
            Boosted results
        """
        if not feedback_sop_ids:
            return derive_results
        
        for result in derive_results:
            sop_id = result.get("sop_id")
            if sop_id and sop_id in feedback_sop_ids:
                # Boost confidence for SOPs user liked
                result["confidence_score"] = min(result["confidence_score"] * 1.2, 1.0)
                result["feedback_boost_applied"] = True
                logger.debug(f"Applied feedback boost to SOP: {sop_id}")
        
        # Re-sort
        derive_results.sort(key=lambda x: x["confidence_score"], reverse=True)
        
        return derive_results

    def clear_caches(self):
        """Clear all internal caches."""
        self.embedding_cache.clear()
        self.results_cache.clear()
        logger.info("Caches cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "embedding_cache": {
                "size": len(self.embedding_cache),
                "maxsize": self.embedding_cache.maxsize,
                "ttl": self.embedding_cache.ttl
            },
            "results_cache": {
                "size": len(self.results_cache),
                "maxsize": self.results_cache.maxsize,
                "ttl": self.results_cache.ttl
            }
        }


# Convenience factory function
def create_derive_engine(embedding_service, qdrant_service, **config_overrides):
    """
    Create a DeriveEngine instance with optional configuration overrides.
    
    Args:
        embedding_service: Embedding service instance
        qdrant_service: Qdrant service instance
        **config_overrides: Override default configuration values
        
    Returns:
        Configured DeriveEngine instance
    """
    config = DeriveConfig()
    
    # Apply overrides
    for key, value in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown configuration key: {key}")
    
    return DeriveEngine(embedding_service, qdrant_service, config)
