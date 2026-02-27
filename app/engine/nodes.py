"""
CPU-optimized node implementations for enterprise RAG.
All nodes preserve full functionality but use lighter techniques.
"""

import asyncio
import os
import time
import re
import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from collections import defaultdict
import numpy as np

# Prometheus metrics (lightweight)
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Cache
from cachetools import TTLCache

# Local imports
from .config import settings
from app.services.token_service import token_counter
from app.services.llm_service import ResponseMode
from app.services.memory_service import memory_manager

logger = structlog.get_logger()

# Prometheus metrics
NODE_REQUESTS = Counter('node_requests_total', 'Node requests', ['node'])
NODE_LATENCY = Histogram('node_latency_seconds', 'Node latency', ['node'])
NODE_CACHE_HITS = Counter('node_cache_hits_total', 'Cache hits', ['node'])

# ============================================================================
# OPTIMIZATION 1: Lightweight Intent Classifier
# ============================================================================

class LightweightIntentClassifier:
    """
    CPU-optimized intent classifier.
    - No transformer models
    - Optional spaCy (lazy loaded)
    - Regex + lightweight ML only
    - Model loaded only when needed
    """
    
    def __init__(self):
        self.cache = TTLCache(
            maxsize=settings.INTENT_CACHE_SIZE,
            ttl=settings.CACHE_MEMORY_TTL
        )
        self._ml_model = None
        self._vectorizer = None
        self._nlp = None
        
        # Patterns (always available, no loading cost)
        self.patterns = {
            "sop_question": [
                r'\b(how\s+do\s+I|procedure|process|policy|rule|regulation|guideline|tier|gap)\b',
                r'\b(sop|standard\s+operating\s+procedure|protocol|workflow|report|document)\b',
                r'\b(what\s+is\s+the\s+(policy|procedure|rule|tier|finding|gap))\b',
                r'\b(can\s+you\s+explain|tell\s+me\s+about)\b.*\b(procedure|process|policy|report)\b',
                r'\b(not\s+working|issue|problem|fix|troubleshoot|why\s+is\s+my|how\s+to)\b',
                r'\b(how\s+can\s+I|where\s+can\s+I|step\s+by\s+step)\b'
            ],
            "greeting": [
                r'^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))$',
                r'\bhowdy\b|\bwhat\'s\s+up\b'
            ],
            "farewell": [
                r'\b(bye|goodbye|see\s+you|talk\s+to\s+you\s+later|take\s+care)\b',
                r'\bhave\s+a\s+(good|great|nice)\s+(day|night|evening)\b'
            ],
            "thanks": [
                r'\b(thanks|thank\s+you|appreciate|grateful)\b',
                r'\bthank you very much\b'
            ],
            "feedback": [
                r'\b(feedback|suggestion|improve|rating)\b',
                r'\b(this\s+(is|was)\s+(good|bad|helpful|unhelpful))\b'
            ]
        }
        
        # Thresholds
        self.thresholds = {
            "sop_question": 0.3 if settings.IS_LOW_RESOURCE else 0.6,
            "greeting": 0.7,
            "farewell": 0.7,
            "thanks": 0.7,
            "feedback": 0.5
        }
    
    def _get_ml_model(self):
        """Lazy load ML model only when needed"""
        if self._ml_model is None and settings.INTENT_USE_ML:
            try:
                import joblib
                if os.path.exists(settings.INTENT_MODEL_PATH):
                    self._ml_model = joblib.load(settings.INTENT_MODEL_PATH)
                    self._vectorizer = joblib.load(
                        settings.INTENT_MODEL_PATH.replace('.joblib', '_vectorizer.joblib')
                    )
                    logger.info("ML intent model loaded")
            except Exception as e:
                logger.warning(f"Failed to load ML model: {e}")
        return self._ml_model, self._vectorizer
    
    def _get_spacy(self):
        """Lazy load spaCy only when needed"""
        if self._nlp is None and settings.INTENT_USE_SPACY:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
                logger.info("spaCy model loaded")
            except Exception as e:
                logger.warning(f"Failed to load spaCy: {e}")
        return self._nlp
    
    async def classify(self, query: str) -> Tuple[str, float]:
        """
        Classify intent using multiple methods.
        Returns (intent, confidence)
        """
        # Check cache first
        cache_key = f"intent:{hashlib.md5(query.encode()).hexdigest()}"
        if cache_key in self.cache:
            NODE_CACHE_HITS.labels(node="intent_classifier").inc()
            return self.cache[cache_key]
        
        confidences = defaultdict(float)
        weights = []
        
        # Method 1: Regex patterns (always available, no loading cost)
        regex_intent, regex_conf = self._classify_by_pattern(query)
        confidences[regex_intent] += regex_conf * 0.5  # Weight: 0.5
        weights.append(0.5)
        
        # Method 2: Lightweight ML (if enabled and available)
        if settings.INTENT_USE_ML:
            ml_model, vectorizer = self._get_ml_model()
            if ml_model and vectorizer:
                ml_intent, ml_conf = self._classify_with_ml(query, ml_model, vectorizer)
                confidences[ml_intent] += ml_conf * 0.3  # Weight: 0.3
                weights.append(0.3)
        
        # Method 3: Simple keyword counting (always available)
        keyword_intent, keyword_conf = self._classify_by_keywords(query)
        confidences[keyword_intent] += keyword_conf * 0.2  # Weight: 0.2
        weights.append(0.2)
        
        # Get top intent
        if not confidences:
            result = ("unknown", 0.0)
        else:
            # Normalize by total weight
            total_weight = sum(weights)
            for intent in confidences:
                confidences[intent] /= total_weight
            
            top_intent = max(confidences.items(), key=lambda x: x[1])
            
            # Apply threshold
            threshold = self.thresholds.get(top_intent[0], 0.3)
            if top_intent[1] < threshold:
                result = ("unknown", top_intent[1])
            else:
                result = top_intent
        
        # Cache result
        self.cache[cache_key] = result
        
        # Post-process for multi-turn (follow-up)
        # If intent is unknown but looks like a follow-up, and we have history
        # (This logic is better handled in the node using the state)
        return result

    def is_follow_up(self, query: str, history: List[Dict[str, str]]) -> bool:
        """Heuristic to detect follow-up questions"""
        if not history:
            return False
            
        query_lower = query.lower()
        follow_up_indicators = [
            'this', 'that', 'it', 'step', 'more', 'detail', 'explain', 
            'tell me more', 'what is the', 'first', 'second', 'next', 'then'
        ]
        
        # If query is starts with an indicator or contains them
        for indicator in follow_up_indicators:
            if query_lower.startswith(indicator) or f" {indicator} " in f" {query_lower} ":
                return True
        return False
    
    def _classify_by_pattern(self, query: str) -> Tuple[str, float]:
        """Classify using regex patterns"""
        query_lower = query.lower().strip()
        
        best_intent = "unknown"
        best_score = 0.0
        
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    # Score based on pattern specificity
                    score = 0.8
                    if len(pattern) > 30:  # Longer patterns are more specific
                        score = 0.9
                    
                    if score > best_score:
                        best_score = score
                        best_intent = intent
                    break
        
        return best_intent, best_score
    
    def _classify_with_ml(self, query: str, model, vectorizer) -> Tuple[str, float]:
        """Classify using ML model"""
        try:
            X = vectorizer.transform([query])
            probs = model.predict_proba(X)[0]
            pred_class = model.classes_[np.argmax(probs)]
            confidence = float(np.max(probs))
            return pred_class, confidence
        except Exception:
            return "unknown", 0.0
    
    def _classify_by_keywords(self, query: str) -> Tuple[str, float]:
        """Simple keyword-based classification"""
        query_lower = query.lower()
        
        # SOP keywords
        sop_keywords = ['how', 'what', 'procedure', 'process', 'policy', 
                       'sop', 'standard', 'guideline', 'rule', 'regulation']
        sop_count = sum(1 for kw in sop_keywords if kw in query_lower)
        
        # Greeting keywords
        greeting_keywords = ['hi', 'hello', 'hey', 'greetings']
        greeting_count = sum(1 for kw in greeting_keywords if kw in query_lower)
        
        # Feedback keywords
        feedback_keywords = ['feedback', 'suggestion', 'improve', 'rating']
        feedback_count = sum(1 for kw in feedback_keywords if kw in query_lower)
        
        total_words = len(query_lower.split())
        if total_words == 0:
            return "unknown", 0.0
        
        sop_ratio = sop_count / total_words
        greeting_ratio = greeting_count / total_words
        feedback_ratio = feedback_count / total_words
        
        if sop_ratio > 0.2:
            return "sop_question", sop_ratio
        elif greeting_ratio > 0.3:
            return "greeting", greeting_ratio
        elif feedback_ratio > 0.2:
            return "feedback", feedback_ratio
        
        return "unknown", 0.1

# ============================================================================
# OPTIMIZATION 2: Lightweight Token Counter
# ============================================================================

# Token counter is now managed by separate service to avoid circular imports

# ============================================================================
# OPTIMIZATION 3: Node Base Class
# ============================================================================

class BaseNode:
    """Base class for all nodes with common functionality"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logger.bind(node=name)
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the state"""
        raise NotImplementedError
    
    def _record_metrics(self, start_time: float, success: bool = True):
        """Record node metrics"""
        duration = time.perf_counter() - start_time
        NODE_LATENCY.labels(node=self.name).observe(duration)
        NODE_REQUESTS.labels(node=self.name).inc()
        
        if not success:
            NODE_REQUESTS.labels(node=f"{self.name}_error").inc()

# ============================================================================
# OPTIMIZATION X: Cache Check Node
# ============================================================================

from app.services.cache_service import cache_service

class CacheCheckNode(BaseNode):
    """Checks the semantic cache for exactly identical queries to bypass the LLM"""
    
    def __init__(self):
        super().__init__("cache_check")
        
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check cache for identical query"""
        start = time.perf_counter()
        query = state.get("query", "")
        
        try:
            cached_response = cache_service.get_cached_response(query)
            
            if cached_response:
                self._record_metrics(start)
                return {
                    "cache_hit": True,
                    "response": cached_response.get("response", ""),
                    "active_sop": cached_response.get("active_sop"),
                    "intent": cached_response.get("intent", "unknown"),
                    "sop_count": cached_response.get("sop_count", 0),
                    # Put sources in context_chunks so RAGState handles them identically
                    "context_chunks": cached_response.get("sources", []),
                    "metadata": cached_response.get("metadata", {})
                }
                
            self._record_metrics(start)
            return {"cache_hit": False}
            
        except Exception as e:
            self.logger.error("Cache check failed", error=str(e))
            self._record_metrics(start, success=False)
            return {"cache_hit": False}

# ============================================================================
# OPTIMIZATION 4: Intent Classifier Node
# ============================================================================

class IntentClassifierNode(BaseNode):
    """Intent classification node with lightweight classifier"""
    
    def __init__(self):
        super().__init__("intent_classifier")
        self.classifier = LightweightIntentClassifier()
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process intent classification"""
        start = time.perf_counter()
        
        try:
            query = state.get("query", "")
            history = state.get("history", [])
            intent, confidence = await self.classifier.classify(query)
            
            # Simple follow-up detection: if unknown but looks like follow-up, 
            # and last intent was sop_question, stick with it.
            if intent == "unknown" and self.classifier.is_follow_up(query, history):
                # Look back in state for last solid intent
                # Note: State from checkpointer will have the last turn's intent
                if state.get("intent") == "sop_question":
                    intent = "sop_question"
                    confidence = 0.5 # Boost confidence
            
            result = {
                "intent": intent,
                "confidence": confidence,
                "cache_key": state.get("cache_key", "")
            }
            
            self._record_metrics(start)
            return result
            
        except Exception as e:
            self.logger.error("Classification failed", error=str(e))
            self._record_metrics(start, success=False)
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "cache_key": state.get("cache_key", "")
            }

# ============================================================================
# OPTIMIZATION 5: Retriever Node with Multi-Strategy
# ============================================================================

class RetrieverNode(BaseNode):
    """Document retrieval node with multiple strategies"""
    
    def __init__(self, qdrant_service):
        super().__init__("retriever")
        self.qdrant = qdrant_service
        self.cache = TTLCache(maxsize=200, ttl=300)
        
        # Strategy weights (optimized for CPU)
        self.strategy_weights = {
            "dense": 0.7,    # Primary strategy
            "keyword": 0.3    # Secondary strategy
        }
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-strategy retrieval"""
        start = time.perf_counter()
        
        try:
            query = state.get("query", "")
            query_embedding = state.get("query_embedding")
            
            # Check cache
            cache_key = f"retrieval:{state.get('cache_key', '')}"
            if cache_key in self.cache:
                NODE_CACHE_HITS.labels(node=self.name).inc()
                return {"retrieved_docs": self.cache[cache_key]}
            
            # Execute strategies in parallel
            tasks = []
            strategies = settings.RETRIEVAL_STRATEGIES
            
            if "dense" in strategies and query_embedding is not None:
                loop = asyncio.get_event_loop()
                tasks.append(loop.run_in_executor(
                    None,
                    lambda: self.qdrant.search_sops(
                        query_embedding,
                        top_k=settings.RETRIEVAL_DENSE_K
                    )
                ))
            
            if "keyword" in strategies:
                loop = asyncio.get_event_loop()
                tasks.append(loop.run_in_executor(
                    None,
                    lambda: self.qdrant.search_by_keywords(
                        self._extract_keywords(query),
                        top_k=settings.RETRIEVAL_KEYWORD_K
                    )
                ))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            all_docs = []
            seen_ids = set()
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.warning(f"Strategy failed", error=str(result))
                    continue
                
                strategy = strategies[i] if i < len(strategies) else "unknown"
                weight = self.strategy_weights.get(strategy, 0.5)
                
                for doc in result:
                    doc_id = doc.get("id")
                    if doc_id not in seen_ids:
                        doc["score"] = doc.get("score", 0) * weight
                        all_docs.append(doc)
                        seen_ids.add(doc_id)
            
            # Sort by score
            all_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
            # Use broader retrieval when re-ranking is enabled
            top_k_limit = settings.RETRIEVAL_INITIAL_K if settings.RERANKER_ENABLED else settings.RETRIEVAL_TOP_K
            all_docs = all_docs[:top_k_limit]
            
            # Cache results
            self.cache[cache_key] = all_docs
            
            self._record_metrics(start)
            return {"retrieved_docs": all_docs}
            
        except Exception as e:
            self.logger.error("Retrieval failed", error=str(e))
            self._record_metrics(start, success=False)
            return {"retrieved_docs": []}
    
    async def _dense_retrieval(self, query_embedding: List[float]) -> List[Dict]:
        """Dense retrieval using embeddings"""
        try:
            results = await self.qdrant.search_sops(
                query_embedding,
                top_k=settings.RETRIEVAL_DENSE_K,
                score_threshold=0.2
            )
            return results
        except Exception as e:
            self.logger.warning(f"Dense retrieval failed", error=str(e))
            return []
    
    async def _keyword_retrieval(self, query: str) -> List[Dict]:
        """Keyword-based retrieval"""
        try:
            keywords = self._extract_keywords(query)
            if not keywords:
                return []
            
            results = await self.qdrant.search_by_keywords(
                keywords,
                top_k=settings.RETRIEVAL_KEYWORD_K
            )
            return results
        except Exception as e:
            self.logger.warning(f"Keyword retrieval failed", error=str(e))
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction (no heavy NLP)"""
        # Simple word splitting and filtering
        words = text.lower().split()
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was'}
        
        keywords = [
            word.strip('.,!?;:') for word in words
            if len(word) > 3 and word not in stop_words
        ]
        
        return keywords[:5]

# ============================================================================
# OPTIMIZATION 6: Context Builder Node
# ============================================================================

class ContextBuilderNode(BaseNode):
    """Context building node with token optimization"""
    
    def __init__(self):
        super().__init__("context_builder")
        self.max_tokens = settings.MODEL_CONTEXT_SIZE - 200  # Reserve for response
        self.cache = TTLCache(maxsize=200, ttl=300)
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build optimized context from retrieved docs"""
        start = time.perf_counter()
        
        try:
            docs = state.get("retrieved_docs", [])
            if not docs:
                return {"context_chunks": []}
            
            # Check cache
            cache_key = f"context:{state.get('cache_key', '')}"
            if cache_key in self.cache:
                NODE_CACHE_HITS.labels(node=self.name).inc()
                return {"context_chunks": self.cache[cache_key]}
            
            # Build context chunks
            chunks = []
            total_tokens = 0
            
            for doc in docs:
                content = doc.get("content", "")
                estimated_tokens = token_counter.count(content)
                
                if total_tokens + estimated_tokens <= self.max_tokens:
                    chunks.append({
                        "content": content,
                        "metadata": {
                            "title": doc.get("title", "Unknown"),
                            "source": doc.get("source", ""),
                            "relevance_score": doc.get("score", 0)
                        }
                    })
                    total_tokens += estimated_tokens
                else:
                    # Truncate if it's important
                    if not chunks and estimated_tokens > self.max_tokens:
                        truncated = self._truncate_content(content, self.max_tokens)
                        chunks.append({
                            "content": truncated,
                            "metadata": {
                                "title": doc.get("title", "Unknown"),
                                "source": doc.get("source", ""),
                                "relevance_score": doc.get("score", 0),
                                "truncated": True
                            }
                        })
                    break
            
            # Cache results
            self.cache[cache_key] = chunks
            
            self._record_metrics(start)
            return {"context_chunks": chunks}
            
        except Exception as e:
            self.logger.error("Context building failed", error=str(e))
            self._record_metrics(start, success=False)
            return {"context_chunks": []}
    
    def _truncate_content(self, content: str, target_tokens: int) -> str:
        """Intelligently truncate content"""
        target_chars = target_tokens * 4
        
        if len(content) <= target_chars:
            return content
        
        # Keep first 70% and last 20%
        first_len = int(target_chars * 0.7)
        last_len = int(target_chars * 0.2)
        
        first_part = content[:first_len]
        last_part = content[-last_len:]
        
        return f"{first_part}...{last_part}"

# ============================================================================
# OPTIMIZATION 7: Answer Generator Node
# ============================================================================

class AnswerGeneratorNode(BaseNode):
    """Answer generation node with LLM"""
    
    def __init__(self, llm_service):
        super().__init__("answer_generator")
        self.llm = llm_service
        self.cache = TTLCache(maxsize=200, ttl=3600)
        
        # Update token counter with LLM
        token_counter.set_llm(llm_service.llm)
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate answer using LLM"""
        start = time.perf_counter()
        
        try:
            # Check cache
            cache_key = f"answer:{state.get('cache_key', '')}"
            if cache_key in self.cache:
                NODE_CACHE_HITS.labels(node=self.name).inc()
                cached = self.cache[cache_key]
                return {
                    "response": cached["response"],
                    "tokens_used": cached["tokens"]
                }
            
            # Prepare prompt
            prompt = self._build_prompt(state)
            
            # Run LLM in thread executor (CPU-bound)
            loop = asyncio.get_event_loop()
            
            # Use synthesize_rag_response for better RAG + History integration
            # This uses the underlying rag_base.j2 template correctly
            response = await loop.run_in_executor(
                None,
                lambda: self.llm.synthesize_rag_response(
                    query=state.get("query", ""),
                    context_chunks=state.get("context_chunks", []),
                    mode=ResponseMode.CONCISE,
                    user_id=state.get("user_id", "anonymous"),
                    history=state.get("history", [])
                )
            )
            
            # Note: synthesize_rag_response might return a dict if include_metadata=True
            if isinstance(response, dict):
                response_text = response.get("answer", "")
            else:
                response_text = response
            
            tokens_used = token_counter.count(response_text)
            
            # Update history in state using MemoryWindow
            new_history = list(state.get("history", []))
            new_history.append({"role": "user", "content": state.get("query", "")})
            new_history.append({"role": "assistant", "content": response_text})
            
            # Use MemoryManager for token-aware sliding window
            new_history = memory_manager.get_sliding_window(new_history)
            
            # Cache response
            self.cache[cache_key] = {
                "response": response_text,
                "tokens": tokens_used
            }
            
            self._record_metrics(start)
            return {
                "response": response_text,
                "tokens_used": tokens_used,
                "history": new_history
            }
            
        except Exception as e:
            self.logger.error("Answer generation failed", error=str(e))
            self._record_metrics(start, success=False)
            return {
                "response": "I encountered an error generating a response.",
                "tokens_used": 0
            }
    
    def _build_prompt(self, state: Dict[str, Any]) -> str:
        """Build prompt from state"""
        query = state.get("query", "")
        chunks = state.get("context_chunks", [])
        
        if not chunks:
            return query
        
        context = "\n\n".join([
            f"Source: {chunk['metadata']['title']}\n{chunk['content']}"
            for chunk in chunks
        ])
        
        return f"Context:\n{context}\n\nQuestion: {query}"
    
    def _get_system_prompt(self, state: Dict[str, Any]) -> str:
        """Get system prompt based on intent"""
        intent = state.get("intent", "unknown")
        
        if intent == "sop_question":
            return (
                "You are a precise SOP assistant. Answer using ONLY the provided context. "
                "If the answer isn't in the context, say it's not available. Be concise."
            )
        else:
            return "You are a helpful assistant. Be friendly and helpful."

# ============================================================================
# OPTIMIZATION 8: General Chat Node
# ============================================================================

class GeneralChatNode(BaseNode):
    """General chat node for non-SOP queries"""
    
    def __init__(self, llm_service):
        super().__init__("general_chat")
        self.llm = llm_service
        self.cache = TTLCache(maxsize=200, ttl=300)
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general chat"""
        start = time.perf_counter()
        
        try:
            intent = state.get("intent", "unknown")
            query = state.get("query", "")
            
            # Template responses for common intents
            templates = {
                "greeting": [
                    "Hello! How can I help you with our procedures today?",
                    "Hi there! What can I assist you with?",
                    "Greetings! Feel free to ask about any SOP."
                ],
                "thanks": [
                    "You're welcome! Happy to help with any procedures.",
                    "My pleasure! Let me know if you need anything else."
                ],
                "farewell": [
                    "Goodbye! Feel free to return if you have more questions.",
                    "Take care! I'm here if you need help with any procedures."
                ]
            }
            
            if intent in templates:
                import random
                response = random.choice(templates[intent])
                tokens_used = token_counter.count(response)
            else:
                # Generate response using LLM
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.llm.generate_response(
                        query=query,
                        system_prompt="You are a helpful assistant. Be conversational.",
                        history=state.get("history", []),
                        max_tokens=150,
                        temperature=0.7
                    )
                )
                tokens_used = token_counter.count(response)
            
            # Update history in state using MemoryWindow
            new_history = list(state.get("history", []))
            new_history.append({"role": "user", "content": query})
            new_history.append({"role": "assistant", "content": response})
            
            # Use MemoryManager for token-aware sliding window
            new_history = memory_manager.get_sliding_window(new_history)
            
            self._record_metrics(start)
            return {
                "response": response,
                "tokens_used": tokens_used,
                "history": new_history
            }
            
        except Exception as e:
            self.logger.error("General chat failed", error=str(e))
            self._record_metrics(start, success=False)
            return {
                "response": "I'm here to help. Could you please rephrase?",
                "tokens_used": 0
            }

# ============================================================================
# OPTIMIZATION 9: Formatter Node
# ============================================================================

class FormatterNode(BaseNode):
    """Response formatting node"""
    
    def __init__(self):
        super().__init__("formatter")
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Format final response"""
        start = time.perf_counter()
        
        try:
            response = state.get("response", "")
            chunks = state.get("context_chunks", [])
            query = state.get("query", "")
            
            # Add sources if available
            # [LOGIC REMOVED: sources are now handled by UI using structured metadata]
            
            metadata = {
                "sources_used": len(chunks),
                "intent": state.get("intent", "unknown"),
                "confidence": state.get("intent_confidence", 0.0)
            }
            
            # Store in cache if not a hit and we have a valid response
            if not state.get("cache_hit") and state.get("intent") != "unknown" and response:
                cache_service.store_response(query, {
                    "response": response,
                    "active_sop": chunks[0].get("metadata", {}).get("title") if chunks else None,
                    "intent": state.get("intent"),
                    "sop_count": len(chunks),
                    "sources": chunks,
                    "metadata": metadata
                })
            
            self._record_metrics(start)
            return {
                "response": response,
                "metadata": metadata
            }
            
        except Exception as e:
            self.logger.error("Formatting failed", error=str(e))
            self._record_metrics(start, success=False)
            return {
                "response": state.get("response", ""),
                "metadata": {}
            }

# ============================================================================
# OPTIMIZATION 10: Feedback Collector Node
# ============================================================================

class FeedbackCollectorNode(BaseNode):
    """Feedback collection node"""
    
    def __init__(self):
        super().__init__("feedback")
        self.feedback_store = []
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Collect feedback"""
        start = time.perf_counter()
        
        try:
            # Store feedback (in production, would send to DB)
            feedback = {
                "query": state.get("query", ""),
                "response": state.get("response", ""),
                "intent": state.get("intent", ""),
                "user_id": state.get("user_id", ""),
                "timestamp": datetime.now().isoformat()
            }
            
            self.feedback_store.append(feedback)
            
            # Keep only last 100 in memory
            if len(self.feedback_store) > 100:
                self.feedback_store = self.feedback_store[-100:]
            
            self._record_metrics(start)
            return {
                "response": "Thank you for your feedback! It helps us improve."
            }
            
        except Exception as e:
            self.logger.error("Feedback collection failed", error=str(e))
            self._record_metrics(start, success=False)
            return {
                "response": "Thank you for your feedback."
            }