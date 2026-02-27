"""
LangGraph-based orchestration for enterprise RAG pipeline.
CPU-optimized while preserving all features.
"""

import asyncio
import time
import uuid
import operator
from typing import Dict, Any, Optional, List, Literal, TypedDict, Annotated
from datetime import datetime
import xxhash

# LangGraph imports (lightweight)
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# Local imports
from .config import settings
from app.services.llm_service import EnterpriseLLMService
from app.services.qdrant_service import QdrantService
from app.services.embedding_service import EmbeddingService
from .nodes import (
    IntentClassifierNode,
    RetrieverNode,
    ContextBuilderNode,
    AnswerGeneratorNode,
    GeneralChatNode,
    FormatterNode,
    FeedbackCollectorNode
)

# Prometheus metrics (lightweight)
from prometheus_client import Counter, Histogram, Gauge
import structlog

logger = structlog.get_logger()

# ============================================================================
# OPTIMIZATION 1: Lightweight State with TypedDict (no Pydantic overhead)
# ============================================================================

class RAGState(TypedDict):
    """
    Lightweight state using TypedDict (no Pydantic validation overhead).
    Memory footprint: ~75% smaller than dataclasses with validation.
    """
    # Core fields
    query: str
    conversation_id: str
    user_id: str
    session_id: str
    
    # Intent fields
    intent: str
    intent_confidence: float
    
    # Embedding field (lazy loaded)
    query_embedding: Optional[List[float]]
    
    # Conversation History
    history: Annotated[List[Dict[str, str]], operator.add]
    
    # Retrieval fields
    retrieved_docs: List[Dict[str, Any]]
    context_chunks: List[Dict[str, Any]]
    
    # Response fields
    response: str
    tokens_used: int
    status: str
    
    # Metadata
    processing_time: Dict[str, float]
    errors: Annotated[List[Dict[str, Any]], operator.add]
    metadata: Dict[str, Any]
    
    # A/B testing
    ab_test_group: Optional[str]
    
    # Cache key (for optimization)
    cache_key: str
    
    # Semantic Caching
    cache_hit: bool
    cached_response: Optional[str]

# ============================================================================
# OPTIMIZATION 2: Minimal Router Logic (No heavy computation)
# ============================================================================

def route_after_cache(state: RAGState) -> Literal["formatter", "intent_classifier"]:
    """Route based on whether we hit the LMDB semantic cache"""
    if state.get("cache_hit"):
        return "formatter"
    return "intent_classifier"

def route_after_intent(state: RAGState) -> Literal["retriever", "general_chat", "feedback"]:
    """
    Lightweight router based on intent.
    No heavy computation, pure string comparison.
    """
    intent = state.get("intent", "unknown")
    
    if intent == "sop_question":
        return "retriever"
    elif intent == "feedback":
        return "feedback"
    else:
        return "general_chat"

def route_after_retrieval(state: RAGState) -> Literal["reranker", "context_builder", "formatter"]:
    """Route based on retrieval success"""
    if state.get("retrieved_docs"):
        if settings.RERANKER_ENABLED:
            return "reranker"
        return "context_builder"
    return "formatter"  # Skip to formatter with no context

def route_after_context(state: RAGState) -> Literal["answer_generator", "formatter"]:
    """Route based on context availability"""
    if state.get("context_chunks"):
        return "answer_generator"
    return "formatter"

def route_after_answer(state: RAGState) -> Literal["formatter"]:
    """Always go to formatter after answer generation"""
    return "formatter"

# ============================================================================
# OPTIMIZATION 3: Lazy Node Loading (Only initialize when needed)
# ============================================================================

class LazyNodeLoader:
    """
    Lazily initialize nodes only when first used.
    Accepts pre-initialized services to avoid duplicate loading.
    """
    
    def __init__(self):
        self._nodes = {}
        self._qdrant = None
        self._llm = None
        self._embedding = None
    
    def set_services(
        self,
        qdrant=None,
        llm=None,
        embedding=None
    ):
        """Inject pre-initialized services to avoid re-loading."""
        if qdrant is not None:
            self._qdrant = qdrant
        if llm is not None:
            self._llm = llm
        if embedding is not None:
            self._embedding = embedding
        # Clear cached nodes so they pick up new services
        self._nodes = {}
    
    def get_qdrant(self):
        """Lazy load Qdrant service"""
        if self._qdrant is None:
            from app.services.qdrant_service import QdrantService
            self._qdrant = QdrantService()
        return self._qdrant
    
    def get_llm(self):
        """Reuse pre-initialized LLM or lazy load (heavy)"""
        if self._llm is None:
            from app.services.llm_service import EnterpriseLLMService
            self._llm = EnterpriseLLMService(
                model_path=settings.MODEL_PATH,
                enable_monitoring=settings.MONITORING_ENABLED
            )
        return self._llm
    
    def get_embedding(self):
        """Lazy load embedding service"""
        if self._embedding is None:
            from app.services.embedding_service import EmbeddingService
            self._embedding = EmbeddingService()
        return self._embedding
    
    def get_node(self, node_name: str):
        """Get or create node lazily"""
        if node_name not in self._nodes:
            if node_name == "cache_check":
                from app.engine.nodes import CacheCheckNode
                self._nodes[node_name] = CacheCheckNode()
            elif node_name == "intent_classifier":
                from app.engine.nodes import IntentClassifierNode
                self._nodes[node_name] = IntentClassifierNode()
            elif node_name == "retriever":
                from app.engine.nodes import RetrieverNode
                self._nodes[node_name] = RetrieverNode(self.get_qdrant())
            elif node_name == "context_builder":
                from app.engine.nodes import ContextBuilderNode
                self._nodes[node_name] = ContextBuilderNode()
            elif node_name == "answer_generator":
                from app.engine.nodes import AnswerGeneratorNode
                self._nodes[node_name] = AnswerGeneratorNode(self.get_llm())
            elif node_name == "general_chat":
                from app.engine.nodes import GeneralChatNode
                self._nodes[node_name] = GeneralChatNode(self.get_llm())
            elif node_name == "formatter":
                from app.engine.nodes import FormatterNode
                self._nodes[node_name] = FormatterNode()
            elif node_name == "feedback":
                from app.engine.nodes import FeedbackCollectorNode
                self._nodes[node_name] = FeedbackCollectorNode()
            elif node_name == "reranker":
                from app.services.reranker_service import RerankerService
                self._nodes[node_name] = RerankerService()
        
        return self._nodes[node_name]

# Global lazy loader
node_loader = LazyNodeLoader()

# ============================================================================
# OPTIMIZATION 4: Node Functions (Thin wrappers around node logic)
# ============================================================================

async def cache_check_node(state: RAGState) -> RAGState:
    """Check LMDB cache before processing"""
    start = time.perf_counter()
    
    try:
        node = node_loader.get_node("cache_check")
        result = await node.process(state)
        
        if result.get("cache_hit"):
            state["cache_hit"] = True
            state["response"] = result.get("response", "")
            state["active_sop"] = result.get("active_sop")
            state["intent"] = result.get("intent", "unknown")
            state["sop_count"] = result.get("sop_count", 0)
            state["context_chunks"] = result.get("context_chunks", [])
            state["metadata"] = result.get("metadata", {})
        else:
            state["cache_hit"] = False
            
    except Exception as e:
        logger.error("Cache check node failed", error=str(e))
        state["cache_hit"] = False
        
    finally:
        state["processing_time"]["cache_check"] = time.perf_counter() - start
        
    return state

async def intent_classifier_node(state: RAGState) -> RAGState:
    """Intent classification node"""
    start = time.perf_counter()
    
    try:
        node = node_loader.get_node("intent_classifier")
        result = await node.process(state)
        
        # Update state
        state["intent"] = result["intent"]
        state["intent_confidence"] = result["confidence"]
        state["cache_key"] = result.get("cache_key", state["cache_key"])
        
    except Exception as e:
        logger.error("Intent classification failed", error=str(e))
        state["errors"].append({
            "node": "intent_classifier",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        state["intent"] = "unknown"
        state["intent_confidence"] = 0.0
    
    finally:
        state["processing_time"]["intent_classifier"] = time.perf_counter() - start
    
    return state

async def retriever_node(state: RAGState) -> RAGState:
    """Document retrieval node"""
    start = time.perf_counter()
    
    try:
        node = node_loader.get_node("retriever")
        
        # Generate embedding if needed (lazy)
        if not state.get("query_embedding"):
            embedding_service = node_loader.get_embedding()
            loop = asyncio.get_event_loop()
            state["query_embedding"] = await loop.run_in_executor(
                None,
                lambda: embedding_service.generate_query_embedding(state["query"])
            )
        
        result = await node.process(state)
        state["retrieved_docs"] = result.get("retrieved_docs", [])
        
    except Exception as e:
        logger.error("Retrieval failed", error=str(e))
        state["errors"].append({
            "node": "retriever",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        state["retrieved_docs"] = []
    
    finally:
        state["processing_time"]["retriever"] = time.perf_counter() - start
    
    return state

async def reranker_node(state: RAGState) -> RAGState:
    """Re-rank retrieved documents using Cross-Encoder for better precision."""
    start = time.perf_counter()
    
    try:
        reranker = node_loader.get_node("reranker")
        query = state.get("query", "")
        docs = state.get("retrieved_docs", [])
        
        if docs and reranker:
            loop = asyncio.get_event_loop()
            reranked = await loop.run_in_executor(
                None,
                lambda: reranker.rerank(query, docs)
            )
            state["retrieved_docs"] = reranked
            logger.info(
                "Re-ranking complete",
                original_count=len(docs),
                reranked_count=len(reranked)
            )
    
    except Exception as e:
        logger.error("Re-ranking failed, keeping original docs", error=str(e))
        state["errors"].append({
            "node": "reranker",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
    
    finally:
        state["processing_time"]["reranker"] = time.perf_counter() - start
    
    return state

async def context_builder_node(state: RAGState) -> RAGState:
    """Context building node"""
    start = time.perf_counter()
    
    try:
        node = node_loader.get_node("context_builder")
        result = await node.process(state)
        state["context_chunks"] = result.get("context_chunks", [])
        
    except Exception as e:
        logger.error("Context building failed", error=str(e))
        state["errors"].append({
            "node": "context_builder",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        state["context_chunks"] = []
    
    finally:
        state["processing_time"]["context_builder"] = time.perf_counter() - start
    
    return state

async def answer_generator_node(state: RAGState) -> RAGState:
    """Answer generation node"""
    start = time.perf_counter()
    
    try:
        node = node_loader.get_node("answer_generator")
        result = await node.process(state)
        state["response"] = result.get("response", "")
        state["tokens_used"] = result.get("tokens_used", 0)
        
    except Exception as e:
        logger.error("Answer generation failed", error=str(e))
        state["errors"].append({
            "node": "answer_generator",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        state["response"] = "I encountered an error generating a response."
    
    finally:
        state["processing_time"]["answer_generator"] = time.perf_counter() - start
    
    return state

async def general_chat_node(state: RAGState) -> RAGState:
    """General chat node"""
    start = time.perf_counter()
    
    try:
        node = node_loader.get_node("general_chat")
        result = await node.process(state)
        state["response"] = result.get("response", "")
        state["tokens_used"] = result.get("tokens_used", 0)
        
    except Exception as e:
        logger.error("General chat failed", error=str(e))
        state["errors"].append({
            "node": "general_chat",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        state["response"] = "I'm here to help. Please try again."
    
    finally:
        state["processing_time"]["general_chat"] = time.perf_counter() - start
    
    return state

async def formatter_node(state: RAGState) -> RAGState:
    """Response formatting node"""
    start = time.perf_counter()
    
    try:
        node = node_loader.get_node("formatter")
        result = await node.process(state)
        state["response"] = result.get("response", state["response"])
        state["metadata"] = result.get("metadata", {})
        
    except Exception as e:
        logger.error("Formatting failed", error=str(e))
        state["errors"].append({
            "node": "formatter",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
    
    finally:
        state["processing_time"]["formatter"] = time.perf_counter() - start
    
    return state

async def feedback_node(state: RAGState) -> RAGState:
    """Feedback collection node"""
    start = time.perf_counter()
    
    try:
        node = node_loader.get_node("feedback")
        result = await node.process(state)
        state["response"] = result.get("response", "Thank you for your feedback!")
        
    except Exception as e:
        logger.error("Feedback collection failed", error=str(e))
        state["errors"].append({
            "node": "feedback",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        state["response"] = "Thank you for your feedback."
    
    finally:
        state["processing_time"]["feedback"] = time.perf_counter() - start
    
    return state

# ============================================================================
# OPTIMIZATION 5: Conditional Edge Functions (No heavy computation)
# ============================================================================

def should_continue_to_retrieval(state: RAGState) -> bool:
    """Check if we should proceed to retrieval"""
    return state.get("intent") == "sop_question" and state.get("intent_confidence", 0) > 0.3

def should_use_context(state: RAGState) -> bool:
    """Check if we have valid context"""
    return len(state.get("context_chunks", [])) > 0

# ============================================================================
# OPTIMIZATION 6: LangGraph Graph Builder
# ============================================================================

class RagGraph:
    """
    LangGraph-powered RAG pipeline.
    CPU-optimized while preserving all enterprise features.
    """
    
    def __init__(self):
        self.graph = self._build_graph()
        self.checkpointer = MemorySaver() if settings.LANGGRAPH_CHECKPOINT_ENABLED else None
        self.compiled_graph = None
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph"""
        
        # Create graph with our state schema
        graph = StateGraph(RAGState)
        
        # ====================================================================
        # Add all nodes (lazily loaded)
        # ====================================================================
        graph.add_node("cache_check", cache_check_node)
        graph.add_node("intent_classifier", intent_classifier_node)
        graph.add_node("retriever", retriever_node)
        graph.add_node("reranker", reranker_node)
        graph.add_node("context_builder", context_builder_node)
        graph.add_node("answer_generator", answer_generator_node)
        graph.add_node("general_chat", general_chat_node)
        graph.add_node("formatter", formatter_node)
        graph.add_node("feedback", feedback_node)
        
        # ====================================================================
        # Add edges with conditional routing
        # ====================================================================
        
        # Start -> Cache Check
        graph.add_edge(START, "cache_check")
        
        # Cache Check -> Router (Hit = Formatter, Miss = Intent Classifier)
        graph.add_conditional_edges(
            "cache_check",
            route_after_cache,
            {
                "formatter": "formatter",
                "intent_classifier": "intent_classifier"
            }
        )
        
        # Intent Classifier -> Router
        graph.add_conditional_edges(
            "intent_classifier",
            route_after_intent,
            {
                "retriever": "retriever",
                "general_chat": "general_chat",
                "feedback": "feedback"
            }
        )
        
        # Retriever -> Reranker or Context Builder (conditional)
        graph.add_conditional_edges(
            "retriever",
            route_after_retrieval,
            {
                "reranker": "reranker",
                "context_builder": "context_builder",
                "formatter": "formatter"
            }
        )
        
        # Reranker -> Context Builder
        graph.add_edge("reranker", "context_builder")
        
        # Context Builder -> Answer Generator (conditional)
        graph.add_conditional_edges(
            "context_builder",
            route_after_context,
            {
                "answer_generator": "answer_generator",
                "formatter": "formatter"
            }
        )
        
        # Answer Generator -> Formatter
        graph.add_edge("answer_generator", "formatter")
        
        # General Chat -> Formatter
        graph.add_edge("general_chat", "formatter")
        
        # Feedback -> Formatter
        graph.add_edge("feedback", "formatter")
        
        # Formatter -> END
        graph.add_edge("formatter", END)
        
        return graph
    
    def get_compiled_graph(self):
        """Get compiled graph (lazy compilation)"""
        if self.compiled_graph is None:
            self.compiled_graph = self.graph.compile(
                checkpointer=self.checkpointer
            )
        return self.compiled_graph
    
    async def ainvoke(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        user_id: str = "anonymous",
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Async invoke the graph with a query.
        """
        start_time = time.perf_counter()
        
        # Generate IDs
        conversation_id = conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
        session_id = session_id or str(uuid.uuid4())
        
        # Create initial state (minimal)
        initial_state = RAGState(
            query=query,
            conversation_id=conversation_id,
            user_id=user_id,
            session_id=session_id,
            intent="unknown",
            intent_confidence=0.0,
            query_embedding=None,
            history=[],
            retrieved_docs=[],
            context_chunks=[],
            response="",
            tokens_used=0,
            status="processing",
            processing_time={},
            errors=[],
            metadata={},
            ab_test_group=None,
            cache_key=f"{xxhash.xxh64(query.encode()).hexdigest()}:{conversation_id}",
            cache_hit=False,
            cached_response=None
        )
        
        # Add A/B test group if enabled
        if settings.AB_TESTING_ENABLED:
            import random
            initial_state["ab_test_group"] = random.choice(settings.AB_TEST_GROUPS)
        
        # Execute graph
        try:
            compiled = self.get_compiled_graph()
            
            # Add config with checkpoint if enabled
            config = {"configurable": {"thread_id": conversation_id}}
            
            # Prepare input data. 
            # Note: We do NOT pass an empty 'history' list here, 
            # as LangGraph will merge (add) it with the existing history in the checkpoint.
            input_data = {
                "query": query,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "session_id": session_id,
                "status": "processing",
                "cache_key": f"{xxhash.xxh64(query.encode()).hexdigest()}:{conversation_id}",
                "processing_time": {},
                "retrieved_docs": [],
                "context_chunks": [],
                "metadata": {}
            }
            
            final_state = await compiled.ainvoke(input_data, config=config)
            
            # Add total processing time
            final_state["processing_time"]["total"] = time.perf_counter() - start_time
            
            return self._format_output(final_state)
            
        except Exception as e:
            logger.error("Graph execution failed", error=str(e))
            return {
                "response": "I encountered an error processing your request.",
                "conversation_id": conversation_id,
                "status": "error",
                "has_error": True,
                "error": str(e)
            }
    
    def _format_output(self, state: RAGState) -> Dict[str, Any]:
        """Format state into output dictionary"""
        sources = state.get("context_chunks", [])
        return {
            "response": state.get("response", ""),
            "conversation_id": state.get("conversation_id", ""),
            "session_id": state.get("session_id", ""),
            "intent": state.get("intent", "unknown"),
            "confidence": state.get("intent_confidence", 0.0),
            "status": "success" if not state.get("errors") else "partial_success",
            "has_error": len(state.get("errors", [])) > 0,
            "sop_count": len(sources),
            "active_sop": sources[0].get("metadata", {}).get("title") if sources else None,
            "processing_time": state.get("processing_time", {}),
            "sources": sources,
            "metadata": {
                "tokens_used": state.get("tokens_used", 0),
                "sources_used": len(sources),
                "ab_test_group": state.get("ab_test_group"),
                "errors": state.get("errors", [])
            }
        }
    

# Global graph instance (lazy loaded)
_rag_graph = None

def get_rag_graph(qdrant=None, llm=None, embedding=None) -> RagGraph:
    """Get or create the RAG graph (lazy loading). Injects pre-initialized services."""
    global _rag_graph
    if _rag_graph is None:
        # Inject services BEFORE building graph so nodes reuse them
        if any([qdrant, llm, embedding]):
            node_loader.set_services(qdrant=qdrant, llm=llm, embedding=embedding)
        _rag_graph = RagGraph()
    return _rag_graph