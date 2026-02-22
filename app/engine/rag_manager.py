"""
Updated Enterprise RAG Manager with LangGraph integration.
CPU-optimized while preserving all features.
"""

import asyncio
import time
from typing import Dict, Any, Optional
import uuid

from .config import settings
from .rag_graph import get_rag_graph
from app.services.llm_service import EnterpriseLLMService
from app.services.qdrant_service import QdrantService
from app.services.embedding_service import EmbeddingService

import structlog
logger = structlog.get_logger()

class EnterpriseRagManager:
    """
    Enterprise RAG Manager with LangGraph orchestration.
    CPU-optimized for i5-4200U while preserving all features.
    """
    
    def __init__(
        self,
        qdrant_service: Optional[QdrantService] = None,
        llm_service: Optional[EnterpriseLLMService] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        # Services are lazily initialized by the graph when needed
        self.qdrant_service = qdrant_service
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        
        # Get LangGraph instance, injecting pre-initialized services
        self.graph = get_rag_graph(
            qdrant=qdrant_service,
            llm=llm_service,
            embedding=embedding_service
        )
        
        logger.info(
            "EnterpriseRagManager initialized",
            resource_mode=settings.RESOURCE_MODE.value,
            features={
                "intent_ml": settings.INTENT_USE_ML,
                "intent_spacy": settings.INTENT_USE_SPACY,
                "intent_transformer": settings.INTENT_USE_TRANSFORMER,
                "retrieval_strategies": settings.RETRIEVAL_STRATEGIES,
                "ab_testing": settings.AB_TESTING_ENABLED,
                "monitoring": settings.MONITORING_ENABLED
            }
        )
    
    async def chat(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """
        Process chat request through LangGraph pipeline.
        
        Args:
            query: User query
            conversation_id: Optional conversation ID
            user_id: Optional user ID
            
        Returns:
            Dict with response and metadata
        """
        start_time = time.perf_counter()
        
        try:
            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
            
            # Execute graph
            result = await self.graph.ainvoke(
                query=query,
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            # Add total processing time
            result["total_time"] = time.perf_counter() - start_time
            
            logger.info(
                "Chat processed",
                conversation_id=conversation_id,
                intent=result.get("intent"),
                status=result.get("status"),
                time=result["total_time"]
            )
            
            return result
            
        except Exception as e:
            logger.exception("Chat processing failed", error=str(e))
            return {
                "response": "I encountered an error processing your request.",
                "conversation_id": conversation_id or "unknown",
                "status": "error",
                "has_error": True,
                "error": str(e),
                "total_time": time.perf_counter() - start_time
            }
    
    async def submit_feedback(
        self,
        conversation_id: str,
        feedback: Dict[str, Any],
        user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """Submit feedback for a conversation"""
        try:
            # Create a feedback-specific query
            result = await self.graph.ainvoke(
                query="feedback",
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            return {
                "status": "success",
                "message": "Feedback received",
                "conversation_id": conversation_id
            }
            
        except Exception as e:
            logger.error("Feedback submission failed", error=str(e))
            return {
                "status": "error",
                "message": str(e),
                "conversation_id": conversation_id
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "resource_mode": settings.RESOURCE_MODE.value,
            "config": {
                "model_threads": settings.MODEL_THREADS,
                "model_context": settings.MODEL_CONTEXT_SIZE,
                "cache_size": settings.CACHE_MEMORY_MAXSIZE,
                "retrieval_strategies": settings.RETRIEVAL_STRATEGIES,
                "ab_testing": settings.AB_TESTING_ENABLED
            }
        }

# Async context manager for easy usage
class RagSession:
    """Context manager for RAG sessions"""
    
    def __init__(self, manager: EnterpriseRagManager):
        self.manager = manager
        self.conversation_id = None
    
    async def __aenter__(self):
        self.conversation_id = f"session_{uuid.uuid4().hex[:8]}"
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def chat(self, query: str, user_id: str = "anonymous") -> Dict[str, Any]:
        """Send a chat message in this session"""
        return await self.manager.chat(
            query=query,
            conversation_id=self.conversation_id,
            user_id=user_id
        )

# Usage example
async def example():
    # Initialize manager
    manager = EnterpriseRagManager()
    
    # Simple chat
    result = await manager.chat(
        query="How do I submit an expense report?",
        user_id="user123"
    )
    print(f"Response: {result['response']}")
    print(f"Intent: {result.get('intent')}")
    print(f"Time: {result.get('total_time'):.2f}s")
    
    # Session-based chat
    async with RagSession(manager) as session:
        result1 = await session.chat("Hello")
        result2 = await session.chat("What's the return policy?")
    
    # Submit feedback
    await manager.submit_feedback(
        conversation_id=result["conversation_id"],
        feedback={"rating": 5, "comment": "Helpful"}
    )

if __name__ == "__main__":
    asyncio.run(example())