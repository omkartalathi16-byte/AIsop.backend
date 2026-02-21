import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class RagManager:
    def __init__(self, qdrant_service, llm_service, embedding_service):
        self.qdrant_service = qdrant_service
        self.llm_service = llm_service
        self.embedding_service = embedding_service

    def chat(self, query: str, conversation_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        Streamlined RAG flow: Query -> Search -> Synthesize.
        Includes a "short-circuit" for simple greetings to improve responsiveness.
        """
        try:
            query_lower = query.lower().strip()
            
            # 1. Detect Greetings/Conversation (Fast Path)
            greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "thanks", "thank you"]
            if query_lower in greetings or len(query_lower) < 3:
                logger.info(f"Greeting detected for: {query}. Skipping RAG.")
                # Call LLM directly without context for conversational speed
                response_text = self.llm_service.generate_response(
                    query, 
                    system_prompt="You are a helpful SOP assistant. Greet the user and ask how you can help them with procedures today."
                )
                return {
                    "response": response_text,
                    "conversation_id": conversation_id or "default",
                    "active_sop": None,
                    "intent": "greeting",
                    "has_error": False,
                    "sop_count": 0
                }

            # 2. Generate Query Embedding
            logger.info(f"Generating embedding for query: {query}")
            query_embedding = self.embedding_service.generate_query_embedding(query)

            # 3. Search Qdrant for SOPs
            logger.info("Searching Qdrant for relevant SOP chunks")
            search_results = self.qdrant_service.search_sops(query_embedding, top_k=2)
            
            # Filter low-score results if score available to avoid noise
            if search_results:
                # Only keep results with reasonable score (> 0.2)
                search_results = [r for r in search_results if r.get("score", 1.0) > 0.2]

            if not search_results:
                logger.info("No relevant chunks found in Qdrant")
                return {
                    "response": "I couldn't find any relevant SOPs for that request. Could you please rephrase or mention a specific procedure?",
                    "conversation_id": conversation_id or "default",
                    "active_sop": None,
                    "intent": "search_failed",
                    "has_error": False,
                    "sop_count": 0
                }

            # 4. Synthesize Response using LLM
            logger.info(f"Synthesizing response using {len(search_results)} chunks")
            context_chunks = []
            for res in search_results:
                context_chunks.append({
                    "content": res.get("content", ""),
                    "metadata": {
                        "title": res.get("title", "Unknown"),
                        "sop_link": res.get("sop_link", "")
                    }
                })

            response_text = self.llm_service.synthesize_rag_response(query, context_chunks)
            
            # 4. Extract active SOP title (highest score)
            active_sop = search_results[0].get("title")
            sop_link = search_results[0].get("sop_link")
            
            # Append source link if available
            if sop_link:
                response_text += f"\n\n🔗 **Source:** [View Document]({sop_link})"

            return {
                "response": response_text,
                "conversation_id": conversation_id or "default",
                "active_sop": active_sop,
                "intent": "rag_success",
                "has_error": False,
                "sop_count": len(search_results)
            }

        except Exception as e:
            logger.exception(f"Error in RagManager chat: {e}")
            return {
                "response": f"I encountered an error processing your request: {str(e)}",
                "conversation_id": conversation_id or "default",
                "active_sop": None,
                "intent": "error",
                "has_error": True,
                "sop_count": 0
            }
