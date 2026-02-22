import asyncio
import logging
from app.engine.rag_manager import EnterpriseRagManager
from app.services.llm_service import EnterpriseLLMService
from app.services.qdrant_service import QdrantService
from app.services.embedding_service import EmbeddingService

logging.basicConfig(level=logging.INFO)

async def main():
    print("Initializing services...")
    embedding_service = EmbeddingService()
    qdrant_service = QdrantService()
    llm_service = EnterpriseLLMService()
    
    rag_manager = EnterpriseRagManager(qdrant_service, llm_service, embedding_service)
    
    print("\nAttempting chat...")
    try:
        result = await rag_manager.chat(
            query="Hello, how are you?",
            conversation_id="test_repro"
        )
        print("\nResult:")
        print(result)
    except Exception as e:
        print(f"\nCaught Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
