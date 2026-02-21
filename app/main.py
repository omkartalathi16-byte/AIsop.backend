import logging
from fastapi import FastAPI, HTTPException
from app.models import SOPCreate, SOPSearchResult, QueryRequest, ChatRequest, ChatResponse
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.services.chunk_service import ChunkService
from app.engine.rag_manager import RagManager
from app.services.llm_service import LLMService

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="SOP Fetching Engine")

# Add CORS Middleware
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ─── Initialize Services ──────────────────────────────────────────
embedding_service = EmbeddingService()
qdrant_service = QdrantService()
chunk_service = ChunkService()

# Initialize LLM Service (Load Qwen 2.5)
llm_service = LLMService()

# Initialize Simplified RAG Manager
rag_manager = RagManager(qdrant_service, llm_service, embedding_service)


# ─── SOP Ingestion ────────────────────────────────────────────────
@app.post("/sops/", response_model=dict)
async def add_sop(sop: SOPCreate):
    try:
        chunks = chunk_service.chunk_text(sop.content)
        embeddings = embedding_service.generate_embeddings(chunks)
        qdrant_service.insert_sop(
            title=sop.title,
            chunks=chunks,
            embeddings=embeddings,
            sop_link=sop.sop_link or "",
            threat_type=sop.threat_type or "",
            category=sop.category or ""
        )
        return {"message": "SOP added successfully", "chunks_count": len(chunks)}
    except Exception as e:
        logging.error(f"Error adding SOP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── SOP Search (direct similarity) ───────────────────────────────
@app.post("/search/", response_model=list[SOPSearchResult])
async def search_sops(request: QueryRequest):
    try:
        query_embedding = embedding_service.generate_query_embedding(request.query)
        results = qdrant_service.search_sops(query_embedding, request.top_k)
        return results
    except Exception as e:
        logging.error(f"Error searching SOPs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Chat Assistant ────────────────────────────────────────────────
@app.post("/chat/", response_model=ChatResponse)
async def chat_interaction(request: ChatRequest):
    try:
        response = rag_manager.chat(
            query=request.query,
            conversation_id=request.conversation_id,
            user_id=request.user_id
        )
        return response
    except Exception as e:
        logging.error(f"Error in chat interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Health ────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
