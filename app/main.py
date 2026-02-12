import uuid
import logging
from fastapi import FastAPI, HTTPException
from app.models import SOPCreate, SOPSearchResult, QueryRequest, ChatRequest, ChatResponse
from app.services.embedding_service import EmbeddingService
from app.services.milvus_service import MilvusService
from app.services.chunk_service import ChunkService
from app.engine.nlp_engine import NLPEngine
from app.engine.nn_engine import NNEngine
from app.engine.graph import build_chat_graph

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="SOP Fetching Engine")

# ─── Initialize Services ──────────────────────────────────────────
embedding_service = EmbeddingService()
milvus_service = MilvusService()
chunk_service = ChunkService()

# ─── Initialize Engines ───────────────────────────────────────────
nlp_engine = NLPEngine(embedding_service, milvus_service)
nn_engine = NNEngine()

# ─── Build LangGraph ──────────────────────────────────────────────
chat_graph = build_chat_graph(nlp_engine, nn_engine)


# ─── SOP Ingestion ────────────────────────────────────────────────
@app.post("/sops/", response_model=dict)
async def add_sop(sop: SOPCreate):
    try:
        chunks = chunk_service.chunk_text(sop.content)
        embeddings = embedding_service.generate_embeddings(chunks)
        link = sop.link or ""
        milvus_service.insert_sop(sop.title, chunks, embeddings, link)
        return {"message": "SOP added successfully", "chunks_count": len(chunks)}
    except Exception as e:
        logging.error(f"Error adding SOP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── SOP Search (direct) ──────────────────────────────────────────
@app.post("/search/", response_model=list[SOPSearchResult])
async def search_sops(request: QueryRequest):
    try:
        query_embedding = embedding_service.generate_query_embedding(request.query)
        results = milvus_service.search_sops(query_embedding, request.top_k)
        return results
    except Exception as e:
        logging.error(f"Error searching SOPs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Chat Assistant ────────────────────────────────────────────────
@app.post("/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())

        initial_state = {
            "message": request.message,
            "session_id": session_id,
            "is_followup": False,
            "intent": None,
            "sop_results": None,
            "sop_context": None,
            "sop_title": None,
            "sop_link": None,
            "summary": None,
            "steps": None,
            "answer": None,
            "chat_history": None,
        }

        result = chat_graph.invoke(initial_state)

        return ChatResponse(
            answer=result.get("answer", ""),
            sop_title=result.get("sop_title"),
            summary=result.get("summary"),
            steps=result.get("steps"),
            link=result.get("sop_link"),
            session_id=session_id
        )
    except Exception as e:
        logging.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Health ────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
