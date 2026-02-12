import logging
from fastapi import FastAPI, HTTPException
from app.models import SOPCreate, SOPSearchResult, QueryRequest, DeriveRequest, DeriveResponse
from app.services.embedding_service import EmbeddingService
from app.services.milvus_service import MilvusService
from app.services.chunk_service import ChunkService
from app.services.derive_engine import DeriveEngine

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="SOP Fetching Engine")

# ─── Initialize Services ──────────────────────────────────────────
embedding_service = EmbeddingService()
milvus_service = MilvusService()
chunk_service = ChunkService()
derive_engine = DeriveEngine(embedding_service, milvus_service)


# ─── SOP Ingestion ────────────────────────────────────────────────
@app.post("/sops/", response_model=dict)
async def add_sop(sop: SOPCreate):
    try:
        chunks = chunk_service.chunk_text(sop.content)
        embeddings = embedding_service.generate_embeddings(chunks)
        milvus_service.insert_sop(
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
        results = milvus_service.search_sops(query_embedding, request.top_k)
        return results
    except Exception as e:
        logging.error(f"Error searching SOPs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── SOP Derive (hybrid retrieval) ────────────────────────────────
@app.post("/derive/", response_model=DeriveResponse)
async def derive_sop(request: DeriveRequest):
    try:
        result = derive_engine.derive(
            query=request.query,
            top_k=request.top_k,
            threat_type=request.threat_type,
            category=request.category
        )
        return result
    except Exception as e:
        logging.error(f"Error deriving SOP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Health ────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
