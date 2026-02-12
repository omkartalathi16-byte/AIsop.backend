from fastapi import FastAPI, HTTPException
from app.models import SOPCreate, SOPSearchResult, QueryRequest
from app.services.embedding_service import EmbeddingService
from app.services.milvus_service import MilvusService
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="SOP Fetching Engine")

# Initialize services
# Note: Embedding service might take a while to load the model on first run
embedding_service = EmbeddingService()
milvus_service = MilvusService()

@app.post("/sops/", response_model=dict)
async def add_sop(sop: SOPCreate):
    try:
        embedding = embedding_service.generate_query_embedding(sop.content)
        milvus_service.insert_sop(sop.title, sop.content, embedding)
        return {"message": "SOP added successfully"}
    except Exception as e:
        logging.error(f"Error adding SOP: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/", response_model=list[SOPSearchResult])
async def search_sops(request: QueryRequest):
    try:
        query_embedding = embedding_service.generate_query_embedding(request.query)
        results = milvus_service.search_sops(query_embedding, request.top_k)
        return results
    except Exception as e:
        logging.error(f"Error searching SOPs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
