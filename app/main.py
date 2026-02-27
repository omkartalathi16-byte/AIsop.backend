import os
import time
import secrets
import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from app.engine.config import settings
from app.models import (
    SOPCreate, SOPBatchCreate, SOPSearchResult,
    QueryRequest, ChatRequest, ChatResponse,
)
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.services.chunk_service import ChunkService
from app.engine.rag_manager import EnterpriseRagManager
from app.services.llm_service import EnterpriseLLMService

# ─── Structured Logging ───────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger("main")

# ─── Environment Validation ───────────────────────────────────────────────────
def _validate_env():
    errors = []
    if not settings.OPENROUTER_API_KEY:
        errors.append("OPENROUTER_API_KEY is not set")
    if not settings.API_KEY or settings.API_KEY == "change-me-to-a-strong-secret":
        logger.warning("API_KEY is not set or is still the default placeholder — API is unprotected!")
    if errors:
        for e in errors:
            logger.error(f"[ENV] {e}")
        raise RuntimeError(f"Missing required environment variables: {', '.join(errors)}")

# ─── Prometheus Metrics ───────────────────────────────────────────────────────
HTTP_REQUESTS   = Counter("http_requests_total",    "Total HTTP requests",       ["method", "path", "status"])
HTTP_LATENCY    = Histogram("http_request_duration_seconds", "HTTP request latency", ["path"])
CHAT_REQUESTS   = Counter("chat_requests_total",    "Total chat requests",       ["status"])
INGEST_REQUESTS = Counter("ingest_requests_total",  "Total ingestion requests",  ["status"])
ACTIVE_CONNS    = Gauge("active_connections",        "Currently active connections")

# ─── In-Memory Rate Limiter ───────────────────────────────────────────────────
_rate_buckets: dict[str, list[float]] = defaultdict(list)

def _is_rate_limited(client_ip: str) -> bool:
    if not settings.RATE_LIMIT_ENABLED:
        return False
    now = time.time()
    window = settings.RATE_LIMIT_WINDOW
    bucket = _rate_buckets[client_ip]
    # Drop timestamps outside the window
    _rate_buckets[client_ip] = [t for t in bucket if now - t < window]
    if len(_rate_buckets[client_ip]) >= settings.RATE_LIMIT_REQUESTS:
        return True
    _rate_buckets[client_ip].append(now)
    return False

# ─── Services (initialised at startup) ───────────────────────────────────────
embedding_service: Optional[EmbeddingService] = None
qdrant_service:    Optional[QdrantService]     = None
chunk_service:     Optional[ChunkService]      = None
llm_service:       Optional[EnterpriseLLMService] = None
rag_manager:       Optional[EnterpriseRagManager] = None

# ─── Background-task status tracking ─────────────────────────────────────────
_ingestion_status: dict[str, dict] = {}   # job_id -> {status, detail, ts}

# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_service, qdrant_service, chunk_service, llm_service, rag_manager
    _validate_env()
    logger.info("Initialising services...")
    embedding_service = EmbeddingService()
    qdrant_service    = QdrantService()
    chunk_service     = ChunkService()
    llm_service       = EnterpriseLLMService()
    rag_manager       = EnterpriseRagManager(qdrant_service, llm_service, embedding_service)
    logger.info("All services ready.")
    yield
    logger.info("Shutting down.")

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="AIsop Backend", version="1.0.0", lifespan=lifespan)

# CORS — hard origins from .env
allowed_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False if "*" in allowed_origins else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Middleware: metrics + rate limit ────────────────────────────────────────
@app.middleware("http")
async def metrics_and_rate_limit(request: Request, call_next):
    ACTIVE_CONNS.inc()
    client_ip = request.client.host if request.client else "unknown"
    path      = request.url.path

    # Rate limit (skip /health and /metrics)
    if path not in ("/health", "/metrics") and _is_rate_limited(client_ip):
        ACTIVE_CONNS.dec()
        HTTP_REQUESTS.labels(method=request.method, path=path, status=429).inc()
        return Response(content='{"detail":"Rate limit exceeded"}', status_code=429, media_type="application/json")

    start = time.time()
    response = await call_next(request)
    elapsed  = time.time() - start

    HTTP_REQUESTS.labels(method=request.method, path=path, status=response.status_code).inc()
    HTTP_LATENCY.labels(path=path).observe(elapsed)
    ACTIVE_CONNS.dec()
    return response

# ─── API Key dependency ───────────────────────────────────────────────────────
async def require_api_key(request: Request):
    if not settings.API_KEY:
        return  # No key configured — open access (dev mode)
    provided = request.headers.get("X-API-KEY", "")
    if not secrets.compare_digest(provided, settings.API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ─── Background Ingestion ─────────────────────────────────────────────────────
def _run_ingestion(job_id: str, sop: SOPCreate):
    _ingestion_status[job_id] = {"status": "running", "title": sop.title, "ts": time.time()}
    try:
        logger.info("ingestion_started", job_id=job_id, title=sop.title)
        chunks     = chunk_service.chunk_text(sop.content)
        embeddings = embedding_service.generate_embeddings(chunks)
        qdrant_service.insert_sop(
            title=sop.title,
            chunks=chunks,
            embeddings=embeddings,
            sop_link=sop.sop_link or "",
            threat_type=sop.threat_type or "",
            category=sop.category or "",
        )
        _ingestion_status[job_id] = {"status": "done", "title": sop.title, "chunks": len(chunks), "ts": time.time()}
        INGEST_REQUESTS.labels(status="success").inc()
        logger.info("ingestion_done", job_id=job_id, chunks=len(chunks))
    except Exception as exc:
        _ingestion_status[job_id] = {"status": "error", "title": sop.title, "error": str(exc), "ts": time.time()}
        INGEST_REQUESTS.labels(status="error").inc()
        logger.error("ingestion_failed", job_id=job_id, error=str(exc))

# ─── Routes ───────────────────────────────────────────────────────────────────

# --- Ingestion ---
@app.post("/sops/", status_code=202, dependencies=[Depends(require_api_key)])
async def add_sop(sop: SOPCreate, background_tasks: BackgroundTasks):
    job_id = f"sop-{int(time.time()*1000)}"
    background_tasks.add_task(_run_ingestion, job_id, sop)
    return {"message": "SOP ingestion started", "job_id": job_id, "title": sop.title}

@app.post("/sops/batch/", status_code=202, dependencies=[Depends(require_api_key)])
async def add_sop_batch(batch: SOPBatchCreate, background_tasks: BackgroundTasks):
    job_ids = []
    for sop in batch.items:
        job_id = f"sop-{sop.title[:20].replace(' ', '_')}-{int(time.time()*1000)}"
        background_tasks.add_task(_run_ingestion, job_id, sop)
        job_ids.append(job_id)
    return {"message": "Batch ingestion started", "count": len(batch.items), "job_ids": job_ids}

@app.get("/sops/status/{job_id}", dependencies=[Depends(require_api_key)])
async def ingestion_status(job_id: str):
    status = _ingestion_status.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status

# --- Search ---
@app.post("/search/", response_model=list[SOPSearchResult], dependencies=[Depends(require_api_key)])
async def search_sops(request: QueryRequest):
    try:
        query_embedding = embedding_service.generate_query_embedding(request.query)
        results         = qdrant_service.search_sops(query_embedding, request.top_k)
        return results
    except Exception as exc:
        logger.error("search_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

# --- Chat ---
@app.post("/chat/", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
async def chat_interaction(request: ChatRequest):
    try:
        response = await rag_manager.chat(
            query=request.query,
            conversation_id=request.conversation_id,
            user_id=request.user_id,
        )
        CHAT_REQUESTS.labels(status="success").inc()
        return response
    except Exception as exc:
        CHAT_REQUESTS.labels(status="error").inc()
        logger.error("chat_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

# --- Health ---
@app.get("/health")
async def health_check():
    qdrant_ok = False
    qdrant_detail = ""
    try:
        collections = qdrant_service.client.get_collections()
        names = [c.name for c in collections.collections]
        qdrant_ok = settings.QDRANT_COLLECTION in names
        qdrant_detail = f"collection '{settings.QDRANT_COLLECTION}' found"
    except Exception as exc:
        qdrant_detail = str(exc)

    llm_status = llm_service.health_status if llm_service else {}
    llm_ok = llm_status.get("status") == "healthy"

    overall = "healthy" if (qdrant_ok and llm_ok) else "degraded"
    return {
        "status": overall,
        "qdrant": {"ok": qdrant_ok, "detail": qdrant_detail},
        "llm":    {"ok": llm_ok,    "detail": llm_status.get("error", "")},
    }

# --- Prometheus metrics ---
@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
