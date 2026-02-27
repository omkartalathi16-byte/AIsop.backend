# AIsop Backend

A robust RAG (Retrieval-Augmented Generation) backend for Standard Operating Procedures using FastAPI, Qdrant, and OpenRouter.

## Features
- **Vector Storage**: Qdrant (Docker) for fast similarity search.
- **Embeddings**: High-performance local embeddings (`bge-small-en-v1.5`).
- **LLM Integration**: Context-aware chat using OpenRouter.
- **Production Ready**: API Key auth, Rate limiting, Prometheus metrics, and Structured logging.
- **Async Ingestion**: Background tasks for handling large documents.

## Prerequisites
- Docker Desktop
- Python 3.9+
- OpenRouter API Key

## Setup

1. **Configure Environment:**
   Copy `.env.example` (if available) or create `.env` based on the provided settings.

2. **Start Qdrant:**
   ```bash
   docker-compose up -d
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API:**
   ```bash
   python -m app.main
   ```

## API Endpoints

All protected endpoints require the `X-API-KEY` header.

### 1. Chat (RAG)
`POST /chat/`
```json
{
  "query": "How do I reset my password?",
  "conversation_id": "optional-uuid",
  "user_id": "optional-user-id"
}
```

### 2. Add SOP (Async)
`POST /sops/`
```json
{
  "title": "IT Policy",
  "content": "Full text of the SOP...",
  "category": "IT",
  "threat_type": "Security"
}
```

### 3. Search SOPs
`POST /search/`
```json
{
  "query": "password policy",
  "top_k": 3
}
```

### 4. Observability
- `GET /health` - System health status.
- `GET /metrics` - Prometheus metrics.
- `GET /sops/status/{job_id}` - Check ingestion progress.
