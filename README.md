# SOP Fetching Engine

A semantic search backend for Standard Operating Procedures using FastAPI, Milvus, and SentenceTransformers.

## Features
- Vector storage using Milvus (Docker).
- Semantic embeddings using `all-MiniLM-L6-v2`.
- FastAPI endpoints for adding and searching SOPs.

## Prerequisites
- Docker Desktop
- Python 3.9+

## Setup

1. **Start Milvus:**
   ```bash
   docker-compose up -d
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API:**
   ```bash
   python -m app.main
   ```

## API Endpoints

### 1. Add SOP
`POST /sops/`
```json
{
  "title": "IT Policy",
  "content": "All employees must change their passwords every 90 days."
}
```

### 2. Search SOPs
`POST /search/`
```json
{
  "query": "password change policy",
  "top_k": 5
}
```

### 3. Health Check
`GET /health`
