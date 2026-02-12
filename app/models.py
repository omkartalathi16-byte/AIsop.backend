from pydantic import BaseModel
from typing import List, Optional


class SOPCreate(BaseModel):
    title: str
    content: str
    link: Optional[str] = None
    metadata: Optional[dict] = None


class SOPSearchResult(BaseModel):
    title: str
    content: str
    score: float
    link: Optional[str] = None
    metadata: Optional[dict] = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sop_title: Optional[str] = None
    summary: Optional[str] = None
    steps: Optional[List[str]] = None
    link: Optional[str] = None
    session_id: str
