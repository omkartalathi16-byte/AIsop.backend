from pydantic import BaseModel
from typing import List, Optional


class SOPCreate(BaseModel):
    title: str
    content: str
    sop_link: Optional[str] = None
    threat_type: Optional[str] = None
    category: Optional[str] = None


class SOPSearchResult(BaseModel):
    title: str
    content: str
    score: float
    sop_link: Optional[str] = None
    threat_type: Optional[str] = None
    category: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class DeriveRequest(BaseModel):
    query: str
    top_k: int = 5
    threat_type: Optional[str] = None
    category: Optional[str] = None


class MatchedChunk(BaseModel):
    content: str
    similarity_score: float


class DeriveResult(BaseModel):
    title: str
    sop_link: Optional[str] = None
    confidence_score: float
    matched_chunks: List[MatchedChunk]
    metadata: dict


class DeriveResponse(BaseModel):
    query: str
    results: List[DeriveResult]
    total_results: int
