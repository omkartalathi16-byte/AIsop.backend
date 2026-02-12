from pydantic import BaseModel
from typing import List, Optional

class SOPCreate(BaseModel):
    title: str
    content: str
    metadata: Optional[dict] = None

class SOPSearchResult(BaseModel):
    title: str
    content: str
    score: float
    metadata: Optional[dict] = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
