from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
import time
from typing import List, Dict, Any, Optional

class QdrantService:
    def __init__(self, host="127.0.0.1", port=6333, collection_name="sop_collection", dim=384):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self.client = QdrantClient(host=self.host, port=self.port)
        self._set_up_collection()

    def _set_up_collection(self):
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
                )
                logging.info(f"Created new Qdrant collection: {self.collection_name}")
            else:
                logging.info(f"Using existing Qdrant collection: {self.collection_name}")
        except Exception as e:
            logging.error(f"Error setting up Qdrant collection: {e}")

    def insert_sop(self, title: str, chunks: list[str], embeddings: list[list[float]], 
                   sop_link: str = "", threat_type: str = "", category: str = ""):
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            payload = {
                "title": title,
                "content": chunk,
                "sop_link": sop_link,
                "threat_type": threat_type,
                "category": category,
                "timestamp": time.time()
            }
            # Use a unique ID (random uuid or hash of content)
            import uuid
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logging.info(f"Inserted {len(points)} points into Qdrant collection: {self.collection_name}")

    def search_sops(self, query_embedding: list[float], top_k: int = 5):
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )

        formatted_results = []
        for hit in search_result:
            formatted_results.append({
                "id": hit.id,
                "title": hit.payload.get("title"),
                "content": hit.payload.get("content"),
                "sop_link": hit.payload.get("sop_link"),
                "threat_type": hit.payload.get("threat_type"),
                "category": hit.payload.get("category"),
                "score": hit.score
            })
        return formatted_results
