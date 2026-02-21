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
        """Search for SOPs using Qdrant - version compatible implementation"""
        try:
            def extract_hits(result):
                if result is None:
                    return []
                if hasattr(result, "points"):
                    return result.points
                if isinstance(result, dict):
                    return result.get("result", []) or result.get("points", [])
                return result

            search_result = None
            if hasattr(self.client, 'search'):
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    with_payload=True
                )
            elif hasattr(self.client, 'query_points'):
                search_result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    limit=top_k,
                    with_payload=True
                )

            hits = extract_hits(search_result)
            if hits:
                formatted_results = []
                for hit in hits:
                    formatted_results.append({
                        "id": hit.id,
                        "title": hit.payload.get("title") if hit.payload else "",
                        "content": hit.payload.get("content") if hit.payload else "",
                        "sop_link": hit.payload.get("sop_link") if hit.payload else "",
                        "threat_type": hit.payload.get("threat_type") if hit.payload else "",
                        "category": hit.payload.get("category") if hit.payload else "",
                        "score": hit.score
                    })
                return formatted_results

            logging.warning("Falling back to manual similarity for Qdrant search")
            all_points = []
            if hasattr(self.client, 'get_all'):
                all_points = self.client.get_all(collection_name=self.collection_name)
            elif hasattr(self.client, 'scroll'):
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    with_payload=True,
                    with_vectors=True
                )
                all_points = scroll_result[0]

            import numpy as np
            query_norm = np.linalg.norm(query_embedding)
            scored_results = []
            for point in all_points:
                if hasattr(point, 'vector') and point.vector:
                    point_vector = point.vector
                    point_norm = np.linalg.norm(point_vector)
                    if query_norm > 0 and point_norm > 0:
                        similarity = np.dot(query_embedding, point_vector) / (query_norm * point_norm)
                        scored_results.append({
                            "id": point.id,
                            "title": point.payload.get("title") if point.payload else "",
                            "content": point.payload.get("content") if point.payload else "",
                            "sop_link": point.payload.get("sop_link") if point.payload else "",
                            "threat_type": point.payload.get("threat_type") if point.payload else "",
                            "category": point.payload.get("category") if point.payload else "",
                            "score": similarity
                        })

            scored_results.sort(key=lambda x: x["score"], reverse=True)
            return scored_results[:top_k]

        except Exception as e:
            logging.error(f"Qdrant search failed: {e}")
            # Return empty results on error
            return []
