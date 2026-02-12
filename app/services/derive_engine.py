"""
Derive Engine — Hybrid SOP Derivation
Combines embedding similarity search with metadata keyword boosting
to return structured SOP results with confidence scores.
"""
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Keyword boost weight — how much metadata matches influence the final score
METADATA_BOOST = 0.15


class DeriveEngine:
    """Hybrid derivation: embedding similarity + metadata keyword boosting."""

    def __init__(self, embedding_service, milvus_service):
        self.embedding_service = embedding_service
        self.milvus_service = milvus_service

    def _compute_metadata_boost(self, result: dict, query: str,
                                 threat_type: str = None, category: str = None) -> float:
        """Compute a metadata boost score based on keyword overlap."""
        boost = 0.0
        query_lower = query.lower()

        # Boost if threat_type filter matches
        result_threat = (result.get("threat_type") or "").lower()
        if threat_type and result_threat:
            if threat_type.lower() in result_threat or result_threat in threat_type.lower():
                boost += METADATA_BOOST

        # Boost if category filter matches
        result_category = (result.get("category") or "").lower()
        if category and result_category:
            if category.lower() in result_category or result_category in category.lower():
                boost += METADATA_BOOST

        # Boost if query keywords appear in title
        title_lower = (result.get("title") or "").lower()
        query_words = [w for w in query_lower.split() if len(w) > 3]
        if query_words:
            matches = sum(1 for w in query_words if w in title_lower)
            boost += (matches / len(query_words)) * METADATA_BOOST

        return min(boost, METADATA_BOOST * 3)  # cap at 3x boost

    def derive(self, query: str, top_k: int = 5,
               threat_type: str = None, category: str = None) -> dict:
        """
        Full derivation pipeline:
        1. Generate query embedding
        2. Search Milvus for similar chunks
        3. Apply metadata keyword boosting
        4. Group chunks by SOP title
        5. Return structured results
        """
        # 1. Generate query embedding
        query_embedding = self.embedding_service.generate_query_embedding(query)

        # 2. Search Milvus — fetch more than top_k to allow grouping
        raw_results = self.milvus_service.search_sops(query_embedding, top_k=top_k * 3)

        if not raw_results:
            return {"query": query, "results": [], "total_results": 0}

        # 3. Apply metadata boosting and group by SOP title
        grouped = defaultdict(lambda: {
            "chunks": [],
            "sop_link": "",
            "threat_type": "",
            "category": "",
            "max_score": 0.0,
            "total_boosted_score": 0.0
        })

        for result in raw_results:
            title = result["title"]
            base_score = result["score"]
            boost = self._compute_metadata_boost(result, query, threat_type, category)
            boosted_score = min(base_score + boost, 1.0)  # COSINE scores are 0-1

            group = grouped[title]
            group["chunks"].append({
                "content": result["content"],
                "similarity_score": round(base_score, 4)
            })
            group["sop_link"] = result.get("sop_link") or group["sop_link"]
            group["threat_type"] = result.get("threat_type") or group["threat_type"]
            group["category"] = result.get("category") or group["category"]
            group["max_score"] = max(group["max_score"], boosted_score)
            group["total_boosted_score"] += boosted_score

        # 4. Build structured results sorted by confidence
        derive_results = []
        for title, group in grouped.items():
            avg_score = group["total_boosted_score"] / len(group["chunks"])
            confidence = round((group["max_score"] * 0.6 + avg_score * 0.4), 4)

            derive_results.append({
                "title": title,
                "sop_link": group["sop_link"],
                "confidence_score": confidence,
                "matched_chunks": group["chunks"][:5],  # limit chunks returned
                "metadata": {
                    "threat_type": group["threat_type"],
                    "category": group["category"],
                    "total_chunks_matched": len(group["chunks"])
                }
            })

        # Sort by confidence descending
        derive_results.sort(key=lambda x: x["confidence_score"], reverse=True)

        return {
            "query": query,
            "results": derive_results[:top_k],
            "total_results": len(derive_results)
        }
