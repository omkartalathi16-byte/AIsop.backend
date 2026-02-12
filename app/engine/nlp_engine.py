"""
NLP Engine — Part 1: Intent Extraction & SOP Retrieval
Uses keyword matching + embedding search to find relevant SOPs.
"""
import re
import logging

logger = logging.getLogger(__name__)

# Threat categories with associated keywords for intent detection
THREAT_KEYWORDS = {
    "phishing": ["phishing", "phish", "suspicious email", "fake email", "spoofed", "credential harvest"],
    "malware": ["malware", "virus", "trojan", "ransomware", "worm", "infected", "payload"],
    "data_breach": ["data breach", "data leak", "exposed data", "unauthorized access", "compromised"],
    "insider_threat": ["insider", "employee misconduct", "privilege abuse", "data exfiltration"],
    "ddos": ["ddos", "denial of service", "traffic flood", "dos attack"],
    "network_intrusion": ["intrusion", "unauthorized network", "lateral movement", "port scan", "brute force"],
    "social_engineering": ["social engineering", "impersonation", "pretexting", "baiting", "tailgating"],
    "vulnerability": ["vulnerability", "cve", "exploit", "patch", "zero-day", "unpatched"],
}


class NLPEngine:
    """Extracts threat intent from user messages and retrieves matching SOPs."""

    def __init__(self, embedding_service, milvus_service):
        self.embedding_service = embedding_service
        self.milvus_service = milvus_service

    def extract_intent(self, message: str) -> dict:
        """Extract threat type and keywords from user message."""
        message_lower = message.lower()
        detected_threats = []
        matched_keywords = []

        for threat_type, keywords in THREAT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in message_lower:
                    if threat_type not in detected_threats:
                        detected_threats.append(threat_type)
                    matched_keywords.append(keyword)

        return {
            "threats": detected_threats if detected_threats else ["general"],
            "keywords": matched_keywords,
            "original_message": message
        }

    def build_search_query(self, intent: dict) -> str:
        """Build an enriched search query from the extracted intent."""
        parts = [intent["original_message"]]
        if intent["keywords"]:
            parts.append(" ".join(intent["keywords"]))
        if intent["threats"] and intent["threats"] != ["general"]:
            parts.append(" ".join(intent["threats"]).replace("_", " "))
        return " ".join(parts)

    def retrieve_sops(self, message: str, top_k: int = 3) -> dict:
        """Full retrieval pipeline: extract intent → search Milvus → return results."""
        intent = self.extract_intent(message)
        search_query = self.build_search_query(intent)

        logger.info(f"Intent: {intent['threats']}, Query: {search_query[:80]}...")

        query_embedding = self.embedding_service.generate_query_embedding(search_query)
        results = self.milvus_service.search_sops(query_embedding, top_k=top_k)

        return {
            "intent": intent,
            "results": results
        }
