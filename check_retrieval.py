import logging
import sys
import os
import traceback
from app.services.qdrant_service import QdrantService
from app.services.embedding_service import EmbeddingService

# Configure logging to handle system encoding issues gracefully
logging.basicConfig(level=logging.INFO)

def check_retrieval(query: str):
    print(f"\n--- Checking Retrieval for: '{query}' ---")
    
    try:
        # Initialize services
        print("1. Initializing services...")
        embedding_service = EmbeddingService()
        qdrant_service = QdrantService()
        
        # Generate embedding
        print("2. Generating query embedding...")
        query_embedding = embedding_service.generate_query_embedding(query)
        
        # Search Qdrant
        print("3. Searching Qdrant...")
        results = qdrant_service.search_sops(query_embedding, top_k=3)
        
        if not results:
            print("[INFO] No results found in Qdrant.")
            print("Try checking if the 'sop_collection' exists and has points.")
            return

        print(f"[SUCCESS] Found {len(results)} results:")
        for i, res in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"  Title: {res.get('title')}")
            print(f"  Score: {res.get('score'):.4f}")
            # Use ascii-safe slice for content
            content = res.get('content', '')[:100].encode('ascii', 'ignore').decode()
            print(f"  Content snippet: {content}...")
            print(f"  Link: {res.get('sop_link')}")

    except Exception as e:
        print(f"[ERROR] Error during retrieval check: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_query = "how to handle security incident"
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    
    # Force output to be ascii-safe if needed or just handle errors in print
    check_retrieval(test_query)
