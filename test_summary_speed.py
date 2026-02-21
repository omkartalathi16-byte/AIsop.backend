import time
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.services.llm_service import LLMService

def test_speed():
    print("Initialize components...")
    start_init = time.time()
    embedding_service = EmbeddingService()
    qdrant_service = QdrantService()
    llm_service = LLMService()
    
    # Wait until fully initialized and print time
    print(f"Initialization took: {time.time() - start_init:.2f} seconds")

    query = "What is the procedure to report a security incident?"
    
    print("\nStarting Retrieval & Generation...")
    start_total = time.time()

    # Step 1: Embedding
    start_embed = time.time()
    query_embedding = embedding_service.generate_query_embedding(query)
    embed_time = time.time() - start_embed
    print(f"Embedding generation took: {embed_time:.2f} seconds")

    # Step 2: Retrieval
    start_retrieve = time.time()
    search_results = qdrant_service.search_sops(query_embedding, top_k=1)
    retrieve_time = time.time() - start_retrieve
    print(f"Qdrant retrieval took: {retrieve_time:.2f} seconds")
    print(f"Found {len(search_results)} chunks.")

    # Format chunks for LLM
    context_chunks = []
    for hit in search_results:
        context_chunks.append({
            "metadata": {"title": hit.get("title", "Unknown")},
            "content": hit.get("content", "")
        })

    # Step 3: Generation
    start_gen = time.time()
    response = llm_service.synthesize_rag_response(query, context_chunks)
    gen_time = time.time() - start_gen
    print(f"LLM generation took: {gen_time:.2f} seconds")

    total_time = time.time() - start_total
    
    print("\n--- Summary ---")
    print(response.encode('ascii', 'ignore').decode())
    print("\n--- Speed Metrics ---")
    print(f"Total time for question: {total_time:.2f} seconds")
    print(f"Embedding: {embed_time:.2f}s | Retrieval: {retrieve_time:.2f}s | Generation: {gen_time:.2f}s")

if __name__ == "__main__":
    test_speed()
