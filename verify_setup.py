import requests
import time

BASE_URL = "http://localhost:8000"

def test_health():
    print("Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}, Response: {response.json()}")

def test_add_sop():
    print("\nTesting /sops/ (Add SOP)...")
    # meaningful long text to trigger chunking (assuming 5 sentences per chunk)
    long_content = (
        "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5. "
        "Sentence 6: This is the second chunk. Sentence 7. Sentence 8. Sentence 9. Sentence 10. "
        "Sentence 11: This is the third chunk and it contains specific hidden details."
    )
    payload = {
        "title": "Long SOP for Chunking Test",
        "content": long_content
    }
    response = requests.post(f"{BASE_URL}/sops/", json=payload)
    print(f"Status: {response.status_code}, Response: {response.json()}")

def test_search():
    print("\nTesting /search/...")
    # Search for something in the third chunk
    payload = {
        "query": "hidden details",
        "top_k": 3
    }
    response = requests.post(f"{BASE_URL}/search/", json=payload)
    print(f"Status: {response.status_code}")
    results = response.json()
    for i, res in enumerate(results):
        print(f"Result {i+1}: {res['title']} (Score: {res['score']:.4f})")
        print(f"Content: {res['content']}...")
        print("-" * 20)

if __name__ == "__main__":
    print("Waiting for server to be ready (run 'python -m app.main' in another terminal)...")
    # In a real scenario, you'd start the server here or assume it's running
    try:
        test_health()
        test_add_sop()
        time.sleep(2) # Give Qdrant a moment to flush if needed
        test_search()
    except Exception as e:
        print(f"Verification failed: {e}")
        print("Make sure the FastAPI server is running on http://localhost:8000")
