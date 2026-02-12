import requests
import time

BASE_URL = "http://localhost:8000"

def test_health():
    print("Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}, Response: {response.json()}")

def test_add_sop():
    print("\nTesting /sops/ (Add SOP)...")
    payload = {
        "title": "Data Privacy Policy",
        "content": "All personal data must be encrypted at rest and in transit. Access is limited to authorized personnel only."
    }
    response = requests.post(f"{BASE_URL}/sops/", json=payload)
    print(f"Status: {response.status_code}, Response: {response.json()}")

def test_search():
    print("\nTesting /search/...")
    payload = {
        "query": "How should we handle personal data?",
        "top_k": 2
    }
    response = requests.post(f"{BASE_URL}/search/", json=payload)
    print(f"Status: {response.status_code}")
    results = response.json()
    for i, res in enumerate(results):
        print(f"Result {i+1}: {res['title']} (Score: {res['score']})")
        print(f"Content: {res['content'][:100]}...")

if __name__ == "__main__":
    print("Waiting for server to be ready (run 'python -m app.main' in another terminal)...")
    # In a real scenario, you'd start the server here or assume it's running
    try:
        test_health()
        test_add_sop()
        time.sleep(2) # Give Milvus a moment to flush if needed
        test_search()
    except Exception as e:
        print(f"Verification failed: {e}")
        print("Make sure the FastAPI server is running on http://localhost:8000")
