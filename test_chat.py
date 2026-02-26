import requests
import json

def test_chat():
    url = "http://localhost:8000/chat/"
    payload = {
        "query": "vps sop procedures what are the general steps",
        "user_id": "verify_prompt_update_99"
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload, timeout=180)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n--- Assistant Response ---")
            # Handle encoding for Windows terminal
            resp_text = result.get("response", "").encode('ascii', 'ignore').decode()
            print(resp_text)
            print("\n--- Metadata ---")
            print(f"Active SOP: {result.get('active_sop')}")
            print(f"SOP Count: {result.get('sop_count')}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_chat()
