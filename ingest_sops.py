import os
import requests
import json
import sys
from unstructured.partition.auto import partition
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configuration (from environment or defaults)
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = os.getenv("API_URL", f"http://{API_HOST}:{API_PORT}/sops/")
BATCH_API_URL = os.getenv("BATCH_API_URL", f"http://{API_HOST}:{API_PORT}/sops/batch/")
DATA_DIR = os.getenv("DATA_DIR", "data/sops")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))

def extract_text_with_unstructured(filepath):
    """Extracts text from various file formats using the unstructured library."""
    try:
        # If it's a JSON file, handle it separately to avoid NDJSON picky errors
        if filepath.lower().endswith(".json"):
            with open(filepath, 'r', encoding='utf-8') as f:
                # If it's valid JSON, pretty print it for context
                data = json.load(f)
                return json.dumps(data, indent=2)
                
        # partition(filename=...) handles many formats automatically (PDF, DOCX, DOC, TXT, etc.)
        elements = partition(filename=filepath)
        text = "\n\n".join([str(el) for el in elements])
        return text.strip()
    except Exception as e:
        # Fallback for JSON if json.load fails
        if filepath.lower().endswith(".json"):
             with open(filepath, 'r', encoding='utf-8') as f:
                 return f.read()
        print(f"Error processing {filepath} with unstructured: {e}")
        return ""

def ingest_documents():
    """Reads documents from DATA_DIR and sends them to the API in batches."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")
        print("Please place your SOP files there and run this script again.")
        return

    supported_extensions = {
        ".pdf", ".docx", ".doc", ".txt", ".md", ".rtf", 
        ".html", ".htm", ".eml", ".msg", ".odt", ".pptx", ".ppt", ".json"
    }
    
    print(f"Scanning {DATA_DIR} for documents...")
    
    supported_files = []
    for root, dirs, filenames in os.walk(DATA_DIR):
        for filename in filenames:
            if os.path.splitext(filename.lower())[1] in supported_extensions:
                supported_files.append(os.path.join(root, filename))
    
    if not supported_files:
        print(f"No supported files found in {DATA_DIR}")
        print(f"Supported extensions: {', '.join(sorted(supported_extensions))}")
        return

    print(f"Found {len(supported_files)} documents. Starting batch ingestion...")
    
    current_batch = []
    
    for filepath in supported_files:
        filename = os.path.basename(filepath)
        print(f"Extracting: {filename}...")
        content = extract_text_with_unstructured(filepath)

        if not content:
            print(f"Skipping {filename}: No text content extracted.")
            continue
        
        title = os.path.splitext(filename)[0].replace("_", " ").title()
        mock_link = f"file:///{os.path.abspath(filepath).replace('\\', '/')}"
        
        payload_item = {
            "title": title,
            "content": content,
            "sop_link": mock_link,
            "threat_type": "General",
            "category": "SOP"
        }
        
        current_batch.append(payload_item)
        
        # Send batch if it reaches the limit
        if len(current_batch) >= BATCH_SIZE:
            send_batch(current_batch)
            current_batch = []
            
    # Send any remaining items
    if current_batch:
        send_batch(current_batch)

def send_batch(batch):
    """Sends a batch of documents to the API."""
    try:
        print(f"Sending batch of {len(batch)} documents...")
        response = requests.post(BATCH_API_URL, json={"items": batch})
        if response.status_code == 202:
            print(f"Successfully queued batch of {len(batch)} documents.")
        else:
            print(f"Failed to queue batch: {response.text}")
    except Exception as e:
        print(f"Error connecting to API for batch: {e}")

if __name__ == "__main__":
    ingest_documents()
