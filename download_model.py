import requests
import os
import sys

def download_model(url, filename):
    file_size = 0
    if os.path.exists(filename):
        file_size = os.path.getsize(filename)
    
    headers = {"Range": f"bytes={file_size}-"}
    response = requests.get(url, headers=headers, stream=True)
    
    # If server doesn't support Range or file is already complete
    if response.status_code == 416: # Range Not Satisfiable
        print(f"Model file at {filename} is already complete or exceeds source size.")
        return
    
    total_size = int(response.headers.get('content-length', 0)) + file_size
    mode = 'ab' if file_size > 0 else 'wb'
    
    if file_size > 0:
        print(f"Resuming download from {file_size/1024/1024:.2f} MB...")
    else:
        print(f"Downloading {filename}...")

    with open(filename, mode) as f:
        downloaded = file_size
        for data in response.iter_content(chunk_size=8192):
            downloaded += len(data)
            f.write(data)
            done = int(50 * downloaded / total_size)
            sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {100 * downloaded / total_size:.2f}% ({downloaded/1024/1024:.2f}MB / {total_size/1024/1024:.2f}MB)")
            sys.stdout.flush()
    print("\nDownload complete!")

if __name__ == "__main__":
    model_dir = os.path.join(os.getcwd(), "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_url = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
    model_path = os.path.join(model_dir, "qwen2.5-3b-instruct-q4_k_m.gguf")
    
    # Simple check for completion: if file size > 1.8GB, consider complete for now
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1900000000:
        print(f"Model already exists and appears complete at {model_path}")
    else:
        download_model(model_url, model_path)
