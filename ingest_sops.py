import os
import requests
import json
from PyPDF2 import PdfReader
from docx import Document
import sys

# Optional dependency for legacy .doc support on Windows
try:
    import win32com.client
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

# Configuration
API_URL = "http://127.0.0.1:8000/sops/"
DATA_DIR = "data/sops"

def extract_text_from_pdf(filepath):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
        return ""

def extract_text_from_docx(filepath):
    """Extracts text from a DOCX file."""
    try:
        doc = Document(filepath)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        print(f"Error reading DOCX {filepath}: {e}")
        return ""

def extract_text_from_doc(filepath):
    """Extracts text from legacy .doc using Word COM interface (Windows only)."""
    if not HAS_WIN32:
        print(f"⚠️ Cannot read .doc file '{os.path.basename(filepath)}'.")
        print("Please install 'pywin32' (pip install pywin32) OR save as .docx.")
        return ""
    
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(os.path.abspath(filepath))
        text = doc.Content.Text
        doc.Close()
        word.Quit()
        return text.strip()
    except Exception as e:
        print(f"Error reading legacy .doc {filepath}: {e}")
        return ""

def ingest_documents():
    """Reads multiple file formats from DATA_DIR and sends them to the API."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")
        print("Please place your SOP files there and run this script again.")
        return

    # Categorize extensions
    text_extensions = {".txt", ".py", ".md", ".json", ".csv", ".log", ".html", ".js", ".ts", ".css"}
    modern_word = {".docx"}
    legacy_word = {".doc"}
    pdf_ext = {".pdf"}
    
    all_extensions = text_extensions | modern_word | legacy_word | pdf_ext
    
    files = os.listdir(DATA_DIR)
    supported_files = [f for f in files if os.path.splitext(f.lower())[1] in all_extensions]
    
    if not supported_files:
        print(f"No supported files found in {DATA_DIR}")
        print(f"Supported extensions: {', '.join(sorted(all_extensions))}")
        return

    print(f"Found {len(supported_files)} documents. Starting ingestion...")

    for filename in supported_files:
        filepath = os.path.join(DATA_DIR, filename)
        ext = os.path.splitext(filename.lower())[1]
        
        content = ""
        try:
            if ext in text_extensions:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            elif ext in modern_word:
                content = extract_text_from_docx(filepath)
            elif ext in legacy_word:
                content = extract_text_from_doc(filepath)
            elif ext in pdf_ext:
                content = extract_text_from_pdf(filepath)
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue

        if not content:
            print(f"⚠️ Skipping {filename}: No text content extracted.")
            continue
        
        # Use filename as title (without extension)
        title = os.path.splitext(filename)[0].replace("_", " ").title()
        
        # Link generation
        mock_link = f"file:///{os.path.abspath(filepath).replace('\\', '/')}"
        
        payload = {
            "title": title,
            "content": content,
            "sop_link": mock_link,
            "threat_type": "General",
            "category": "SOP"
        }

        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                print(f"✅ Successfully ingested: {title} ({ext})")
            else:
                print(f"❌ Failed to ingest {title}: {response.text}")
        except Exception as e:
            print(f"⚠️ Error connecting to API for {title}: {e}")

if __name__ == "__main__":
    ingest_documents()
