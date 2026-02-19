# Enterprise SOP Assistant Backend Architecture

This document defines the current and target architecture. It prioritizes a **Single-Container Vector DB (Qdrant)** for simplicity, speed, and production readiness.

## System Architecture Diagram

```mermaid
graph TD

%% ================= USER LAYER =================
User["User / SOC Analyst"]

%% ================= ORCHESTRATION =================
subgraph Orchestration ["n8n - Central Nervous System"]
    n8n["Workflow Engine (Target)"]
end

%% ================= DATA SOURCE =================
subgraph Data_Source ["Data Layer"]
    Files["SOP Documents (.pdf .docx .txt .py)"]
end

%% ================= INGESTION =================
subgraph Ingestion ["Ingestion Pipeline (Implemented)"]
    Reader["Universal Reader"]
    Chunker["NLTK Chunking"]
    Embed_I["Embedding Service (Shared)"]
end

%% ================= AI CORE =================
subgraph AI_Core ["AI Engine (Implemented)"]
    API["FastAPI (/chat /derive)"]

    subgraph LangGraph ["LangGraph Workflow"]
        Intent["Intent Classifier"]
        Logic["Routing Logic"]
        Summ["Summarizer"]
        Extract["Instruction Extractor"]
        Format["Response Formatter"]
    end
end

%% ================= RETRIEVAL =================
subgraph Retrieval ["Knowledge Retrieval"]
    Hybrid["Hybrid Search"]
    Embed_Q["Query Embedding"]
end

%% ================= VECTOR INFRA =================
subgraph Infra ["Vector Infrastructure (Implemented)"]
    Qdrant["Qdrant Vector DB"]
end

%% ================= QUERY FLOW =================
User --> n8n
n8n --> API
API --> Intent
Intent --> Logic
Logic --> Hybrid
Hybrid --> Embed_Q
Hybrid --> Qdrant
Qdrant --> Hybrid
Hybrid --> Summ
Summ --> Extract
Extract --> Format
Format --> n8n

%% ================= INGESTION FLOW =================
Files --> n8n
n8n --> Reader
Reader --> Chunker
Chunker --> Embed_I
Embed_I --> Qdrant

%% ================= INFRA LINKS =================
Qdrant --> Volumes["Local Persistence"]
```

## Component Overview: Current vs. Target

### 1. n8n Orchestration (Target Interface)
*   **Workflow Engine**: **[PLANNING]** This will act as the gatekeeper, connecting to Google Drive/WhatsApp/Slack and routing data to our FastAPI.

### 2. Ingestion Utility (`ingest_sops.py`)
*   **Status**: **[IMPLEMENTED]** 
*   **Logic**: Uses NLTK for semantic chunking and `pywin32` for Word doc extraction.
*   **Storage**: Pushes directly to **Qdrant**.

### 3. AI Engine (`sop_assistant_enterprise.py`)
*   **Status**: **[IMPLEMENTED]**
*   **FastAPI**: Active layer that handles requests for `/chat` and `/derive`.
*   **LangGraph**: Active workflow that manages the logic of *Identify Intent -> Retrieve Path -> Summarize -> Format*.

### 4. Vector Infrastructure (Docker)
*   **Qdrant**: **[IMPLEMENTED]** Our unified "Brain." It handles 100% of the vector storage, search, and metadata management in a single lightweight container.
*   **Efficiency**: Replaces the redundant 3-container Milvus setup with a highly optimized Rust-based engine.

---
> [!IMPORTANT]
> **Production Ready**: Qdrant is configured with local persistence and healthy memory limits. It is ready for the next phase of n8n integration.
