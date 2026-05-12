# 🚀 Graph-Enriched Agentic RAG System

A state-of-the-art Retrieval-Augmented Generation (RAG) system designed for high-precision document analysis. This system moves beyond simple vector search by implementing **Adaptive Structural Chunking** and a **Relational Chunk Graph** to preserve context and document hierarchy.

---

## 🏗️ Project Overview

This project implements an "Agentic RAG" pipeline that combines:
1.  **Hybrid Structural Retrieval**: Uses both vector similarity and graph-based relationships.
2.  **Adaptive Chunking**: Intelligently segments documents based on their actual structure (headings, tables, lists).
3.  **Graph Enrichment**: Links chunks via sequential, hierarchical, semantic, and entity-based relationships.
4.  **Agentic Diagnostics**: Tools designed for complex document troubleshooting and information extraction.

---

## 🧠 Advanced Chunking Strategy

The core strength of this system lies in its **Adaptive Chunking Engine**. Unlike standard RAG systems that use fixed-size character or token windows, this system treats documents as structured data.

### 1. Structural Awareness
The `AdaptiveChunker` (in `rag/chunking.py`) analyzes the document layout to identify:
*   **Headings**: Detected using regex patterns (Markdown, Numbered, All-Caps). Headings become "Anchor Chunks" that serve as parents to subsequent content.
*   **Lists**: Bulleted and numbered lists are kept together in single chunks to maintain the internal logic and sequence of the list items.
*   **Tables**: The system uses `pdfplumber` (for PDFs) and specialized regex (for text) to extract tables as structured JSON. These are then serialized into a special format for embedding, ensuring the LLM understands row/column relationships.

### 2. LLM-Assisted Structure Detection
Before chunking, the system optionally uses an LLM (`rag/structure_detector.py`) to "pre-read" the document. It identifies:
*   Main sections and hierarchy levels.
*   Key entities mentioned in specific sections.
*   Overall document type (e.g., Service Manual, Specification).

### 3. Sliding Window for Continuity
For large paragraphs that exceed `MAX_CHUNK_TOKENS`, a sliding window approach with configurable overlap is used. This ensures that semantic meaning is never cut off mid-sentence and context is preserved across chunk boundaries.

---

## 🔗 Graph-Enriched Indexing

Once chunked, the documents are transformed into a **Directed Relationship Graph** using `networkx`. This allows the retriever to "walk" the graph to find related context that simple vector search might miss.

| Link Type | Description | Purpose |
| :--- | :--- | :--- |
| **NEXT** | Sequential connection between consecutive chunks. | Preserves document flow and narrative. |
| **PARENT** | Connects a content chunk to its section heading. | Provides high-level context to specific details. |
| **SIMILAR** | Links chunks with high cosine similarity (>0.82) across different sections. | Discovers cross-references and related topics. |
| **ENTITY_MATCH** | Links chunks sharing significant keywords/entities. | Identifies all occurrences of a specific component or term. |

---

## 🛠️ Tech Stack & Dependencies

*   **LLM/Embeddings**: Azure OpenAI (GPT-4o, text-embedding-3-large)
*   **Vector Store**: FAISS (for high-speed similarity search)
*   **Graph Engine**: NetworkX
*   **UI**: Streamlit
*   **PDF Processing**: pdfplumber
*   **Core Libraries**: NumPy, scikit-learn, beautifulsoup4

---

## 🚀 Getting Started

### 1. Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory with your Azure OpenAI credentials:
```env
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_AD_TOKEN=your_token
OPENAI_API_VERSION=2024-02-15-preview
```

### 3. Indexing Documents
Place your documents (.pdf, .txt, .md) in `data/docs/` and run:
```bash
python index_documents.py --reset
```

### 4. Running the UI
```bash
streamlit run ui/app.py
```

---

## 📁 Project Structure

```text
├── agents/             # Agentic logic and tool definitions
├── config/             # System settings and constants
├── core/               # Base models and LLM clients
├── data/               # Raw documents and indexed stores
├── rag/                # The RAG Engine
│   ├── chunking.py     # Adaptive chunking logic
│   ├── graph.py        # Relationship graph implementation
│   ├── embedder.py     # Embedding and FAISS logic
│   └── table_extractor.py # Structured data extraction
├── tools/              # Utility tools for the agents
└── ui/                 # Streamlit interface
```
