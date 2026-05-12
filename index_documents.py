"""
index_documents.py — CLI pipeline: Load → Detect → Chunk → Embed → Graph → Save FAISS
Usage:
    python index_documents.py --docs ./data/docs/
    python index_documents.py --docs ./data/docs/ --reset
"""
import sys
import os
import argparse
import logging

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("indexer")

from rag.document_loader import load_directory
from rag.structure_detector import detect_structure
from rag.chunking import AdaptiveChunker
from rag.embedder import embed_chunks, build_vector_store
from rag.graph import ChunkGraph
from core.vector_store import VectorStore
from config.settings import DOCS_DIR, FAISS_INDEX_DIR


def main():
    parser = argparse.ArgumentParser(description="Index documents into the RAG system.")
    parser.add_argument("--docs", default=DOCS_DIR, help="Directory containing documents to index.")
    parser.add_argument("--reset", action="store_true", help="Reset existing index before indexing.")
    args = parser.parse_args()

    docs_dir = args.docs

    if not os.path.isdir(docs_dir):
        logger.error(f"Document directory not found: {docs_dir}")
        sys.exit(1)

    if args.reset and os.path.exists(FAISS_INDEX_DIR):
        import shutil
        shutil.rmtree(FAISS_INDEX_DIR)
        logger.info("Existing index removed.")

    # ── Step 1: Load documents ────────────────────────────────────────────────
    logger.info(f"📂 Loading documents from: {docs_dir}")
    docs = load_directory(docs_dir)
    if not docs:
        logger.error("No documents found. Supported formats: .pdf, .txt, .html, .md")
        sys.exit(1)
    logger.info(f"Loaded {len(docs)} documents.")

    # ── Step 2: Structure detection + chunking ────────────────────────────────
    all_chunks = []
    for fname, text, fmt in docs:
        logger.info(f"  📄 Processing '{fname}' (format: {fmt})")
        structure = detect_structure(text)
        chunker = AdaptiveChunker(source=fname)
        chunks = chunker.chunk(text, structure)
        logger.info(f"     → {len(chunks)} chunks produced")
        all_chunks.extend(chunks)

    logger.info(f"Total chunks: {len(all_chunks)}")

    # ── Step 3: Embed all chunks ───────────────────────────────────────────────
    logger.info("🔢 Embedding chunks with Azure OpenAI...")
    all_chunks = embed_chunks(all_chunks)

    # ── Step 4: Build FAISS index ──────────────────────────────────────────────
    logger.info("🗂️  Building FAISS vector store...")
    store = build_vector_store(all_chunks)
    store.save()
    logger.info(f"✅ FAISS index saved to: {FAISS_INDEX_DIR}")

    # ── Step 5: Build chunk graph ──────────────────────────────────────────────
    logger.info("🔗 Building chunk relationship graph...")
    graph = ChunkGraph()
    graph.build(all_chunks)

    # Save graph alongside index
    import pickle
    graph_path = os.path.join(FAISS_INDEX_DIR, "graph.pkl")
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)
    logger.info(f"✅ Graph saved to: {graph_path}")

    logger.info("\n🎉 Indexing complete!")
    logger.info(f"   Documents: {len(docs)}")
    logger.info(f"   Chunks: {len(all_chunks)}")
    logger.info(f"   Graph edges: {graph.graph.number_of_edges()}")
    logger.info(f"\n▶ Now run: streamlit run ui/app.py")


if __name__ == "__main__":
    main()
