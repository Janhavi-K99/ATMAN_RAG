#!/usr/bin/env python
"""
Phase 1 Pipeline: PDF Ingestion -> Chunking -> Embeddings -> Vector Store
Run this script to build the complete vector database from PDFs.
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ingestion.pdf_extractor import extract_all_pdfs
from src.ingestion.chunker import chunk_all_pages
from src.embeddings.embedder import EmbeddingGenerator
from src.embeddings.vector_store import FAISSVectorStore, build_vector_store


def main():
    print("=" * 60)
    print("PHASE 1: RAG Pipeline - Document Ingestion")
    print("=" * 60)
    
    data_dir = Path(__file__).parent / "data"
    print(f"Data directory: {data_dir}")
    
    # Step 1: Extract text from PDFs
    print("\n[1/4] Extracting text from PDFs...")
    pages = extract_all_pdfs(data_dir)
    print(f"       Extracted {len(pages)} pages from {len(set(p.doc_name for p in pages))} documents")
    
    # Step 2: Chunk the documents
    print("\n[2/4] Chunking documents...")
    chunks = chunk_all_pages(pages, chunk_size=1000, chunk_overlap=200)
    print(f"       Created {len(chunks)} chunks")
    
    # Step 3: Generate embeddings
    print("\n[3/4] Generating embeddings...")
    embedder = EmbeddingGenerator()
    embeddings = embedder.generate_embeddings(chunks)
    print(f"       Embedding shape: {embeddings.shape}")
    
    # Step 4: Store in vector database
    print("\n[4/4] Storing in FAISS...")
    store = FAISSVectorStore()
    store.initialize()
    store.add_chunks(chunks, embeddings)
    
    # Verify
    stats = store.get_stats()
    print(f"\n[SUCCESS] Phase 1 Complete!")
    print(f"   Total chunks in vector store: {stats['total_chunks']}")
    print(f"   Collection: {stats['collection_name']}")
    print(f"   Persist dir: {stats['persist_dir']}")


if __name__ == "__main__":
    main()