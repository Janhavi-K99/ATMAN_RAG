import faiss
import numpy as np
import pickle
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.ingestion.chunker import Chunk
from src.embeddings.embedder import EmbeddingGenerator
from dotenv import load_dotenv

load_dotenv()


class FAISSVectorStore:
    def __init__(
        self,
        persist_dir: str = None,
        collection_name: str = None,
        embedding_model: str = None
    ):
        self.persist_dir = Path(persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "atman_docs")
        self.embedding_model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        
        self.index = None
        self.chunks_metadata = []  # List of dicts with doc_name, page_num, chunk_id, text
        self.embedder = EmbeddingGenerator(self.embedding_model_name)
        
        self.index_path = self.persist_dir / f"{self.collection_name}.faiss"
        self.metadata_path = self.persist_dir / f"{self.collection_name}_metadata.pkl"
    
    def initialize(self):
        """Initialize or load FAISS index and metadata."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        if self.index_path.exists() and self.metadata_path.exists():
            print(f"Loading existing FAISS index from {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'rb') as f:
                self.chunks_metadata = pickle.load(f)
            print(f"Loaded index with {self.index.ntotal} vectors")
        else:
            print("Creating new FAISS index")
            self.index = None
            self.chunks_metadata = []
    
    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray):
        """Add chunks and their embeddings to the FAISS index."""
        self.initialize()
        
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        start_id = self.index.ntotal
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        for i, chunk in enumerate(chunks):
            self.chunks_metadata.append({
                "doc_name": chunk.doc_name,
                "page_num": chunk.page_num,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text
            })
        
        print(f"Added {len(chunks)} chunks. Total vectors: {self.index.ntotal}")
        self._save()
    
    def _save(self):
        """Persist index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.chunks_metadata, f)
        print(f"Saved index to {self.index_path}")
    
    def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Search for similar chunks using cosine similarity."""
        self.initialize()
        
        if self.index is None or self.index.ntotal == 0:
            print("Index is empty")
            return []
        
        query_embedding = self.embedder.embed_query(query).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            similarity = float(score)  # Inner product = cosine similarity for normalized vectors
            if similarity >= similarity_threshold:
                meta = self.chunks_metadata[idx]
                results.append({
                    "text": meta["text"],
                    "doc_name": meta["doc_name"],
                    "page_num": meta["page_num"],
                    "chunk_id": meta["chunk_id"],
                    "similarity": similarity
                })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        self.initialize()
        return {
            "total_chunks": self.index.ntotal if self.index else 0,
            "collection_name": self.collection_name,
            "persist_dir": str(self.persist_dir),
            "embedding_model": self.embedding_model_name
        }


def build_vector_store(
    data_dir: str = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    force_rebuild: bool = False
) -> FAISSVectorStore:
    """Build or load the vector store from PDFs."""
    from pathlib import Path
    from src.ingestion.pdf_extractor import extract_all_pdfs
    from src.ingestion.chunker import chunk_all_pages
    
    data_path = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent / "data"
    
    store = FAISSVectorStore()
    store.initialize()
    
    if not force_rebuild and store.index is not None and store.index.ntotal > 0:
        print(f"Vector store already has {store.index.ntotal} chunks. Skipping rebuild.")
        return store
    
    print("Building vector store from PDFs...")
    pages = extract_all_pdfs(data_path)
    chunks = chunk_all_pages(pages, chunk_size, chunk_overlap)
    embeddings = store.embedder.generate_embeddings(chunks)
    store.add_chunks(chunks, embeddings)
    
    return store


if __name__ == "__main__":
    store = build_vector_store(force_rebuild=True)
    stats = store.get_stats()
    print(f"Build complete: {stats}")