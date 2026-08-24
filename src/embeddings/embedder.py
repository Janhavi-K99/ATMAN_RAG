from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np
from src.ingestion.chunker import Chunk


class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
    
    def load_model(self):
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def generate_embeddings(self, chunks: List[Chunk]) -> np.ndarray:
        """Generate embeddings for a list of chunks."""
        self.load_model()
        
        texts = [chunk.text for chunk in chunks]
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        self.load_model()
        return self.model.encode([query], convert_to_numpy=True)[0]


if __name__ == "__main__":
    from src.ingestion.pdf_extractor import extract_all_pdfs
    from src.ingestion.chunker import chunk_all_pages
    from pathlib import Path
    
    data_dir = Path(__file__).parent.parent.parent / "data"
    pages = extract_all_pdfs(data_dir)
    chunks = chunk_all_pages(pages)
    
    generator = EmbeddingGenerator()
    embeddings = generator.generate_embeddings(chunks)
    print(f"Done. Embedding dim: {embeddings.shape[1]}")