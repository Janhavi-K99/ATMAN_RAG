"""Global singleton cache for heavy resources."""
from src.embeddings.vector_store import FAISSVectorStore
from src.embeddings.embedder import EmbeddingGenerator
from src.retrieval.retriever import Retriever
from src.generation.generator import LLMGenerator
import threading


_vector_store = None
_embedder = None
_retriever = None
_generator = None
_lock = threading.Lock()


def get_vector_store() -> FAISSVectorStore:
    global _vector_store
    if _vector_store is None:
        with _lock:
            if _vector_store is None:
                _vector_store = FAISSVectorStore()
                _vector_store.initialize()
    return _vector_store


def get_embedder() -> EmbeddingGenerator:
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                _embedder = EmbeddingGenerator()
                _embedder.load_model()
    return _embedder


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                _retriever = Retriever(vector_store=get_vector_store())
    return _retriever


def get_generator() -> LLMGenerator:
    global _generator
    if _generator is None:
        with _lock:
            if _generator is None:
                _generator = LLMGenerator()
    return _generator


def preload_all():
    """Preload all heavy resources at startup."""
    print("Preloading vector store...")
    get_vector_store()
    print("Preloading embedder...")
    get_embedder()
    print("Preloading generator...")
    get_generator()
    print("Warming up LLM...")
    generator = get_generator()
    # Warmup call to load model into memory
    try:
        generator.generate("test", "test context")
        print("LLM warmed up.")
    except Exception as e:
        print(f"LLM warmup failed: {e}")
    print("All resources loaded.")


if __name__ == "__main__":
    preload_all()