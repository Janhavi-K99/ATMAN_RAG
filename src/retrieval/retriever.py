from typing import List, Dict, Any, Optional
from src.embeddings.vector_store import FAISSVectorStore
from dotenv import load_dotenv
import os

load_dotenv()


class Retriever:
    def __init__(
        self,
        vector_store: FAISSVectorStore = None,
        top_k: int = None,
        similarity_threshold: float = None
    ):
        self.vector_store = vector_store or FAISSVectorStore()
        self.vector_store.initialize()
        
        self.top_k = top_k or int(os.getenv("TOP_K", "5"))
        self.similarity_threshold = similarity_threshold or float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query, deduplicated by (doc, page)."""
        # Fetch more than top_k to allow for deduplication
        fetch_k = max(self.top_k * 3, 15)
        results = self.vector_store.search(
            query=query,
            top_k=fetch_k,
            similarity_threshold=self.similarity_threshold
        )
        
        # Deduplicate by (doc_name, page_num) - keep highest similarity
        seen = {}
        for r in results:
            key = (r["doc_name"], r["page_num"])
            if key not in seen or r["similarity"] > seen[key]["similarity"]:
                seen[key] = r
        
        # Sort by similarity descending and take top_k
        deduped = sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)
        return deduped[:self.top_k]
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string with citations."""
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, r in enumerate(results):
            citation = f"[Source {i+1}: {r['doc_name']}, Page {r['page_num']}]"
            context_parts.append(f"{citation}\n{r['text']}")
        
        return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    retriever = Retriever()
    
    test_queries = [
        "What is the leave policy?",
        "How do I configure the API endpoint?",
        "What are the pricing tiers?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query)
        print(f"Found {len(results)} relevant chunks")
        for r in results:
            print(f"  [{r['similarity']:.3f}] {r['doc_name']} p{r['page_num']}: {r['text'][:100]}...")