from src.retrieval.retriever import Retriever
from src.embeddings.vector_store import FAISSVectorStore
from src.generation.generator import LLMGenerator

store = FAISSVectorStore()
store.initialize()

retriever = Retriever(vector_store=store)
generator = LLMGenerator()

queries = [
    "What is the leave policy?",
    "How do I configure the API endpoint?",
    "What are the pricing tiers?",
    "What is the company's mission statement?",  # Unanswerable
    "How do I bake a chocolate cake?",  # Unanswerable
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    chunks = retriever.retrieve(query)
    result = generator.generate_with_sources(query, chunks)
    
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])}")
    for s in result['sources']:
        print(f"  [Source {s['source_num']}] {s['doc_name']} p{s['page_num']} (sim: {s['similarity']:.3f})")