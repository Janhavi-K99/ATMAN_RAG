from src.retrieval.retriever import Retriever
from src.embeddings.vector_store import FAISSVectorStore

store = FAISSVectorStore()
store.initialize()
retriever = Retriever(vector_store=store)

queries = ['How do I configure the API endpoint?', 'What is the leave policy?', 'What are the pricing tiers?']

for q in queries:
    print(f'\nQuery: {q}')
    results = retriever.retrieve(q)
    print(f'Found {len(results)} unique chunks:')
    for r in results:
        print(f'  [{r["similarity"]:.3f}] {r["doc_name"]} p{r["page_num"]}: {r["text"][:80]}...')