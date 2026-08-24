import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import ollama

load_dotenv()


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context documents.

RULES:
1. Answer ONLY using information from the provided context.
2. The context below CONTAINS RELEVANT INFORMATION for the question. You MUST answer using it.
3. ONLY say "I cannot answer this question from the provided documents" if the context is completely empty or totally unrelated.
4. If the context has authentication, endpoints, rate limits, or configuration details — that IS configuration information. Answer the question using it.
5. Cite sources using [Source X].
6. Be concise and direct.
7. If multiple sources contain relevant information, synthesize them.

FORMAT YOUR ANSWER FOR READABILITY:
- Use **bold** for key terms and section headers
- Use bullet points (•) for lists
- Use numbered lists (1., 2., 3.) for steps or ordered items
- Use `code blocks` for technical values (endpoints, parameters, commands)
- Separate distinct topics with blank lines
- Keep paragraphs short (2-3 sentences max)

CONTEXT:
{context}"""


def build_prompt(query: str, context: str, conversation_history: List[Dict[str, str]] = None) -> str:
    """Build the full prompt with system instructions, context, and conversation history."""
    history_text = ""
    if conversation_history:
        history_text = "\n\nCONVERSATION HISTORY:\n"
        for turn in conversation_history[-4:]:  # Last 4 turns
            history_text += f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}\n"
        history_text += "\n"
    
    return SYSTEM_PROMPT.format(context=context) + history_text + f"\nQUESTION: {query}\n\nANSWER:"


class LLMGenerator:
    def __init__(
        self,
        provider: str = None,
        model: str = None,
        ollama_base_url: str = None
    ):
        self.provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        if self.provider == "ollama":
            self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:latest")
            self.client = ollama.Client(host=self.ollama_base_url)
        elif self.provider == "openai":
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            import openai
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate(self, query: str, context: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Generate an answer grounded in the context."""
        prompt = build_prompt(query, context, conversation_history)
        
        try:
            if self.provider == "ollama":
                response = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": 0.1,
                        "num_predict": 256,
                        "num_ctx": 1024,
                        "num_thread": 4,
                        "keep_alive": "30m",
                    }
                )
                return response["message"]["content"].strip()
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def generate_with_sources(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Generate answer with source attribution."""
        if not retrieved_chunks:
            return {
                "answer": "I cannot answer this question from the provided documents.",
                "sources": [],
                "context_used": ""
            }
        
        context = self.format_context_for_prompt(retrieved_chunks)
        answer = self.generate(query, context, conversation_history)
        
        # Extract source citations from answer
        sources = []
        for i, chunk in enumerate(retrieved_chunks):
            sources.append({
                "source_num": i + 1,
                "doc_name": chunk["doc_name"],
                "page_num": chunk["page_num"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"][:500],
                "similarity": chunk["similarity"]
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "context_used": context
        }
    
    def format_context_for_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks for the prompt with numbered sources."""
        if not chunks:
            return "No relevant documents found."
        
        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(f"[Source {i+1}: {chunk['doc_name']}, Page {chunk['page_num']}]\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    from src.retrieval.retriever import Retriever
    from src.embeddings.vector_store import FAISSVectorStore
    
    # Test the full pipeline
    store = FAISSVectorStore()
    store.initialize()
    
    retriever = Retriever(vector_store=store)
    generator = LLMGenerator()
    
    query = "What is the leave policy?"
    print(f"Query: {query}")
    
    chunks = retriever.retrieve(query)
    result = generator.generate_with_sources(query, chunks)
    
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources:")
    for s in result['sources']:
        print(f"  [Source {s['source_num']}] {s['doc_name']} p{s['page_num']} (sim: {s['similarity']:.3f})")