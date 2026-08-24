# RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) pipeline for answering natural-language questions over a set of internal documents (product manuals, policies, technical specs) with source-grounded answers.

## Architecture Overview

```
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   PDFs      │───▶│  Extract    │───▶│   Chunk      │───▶│  Embed      │───▶│  Vector DB  │
│  (7 docs)   │    │  Text +     │    │  (1000/200)  │    │  (MiniLM)   │    │  (FAISS)    │
└─────────────┘    │  Tables     │    └──────────────┘    └─────────────┘    └──────────────┘
                   └─────────────┘                              ▲                   │
                                                                │                   ▼
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   User      │◀───│  LLM        │◀───│  Retrieve    │◀───│  Search     │◀───│  Query      │
│  Question   │    │  (Ollama)   │    │  Top-K       │    │  (Cosine)   │    │  Embedding  │
└─────────────┘    └─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                          │                   │
                          ▼                   ▼
                   ┌─────────────┐    ┌──────────────┐
                   │  Answer +   │    │  Source      │
                   │  Citations  │    │  Metadata    │
                   └─────────────┘    └──────────────┘
```

### Pipeline Steps

1. **PDF Ingestion** (`src/ingestion/pdf_extractor.py`): Uses `pdfplumber` to extract text and tables from PDFs, preserving page numbers.
2. **Chunking** (`src/ingestion/chunker.py`): RecursiveCharacterTextSplitter with chunk_size=1000, overlap=200. Respects paragraph boundaries.
3. **Embeddings** (`src/embeddings/embedder.py`): `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local, fast CPU inference).
4. **Vector Store** (`src/embeddings/vector_store.py`): FAISS IndexFlatIP with cosine similarity (normalized vectors).
5. **Retrieval** (`src/retrieval/retriever.py`): Top-K semantic search with similarity threshold (default 0.3).
6. **Generation** (`src/generation/generator.py`): Ollama LLM (llama3.2:latest) with strict grounding prompt.
7. **API** (`src/api/main.py`): FastAPI with `/ask` endpoint returning answer + sources.
8. **UI** (`src/ui/app.py`): Streamlit chat interface with source citation display.

## Design Decisions

### Chunking Strategy
- **Chunk size: 1000 chars** — Balances context window (fits multiple chunks) with semantic coherence
- **Overlap: 200 chars** — Preserves context across boundaries, mitigates splitting mid-concept
- **Recursive splitter** — Splits on `\n\n` → `\n` → `. ` → ` ` → ``, preserving document structure
- **Tables included** — Extracted tables formatted as markdown and appended to page text before chunking

### Embedding Model: `all-MiniLM-L6-v2`
- 384 dimensions, 22M parameters
- Strong MTEB benchmarks, fast CPU inference (~1000 docs/sec)
- No API key required, fully local

### Vector Database: FAISS
- In-memory index with disk persistence (pickle + .faiss)
- IndexFlatIP (inner product) with L2-normalized vectors = cosine similarity
- Simple, fast, no external dependencies

### LLM: Ollama (llama3.2:latest)
- Local, free, no API costs
- 2B parameter model, good instruction following
- Temperature 0.1 for deterministic, grounded answers

### Prompt Design
- Explicit "answer ONLY from context" instruction
- Mandatory "I cannot answer..." response for out-of-scope questions
- Numbered source citations [Source 1], [Source 2] in context
- Low temperature (0.1) to reduce hallucination

### Source Attribution
Every answer returns:
- `doc_name` (e.g., "Employee_Handbook")
- `page_num` (1-indexed)
- `chunk_id` (global chunk index)
- `similarity` score (cosine)
- Chunk text preview

## Setup & Run Instructions

### Prerequisites
- Python 3.10+
- Ollama installed and running (`ollama serve`)
- Model pulled: `ollama pull llama3.2:latest`

### Quick Start (Any OS)
```bash
# 1. Clone repository
git clone https://github.com/Janhavi-K99/ATMAN_RAG.git
cd ATMAN_RAG

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build vector store (one-time, ~30 sec)
python run_phase1.py

# 4. Start API + UI (one command)
python run.py
```

### Manual Start (2 Terminals)
```bash
# Terminal 1 - API
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 - UI
python -m streamlit run src/ui/app.py --server.port 8501
```

### Verify
- Open browser: http://localhost:8501
- API health: http://localhost:8000/health
- API docs: http://localhost:8000/docs
```

### Using the API
```bash
# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the leave policy?", "top_k": 5}'

# Health check
curl http://localhost:8000/health

# List documents
curl http://localhost:8000/documents
```

### API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Configuration

Copy `.env.example` to `.env` and adjust:

```env
# LLM Provider: "ollama" (local) or "openai"
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest

# Embedding
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector DB
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=atman_docs

# Retrieval
TOP_K=5
SIMILARITY_THRESHOLD=0.3

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## Trade-offs & Limitations

| Aspect | Decision | Trade-off |
|--------|----------|-----------|
| **Chunking** | Fixed-size recursive | May split related content; no semantic boundary detection |
| **Embeddings** | MiniLM-L6-v2 | Smaller model; less nuanced than larger models (e.g., BGE-large, E5) |
| **Retrieval** | Pure semantic (cosine) | No keyword/BM25 hybrid; misses exact-match terms |
| **Reranking** | None | Top-K retrieved directly; no cross-encoder refinement |
| **Context window** | All top-K chunks | No dynamic context budgeting; may exceed LLM context |
| **Conversation** | Stateless | No chat history / multi-turn support |
| **Evaluation** | Manual only | No automated metrics (faithfulness, relevance, etc.) |

### Known Weaknesses
1. **No hybrid search** — Pure dense retrieval misses exact keywords (e.g., error codes, API names)
2. **No reranking** — Cross-encoder would improve precision at top-K
3. **Fixed chunking** — Doesn't adapt to document type (code vs. prose vs. tables)
4. **Single-turn** — No conversation memory or follow-up handling
5. **No evaluation harness** — Cannot measure retrieval quality or answer faithfulness automatically
6. **Ollama dependency** — Requires local GPU/CPU; fallback to OpenAI needs API key

### With More Time Would Implement
1. **Hybrid retrieval** — BM25 + dense fusion (reciprocal rank fusion)
2. **Cross-encoder reranking** — ms-marco-MiniLM-L-6-v2 for top-20 → top-5
3. **Semantic chunking** — LLM-based or embedding-based boundary detection
4. **Conversation history** — Session-aware retrieval with query rewriting
5. **Evaluation suite** — RAGAS metrics (faithfulness, answer_relevancy, context_precision)
6. **Streaming answers** — Token-by-token streaming for better UX
7. **Document versioning** — Incremental updates, deletion, metadata filtering
8. **Multi-modal** — Image/table extraction with vision LLM for diagrams

## Project Structure

```
ATMAN_ASSIGNMENT/
├── data/                       # Source PDFs (7 documents)
├── chroma_db/                  # FAISS index + metadata (persisted)
├── src/
│   ├── ingestion/
│   │   ├── pdf_extractor.py    # PDF → text + tables + page nums
│   │   └── chunker.py          # Recursive chunking
│   ├── embeddings/
│   │   ├── embedder.py         # SentenceTransformer wrapper
│   │   └── vector_store.py     # FAISS index + persistence
│   ├── retrieval/
│   │   └── retriever.py        # Top-K search + context formatting
│   ├── generation/
│   │   └── generator.py        # LLM prompt + grounded answer
│   ├── api/
│   │   └── main.py             # FastAPI endpoints
│   └── ui/
│       └── app.py              # Streamlit chat interface
├── run_phase1.py               # Build vector store from PDFs
├── run.py                      # One-command API + UI startup
├── test_pipeline.py            # End-to-end pipeline test
├── requirements.txt            # Python dependencies
├── .env.example                # Config template (no secrets)
├── .env                        # Local config (gitignored)
├── sample_qa.log               # 10+ Q&A examples
└── README.md                   # This file
```

## Sample Q&A Log

See `sample_qa.log` for 10+ example questions with answers and sources.

## AI Assistant Usage

This project was developed with assistance from AI coding assistants (Claude, Copilot) for:
- Boilerplate code generation (FastAPI, Streamlit, Pydantic models)
- FAISS index management patterns
- Prompt engineering iterations
- Debugging dependency conflicts

All architectural decisions, code review, and integration were performed by the author.

## License

MIT License — for evaluation purposes only.