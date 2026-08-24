from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

from src.retrieval.retriever import Retriever
from src.generation.generator import LLMGenerator
from src.cache import get_retriever, get_generator, preload_all


app = FastAPI(
    title="RAG Document Q&A API",
    description="Retrieval-Augmented Generation API for document question answering with conversation memory",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conversation_sessions: Dict[str, List[Dict[str, str]]] = {}


class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None
    session_id: Optional[str] = None


class SourceResponse(BaseModel):
    source_num: int
    doc_name: str
    page_num: int
    chunk_id: int
    text: str
    similarity: float


class QuestionResponse(BaseModel):
    answer: str
    sources: List[SourceResponse]
    context_used: str
    session_id: str


@app.on_event("startup")
async def startup_event():
    preload_all()


def get_session_history(session_id: str) -> List[Dict[str, str]]:
    return conversation_sessions.get(session_id, [])


def update_session_history(session_id: str, user_q: str, assistant_a: str):
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = []
    conversation_sessions[session_id].append({
        "user": user_q,
        "assistant": assistant_a
    })
    if len(conversation_sessions[session_id]) > 6:
        conversation_sessions[session_id] = conversation_sessions[session_id][-6:]


@app.get("/")
async def root():
    return {
        "message": "RAG Document Q&A API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    from src.cache import get_vector_store
    vs = get_vector_store()
    stats = vs.get_stats()
    return {
        "status": "healthy",
        "vector_store": stats,
        "active_sessions": len(conversation_sessions)
    }


@app.post("/session/new")
async def new_session():
    session_id = str(uuid.uuid4())[:8]
    conversation_sessions[session_id] = []
    return {"session_id": session_id, "message": "New session created"}


@app.get("/session/{session_id}/history")
async def get_session_history_endpoint(session_id: str):
    if session_id not in conversation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "history": conversation_sessions[session_id],
        "turn_count": len(conversation_sessions[session_id])
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id not in conversation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del conversation_sessions[session_id]
    return {"message": "Session deleted"}


@app.get("/sessions")
async def list_sessions():
    sessions = []
    for sid, history in conversation_sessions.items():
        sessions.append({
            "session_id": sid,
            "turn_count": len(history),
            "last_message": history[-1]["user"][:50] + "..." if history else "Empty"
        })
    return {"sessions": sessions}


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    try:
        retriever = get_retriever()
        generator = get_generator()

        session_id = request.session_id or str(uuid.uuid4())[:8]
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = []

        conversation_history = get_session_history(session_id)

        if request.top_k is not None:
            retriever.top_k = request.top_k
        if request.similarity_threshold is not None:
            retriever.similarity_threshold = request.similarity_threshold

        chunks = retriever.retrieve(request.question)

        result = generator.generate_with_sources(
            request.question,
            chunks,
            conversation_history=conversation_history
        )

        update_session_history(session_id, request.question, result["answer"])

        return QuestionResponse(
            answer=result["answer"],
            sources=[SourceResponse(**s) for s in result["sources"]],
            context_used=result["context_used"],
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    from src.cache import get_vector_store
    vs = get_vector_store()
    if vs.index is None or vs.index.ntotal == 0:
        return {"documents": []}
    doc_names = set()
    for meta in vs.chunks_metadata:
        doc_names.add(meta["doc_name"])
    return {"documents": sorted(list(doc_names))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False
    )