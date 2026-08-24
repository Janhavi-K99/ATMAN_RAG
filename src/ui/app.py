import streamlit as st
import requests
import os
import uuid
from dotenv import load_dotenv
from typing import Dict, Any, List

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .main-header {
        background: linear-gradient(90deg, #2c3e50 0%, #4a6491 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(44, 62, 80, 0.4);
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p { margin: 0.5rem 0 0; opacity: 0.9; }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #2c3e50;
    }
    .source-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .source-header {
        font-weight: 600;
        color: #495057;
        margin-bottom: 0.5rem;
    }
    .source-text {
        font-family: 'Monospace', monospace;
        font-size: 0.85rem;
        color: #6c757d;
        background: white;
        padding: 0.75rem;
        border-radius: 6px;
        border: 1px solid #dee2e6;
        max-height: 200px;
        overflow-y: auto;
    }
    .answer-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #27ae60;
        margin: 1rem 0;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: linear-gradient(90deg, #2c3e50 0%, #4a6491 100%);
        color: white;
        margin-left: 2rem;
    }
    .assistant-message {
        background: white;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .sidebar .stButton > button {
        width: 100%;
        text-align: left;
        justify-content: flex-start;
    }
    .divider {
        border-top: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    :focus-visible {
        outline: 3px solid #2c3e50 !important;
        outline-offset: 2px !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📚 RAG Document Q&A System</h1>
    <p>Ask questions about internal documents • Get grounded answers with verified sources</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_healthy" not in st.session_state:
    st.session_state.api_healthy = False
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "similarity_threshold" not in st.session_state:
    st.session_state.similarity_threshold = 0.3
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

# Auto-check API health on first load
if not st.session_state.api_healthy:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.api_healthy = True
            st.session_state.chunk_count = data['vector_store']['total_chunks']
    except Exception:
        pass

# Sample questions organized by category (based on available documents)
SAMPLE_QUESTIONS = {
    "📋 Policies & HR": [
        "What is the leave policy?",
        "What is the code of conduct for employees?",
        "What is the onboarding process for new employees?",
        "What are the working hours and overtime policy?",
        "What is the performance review process?",
        "What are the employee benefits?",
        "What is the resignation notice period?",
    ],
    "💰 Pricing & SLA": [
        "What are the pricing tiers?",
        "What are the SLA terms?",
        "What are the API rate limits?",
        "What is included in the enterprise plan?",
        "What are the payment terms?",
        "What is the refund policy?",
    ],
    "🔐 Security & Compliance": [
        "What is the security policy for data handling?",
        "What encryption standards are used?",
        "What is the incident response procedure?",
        "What compliance certifications does the company have?",
        "How is access control managed?",
    ],
    "🔧 Technical & Support": [
        "How do I troubleshoot device connection issues?",
        "What is the product return policy?",
        "How do I set up the API integration?",
        "What are the system requirements?",
        "How do I configure webhooks?",
        "What is the troubleshooting guide for common errors?",
    ],
    "📚 Product & API Reference": [
        "What are the main product features?",
        "What API endpoints are available?",
        "What authentication methods does the API support?",
        "What are the request/response formats?",
        "What are the error codes and their meanings?",
    ],
}

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.top_k = st.slider(
            "Top K", 
            1, 10, 
            st.session_state.top_k, 
            help="Number of chunks to retrieve"
        )
    with col2:
        st.session_state.similarity_threshold = st.slider(
            "Similarity", 
            0.0, 1.0, 
            st.session_state.similarity_threshold, 
            0.05, 
            help="Minimum similarity threshold"
        )
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Health check
    st.markdown("### 🏥 System Health")
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🔄 Check API Health", use_container_width=True):
            try:
                resp = requests.get(f"{API_URL}/health", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.api_healthy = True
                    st.session_state.chunk_count = data['vector_store']['total_chunks']
                    st.success(f"✅ Healthy — {data['vector_store']['total_chunks']} chunks loaded")
                else:
                    st.session_state.api_healthy = False
                    st.error(f"API Error: {resp.status_code}")
            except Exception as e:
                st.session_state.api_healthy = False
                st.error(f"Cannot connect: {e}")
    with col2:
        if st.session_state.api_healthy:
            st.markdown("🟢")
        else:
            st.markdown("🔴")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Stats
    st.markdown("### 📊 Knowledge Base")
    st.metric("Documents", "7")
    st.metric("Chunks Indexed", st.session_state.chunk_count or "—")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Sample Questions Section
    st.markdown("### 💡 Sample Questions")
    
    for category, questions in SAMPLE_QUESTIONS.items():
        with st.expander(category, expanded=False):
            for i, question in enumerate(questions):
                if st.button(
                    question,
                    key=f"sample_{category}_{i}",
                    use_container_width=True,
                    help=f"Click to ask: {question}"
                ):
                    st.session_state.example_question = question
                    st.rerun()
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Clear chat
    if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Session info
    st.markdown("### 💬 Session")
    st.caption(f"Session: `{st.session_state.session_id}`")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()

# Main chat area
if "example_question" in st.session_state and st.session_state.example_question:
    prompt = st.session_state.example_question
    st.session_state.example_question = None
else:
    prompt = st.chat_input(
        "Ask a question about the documents...", 
        key="chat_input",
        disabled=not st.session_state.api_healthy
    )

# Display chat history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f'<div class="answer-box">{msg["content"]}</div>', unsafe_allow_html=True)
                if "sources" in msg and msg["sources"]:
                    with st.expander(f"📖 Sources ({len(msg['sources'])} unique chunks)", expanded=False):
                        for src in msg["sources"]:
                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-header">
                                    Source {src['source_num']} • {src['doc_name']} • Page {src['page_num']} • Similarity: {src['similarity']:.3f}
                                </div>
                                <div class="source-text">{src['text'][:500]}...</div>
                            </div>
                            """, unsafe_allow_html=True)

# Process new question
if prompt:
    if not st.session_state.api_healthy:
        st.error("❌ API is not available. Please check system health.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Get answer
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching documents & generating answer..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": prompt,
                        "top_k": st.session_state.top_k,
                        "similarity_threshold": st.session_state.similarity_threshold,
                        "session_id": st.session_state.session_id
                    },
                    timeout=120
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    # Update session_id if server returned a new one
                    if "session_id" in data and data["session_id"] != st.session_state.session_id:
                        st.session_state.session_id = data["session_id"]
                    
                    # Display answer
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                    
                    # Display sources
                    if sources:
                        with st.expander(f"📖 Sources ({len(sources)} unique chunks)", expanded=False):
                            for src in sources:
                                st.markdown(f"""
                                <div class="source-card">
                                    <div class="source-header">
                                        Source {src['source_num']} • {src['doc_name']} • Page {src['page_num']} • Similarity: {src['similarity']:.3f}
                                    </div>
                                    <div class="source-text">{src['text'][:500]}...</div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No relevant sources found.")
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    error_msg = f"❌ API Error: {resp.status_code} — {resp.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except requests.exceptions.ConnectionError:
                error_msg = "🔴 Cannot connect to API. Make sure the FastAPI server is running on port 8000."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.85rem; padding: 1rem;">
    RAG Document Q&A System • Built with FastAPI + Streamlit + FAISS + Ollama
</div>
""", unsafe_allow_html=True)