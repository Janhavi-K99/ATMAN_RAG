#!/usr/bin/env python
"""
Run script to start the full RAG system:
1. FastAPI backend on port 8000
2. Streamlit frontend on port 8501
"""
import os
import sys
import subprocess
import time
import signal
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def run_api():
    """Start FastAPI server."""
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "src.api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ], cwd=PROJECT_ROOT)


def run_streamlit():
    """Start Streamlit app."""
    env = os.environ.copy()
    env["API_URL"] = "http://localhost:8000"
    return subprocess.Popen([
        sys.executable, "-m", "streamlit", "run",
        "src/ui/app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ], cwd=PROJECT_ROOT, env=env)


def main():
    print("=" * 60)
    print("Starting Atman RAG Document Q&A System")
    print("=" * 60)
    print("FastAPI:  http://localhost:8000")
    print("Streamlit: http://localhost:8501")
    print("API Docs: http://localhost:8000/docs")
    print("=" * 60)
    
    api_process = run_api()
    time.sleep(3)  # Wait for API to start
    
    streamlit_process = run_streamlit()
    
    print("\nBoth services started. Press Ctrl+C to stop.")
    
    try:
        api_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        api_process.terminate()
        streamlit_process.terminate()
        api_process.wait()
        streamlit_process.wait()
        print("Done.")


if __name__ == "__main__":
    main()