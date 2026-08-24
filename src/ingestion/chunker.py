from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from dataclasses import dataclass
from src.ingestion.pdf_extractor import PageContent, format_tables_for_text


@dataclass
class Chunk:
    doc_name: str
    page_num: int
    chunk_id: int
    text: str
    char_start: int
    char_end: int


def create_chunker(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """Create a recursive character text splitter with sensible defaults."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def chunk_page_content(page: PageContent, chunker: RecursiveCharacterTextSplitter, global_chunk_id: int) -> List[Chunk]:
    """Split a single page's content into chunks."""
    full_text = page.text + format_tables_for_text(page.tables)
    
    if not full_text.strip():
        return []
    
    chunks = chunker.create_documents(
        [full_text],
        metadatas=[{"doc_name": page.doc_name, "page_num": page.page_num}]
    )
    
    result = []
    for i, chunk in enumerate(chunks):
        result.append(Chunk(
            doc_name=page.doc_name,
            page_num=page.page_num,
            chunk_id=global_chunk_id + i,
            text=chunk.page_content,
            char_start=0,
            char_end=len(chunk.page_content)
        ))
    
    return result


def chunk_all_pages(pages: List[PageContent], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Chunk]:
    """Chunk all pages into overlapping segments."""
    chunker = create_chunker(chunk_size, chunk_overlap)
    all_chunks = []
    global_chunk_id = 0
    
    for page in pages:
        page_chunks = chunk_page_content(page, chunker, global_chunk_id)
        all_chunks.extend(page_chunks)
        global_chunk_id += len(page_chunks)
    
    return all_chunks


if __name__ == "__main__":
    from src.ingestion.pdf_extractor import extract_all_pdfs
    from pathlib import Path
    
    data_dir = Path(__file__).parent.parent.parent / "data"
    pages = extract_all_pdfs(data_dir)
    chunks = chunk_all_pages(pages)
    print(f"Total chunks created: {len(chunks)}")
    for c in chunks[:3]:
        print(f"  Chunk {c.chunk_id}: {c.doc_name} p{c.page_num} ({len(c.text)} chars)")