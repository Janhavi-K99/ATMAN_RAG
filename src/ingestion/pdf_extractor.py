import pdfplumber
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class PageContent:
    doc_name: str
    page_num: int
    text: str
    tables: List[List[List[str]]]


def extract_pdf_content(pdf_path: Path) -> List[PageContent]:
    """Extract text and tables from a PDF, preserving page numbers."""
    pages_content = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            
            tables = []
            extracted_tables = page.extract_tables()
            if extracted_tables:
                for table in extracted_tables:
                    if table:
                        cleaned_table = [[cell or "" for cell in row] for row in table]
                        tables.append(cleaned_table)
            
            pages_content.append(PageContent(
                doc_name=pdf_path.stem,
                page_num=page_num,
                text=text,
                tables=tables
            ))
    
    return pages_content


def format_tables_for_text(tables: List[List[List[str]]]) -> str:
    """Convert extracted tables to readable text format."""
    if not tables:
        return ""
    
    formatted = ["\n--- TABLE ---"]
    for table_idx, table in enumerate(tables):
        formatted.append(f"Table {table_idx + 1}:")
        for row in table:
            formatted.append(" | ".join(row))
    formatted.append("--- END TABLE ---\n")
    return "\n".join(formatted)


def extract_all_pdfs(data_dir: Path) -> List[PageContent]:
    """Extract content from all PDFs in the data directory."""
    all_pages = []
    pdf_files = list(data_dir.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_path in pdf_files:
        if pdf_path.name == "RAG_AI_ML_Assignment.pdf":
            print(f"Skipping assignment PDF: {pdf_path.name}")
            continue
            
        print(f"Processing: {pdf_path.name}")
        pages = extract_pdf_content(pdf_path)
        all_pages.extend(pages)
        print(f"  Extracted {len(pages)} pages")
    
    return all_pages


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent / "data"
    pages = extract_all_pdfs(data_dir)
    print(f"\nTotal pages extracted: {len(pages)}")
    for p in pages[:3]:
        print(f"  {p.doc_name} - Page {p.page_num}: {len(p.text)} chars")