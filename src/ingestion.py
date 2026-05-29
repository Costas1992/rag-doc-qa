import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text(pdf_path: str) -> str:
    """Open a PDF and return all of its text as one string."""
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text())
    doc.close()
    return "\n".join(pages_text)


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """Split a long string into overlapping chunks on natural boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


if __name__ == "__main__":
    text = extract_text("data/handbook.pdf")
    print(f"Extracted {len(text)} characters")

    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks")
    print(f"Chunk sizes: {[len(c) for c in chunks]}")

    print("\n----- chunk 0 -----")
    print(chunks[0])
    print("\n----- chunk 1 (watch the overlap) -----")
    print(chunks[1])