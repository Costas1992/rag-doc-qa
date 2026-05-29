import os
import chromadb
from embeddings import embed_texts
from ingestion import extract_text, chunk_text

CHROMA_PATH = "chroma_db"          # folder where Chroma saves to disk
COLLECTION_NAME = "handbook"


def build_store(chunks: list[str]):
    """Embed chunks and store them in a persistent Chroma collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Start fresh each build so re-runs don't pile up duplicates
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    vectors = embed_texts(chunks)                    # NumPy array (N, 384)
    ids = [f"chunk_{i}" for i in range(len(chunks))] # a unique id per chunk

    collection.add(
        ids=ids,
        documents=chunks,                # the original text, stored alongside
        embeddings=vectors.tolist(),     # the vectors, as plain Python lists
    )
    return collection


# tracks the name of whatever document is currently loaded.
# NOTE: a single shared dict = single-active-document model. For true multi-user,
# this would become a session-keyed structure (e.g. {session_id: doc_name}).
_active_doc = {"name": None}


def build_store_from_pdf(pdf_path: str, doc_name: str) -> int:
    """Ingest a PDF file and (re)build the active collection. Returns chunk count."""
    text = extract_text(pdf_path)
    if not text.strip():
        raise ValueError("No extractable text found — is this a scanned PDF?")
    chunks = chunk_text(text)
    build_store(chunks)              # reuses the existing wipe-and-rebuild logic
    _active_doc["name"] = doc_name
    return len(chunks)


def active_document() -> str | None:
    """Return the name of the currently loaded document, or None."""
    return _active_doc["name"]


def query_store(question: str, k: int = 3):
    """Return the k chunks most similar to the question."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    q_vec = embed_texts([question])[0]               # one (384,) vector
    results = collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
    )
    return results


if __name__ == "__main__":
    text = extract_text("data/handbook.pdf")
    chunks = chunk_text(text)

    print("Building vector store...")
    build_store(chunks)
    print(f"Stored {len(chunks)} chunks in '{COLLECTION_NAME}'")

    question = "How many vacation days do employees get?"
    results = query_store(question, k=3)

    print(f"\nQuestion: {question}")
    print(f"Top {len(results['documents'][0])} matches:\n")
    for rank, (doc, dist) in enumerate(
        zip(results["documents"][0], results["distances"][0]), start=1
    ):
        snippet = doc[:90].replace("\n", " ")
        print(f"{rank}. (distance {dist:.3f}) {snippet}...")