from sentence_transformers import SentenceTransformer

_model = None  # module-level cache, filled in on first use


def get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load the embedding model once and reuse it on later calls."""
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def embed_texts(texts: list[str]):
    """Turn a list of strings into a NumPy array of embedding vectors."""
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


if __name__ == "__main__":
    import numpy as np
    from ingestion import extract_text, chunk_text

    text = extract_text("data/handbook.pdf")
    chunks = chunk_text(text)
    print(f"Embedding {len(chunks)} chunks...")

    vectors = embed_texts(chunks)
    print(f"Embeddings shape: {vectors.shape}")  # expect (9, 384)

    question = "How many vacation days do employees get?"
    q_vec = embed_texts([question])[0]

    scores = vectors @ q_vec              # dot product of question vs every chunk
    best = int(np.argmax(scores))         # index of the highest-scoring chunk
    print(f"\nQuestion: {question}")
    print(f"Best match: chunk #{best} (score {scores[best]:.3f})\n")
    print(chunks[best])