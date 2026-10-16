"""rag-autotune config for rag-doc-qa. The ONLY file the agent edits (via `rag-autotune run --set`)."""

CONFIG = {
    "chunk_size": 500,       # characters per chunk (production default in src/ingestion.py)
    "chunk_overlap": 100,    # characters shared between neighbouring chunks
    "top_k": 3,              # chunks retrieved (production: retrieve(question, k=3))
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",  # production model
    "query_prefix": "",      # text prepended to the question before embedding
}
