import os
import sys
import tempfile

# src/ modules import each other with flat names, so put src/ on the import path.
SRC_PATH = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC_PATH)

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import chromadb
from ingestion import extract_text, chunk_text
from vector_store import (
    build_store,
    build_store_from_pdf,
    active_document,
    CHROMA_PATH,
    COLLECTION_NAME,
)
from embeddings import get_model
from generator import get_llm, generate_answer
from retriever import retrieve, format_context

DATA_PATH = os.path.join(SRC_PATH, "..", "data", "handbook.pdf")
MAX_UPLOAD_MB = 10


def ensure_store():
    """Build the vector store from the bundled handbook if it doesn't exist yet."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    if COLLECTION_NAME not in [c.name for c in client.list_collections()]:
        chunks = chunk_text(extract_text(DATA_PATH))
        build_store(chunks)
        print(f"[startup] built vector store with {len(chunks)} chunks")
    else:
        print("[startup] vector store already exists")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] warming up store and models...")
    ensure_store()
    get_model()    # load embedding model into cache, once
    get_llm()      # load LLM into cache, once
    print("[startup] ready to serve")
    yield


app = FastAPI(title="RAG Document Q&A", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- request/response models ----------

class AskRequest(BaseModel):
    question: str
    k: int = 3


class Source(BaseModel):
    rank: int
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


# ---------- API routes (must be defined BEFORE the static mount) ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    return {"document": active_document()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    chunks = retrieve(req.question, k=req.k)
    context = format_context(chunks)
    answer = generate_answer(req.question, context)
    sources = [Source(rank=i, text=c) for i, c in enumerate(chunks, start=1)]
    return AskResponse(answer=answer, sources=sources)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # 1. validate it's a PDF
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    # 2. enforce a size cap to protect the free CPU/RAM
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File is {size_mb:.1f} MB; limit is {MAX_UPLOAD_MB} MB.",
        )

    # 3. write to a temp file, ingest, then clean up
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        n_chunks = build_store_from_pdf(tmp_path, doc_name=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"document": file.filename, "chunks": n_chunks}


# ---------- static frontend (catch-all "/", MUST be last) ----------

DIST_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(DIST_PATH):
    app.mount("/", StaticFiles(directory=DIST_PATH, html=True), name="static")