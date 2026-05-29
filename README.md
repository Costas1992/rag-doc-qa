---
title: RAG Document Q&A
emoji: 📄
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# RAG Document Q&A

A retrieval-augmented document question-answering system. Upload a PDF and ask it questions — every answer is grounded in the source document and shows the passages it was drawn from.

## How it works

PDF → text (PyMuPDF) → chunks (LangChain recursive splitter) → embeddings (sentence-transformers `all-MiniLM-L6-v2`) → vector store (ChromaDB) → retrieval → grounded generation (Qwen2.5-1.5B-Instruct, running locally — no API keys, no external calls).

A FastAPI backend exposes `/ask`, `/upload`, `/status`, and `/health`, and serves a built React (Vite) frontend from the same origin in production.

## Stack

- **Backend** — Python 3.13, FastAPI, Uvicorn
- **RAG** — LangChain text splitters, sentence-transformers, ChromaDB, Hugging Face Transformers (Qwen2.5-1.5B-Instruct)
- **Frontend** — React + Vite, vanilla CSS
- **Packaging** — Multi-stage Docker (Node build → Python runtime, CPU-only torch)

## Notes on this Space

- **CPU-only.** Generation runs in ~10–30s per question on the free tier; the same code runs much faster on local GPU/MPS.
- **Single-active-document model.** Uploads replace the loaded document for everyone on this Space — appropriate for a demo, not for multi-tenant production. The cleanest way to extend this is per-session collections keyed by a session ID.
- **Free Spaces sleep after inactivity** — the first request after idle takes longer while models reload.
- **Source repo:** [github.com/Costas1992](https://github.com/Costas1992)

## Local development

```bash
# backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000

# frontend (in a second terminal)
cd frontend && npm install && npm run dev
```

Or run the whole thing in one container:

```bash
docker build -t rag-doc-qa .
docker run -p 8000:7860 rag-doc-qa
```
