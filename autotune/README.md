# Retrieval tuning (rag-autotune)

Tunes this app's retrieval with [rag-autotune](https://github.com/Costas1992/rag-autotune).

| File | What | Who edits |
|---|---|---|
| `adapter.py` | wraps the production chunker (`src/ingestion.py`), embedding model and Chroma (in-memory, so `chroma_db/` is never touched) | human only, locked |
| `evalset.jsonl` | 32 questions about `data/handbook.pdf`, labelled with handbook sections `s1`-`s8` | human only, locked |
| `config.py` | chunk_size, chunk_overlap, top_k, embedding_model, query_prefix | the agent, via `--set` |

**Metric: recall@3 over handbook sections.** Each retrieved chunk counts as every section it contains, and the list is cut at 3, so a chunk spanning several sections uses several slots. Big chunks can't score well just by containing everything.

Why not plain chunk-level recall@3: the handbook is ~9 chunks, so 3 chunks is a third of the document and almost any config scores ~1.0.

```bash
cd autotune
rag-autotune init --eval evalset.jsonl --adapter adapter.py --config config.py --k 3
rag-autotune run -m "baseline"
```

If a tuned config wins, copy its `chunk_size` / `chunk_overlap` into `src/ingestion.py` and its `top_k` into the `retrieve(...)` call in `app/main.py`, then rebuild `chroma_db/`.
