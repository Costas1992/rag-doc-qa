"""rag-autotune adapter for rag-doc-qa. LOCKED after `init`.

Reuses the production pieces (src/ingestion.py chunking, Chroma, sentence-transformers)
so tuning results transfer to the app. The only deviation: an in-memory Chroma client,
so experiments never touch the app's chroma_db/ folder.

Why section ids, not chunk ids: chunk ids ("chunk_3") change every time chunk_size
changes, so they can't be labels. Each chunk is labelled with every handbook section it
contains (at least MIN_OVERLAP characters of it), largest share first. retrieve() returns
those labels in rank order, and rag-autotune cuts the list at the locked k. So a chunk that
spans 3 sections uses 3 of the k slots: big chunks can't inflate recall for free, they pay
for the context they take up.
"""
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ingestion import extract_text, chunk_text  # noqa: E402  production chunker

PDF = ROOT / "data" / "handbook.pdf"
_models = {}
_text_cache = {}


def _model(name):
    if name not in _models:
        from sentence_transformers import SentenceTransformer
        _models[name] = SentenceTransformer(name)
    return _models[name]


def _sections(text):
    """Return [(start_offset, 's<n>')] for headings like '3. Vacation and leave'."""
    marks = [(m.start(), f"s{m.group(1)}") for m in re.finditer(r"(?m)^\s*(\d)\.\s+[A-Z]", text)]
    return [(0, "s0")] + marks  # s0 = title block before section 1


MIN_OVERLAP = 40  # chars; smaller slivers of a section rarely hold an answer


def _labels(chunk, text, marks, cursor):
    start = text.find(chunk, cursor)
    if start < 0:
        start = text.find(chunk)
    if start < 0:
        return ["s?"], cursor
    end = start + len(chunk)
    bounds = [m[0] for m in marks] + [len(text)]
    hits = []
    for i, (s_start, sid) in enumerate(marks):
        ov = max(0, min(end, bounds[i + 1]) - max(start, s_start))
        if ov >= MIN_OVERLAP and sid != "s0":  # s0 = title block, never an answer
            hits.append((ov, sid))
    hits.sort(reverse=True)
    return [sid for _, sid in hits] or ["s?"], start + 1


def build_index(config):
    import chromadb
    if "text" not in _text_cache:
        _text_cache["text"] = extract_text(str(PDF))
    text = _text_cache["text"]
    chunks = chunk_text(text, chunk_size=int(config["chunk_size"]), chunk_overlap=int(config["chunk_overlap"]))
    marks, cursor, labels = _sections(text), 0, []
    for c in chunks:
        labs, cursor = _labels(c, text, marks, cursor)
        labels.append(",".join(labs))
    model = _model(config["embedding_model"])
    vecs = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    client = chromadb.EphemeralClient()
    col = client.create_collection(f"autotune_{uuid.uuid4().hex[:8]}")
    col.add(ids=[f"c{i}" for i in range(len(chunks))], documents=chunks,
            embeddings=vecs.tolist(), metadatas=[{"section": l} for l in labels])
    return {"col": col, "model": model, "n": len(chunks)}


def retrieve(index, question, config):
    q = (config.get("query_prefix") or "") + question
    qv = index["model"].encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
    res = index["col"].query(query_embeddings=[qv.tolist()],
                             n_results=min(int(config["top_k"]), index["n"]))
    out = []
    for m in res["metadatas"][0]:
        out.extend(m["section"].split(","))
    return out  # rag-autotune dedupes and cuts at the locked k
