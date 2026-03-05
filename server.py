import os, json, glob, csv, requests
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pypdf import PdfReader
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = os.getenv("DATA_DIR", "/data")
INDEX_DIR = "/index"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
DIM = 384

# In-memory cache — loaded once at startup
_index: Optional[faiss.Index] = None
_meta: Optional[List[Dict]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _index, _meta
    _index, _meta = load_index()
    yield


app = FastAPI(lifespan=lifespan)


def read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == ".csv":
        return read_csv_as_text(path)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def read_csv_as_text(path: str) -> str:
    """Convert CSV rows into explicit natural-language paragraphs so small LLMs can't mix up columns."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return ""

    # Detect known route-data schema and render it verbosely
    keys = set(rows[0].keys())
    is_route_schema = {"route", "distance_mi", "distance_km"}.issubset(keys)

    blocks = []
    for row in rows:
        if is_route_schema:
            stage = row.get("stage", "").strip()
            route = row.get("route", "").strip()
            if not route:
                continue  # skip Total / empty rows
            label = f"Stage {stage}: {route}" if stage else route

            dist_mi = row.get("distance_mi", "").strip()
            dist_km = row.get("distance_km", "").strip()
            gain_ft = row.get("gain_ft_avg") or row.get("gain_ft_min") or row.get("Gain_ft", "")
            gain_m  = row.get("gain_m_avg")  or row.get("gain_m_min")  or row.get("Gain_m", "")
            loss_ft = row.get("loss_ft_avg") or row.get("loss_ft_min") or row.get("Loss_ft", "")
            loss_m  = row.get("loss_m_avg")  or row.get("loss_m_min")  or row.get("Loss_m", "")
            conf    = row.get("confidence") or row.get("Confidence", "")

            parts = [label]
            if dist_mi and dist_km:
                parts.append(f"Distance: {dist_mi} miles ({dist_km} km)")
            if gain_ft and gain_m:
                parts.append(f"Elevation gain: {gain_ft} ft ({gain_m} m)")
            if loss_ft and loss_m:
                parts.append(f"Elevation loss: {loss_ft} ft ({loss_m} m)")
            if conf:
                parts.append(f"Data confidence: {conf}")
            blocks.append(". ".join(parts) + ".")
        else:
            # Generic fallback: one "Key: Value" per line per row, separated by blank lines
            lines = [f"{k}: {v}" for k, v in row.items() if str(v).strip()]
            blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    text = " ".join(text.split())
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + max_chars])
        i += max_chars - overlap
    return chunks


def index_paths() -> List[str]:
    exts = ["*.md", "*.txt", "*.pdf", "*.csv", "*.json"]
    search_dirs = [DATA_DIR, "/app/data"]
    paths = []
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        for e in exts:
            paths += glob.glob(os.path.join(base, "**", e), recursive=True)
        if paths:
            break
    return sorted(set(paths))


def save_meta(meta: List[Dict]):
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(os.path.join(INDEX_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_meta() -> List[Dict]:
    with open(os.path.join(INDEX_DIR, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def build_index():
    paths = index_paths()
    vectors = []
    meta = []
    for p in paths:
        text = read_file(p)
        for j, c in enumerate(chunk_text(text)):
            emb = EMB_MODEL.encode(c)
            vectors.append(emb)
            meta.append({"path": p, "chunk_id": j, "text": c})
    if not vectors:
        raise RuntimeError(f"No documents found to index. Searched: {DATA_DIR} and /app/data")
    X = np.array(vectors, dtype="float32")
    idx = faiss.IndexFlatIP(DIM)
    faiss.normalize_L2(X)
    idx.add(X)
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(idx, os.path.join(INDEX_DIR, "faiss.index"))
    save_meta(meta)


def load_index():
    idx_path = os.path.join(INDEX_DIR, "faiss.index")
    if not os.path.exists(idx_path):
        build_index()
    idx = faiss.read_index(idx_path)
    meta = load_meta()
    return idx, meta


def retrieve(query: str, k: int = 8):
    global _index, _meta
    if _index is None or _meta is None:
        _index, _meta = load_index()
    q = EMB_MODEL.encode(query).astype("float32")[None, :]
    faiss.normalize_L2(q)
    scores, ids = _index.search(q, k)
    results = []
    for score, i in zip(scores[0], ids[0]):
        if i == -1:
            continue
        results.append((float(score), _meta[i]))
    return results


def ollama_answer(question: str, contexts: List[Dict]) -> str:
    ctx = "\n\n".join(c["text"] for c in contexts)

    prompt = f"""You are a friendly and knowledgeable trekking assistant. Answer the question below using only the data provided.

Guidelines:
- Write in clear, natural sentences — never output raw CSV values or comma-separated lists.
- Always include relevant numbers with their units (km, mi, meters, feet).
- When comparing stages (e.g. "easiest", "longest", "most elevation gain"), look at the relevant columns, state the winner, and briefly explain why.
- Keep your answer concise: 1–4 sentences is ideal.
- If the data does not contain the answer, say: "I don't have that information in the current dataset."

DATA:
{ctx}

QUESTION: {question}

ANSWER:""".strip()

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        if r.status_code == 404:
            raise requests.HTTPError("chat endpoint not found", response=r)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except requests.HTTPError:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["response"].strip()


@app.post("/reindex")
def reindex():
    global _index, _meta
    build_index()
    _index, _meta = load_index()
    return {"status": "ok"}


@app.get("/ask")
def ask(q: str):
    hits = retrieve(q, k=8)
    strong = [m for s, m in hits if s >= 0.10]
    if not strong:
        return {"answer": "I don't have that information in the current dataset."}
    answer = ollama_answer(q, strong[:6])
    return {"answer": answer}
