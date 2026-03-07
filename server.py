import os, json, glob, csv, re, requests, threading, time, uuid, hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pypdf import PdfReader
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR   = os.getenv("DATA_DIR",   "/app/data")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads")
INDEX_DIR  = os.getenv("INDEX_DIR",  "/tmp/index")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MODEL      = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))
OLLAMA_EXTRACT_TIMEOUT = int(os.getenv("OLLAMA_EXTRACT_TIMEOUT", "120"))
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".csv", ".json"}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

class HashEmbeddingModel:
    """
    Deterministic, dependency-free fallback when SentenceTransformer cannot load.
    Keeps the API available in offline/no-egress environments.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        for token in re.findall(r"\w+", (text or "").lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if (digest[4] & 1) else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode(self, text):
        if isinstance(text, str):
            return self._encode_one(text)
        return np.array([self._encode_one(t) for t in text], dtype="float32")


def _init_embedding_model():
    try:
        return SentenceTransformer(EMBED_MODEL_NAME)
    except Exception as exc:
        print(f"[warn] Embedding model '{EMBED_MODEL_NAME}' unavailable: {exc}")
        print(f"[warn] Falling back to hash embeddings (dim={EMBED_DIM}).")
        return HashEmbeddingModel(dim=EMBED_DIM)


EMB_MODEL = _init_embedding_model()
DIM = int(getattr(EMB_MODEL, "get_sentence_embedding_dimension", lambda: EMBED_DIM)())

# In-memory cache — populated on first request
_index: Optional[faiss.Index] = None
_meta: Optional[List[Dict]] = None
_index_lock = threading.RLock()

# Background job tracking
_job_lock = threading.Lock()
_jobs: Dict[str, Dict] = {}
_executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKER_THREADS", "4")))

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "app", "static")

# ── Internal data schemas ─────────────────────────────────────────────────────
#
# Ollama extracts uploaded content into one of these four categories.
# Each entry is stored as structured JSON and then rendered to rich natural
# language for semantic indexing — the LLM never sees raw uploaded text.
#
# trail      → hiking stage / route facts
# guesthouse → accommodation details
# transport  → bus / taxi / shuttle connections
# general    → anything else (food, gear, local tips, …)

EXTRACTION_PROMPT = """\
You are a data extraction assistant for a Balkan trekking information system.

Read the text below and extract ALL facts into structured JSON.

First decide which single category best describes the content:
- "trail"      : hiking route / stage / path information
- "guesthouse" : accommodation, guesthouse, lodge, hotel, hostel
- "transport"  : bus, taxi, shuttle, ferry, transfer between places
- "general"    : anything else (food, gear, tips, culture, emergency contacts, etc.)

Then return ONLY a valid JSON object — no explanation, no markdown fences:

{{
  "category": "<trail|guesthouse|transport|general>",
  "entries": [
    {{ ...fields for each distinct item found... }}
  ]
}}

Field schemas per category:

trail entry:
  from_location, to_location, distance_km, distance_mi,
  elevation_gain_m, elevation_loss_m, difficulty, duration_hours,
  surface, waypoints, notes

guesthouse entry:
  name, location, price_per_night, rooms, contact,
  facilities, meals, booking_notes, notes

transport entry:
  from_location, to_location, method, price,
  duration, schedule, operator, notes

general entry:
  topic, location, description, tags (list of keywords)

Use null for any field you cannot find. Extract every distinct item as a
separate entry. If the text describes multiple guesthouses, emit one entry
per guesthouse, and so on.

TEXT:
{text}

JSON:"""


def _call_ollama_generate(prompt: str, timeout: int = OLLAMA_EXTRACT_TIMEOUT) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


def extract_structure(raw_text: str, filename: str) -> Optional[Dict]:
    """
    Send raw text to Ollama and get back a validated structured dict.
    Returns None if extraction fails so the caller can fall back to raw indexing.
    """
    prompt = EXTRACTION_PROMPT.format(text=raw_text[:4000])
    try:
        response = _call_ollama_generate(prompt, timeout=OLLAMA_EXTRACT_TIMEOUT)
        # Strip markdown fences if the model added them anyway
        response = re.sub(r"```(?:json)?", "", response).strip().rstrip("`")
        # Find the outermost JSON object
        match = re.search(r"\{[\s\S]*\}", response)
        if not match:
            return None
        data = json.loads(match.group())
        if not isinstance(data.get("entries"), list) or not data.get("category"):
            return None
        data["source"] = filename
        return data
    except Exception:
        return None


def structured_to_text(data: Dict) -> str:
    """
    Render a structured JSON record into clear natural-language paragraphs
    ready for chunking and semantic indexing.
    """
    category = data.get("category", "general")
    source   = data.get("source", "")
    entries  = data.get("entries", [])

    header = f"[Source: {source}] [Category: {category}]"
    blocks = [header]

    for e in entries:
        if not isinstance(e, dict):
            continue
        parts: List[str] = []

        if category == "trail":
            frm, to = e.get("from_location"), e.get("to_location")
            if frm and to:
                parts.append(f"Trail stage from {frm} to {to}")
            elif frm or to:
                parts.append(f"Trail: {frm or to}")
            if e.get("distance_km"):
                d = f"Distance: {e['distance_km']} km"
                if e.get("distance_mi"):
                    d += f" ({e['distance_mi']} miles)"
                parts.append(d)
            if e.get("elevation_gain_m"):
                parts.append(f"Elevation gain: {e['elevation_gain_m']} m")
            if e.get("elevation_loss_m"):
                parts.append(f"Elevation loss: {e['elevation_loss_m']} m")
            if e.get("difficulty"):
                parts.append(f"Difficulty: {e['difficulty']}")
            if e.get("duration_hours"):
                parts.append(f"Estimated duration: {e['duration_hours']} hours")
            if e.get("surface"):
                parts.append(f"Surface: {e['surface']}")
            if e.get("waypoints"):
                parts.append(f"Waypoints: {e['waypoints']}")
            if e.get("notes"):
                parts.append(str(e["notes"]))

        elif category == "guesthouse":
            name = e.get("name") or "Guesthouse"
            loc  = e.get("location")
            parts.append(name + (f" in {loc}" if loc else ""))
            if e.get("price_per_night"):
                parts.append(f"Price: {e['price_per_night']} per night")
            if e.get("rooms"):
                parts.append(f"Rooms: {e['rooms']}")
            if e.get("contact"):
                parts.append(f"Contact: {e['contact']}")
            if e.get("facilities"):
                parts.append(f"Facilities: {e['facilities']}")
            if e.get("meals"):
                parts.append(f"Meals: {e['meals']}")
            if e.get("booking_notes"):
                parts.append(str(e["booking_notes"]))
            if e.get("notes"):
                parts.append(str(e["notes"]))

        elif category == "transport":
            frm    = e.get("from_location", "")
            to     = e.get("to_location", "")
            method = e.get("method") or "Transport"
            parts.append(f"{method.title()} from {frm} to {to}" if frm and to else method)
            if e.get("price"):
                parts.append(f"Price: {e['price']}")
            if e.get("duration"):
                parts.append(f"Duration: {e['duration']}")
            if e.get("schedule"):
                parts.append(f"Schedule: {e['schedule']}")
            if e.get("operator"):
                parts.append(f"Operator: {e['operator']}")
            if e.get("notes"):
                parts.append(str(e["notes"]))

        else:  # general
            if e.get("topic"):
                parts.append(str(e["topic"]))
            if e.get("location"):
                parts.append(f"Location: {e['location']}")
            if e.get("description"):
                parts.append(str(e["description"]))
            if e.get("tags") and isinstance(e["tags"], list):
                parts.append(f"Tags: {', '.join(e['tags'])}")

        if parts:
            blocks.append(". ".join(parts).rstrip(".") + ".")

    return "\n\n".join(blocks)


# ── File reading ──────────────────────────────────────────────────────────────

def read_file(path: str) -> str:
    # Structured JSON files produced by our extraction pipeline
    if path.endswith(".structured.json"):
        with open(path, "r", encoding="utf-8") as f:
            return structured_to_text(json.load(f))

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
    """Convert CSV rows into explicit natural-language paragraphs."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return ""

    keys = set(rows[0].keys())
    is_route_schema = {"route", "distance_mi", "distance_km"}.issubset(keys)

    blocks = []
    for row in rows:
        if is_route_schema:
            stage = row.get("stage", "").strip()
            route = row.get("route", "").strip()
            if not route:
                continue
            label   = f"Stage {stage}: {route}" if stage else route
            dist_mi = row.get("distance_mi", "").strip()
            dist_km = row.get("distance_km", "").strip()
            gain_ft = row.get("gain_ft_avg") or row.get("gain_ft_min") or row.get("Gain_ft", "")
            gain_m  = row.get("gain_m_avg")  or row.get("gain_m_min")  or row.get("Gain_m", "")
            loss_ft = row.get("loss_ft_avg") or row.get("loss_ft_min") or row.get("Loss_ft", "")
            loss_m  = row.get("loss_m_avg")  or row.get("loss_m_min")  or row.get("Loss_m", "")
            conf    = row.get("confidence")  or row.get("Confidence", "")

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
            lines = [f"{k}: {v}" for k, v in row.items() if str(v).strip()]
            blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, source_label: str = "", max_chars: int = 1800) -> List[str]:
    """Paragraph-aware chunking that carries section headings into every chunk."""
    heading_re = re.compile(r"^(#{1,3}\s+.+|[A-Z][A-Z ]{3,}:?)$")
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: List[str] = []
    current_heading = ""
    buffer: List[str] = []
    buffer_len = 0

    def flush(heading: str, buf: List[str]) -> None:
        if not buf:
            return
        body = "\n\n".join(buf)
        prefix_parts = []
        if source_label:
            prefix_parts.append(f"[Source: {source_label}]")
        if heading:
            prefix_parts.append(f"[Section: {heading}]")
        prefix = "\n".join(prefix_parts)
        chunks.append(f"{prefix}\n\n{body}".strip() if prefix else body)

    for para in paragraphs:
        if heading_re.match(para):
            flush(current_heading, buffer)
            buffer, buffer_len = [], 0
            current_heading = para.lstrip("#").strip()
            continue

        if len(para) > max_chars:
            flush(current_heading, buffer)
            buffer, buffer_len = [], 0
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sent_buf, sent_len = [], 0
            for s in sentences:
                if sent_len + len(s) > max_chars and sent_buf:
                    flush(current_heading, sent_buf)
                    sent_buf, sent_len = [], 0
                sent_buf.append(s)
                sent_len += len(s)
            flush(current_heading, sent_buf)
            continue

        if buffer_len + len(para) > max_chars and buffer:
            flush(current_heading, buffer)
            buffer, buffer_len = [], 0

        buffer.append(para)
        buffer_len += len(para)

    flush(current_heading, buffer)
    return chunks


# ── Indexing ──────────────────────────────────────────────────────────────────

def source_label(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    name = re.sub(r"\.structured$", "", name)
    return name.replace("_", " ").replace("-", " ").title()


def index_paths() -> List[str]:
    """
    Collect files to index.
    For the upload directory: prefer .structured.json over the raw original.
    """
    exts = ["*.md", "*.txt", "*.pdf", "*.csv", "*.json"]
    base_paths: List[str] = []
    for base in [DATA_DIR, "/app/data"]:
        if not os.path.isdir(base):
            continue
        for e in exts:
            base_paths += glob.glob(os.path.join(base, "**", e), recursive=True)

    upload_paths: List[str] = []
    if os.path.isdir(UPLOAD_DIR):
        structured = set(glob.glob(os.path.join(UPLOAD_DIR, "**", "*.structured.json"), recursive=True))
        # Raw originals that have already been extracted → skip them
        has_structured = {p.replace(".structured.json", "") for p in structured}
        for e in exts:
            for p in glob.glob(os.path.join(UPLOAD_DIR, "**", e), recursive=True):
                if p in structured:
                    upload_paths.append(p)  # always index structured
                elif not any(p.startswith(h) or p == h + os.path.splitext(p)[1] for h in has_structured):
                    upload_paths.append(p)  # raw fallback only if no structured version

    return sorted(set(base_paths + upload_paths))


def save_meta(meta: List[Dict]):
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(os.path.join(INDEX_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_meta() -> List[Dict]:
    with open(os.path.join(INDEX_DIR, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def build_index_data():
    paths = index_paths()
    vectors, meta = [], []
    for p in paths:
        text  = read_file(p)
        label = source_label(p)
        for j, c in enumerate(chunk_text(text, source_label=label)):
            vectors.append(EMB_MODEL.encode(c))
            meta.append({"path": p, "chunk_id": j, "text": c})
    if not vectors:
        raise RuntimeError(f"No documents found. Searched: {DATA_DIR}, /app/data, {UPLOAD_DIR}")
    X = np.array(vectors, dtype="float32")
    idx = faiss.IndexFlatIP(DIM)
    faiss.normalize_L2(X)
    idx.add(X)
    return idx, meta


def persist_index(idx: faiss.Index, meta: List[Dict]) -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(idx, os.path.join(INDEX_DIR, "faiss.index"))
    save_meta(meta)


def build_index():
    idx, meta = build_index_data()
    persist_index(idx, meta)
    return idx, meta


def load_index():
    idx_path = os.path.join(INDEX_DIR, "faiss.index")
    if not os.path.exists(idx_path):
        idx, meta = build_index()
        return idx, meta
    return faiss.read_index(idx_path), load_meta()


def retrieve(query: str, k: int = 8):
    global _index, _meta
    with _index_lock:
        if _index is None or _meta is None:
            _index, _meta = load_index()
        idx, meta = _index, _meta
    q = EMB_MODEL.encode(query).astype("float32")[None, :]
    faiss.normalize_L2(q)
    scores, ids = idx.search(q, k)
    return [(float(s), meta[i]) for s, i in zip(scores[0], ids[0]) if i != -1]


# ── LLM answer generation ─────────────────────────────────────────────────────

def _build_answer_prompt(question: str, contexts: List[Dict]) -> str:
    ctx = "\n\n".join(c["text"] for c in contexts)
    return f"""You are a friendly and knowledgeable trekking assistant. Answer the question below using only the data provided.

Guidelines:
- Write in clear, natural sentences — never output raw CSV values or comma-separated lists.
- Always include relevant numbers with their units (km, mi, meters, feet, price).
- When comparing stages (e.g. "easiest", "longest"), look at the relevant fields, state the winner, and briefly explain why.
- Keep your answer concise: 1–4 sentences is ideal.
- If the data does not contain the answer, say: "I don't have that information in the current dataset."

DATA:
{ctx}

QUESTION: {question}

ANSWER:""".strip()


def ollama_answer(question: str, contexts: List[Dict]) -> str:
    prompt = _build_answer_prompt(question, contexts)
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        if r.status_code == 404:
            raise requests.HTTPError("chat endpoint not found", response=r)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except requests.HTTPError:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["response"].strip()
    except requests.RequestException:
        return "The language model is taking too long to respond. Please try again in a moment."


def ollama_stream(question: str, contexts: List[Dict]):
    """Generator that yields SSE events with tokens streamed from Ollama."""
    prompt = _build_answer_prompt(question, contexts)

    def _try_chat_stream():
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": True},
            stream=True,
            timeout=OLLAMA_TIMEOUT,
        )
        if r.status_code == 404:
            raise requests.HTTPError("chat endpoint not found", response=r)
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
            if chunk.get("done"):
                break

    def _try_generate_stream():
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL, "prompt": prompt, "stream": True},
            stream=True,
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("response", "")
            if token:
                yield token
            if chunk.get("done"):
                break

    try:
        try:
            for token in _try_chat_stream():
                yield f"data: {json.dumps({'token': token})}\n\n"
        except requests.HTTPError:
            for token in _try_generate_stream():
                yield f"data: {json.dumps({'token': token})}\n\n"
    except requests.RequestException as e:
        yield f"data: {json.dumps({'error': 'Model is not responding. Please try again.'})}\n\n"

    yield "data: [DONE]\n\n"


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_filename(name: str) -> str:
    name = os.path.basename(name)
    return re.sub(r"[^\w.\-]", "_", name)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _create_job(kind: str, payload: Optional[Dict] = None) -> str:
    job_id = uuid.uuid4().hex
    with _job_lock:
        _jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "queued_at": _now_iso(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "payload": payload or {},
        }
        # keep the last 200 jobs to avoid unbounded growth
        if len(_jobs) > 200:
            for jid in list(_jobs.keys())[:-200]:
                _jobs.pop(jid, None)
    return job_id


def _update_job(job_id: str, **fields) -> None:
    with _job_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def _run_reindex_job(job_id: str) -> None:
    _update_job(job_id, status="running", started_at=_now_iso())
    try:
        idx, meta = build_index_data()
        persist_index(idx, meta)
        with _index_lock:
            global _index, _meta
            _index, _meta = idx, meta
        _update_job(job_id, status="done", finished_at=_now_iso())
    except Exception as exc:
        _update_job(job_id, status="failed", finished_at=_now_iso(), error=str(exc))


def _run_upload_job(job_id: str, paths: List[str]) -> None:
    _update_job(job_id, status="running", started_at=_now_iso())
    processed = []
    try:
        for dest in paths:
            try:
                text = read_file(dest)
                structured = extract_structure(text, os.path.basename(dest))
                if structured:
                    struct_path = dest + ".structured.json"
                    with open(struct_path, "w", encoding="utf-8") as f:
                        json.dump(structured, f, ensure_ascii=False, indent=2)
                    processed.append({
                        "file": os.path.basename(dest),
                        "category": structured["category"],
                        "entries_extracted": len(structured["entries"]),
                        "indexed_from": "structured",
                    })
                else:
                    processed.append({
                        "file": os.path.basename(dest),
                        "category": "raw",
                        "entries_extracted": None,
                        "indexed_from": "raw",
                    })
            except Exception as exc:
                processed.append({
                    "file": os.path.basename(dest),
                    "category": "error",
                    "entries_extracted": None,
                    "indexed_from": "raw",
                    "error": str(exc),
                })

        idx, meta = build_index_data()
        persist_index(idx, meta)
        with _index_lock:
            global _index, _meta
            _index, _meta = idx, meta
        _update_job(job_id, status="done", finished_at=_now_iso(), result=processed)
    except Exception as exc:
        _update_job(job_id, status="failed", finished_at=_now_iso(), error=str(exc), result=processed)


# ── Endpoints ─────────────────────────────────────────────────────────────────

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "ok", "message": "UI not built. Place files in app/static."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/files")
def list_files():
    if not os.path.isdir(UPLOAD_DIR):
        return {"files": []}
    files = []
    for p in sorted(glob.glob(os.path.join(UPLOAD_DIR, "**", "*"), recursive=True)):
        if os.path.isfile(p) and not p.endswith(".structured.json"):
            struct_path = p + ".structured.json"
            entry: Dict = {
                "name": os.path.relpath(p, UPLOAD_DIR),
                "size_kb": round(os.path.getsize(p) / 1024, 1),
                "processed": os.path.exists(struct_path),
            }
            if entry["processed"]:
                with open(struct_path) as f:
                    d = json.load(f)
                entry["category"] = d.get("category")
                entry["entries"]  = len(d.get("entries", []))
            files.append(entry)
    return {"files": files}


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    saved_paths = []

    for file in files:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"{file.filename}: unsupported type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            )
        content = await file.read()
        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(status_code=400, detail=f"{file.filename}: exceeds 10 MB limit.")

        # Save raw file
        dest = os.path.join(UPLOAD_DIR, safe_filename(file.filename or "upload"))
        with open(dest, "wb") as f:
            f.write(content)
        saved_paths.append(dest)

    job_id = _create_job("upload", {"files": [os.path.basename(p) for p in saved_paths]})
    _executor.submit(_run_upload_job, job_id, saved_paths)
    return {"status": "queued", "job_id": job_id, "files": [os.path.basename(p) for p in saved_paths]}


@app.post("/reindex")
def reindex():
    job_id = _create_job("reindex")
    _executor.submit(_run_reindex_job, job_id)
    return {"status": "queued", "job_id": job_id}


@app.get("/jobs")
def list_jobs():
    with _job_lock:
        jobs = list(_jobs.values())
    jobs.sort(key=lambda j: j.get("queued_at") or "", reverse=True)
    return {"jobs": jobs[:50]}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with _job_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/ask")
def ask(q: str):
    hits = retrieve(q, k=8)
    strong = [m for s, m in hits if s >= 0.10]
    if not strong:
        return {"answer": "I don't have that information in the current dataset."}
    return {"answer": ollama_answer(q, strong[:6])}


@app.get("/ask/stream")
def ask_stream(q: str):
    """SSE endpoint — tokens arrive as they are generated by Ollama."""
    hits = retrieve(q, k=8)
    strong = [m for s, m in hits if s >= 0.10]
    if not strong:
        def _no_data():
            yield f"data: {json.dumps({'token': 'I don' + chr(39) + 't have that information in the current dataset.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_no_data(), media_type="text/event-stream")
    return StreamingResponse(
        ollama_stream(q, strong[:6]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
