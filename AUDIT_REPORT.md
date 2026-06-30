# GraphRAG Research Assistant — Security & Performance Audit Report

**Prepared by:** Principal QA / Security / Performance Review
**Date:** 2026-06-29
**Scope:** Backend API, Extractor Pipeline, Agent Layer, Deployment Configuration
**Severity scale:** CRITICAL → HIGH → MEDIUM → LOW

---

## Executive Summary

The GraphRAG Research Assistant backend is structurally sound and uses modern
Python (3.11) with Pydantic v2 validation throughout. The main risk surface lies
in **in-memory state management** (job store, trace store), **missing rate
limiting on the query endpoint**, and **CORS/MIME validation gaps** that must be
closed before any internet-facing deployment. No CRITICAL findings were
identified. All HIGH findings are addressable with small, targeted code changes.

---

## Findings

---

### FINDING-001: CORS Wildcard Risk

**Severity:** HIGH
**Location:** `backend/config.py` · `backend/main.py` (CORS middleware setup)
**Issue:**
If `CORS_ORIGINS` is accidentally set to `["*"]` (e.g., left from a dev
shortcut, or if the `.env` file is missing), the FastAPI CORS middleware allows
any origin to send credentialed cross-site requests. This nullifies the
Same-Origin Policy and enables CSRF-class attacks against authenticated users.

**Fix:**

```python
# backend/config.py
from pydantic_settings import BaseSettings
from pydantic import field_validator

class Settings(BaseSettings):
    cors_origins: list[str] = ["http://localhost:5173"]

    @field_validator("cors_origins")
    @classmethod
    def no_wildcard_in_production(cls, v: list[str]) -> list[str]:
        import os
        if os.getenv("APP_ENV", "development") == "production" and "*" in v:
            raise ValueError(
                "CORS wildcard '*' is forbidden in production. "
                "Set CORS_ORIGINS to your specific frontend domain."
            )
        return v
```

---

### FINDING-002: Magic Bytes Validation Missing on File Upload

**Severity:** HIGH
**Location:** `backend/api/routes.py` — `POST /upload` handler
**Issue:**
The current upload handler checks only that the filename ends with `.pdf`.
An attacker can rename any file (HTML, JavaScript, executable) to `exploit.pdf`
and upload it. The file is then parsed by PyMuPDF, which may be exploited via
malformed document attacks (CVE-2023-44xxxx class).

**Fix:**

```python
# backend/api/routes.py
import magic  # python-magic (libmagic binding)

ALLOWED_MIME = {"application/pdf"}
PDF_MAGIC_BYTES = b"%PDF-"

async def upload(file: UploadFile = File(...)):
    data = await file.read()

    # 1. Check magic bytes prefix (fast, zero-dep)
    if not data.startswith(PDF_MAGIC_BYTES):
        raise HTTPException(status_code=400, detail="File is not a valid PDF.")

    # 2. Deep MIME type check via libmagic (catches renamed files)
    detected_mime = magic.from_buffer(data[:2048], mime=True)
    if detected_mime not in ALLOWED_MIME:
        raise HTTPException(
            status_code=400,
            detail=f"Detected MIME type '{detected_mime}' is not allowed.",
        )
```

> [!IMPORTANT]
> `python-magic` requires `libmagic1` installed in the Docker image.
> It is already present in the provided `Dockerfile` (`apt-get install libmagic1`).

---

### FINDING-003: Cypher Dynamic Label in neo4j_writer.py

**Severity:** MEDIUM
**Location:** `backend/infra/neo4j_writer.py` — node creation Cypher
**Issue:**
Entity labels (e.g., `Concept`, `Method`, `Dataset`) are interpolated directly
into Cypher strings:

```python
# Potentially dangerous pattern:
cypher = f"CREATE (n:{node.label} {{name: $name}})"
```

If `node.label` is user-supplied without validation, this is a Cypher injection
vector that could allow arbitrary graph mutations (e.g., label `:ADMIN`).

**Current Mitigation:**
`NodeModel.label` is validated by Pydantic against a `Literal` type constraint in
the extractor. The LLM output is JSON-parsed and model-validated before writing,
so arbitrary labels cannot reach the writer under normal flow.

**Recommended Hardening:**

```python
# backend/infra/neo4j_writer.py
ALLOWED_LABELS = frozenset({
    "Concept", "Method", "Dataset", "Model", "Algorithm",
    "Author", "Institution", "Result", "Metric", "Unknown",
})

def _safe_label(label: str) -> str:
    if label not in ALLOWED_LABELS:
        raise ValueError(f"Disallowed node label: '{label}'")
    return label

# Usage:
cypher = f"MERGE (n:{_safe_label(node.label)} {{entity_id: $entity_id}})"
```

---

### FINDING-004: Job Store is In-Memory — Data Lost on Restart

**Severity:** MEDIUM
**Location:** `backend/api/routes.py` — `_JOB_STORE: dict`
**Issue:**
The job store is a plain Python `dict` held in application memory. Any process
restart, crash, or scale-out event causes all job statuses to disappear.
Clients polling `/status/{job_id}` receive 404 after restart.
Duplicate uploads cannot be detected across restarts.

**Fix:** Replace with Redis:

```python
# backend/infra/job_store.py
import json
import redis.asyncio as aioredis
from backend.config import get_settings

_redis: aioredis.Redis | None = None

async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(get_settings().redis_url, decode_responses=True)
    return _redis

async def set_job(job_id: str, data: dict, ttl: int = 86400) -> None:
    r = await get_redis()
    await r.setex(f"job:{job_id}", ttl, json.dumps(data))

async def get_job(job_id: str) -> dict | None:
    r = await get_redis()
    raw = await r.get(f"job:{job_id}")
    return json.loads(raw) if raw else None
```

Add `REDIS_URL=redis://localhost:6379/0` to `.env.example` and
`redis[asyncio]>=5.0.0` to `pyproject.toml`.

---

### FINDING-005: Trace Store In-Memory — Query Traces Lost on Restart

**Severity:** HIGH
**Location:** `backend/api/routes.py` — `_TRACE_STORE: dict`
**Issue:**
Query execution traces (used for debugging and explainability via `GET /trace/{id}`)
are stored in a plain `dict`. They are lost on any restart, making post-mortem
debugging impossible in production.

**Fix:** Store traces in Redis with a 7-day TTL:

```python
async def save_trace(query_id: str, trace: dict, ttl: int = 604800) -> None:
    r = await get_redis()
    await r.setex(f"trace:{query_id}", ttl, json.dumps(trace))

async def get_trace(query_id: str) -> dict | None:
    r = await get_redis()
    raw = await r.get(f"trace:{query_id}")
    return json.loads(raw) if raw else None
```

---

### FINDING-006: Pinecone top_k Unbounded in Direct API Calls

**Severity:** MEDIUM
**Location:** `backend/agents/vector_agent.py` — `VectorAgent.query()`
**Issue:**
A caller-controlled `top_k` without a cap allows requesting `top_k=10000`,
causing a very large Pinecone payload that saturates backend memory and
response time.

**Fix:**

```python
MAX_TOP_K = 20

async def query(self, text: str, paper_id: str, top_k: int = 5) -> VectorResult:
    top_k = min(top_k, MAX_TOP_K)  # Hard cap; never trust caller value
    ...
```

And in the Pydantic request model:

```python
class QueryRequest(BaseModel):
    paper_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
```

---

### FINDING-007: Neo4j Traversal Depth — Confirm Cap at 3

**Severity:** LOW
**Location:** `backend/agents/graph_agent.py` — `_traverse()` Cypher
**Issue:**
Graph traversals with unbounded depth (`[*]`) cause exponential path explosion
in dense graphs, leading to query timeouts and memory exhaustion in Neo4j.

**Current status:** The traversal query correctly uses `[*1..3]`. This must be
maintained as a named constant.

**Fix:**

```python
MAX_TRAVERSAL_DEPTH = 3  # do NOT increase without load testing

TRAVERSE_CYPHER = (
    f"MATCH (n)-[r*1..{MAX_TRAVERSAL_DEPTH}]-(m) "
    "WHERE n.entity_id IN $ids "
    "AND n.paper_id = $paper_id "
    "RETURN n, r, m LIMIT 50"
)
```

> [!NOTE]
> Neo4j does not accept relationship depth as a bound query parameter.
> String-format only the integer constant (never user input).

---

### FINDING-008: embed_text Called Per-Chunk vs embed_batch During Ingestion

**Severity:** MEDIUM
**Location:** `backend/agents/vector_agent.py` / ingestion pipeline
**Issue:**
Calling `SentenceTransformer.encode()` once per chunk during ingestion
(200+ chunks for a 50-page paper) inflates ingestion time by ~10x compared
to a single batched call.

**Fix:**

```python
# backend/ingestion/embedder.py
def embed_text(text: str) -> list[float]:
    """Single-text embed — for query-time use only."""
    return get_model().encode(text, normalize_embeddings=True).tolist()

def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Batched embed — MUST be used during PDF ingestion."""
    return get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
```

The ingestion pipeline must call `embed_batch(chunk_texts)` not `embed_text()`.

---

### FINDING-009: Missing Rate Limiting on /query and /upload Endpoints

**Severity:** HIGH
**Location:** `backend/api/routes.py` — `POST /query`, `POST /upload`
**Issue:**
The `/query` endpoint triggers three concurrent external calls (Pinecone + Neo4j)
plus one Mistral LLM synthesis call per request. Without rate limiting a single
client can saturate the Mistral API quota, exhaust Neo4j connection pool slots,
and cause cascading 500 errors.

**Fix:** Apply `slowapi` limiter (already in `pyproject.toml`):

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# In main.py:
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

# Decorate endpoints:
@router.post("/query")
@limiter.limit("10/minute")
async def query(request: Request, body: QueryRequest): ...

@router.post("/upload")
@limiter.limit("5/minute")
async def upload(request: Request, file: UploadFile = File(...)): ...
```

---

### FINDING-010: Missing Request ID Correlation for Distributed Tracing

**Severity:** LOW
**Location:** `backend/main.py` · all route handlers
**Issue:**
Logs from the PDF parser, domain detector, entity extractor, and three agents
carry no shared `request_id`, making it impossible to correlate all log lines
for a single request in multi-worker or multi-replica deployments.

**Fix:**

```python
# backend/middleware/request_id.py
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        with structlog.contextvars.bound_contextvars(request_id=request_id):
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

# backend/main.py
from backend.middleware.request_id import RequestIdMiddleware
app.add_middleware(RequestIdMiddleware)
```

---

## Deployment Checklist

> [!CAUTION]
> Do not expose this service to the internet until every HIGH item is resolved
> and every checklist item below is verified.

- [ ] **Set all required `.env` variables** — never leave placeholder values (`your_*_here`)
- [ ] **Run Neo4j schema init** — execute `infra/neo4j/init.cypher` before first launch
- [ ] **Confirm Neo4j persistent volume** — verify `neo4j_data` Docker volume survives restarts
- [ ] **Set `CORS_ORIGINS` to specific frontend domain** — never `["*"]` in production (FINDING-001)
- [ ] **Rotate all API keys** — generate fresh Mistral + Pinecone keys; never reuse `.env.example` values
- [ ] **Verify Pinecone index exists** — create with `dimension=384`, `metric=cosine` before first upload
- [ ] **Test `/health` endpoint** — all upstream services must report `"ok"` before opening to traffic
- [ ] **Run pytest test suite** — `pytest tests/ -v` — all tests must pass
- [ ] **Build frontend production bundle** — `npm run build` in `frontend/`, verify `dist/` is non-empty
- [ ] **Set `uvicorn --workers 4`** — already in `Dockerfile` CMD; tune to 2× vCPU count
- [ ] **Configure log aggregation** — route structlog JSON output to Fluent Bit / Filebeat / external service
- [ ] **Set up Neo4j backup schedule** — automated daily backup of the `neo4j_data` volume
- [ ] **Add API authentication (JWT)** — all routes must require valid JWT before internet exposure
- [ ] **Enable HTTPS via reverse proxy** — deploy nginx or Traefik in front; never serve plain HTTP in production
- [ ] **Set `MAX_UPLOAD_SIZE_MB`** — calibrate to server RAM (3× max PDF size as minimum headroom)
- [ ] **Implement magic-bytes MIME validation** — resolve FINDING-002 before accepting uploads from untrusted clients
- [ ] **Deploy Redis** — replace in-memory job store and trace store before scaling past 1 replica (FINDING-004, FINDING-005)
- [ ] **Apply rate limiting** — configure `slowapi` on `/query` and `/upload` (FINDING-009)
- [ ] **Verify traversal depth cap** — confirm `MAX_TRAVERSAL_DEPTH = 3` is used everywhere in `graph_agent.py` (FINDING-007)
- [ ] **Enable Neo4j query logging** — set `dbms.logs.query.enabled=true` to audit slow Cypher queries

---

*End of Audit Report*
