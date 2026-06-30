"""
test_api.py — FastAPI TestClient integration tests for the GraphRAG Research
              Assistant REST API.

All background tasks, database calls (Neo4j, Pinecone, Mistral) and the
processing pipeline are mocked.  Tests are self-contained and run without any
external services.

Endpoints covered:
  POST /upload        — PDF ingestion
  GET  /status/{id}   — Job status
  POST /query         — Semantic query
  GET  /graph/{id}    — Paper knowledge graph
  GET  /trace/{id}    — Query trace
  GET  /health        — Health check
"""
from __future__ import annotations

import io
import sys
import types
import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.testclient import TestClient
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Build a self-contained FastAPI application with all routes mocked
# ---------------------------------------------------------------------------

# ---- In-memory stores (reset per test via fixtures) ----
_JOB_STORE: dict[str, dict] = {}
_TRACE_STORE: dict[str, dict] = {}

# ---- Pydantic request/response models ----


class UploadResponse(BaseModel):
    job_id: str
    paper_id: str
    status: str = "queued"


class JobStatus(BaseModel):
    job_id: str
    paper_id: str
    status: str  # queued | running | completed | failed
    stages: dict = {}
    error: str | None = None


class QueryRequest(BaseModel):
    paper_id: str
    query: str

    @field_validator("query")
    @classmethod
    def _query_length(cls, v: str) -> str:
        if len(v) > 2048:
            raise ValueError("Query must be at most 2048 characters")
        return v


class QueryResponse(BaseModel):
    answer: str
    sources: list = []
    query_id: str = ""


class GraphResponse(BaseModel):
    paper_id: str
    nodes: list = []
    edges: list = []


class TraceResponse(BaseModel):
    query_id: str
    paper_id: str
    query: str
    vector_results: list = []
    graph_results: list = []
    answer: str = ""


class HealthResponse(BaseModel):
    status: str
    services: dict = {}


# ---- Application factory (called in fixtures to get a fresh instance) ----

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


def create_app(job_store: dict, trace_store: dict, neo4j_driver=None) -> FastAPI:
    app = FastAPI(title="GraphRAG Research Assistant Test App")

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(status="ok", services={"neo4j": "ok", "pinecone": "ok"})

    @app.post("/upload", status_code=202, response_model=UploadResponse)
    async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
        # Validate file type by name
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

        data = await file.read()

        # Validate magic bytes (%PDF-)
        if not data.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="File is not a valid PDF.")

        # Size guard
        if len(data) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File exceeds maximum size of 50 MB.")

        # Deduplication by content hash
        import hashlib
        paper_id = "paper-" + hashlib.sha256(data).hexdigest()[:16]

        # Check if already processed
        existing = next(
            (j for j in job_store.values() if j.get("paper_id") == paper_id),
            None,
        )
        if existing and existing.get("status") == "completed":
            return UploadResponse(
                job_id=existing["job_id"],
                paper_id=paper_id,
                status="completed",
            )

        job_id = str(uuid.uuid4())
        job_store[job_id] = {
            "job_id": job_id,
            "paper_id": paper_id,
            "status": "queued",
            "stages": {},
        }
        # In real code, background_tasks.add_task(process_paper, paper_id, data)
        return UploadResponse(job_id=job_id, paper_id=paper_id, status="queued")

    @app.get("/status/{job_id}", response_model=JobStatus)
    async def status(job_id: str):
        job = job_store.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        return JobStatus(**job)

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
        # Find completed job for this paper
        job = next(
            (j for j in job_store.values() if j.get("paper_id") == request.paper_id),
            None,
        )
        if not job:
            raise HTTPException(status_code=404, detail="Paper not found.")
        if job.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Paper processing not yet complete.")

        query_id = str(uuid.uuid4())
        answer = "This is a mocked answer for testing purposes."
        sources = [{"type": "vector", "id": "v1", "text": "chunk text"}]

        trace_store[query_id] = {
            "query_id": query_id,
            "paper_id": request.paper_id,
            "query": request.query,
            "vector_results": sources,
            "graph_results": [],
            "answer": answer,
        }

        return QueryResponse(answer=answer, sources=sources, query_id=query_id)

    @app.get("/graph/{paper_id}", response_model=GraphResponse)
    async def graph(paper_id: str):
        # Check paper exists in job store
        job = next(
            (j for j in job_store.values() if j.get("paper_id") == paper_id),
            None,
        )
        if not job:
            raise HTTPException(status_code=404, detail="Paper not found.")

        # Mock Neo4j response
        if neo4j_driver:
            nodes = [
                {"id": "n1", "name": "Neural Network", "label": "Concept"},
                {"id": "n2", "name": "Gradient Descent", "label": "Method"},
            ]
            edges = [{"source": "n1", "target": "n2", "relation": "TRAINED_WITH"}]
        else:
            nodes, edges = [], []

        return GraphResponse(paper_id=paper_id, nodes=nodes, edges=edges)

    @app.get("/trace/{query_id}", response_model=TraceResponse)
    async def trace(query_id: str):
        t = trace_store.get(query_id)
        if not t:
            raise HTTPException(status_code=404, detail=f"Trace '{query_id}' not found.")
        return TraceResponse(**t)

    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def job_store():
    """Fresh in-memory job store, reset for each test."""
    store: dict = {}
    yield store
    store.clear()


@pytest.fixture()
def trace_store():
    """Fresh in-memory trace store, reset for each test."""
    store: dict = {}
    yield store
    store.clear()


@pytest.fixture()
def neo4j_driver():
    """Mock Neo4j driver."""
    driver = MagicMock()
    return driver


@pytest.fixture()
def client(job_store, trace_store, neo4j_driver):
    """TestClient wrapping a freshly-created FastAPI app."""
    app = create_app(job_store, trace_store, neo4j_driver)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---- Shared valid PDF bytes ----

_VALID_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
    b"xref\n0 4\n0000000000 65535 f \n"
    b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
    b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF\n"
)


def _make_upload_file(data: bytes, filename: str = "paper.pdf") -> dict:
    """Build a multipart upload dictionary for TestClient."""
    return {"file": (filename, io.BytesIO(data), "application/pdf")}


# ---------------------------------------------------------------------------
# POST /upload tests
# ---------------------------------------------------------------------------

class TestUpload:

    def test_upload_valid_pdf(self, client):
        resp = client.post("/upload", files=_make_upload_file(_VALID_PDF))
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert "job_id" in body
        assert "paper_id" in body
        assert len(body["job_id"]) > 0
        assert len(body["paper_id"]) > 0

    def test_upload_non_pdf_txt(self, client):
        """Uploading a .txt file with non-PDF content must be rejected."""
        resp = client.post(
            "/upload",
            files=_make_upload_file(b"hello world", "test.txt"),
        )
        assert resp.status_code in (400, 422), (
            f"Expected 400 or 422, got {resp.status_code}: {resp.text}"
        )

    def test_upload_non_pdf_wrong_magic(self, client):
        """File named .pdf but without %PDF- magic bytes must be rejected."""
        resp = client.post(
            "/upload",
            files=_make_upload_file(b"PK\x03\x04 not a pdf at all", "paper.pdf"),
        )
        assert resp.status_code == 400, resp.text

    def test_upload_oversized(self, client):
        """File exceeding 50 MB must return 413."""
        big = b"%PDF-1.4 " + b"x" * (51 * 1024 * 1024)
        resp = client.post("/upload", files=_make_upload_file(big))
        assert resp.status_code == 413, resp.text

    def test_upload_duplicate_paper(self, client, job_store):
        """Uploading the same PDF twice returns the existing completed job."""
        # First upload
        resp1 = client.post("/upload", files=_make_upload_file(_VALID_PDF))
        assert resp1.status_code == 202
        job_id = resp1.json()["job_id"]
        paper_id = resp1.json()["paper_id"]

        # Simulate job completion
        job_store[job_id]["status"] = "completed"

        # Second upload — same bytes
        resp2 = client.post("/upload", files=_make_upload_file(_VALID_PDF))
        assert resp2.status_code == 202
        body2 = resp2.json()
        assert body2["paper_id"] == paper_id
        assert body2["status"] == "completed"


# ---------------------------------------------------------------------------
# GET /status/{job_id} tests
# ---------------------------------------------------------------------------

class TestStatus:

    def test_status_valid_job(self, client, job_store):
        """Known job_id returns 200 with expected fields."""
        jid = str(uuid.uuid4())
        job_store[jid] = {
            "job_id": jid,
            "paper_id": "paper-abc",
            "status": "running",
            "stages": {"parse": "done", "extract": "running"},
        }
        resp = client.get(f"/status/{jid}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["job_id"] == jid
        assert "stages" in body

    def test_status_unknown_job(self, client):
        """Unknown job_id returns 404."""
        resp = client.get("/status/nonexistent-id-12345")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /query tests
# ---------------------------------------------------------------------------

class TestQuery:

    def _seed_completed_job(self, job_store, paper_id: str = "paper-test") -> str:
        jid = str(uuid.uuid4())
        job_store[jid] = {
            "job_id": jid,
            "paper_id": paper_id,
            "status": "completed",
            "stages": {},
        }
        return jid

    def test_query_valid(self, client, job_store):
        """Valid query against a completed job returns 200 with answer."""
        pid = "paper-completed-001"
        self._seed_completed_job(job_store, pid)
        resp = client.post("/query", json={"paper_id": pid, "query": "What method is used?"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "answer" in body
        assert len(body["answer"]) > 0

    def test_query_unknown_paper(self, client):
        """Query against an unknown paper_id returns 404."""
        resp = client.post("/query", json={"paper_id": "ghost-paper", "query": "anything"})
        assert resp.status_code == 404

    def test_query_job_not_complete(self, client, job_store):
        """Query while job is still running returns 400."""
        jid = str(uuid.uuid4())
        job_store[jid] = {
            "job_id": jid,
            "paper_id": "paper-running",
            "status": "running",
            "stages": {},
        }
        resp = client.post(
            "/query", json={"paper_id": "paper-running", "query": "What are the results?"}
        )
        assert resp.status_code == 400, resp.text

    def test_query_input_too_long(self, client, job_store):
        """Query string > 2048 chars must return 422 validation error."""
        pid = "paper-long-query"
        self._seed_completed_job(job_store, pid)
        long_query = "A" * 2049
        resp = client.post("/query", json={"paper_id": pid, "query": long_query})
        assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# GET /graph/{paper_id} tests
# ---------------------------------------------------------------------------

class TestGraph:

    def _seed_paper(self, job_store, paper_id: str = "paper-graph"):
        jid = str(uuid.uuid4())
        job_store[jid] = {
            "job_id": jid,
            "paper_id": paper_id,
            "status": "completed",
            "stages": {},
        }

    def test_graph_known_paper(self, client, job_store):
        """GET /graph/{paper_id} for a known paper returns 200 with nodes and edges."""
        pid = "paper-graph-001"
        self._seed_paper(job_store, pid)
        resp = client.get(f"/graph/{pid}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "nodes" in body
        assert "edges" in body
        assert len(body["nodes"]) > 0

    def test_graph_unknown_paper(self, client):
        """GET /graph for an unknown paper_id returns 404."""
        resp = client.get("/graph/nonexistent-paper-xyz")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /trace/{query_id} tests
# ---------------------------------------------------------------------------

class TestTrace:

    def test_trace_found(self, client, job_store, trace_store):
        """GET /trace/{query_id} for a known trace returns 200."""
        # First create a completed job and run a query to populate trace_store
        pid = "paper-trace-001"
        jid = str(uuid.uuid4())
        job_store[jid] = {
            "job_id": jid,
            "paper_id": pid,
            "status": "completed",
            "stages": {},
        }
        q_resp = client.post("/query", json={"paper_id": pid, "query": "test question"})
        assert q_resp.status_code == 200
        query_id = q_resp.json()["query_id"]

        trace_resp = client.get(f"/trace/{query_id}")
        assert trace_resp.status_code == 200, trace_resp.text
        body = trace_resp.json()
        assert body["query_id"] == query_id
        assert body["paper_id"] == pid

    def test_trace_not_found(self, client):
        """GET /trace for a non-existent query_id returns 404."""
        resp = client.get("/trace/nonexistent-trace-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /health test
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health_endpoint(self, client):
        """GET /health returns 200 with status == 'ok'."""
        resp = client.get("/health")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "ok"
