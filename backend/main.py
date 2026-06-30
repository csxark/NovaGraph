"""
FastAPI application entry-point for the GraphRAG Research Assistant.

Provides endpoints for PDF ingestion, job-status polling, knowledge-graph
querying, graph visualisation, and system health checks.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.agents.synthesizer import QueryResponse, run_pipeline
from backend.config import Settings, get_settings
from backend.extractor.domain_detector import detect_domain
from backend.extractor.entity_extractor import extract_all_chunks
from backend.graph.neo4j_queries import get_subgraph, paper_exists
from backend.graph.neo4j_client import get_driver
from backend.graph.schema_init import initialize_schema
from backend.graph.neo4j_writer import (
    batch_write_edges,
    batch_write_nodes,
)
from backend.vector.embedder import embed_nodes_batch, verify_hf_api, close_client
from backend.vector.pinecone_store import get_index, upsert_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory stores (module-level singletons)
# ---------------------------------------------------------------------------

job_store: dict[str, dict] = {}      # job_id -> job state dict
trace_store: dict[str, dict] = {}    # query_id -> trace dict


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class JobStage(BaseModel):
    """Progress information for a single ingestion pipeline stage."""

    stage: str
    status: str                       # 'pending' | 'running' | 'completed' | 'failed'
    duration_ms: Optional[int] = None
    error: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Full status snapshot of an ingestion job."""

    job_id: str
    paper_id: str
    status: str                       # 'pending' | 'processing' | 'completed' | 'failed'
    stages: list[JobStage]
    created_at: int                   # Unix epoch ms
    updated_at: int
    error: Optional[str] = None


class UploadResponse(BaseModel):
    """Immediate response to a successful /upload request."""

    job_id: str
    paper_id: str
    status: str
    message: str


class QueryRequest(BaseModel):
    """Body of a POST /query request."""

    query: str = Field(..., min_length=3, max_length=2048)
    paper_id: str
    top_k: int = Field(default=5, ge=1, le=20)
    include_trace: bool = False


class QueryResponseModel(BaseModel):
    """API response wrapping the pipeline QueryResponse."""

    answer: str
    sources: list[dict] = Field(default_factory=list)
    trace: dict = Field(default_factory=dict)
    query_id: str
    total_nodes_retrieved: int = 0
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error body."""

    detail: str
    error_type: str


# ---------------------------------------------------------------------------
# Ingestion stage helpers
# ---------------------------------------------------------------------------

_INGESTION_STAGES: list[str] = [
    "pdf_parse",
    "domain_detection",
    "entity_extraction",
    "embedding",
    "neo4j_write",
    "pinecone_upsert",
]


def _init_job(job_id: str, paper_id: str) -> dict:
    """Create and store a fresh job entry in job_store."""
    now = int(time.time() * 1000)
    job: dict = {
        "job_id": job_id,
        "paper_id": paper_id,
        "status": "pending",
        "stages": {s: JobStage(stage=s, status="pending") for s in _INGESTION_STAGES},
        "created_at": now,
        "updated_at": now,
        "error": None,
    }
    job_store[job_id] = job
    return job


def _stage_start(job_id: str, stage: str) -> int:
    """Mark a stage as running; return its start time (epoch ms)."""
    t = int(time.time() * 1000)
    job_store[job_id]["stages"][stage] = JobStage(stage=stage, status="running")
    job_store[job_id]["status"] = "processing"
    job_store[job_id]["updated_at"] = t
    return t


def _stage_done(job_id: str, stage: str, start_ms: int) -> None:
    """Mark a stage as completed with its duration."""
    now = int(time.time() * 1000)
    job_store[job_id]["stages"][stage] = JobStage(
        stage=stage,
        status="completed",
        duration_ms=now - start_ms,
    )
    job_store[job_id]["updated_at"] = now


def _stage_fail(job_id: str, stage: str, start_ms: int, error: str) -> None:
    """Mark a stage as failed."""
    now = int(time.time() * 1000)
    job_store[job_id]["stages"][stage] = JobStage(
        stage=stage,
        status="failed",
        duration_ms=now - start_ms,
        error=error,
    )
    job_store[job_id]["status"] = "failed"
    job_store[job_id]["error"] = error
    job_store[job_id]["updated_at"] = now


def _job_to_response(job_id: str) -> JobStatusResponse:
    """Convert the raw job dict to a :class:`JobStatusResponse`."""
    job = job_store[job_id]
    return JobStatusResponse(
        job_id=job["job_id"],
        paper_id=job["paper_id"],
        status=job["status"],
        stages=list(job["stages"].values()),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        error=job.get("error"),
    )


# ---------------------------------------------------------------------------
# Background ingestion pipeline
# ---------------------------------------------------------------------------


async def run_ingestion(
    job_id: str,
    paper_id: str,
    file_bytes: bytes,
    filename: str,
    settings: Settings,
) -> None:
    """Run the full PDF-to-graph ingestion pipeline.

    Stages:
        pdf_parse → domain_detection → entity_extraction →
        embedding → neo4j_write → pinecone_upsert
    """
    from backend.graph.neo4j_client import get_driver
    driver = await get_driver(settings)

    # ------------------------------------------------------------------
    # Stage 1: PDF parse
    # ------------------------------------------------------------------
    t = _stage_start(job_id, "pdf_parse")
    try:
        from backend.parser.pdf_parser import parse_pdf
        parsed = parse_pdf(file_bytes)
        _stage_done(job_id, "pdf_parse", t)
    except Exception as exc:
        _stage_fail(job_id, "pdf_parse", t, str(exc))
        return

    # ------------------------------------------------------------------
    # Stage 2: Domain detection
    # ------------------------------------------------------------------
    t = _stage_start(job_id, "domain_detection")
    try:
        domain_result = await detect_domain(parsed.full_text, settings)
        _stage_done(job_id, "domain_detection", t)
    except Exception as exc:
        _stage_fail(job_id, "domain_detection", t, str(exc))
        return

    # ------------------------------------------------------------------
    # Stage 3: Entity extraction
    # ------------------------------------------------------------------
    t = _stage_start(job_id, "entity_extraction")
    try:
        from backend.parser.pdf_parser import get_chunks
        chunks = get_chunks(parsed)
        graph_schema = await extract_all_chunks(
            chunks=chunks,
            domain_result=domain_result,
            settings=settings,
        )
        _stage_done(job_id, "entity_extraction", t)
    except Exception as exc:
        _stage_fail(job_id, "entity_extraction", t, str(exc))
        return

    # ------------------------------------------------------------------
    # Stage 4: Embedding
    # ------------------------------------------------------------------
    t = _stage_start(job_id, "embedding")
    try:
        embeddings = await embed_nodes_batch(
            nodes=graph_schema.entities,
            settings=settings,
        )
        _stage_done(job_id, "embedding", t)
    except Exception as exc:
        _stage_fail(job_id, "embedding", t, str(exc))
        return

    # ------------------------------------------------------------------
    # Stage 5: Neo4j write
    # ------------------------------------------------------------------
    t = _stage_start(job_id, "neo4j_write")
    try:
        await batch_write_nodes(
            driver=driver,
            nodes=graph_schema.entities,
            paper_id=paper_id,
        )
        name_to_id = {node.name: node.entity_id for node in graph_schema.entities}
        await batch_write_edges(
            driver=driver,
            edges=graph_schema.relationships,
            name_to_id=name_to_id,
            paper_id=paper_id,
        )
        _stage_done(job_id, "neo4j_write", t)
    except Exception as exc:
        _stage_fail(job_id, "neo4j_write", t, str(exc))
        return

    # ------------------------------------------------------------------
    # Stage 6: Pinecone upsert
    # ------------------------------------------------------------------
    t = _stage_start(job_id, "pinecone_upsert")
    try:
        upsert_batch(
            nodes=graph_schema.entities,
            embeddings=embeddings,
            paper_id=paper_id,
            settings=settings,
        )
        _stage_done(job_id, "pinecone_upsert", t)
    except Exception as exc:
        _stage_fail(job_id, "pinecone_upsert", t, str(exc))
        return

    # ------------------------------------------------------------------
    # All stages succeeded
    # ------------------------------------------------------------------
    job_store[job_id]["status"] = "completed"
    job_store[job_id]["updated_at"] = int(time.time() * 1000)
    logger.info("Ingestion completed: job_id=%s paper_id=%s", job_id, paper_id)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: verify Neo4j, initialise schema, warm up embedding model.
    Shutdown: close the Neo4j driver.
    """
    settings = get_settings()
    logger.info("Starting GraphRAG Research Assistant…")

    # Verify Neo4j connectivity
    try:
        driver = await get_driver(settings)
        await driver.verify_connectivity()
        await initialize_schema(driver)
        logger.info("Neo4j connected and schema initialised.")
        app.state.driver = driver
    except Exception as exc:
        logger.error("Neo4j startup failure: %s", exc)
        app.state.driver = None

    # Verify HuggingFace Inference API connectivity
    try:
        hf_ok = await verify_hf_api(settings)
        app.state.embedding_ready = hf_ok
        if hf_ok:
            logger.info("HuggingFace Inference API verified — cloud embeddings ready.")
        else:
            logger.warning("HuggingFace Inference API probe returned unexpected results.")
    except Exception as exc:
        logger.error("HuggingFace Inference API verification failure: %s", exc)
        app.state.embedding_ready = False

    yield

    # Shutdown
    try:
        if app.state.driver:
            await app.state.driver.close()
            logger.info("Neo4j driver closed.")
    except Exception as exc:
        logger.warning("Error closing Neo4j driver: %s", exc)

    # Close the HuggingFace HTTP client
    try:
        await close_client()
        logger.info("HuggingFace HTTP client closed.")
    except Exception as exc:
        logger.warning("Error closing HuggingFace HTTP client: %s", exc)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

settings = get_settings()

app = FastAPI(
    title="GraphRAG Research Assistant",
    version="1.0.0",
    description="LLM-powered knowledge-graph retrieval over academic PDFs.",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail=str(exc),
            error_type=type(exc).__name__,
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="System health check")
async def health() -> dict:
    """Return liveness and readiness information for all external services."""
    settings = get_settings()

    # Neo4j
    neo4j_ok = False
    try:
        driver = await get_driver(settings)
        await driver.verify_connectivity()
        neo4j_ok = True
    except Exception:
        pass

    # Pinecone — attempt to describe index stats
    pinecone_ok = False
    try:
        index = get_index(settings)
        index.describe_index_stats()
        pinecone_ok = True
    except Exception:
        pass

    # Embedding model
    embedding_ok = getattr(app.state, "embedding_ready", False)

    return {
        "status": "ok",
        "neo4j": neo4j_ok,
        "pinecone": pinecone_ok,
        "embedding_model": embedding_ok,
    }


@app.post(
    "/upload",
    response_model=UploadResponse,
    status_code=202,
    summary="Upload a PDF for ingestion",
)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> UploadResponse:
    """Accept a PDF, validate it, and kick off a background ingestion job.

    Returns HTTP 202 immediately with a ``job_id`` that can be polled via
    ``GET /status/{job_id}``.
    """
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    # Content-type / extension validation
    content_type = file.content_type or ""
    filename = file.filename or "upload.pdf"
    if content_type != "application/pdf" and not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are accepted (application/pdf).",
        )

    # Read and size-check
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {len(content) / 1_048_576:.1f} MB "
                f"(max {settings.max_upload_size_mb} MB)."
            ),
        )

    # Deterministic paper ID
    paper_id = hashlib.sha256(content).hexdigest()

    # Check if paper already ingested
    try:
        driver = await get_driver(settings)
        already_exists = await paper_exists(driver=driver, paper_id=paper_id)
    except Exception:
        already_exists = False

    if already_exists:
        # Return a synthetic completed job
        synthetic_job_id = f"existing-{paper_id[:8]}"
        if synthetic_job_id not in job_store:
            now = int(time.time() * 1000)
            job_store[synthetic_job_id] = {
                "job_id": synthetic_job_id,
                "paper_id": paper_id,
                "status": "completed",
                "stages": {
                    s: JobStage(stage=s, status="completed")
                    for s in _INGESTION_STAGES
                },
                "created_at": now,
                "updated_at": now,
                "error": None,
            }
        return UploadResponse(
            job_id=synthetic_job_id,
            paper_id=paper_id,
            status="completed",
            message="Paper already ingested. Use the paper_id to query.",
        )

    # Create fresh job
    job_id = str(uuid.uuid4())
    _init_job(job_id, paper_id)

    # Schedule background ingestion
    background_tasks.add_task(
        run_ingestion,
        job_id=job_id,
        paper_id=paper_id,
        file_bytes=content,
        filename=filename,
        settings=settings,
    )

    logger.info(
        "Upload accepted: job_id=%s paper_id=%s filename=%r size=%d bytes",
        job_id,
        paper_id,
        filename,
        len(content),
    )
    return UploadResponse(
        job_id=job_id,
        paper_id=paper_id,
        status="pending",
        message="Ingestion started. Poll GET /status/{job_id} for progress.",
    )


@app.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll ingestion job status",
)
async def get_status(job_id: str) -> JobStatusResponse:
    """Return the current status and per-stage breakdown for an ingestion job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _job_to_response(job_id)


@app.post(
    "/query",
    response_model=QueryResponseModel,
    summary="Query the knowledge graph for a paper",
)
async def query_paper(body: QueryRequest) -> QueryResponseModel:
    """Run the full RAG pipeline against the knowledge graph for *paper_id*.

    The pipeline executes vector search, graph traversal, and entity resolution
    in parallel, then synthesizes an answer with Mistral large.
    """
    settings = get_settings()
    driver = await get_driver(settings)

    # Validate paper exists
    try:
        exists = await paper_exists(driver=driver, paper_id=body.paper_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {exc}") from exc

    if not exists:
        raise HTTPException(
            status_code=404,
            detail=f"Paper '{body.paper_id}' not found. Please upload it first.",
        )

    # Check that the job (if tracked) has completed
    # Find any job for this paper_id
    paper_jobs = [
        j for j in job_store.values() if j["paper_id"] == body.paper_id
    ]
    if paper_jobs:
        latest = max(paper_jobs, key=lambda j: j["created_at"])
        if latest["status"] in ("pending", "processing"):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Paper is still being processed (status={latest['status']}). "
                    "Please wait until ingestion is completed."
                ),
            )

    # Run the full pipeline
    try:
        result: QueryResponse = await run_pipeline(
            query=body.query,
            paper_id=body.paper_id,
            driver=driver,
            settings=settings,
            include_trace=body.include_trace,
        )
    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    # Persist trace if requested
    if body.include_trace and result.trace:
        trace_store[result.query_id] = result.trace

    return QueryResponseModel(
        answer=result.answer,
        sources=[s.model_dump() for s in result.sources],
        trace=result.trace,
        query_id=result.query_id,
        total_nodes_retrieved=result.total_nodes_retrieved,
        error=result.error,
    )


@app.get(
    "/graph/{paper_id}",
    summary="Retrieve the full knowledge-graph subgraph for a paper",
)
async def get_graph(paper_id: str) -> dict:
    """Return up to 200 nodes and their edges for visualisation."""
    settings = get_settings()
    driver = await get_driver(settings)
 
    try:
        exists = await paper_exists(driver=driver, paper_id=paper_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {exc}") from exc
 
    if not exists:
        raise HTTPException(
            status_code=404,
            detail=f"Paper '{paper_id}' not found.",
        )
 
    try:
        subgraph: dict = await get_subgraph(
            driver=driver,
            paper_id=paper_id,
            max_nodes=200,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph query error: {exc}") from exc

    nodes: list = subgraph.get("nodes", [])
    edges: list = subgraph.get("edges", [])
    truncated: bool = subgraph.get("truncated", False)

    return {
        "paper_id": paper_id,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "truncated": truncated,
    }


@app.get(
    "/trace/{query_id}",
    summary="Retrieve the raw agent trace for a past query",
)
async def get_trace(query_id: str) -> dict:
    """Return the raw per-agent dumps stored when ``include_trace=True`` was set."""
    if query_id not in trace_store:
        raise HTTPException(
            status_code=404,
            detail=f"Trace for query_id '{query_id}' not found.",
        )
    return trace_store[query_id]


@app.delete("/papers/{paper_id}", summary="Delete a paper and all its graph data")
async def delete_paper(paper_id: str) -> dict:
    settings = get_settings()
    driver = await get_driver(settings)

    # Delete all Neo4j nodes for this paper
    async with driver.session() as session:
        await session.run(
            "MATCH (n {paper_id: $paper_id}) DETACH DELETE n",
            paper_id=paper_id,
        )

    # Delete all Pinecone vectors for this paper (namespace = paper_id)
    try:
        index = get_index(settings)
        index.delete(delete_all=True, namespace=paper_id)
    except Exception as exc:
        logger.warning("Pinecone namespace delete failed: %s", exc)

    # Remove from job_store
    to_remove = [jid for jid, j in job_store.items() if j["paper_id"] == paper_id]
    for jid in to_remove:
        del job_store[jid]

    return {"deleted": True, "paper_id": paper_id}
