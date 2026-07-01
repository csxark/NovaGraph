"""
FastAPI application entry-point for the Graphora GraphRAG Research Assistant.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import timezone
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.agents.synthesizer import QueryResponse, run_pipeline
from backend.config import Settings, get_settings
from backend.extractor.domain_detector import detect_domain
from backend.extractor.entity_extractor import extract_all_chunks
from backend.graph.neo4j_queries import get_subgraph, document_exists
from backend.graph.neo4j_client import get_driver
from backend.graph.schema_init import initialize_schema
from backend.graph.neo4j_writer import (
    batch_write_edges,
    batch_write_nodes,
)
from backend.vector.embedder import embed_nodes_batch, verify_hf_api, close_client
from backend.vector.pinecone_store import get_index, upsert_batch

from backend.db.document_registry import (
    init_db,
    mark_stale_jobs_failed,
    create_document,
    get_document_by_hash,
    get_document_by_id,
    update_document_status,
    get_documents_by_user,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

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
    doc_id: str
    status: str                       # 'pending' | 'processing' | 'completed' | 'failed'
    stages: list[JobStage]
    created_at: int                   # Unix epoch ms
    updated_at: int
    error: Optional[str] = None


class UploadResponse(BaseModel):
    """Immediate response to a successful /upload request."""
    job_id: str
    doc_id: str
    status: str
    message: str
    user_id: str


class QueryRequest(BaseModel):
    """Body of a POST /query request."""
    query: str = Field(..., min_length=3, max_length=2048)
    doc_id: str
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
    response_type: str = 'paper_answer'
    prompts: list[dict] = Field(default_factory=list)
    domain: str = ''


class DocumentResponse(BaseModel):
    """Response representing a Document record."""
    doc_id: str
    user_id: str
    sha256_hash: str
    filename: str
    file_size_bytes: int
    page_count: int
    title: str
    paper_domain: str
    processing_status: str
    node_count: int
    edge_count: int
    vector_count: int
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class ErrorResponse(BaseModel):
    """Standard error body."""
    detail: str
    error_type: str


# ---------------------------------------------------------------------------
# Background ingestion pipeline
# ---------------------------------------------------------------------------

async def run_ingestion(
    doc_id: str,
    sha256_hash: str,
    file_bytes: bytes,
    filename: str,
    user_id: str,
    settings: Settings,
) -> None:
    """Run the full PDF-to-graph ingestion pipeline.
    """
    logger.info("Starting ingestion for doc_id=%s filename=%s", doc_id, filename)
    driver = await get_driver(settings)

    try:
        # 1. PDF parse
        await update_document_status(doc_id, "pending")
        start_time = time.time()
        
        from backend.parser.pdf_parser import parse_pdf
        parsed = parse_pdf(file_bytes)
        
        # 2. Domain detection & Entity extraction
        await update_document_status(doc_id, "extracting")
        domain_result = await detect_domain(parsed.full_text, settings)
        
        from backend.parser.pdf_parser import get_chunks
        chunks = get_chunks(parsed)
        graph_schema = await extract_all_chunks(
            chunks=chunks,
            domain_result=domain_result,
            settings=settings,
        )
        
        # Stamp doc_id onto every entity and edge before writing
        for entity in graph_schema.entities:
            entity.paper_id = doc_id
        for rel in graph_schema.relationships:
            rel.paper_id = doc_id

        if not graph_schema.entities:
            raise ValueError("Zero entities extracted — check Mistral API logs")

        # 3. Embedding
        await update_document_status(doc_id, "embedding")
        embeddings = await embed_nodes_batch(
            nodes=graph_schema.entities,
            settings=settings,
        )
        if len(embeddings) != len(graph_schema.entities):
            raise ValueError(f"Embedding count mismatch: {len(embeddings)} != {len(graph_schema.entities)}")

        # 4. Neo4j & Pinecone writes (graph_building)
        await update_document_status(doc_id, "graph_building")
        
        # Write to Neo4j
        nodes_written = await batch_write_nodes(
            driver=driver,
            nodes=graph_schema.entities,
            doc_id=doc_id,
        )
        name_to_id = {node.name: node.entity_id for node in graph_schema.entities}
        edges_written = await batch_write_edges(
            driver=driver,
            edges=graph_schema.relationships,
            name_to_id=name_to_id,
            doc_id=doc_id,
        )
        
        # Write to Pinecone
        vectors_written = upsert_batch(
            nodes=graph_schema.entities,
            embeddings=embeddings,
            doc_id=doc_id,
            user_id=user_id,
            settings=settings,
        )

        # 5. Completed
        await update_document_status(
            doc_id,
            "completed",
            node_count=nodes_written,
            edge_count=edges_written,
            vector_count=vectors_written,
            title=parsed.title or filename,
            paper_domain=domain_result.primary_domain or "general",
            page_count=parsed.page_count,
        )
        logger.info("Ingestion completed successfully for doc_id=%s", doc_id)

    except Exception as exc:
        logger.error("Ingestion failed for doc_id=%s: %s", doc_id, exc, exc_info=True)
        await update_document_status(
            doc_id,
            "failed",
            error_message=str(exc),
        )


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: verify Neo4j, init SQLite, fail stale jobs.
    Shutdown: close drivers.
    """
    settings = get_settings()
    logger.info("Starting GraphRAG Research Assistant…")

    # Init SQLite DB registry
    try:
        await init_db()
        await mark_stale_jobs_failed()
    except Exception as exc:
        logger.error("SQLite DB startup failure: %s", exc)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    neo4j_ok = False
    try:
        driver = await get_driver(settings)
        await driver.verify_connectivity()
        neo4j_ok = True
    except Exception:
        pass

    pinecone_ok = False
    try:
        index = get_index(settings)
        index.describe_index_stats()
        pinecone_ok = True
    except Exception:
        pass

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
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> UploadResponse:
    """Accept a PDF, validate it, and kick off a background ingestion job.
    """
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    # User isolation ID
    user_id = x_user_id or str(uuid.uuid4())

    # Content-type / extension validation
    content_type = file.content_type or ""
    filename = file.filename or "upload.pdf"
    if content_type != "application/pdf" and not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are accepted (application/pdf).",
        )

    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {len(content) / 1_048_576:.1f} MB "
                f"(max {settings.max_upload_size_mb} MB)."
            ),
        )

    sha256_hash = hashlib.sha256(content).hexdigest()

    # Deduplication hash check
    existing = await get_document_by_hash(sha256_hash)
    if existing:
        if existing.processing_status == "completed":
            return UploadResponse(
                job_id=existing.doc_id,
                doc_id=existing.doc_id,
                status="already_exists",
                message="Paper already processed — use doc_id to query",
                user_id=user_id,
            )
        elif existing.processing_status == "failed":
            # Allow re-upload by resetting status and triggering processing
            logger.info("Re-uploading previously failed document: hash=%s", sha256_hash)
            doc_id = existing.doc_id
            await update_document_status(doc_id, "pending", error_message=None)
        else:
            # Active in-flight processing
            return UploadResponse(
                job_id=existing.doc_id,
                doc_id=existing.doc_id,
                status=existing.processing_status,
                message="Paper is currently being processed.",
                user_id=user_id,
            )
    else:
        doc_id = str(uuid.uuid4())
        await create_document(
            doc_id=doc_id,
            user_id=user_id,
            sha256_hash=sha256_hash,
            filename=filename,
            file_size_bytes=len(content),
            page_count=0,
            title=filename,
            paper_domain="general",
            processing_status="pending",
        )

    background_tasks.add_task(
        run_ingestion,
        doc_id=doc_id,
        sha256_hash=sha256_hash,
        file_bytes=content,
        filename=filename,
        user_id=user_id,
        settings=settings,
    )

    return UploadResponse(
        job_id=doc_id,
        doc_id=doc_id,
        status="pending",
        message="Ingestion started. Poll GET /status/{doc_id} for progress.",
        user_id=user_id,
    )


@app.get(
    "/status/{doc_id}",
    response_model=JobStatusResponse,
    summary="Poll ingestion job status",
)
async def get_status(doc_id: str) -> JobStatusResponse:
    """Return the current status and per-stage breakdown for an ingestion job."""
    doc = await get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    status = doc.processing_status
    error_msg = doc.error_message

    # Stages mapping
    # pending -> Stage 1 is running, extract -> Stage 2 is running, etc.
    stages = [
        JobStage(stage="pdf_parse", status="pending"),
        JobStage(stage="entity_extraction", status="pending"),
        JobStage(stage="embedding", status="pending"),
        JobStage(stage="neo4j_write", status="pending"),
        JobStage(stage="pinecone_upsert", status="pending"),
    ]

    if status == "pending":
        stages[0].status = "running"
    elif status == "extracting":
        stages[0].status = "completed"
        stages[1].status = "running"
    elif status == "embedding":
        stages[0].status = "completed"
        stages[1].status = "completed"
        stages[2].status = "running"
    elif status == "graph_building":
        stages[0].status = "completed"
        stages[1].status = "completed"
        stages[2].status = "completed"
        stages[3].status = "running"
        stages[4].status = "running"
    elif status == "completed":
        for s in stages:
            s.status = "completed"
    elif status == "failed":
        # Fallback stage failure mapping
        for s in stages:
            s.status = "failed"
            s.error = error_msg

    created_ms = int(doc.created_at.replace(tzinfo=timezone.utc).timestamp() * 1000)
    updated_ms = int(doc.updated_at.replace(tzinfo=timezone.utc).timestamp() * 1000)

    return JobStatusResponse(
        job_id=doc_id,
        doc_id=doc_id,
        status=status,
        stages=stages,
        created_at=created_ms,
        updated_at=updated_ms,
        error=error_msg,
    )


@app.post(
    "/query",
    response_model=QueryResponseModel,
    summary="Query the knowledge graph for a paper",
)
async def query_paper(body: QueryRequest) -> QueryResponseModel:
    """Run the full RAG pipeline against the knowledge graph for *doc_id*."""
    settings = get_settings()
    driver = await get_driver(settings)

    # Validate document exists in SQLite registry
    doc = await get_document_by_id(body.doc_id)
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{body.doc_id}' not found. Please upload it first.",
        )

    # Validate that it is completed
    if doc.processing_status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Document is in status '{doc.processing_status}'. Ingestion must be completed to query.",
        )

    # Update last_accessed_at in registry
    await update_document_status(body.doc_id, "completed")

    try:
        result: QueryResponse = await run_pipeline(
            query=body.query,
            doc_id=body.doc_id,
            driver=driver,
            settings=settings,
            include_trace=body.include_trace,
            domain=doc.paper_domain,
        )
    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    if body.include_trace and result.trace:
        result.trace["doc_id"] = body.doc_id
        trace_store[result.query_id] = result.trace

    return QueryResponseModel(
        answer=result.answer,
        sources=[s.model_dump() for s in result.sources],
        trace=result.trace,
        query_id=result.query_id,
        total_nodes_retrieved=result.total_nodes_retrieved,
        error=result.error,
        response_type=result.response_type,
        prompts=result.prompts,
        domain=result.domain,
    )


@app.get(
    "/graph/{doc_id}",
    summary="Retrieve the full knowledge-graph subgraph for a paper",
)
async def get_graph(doc_id: str) -> dict:
    """Return up to 200 nodes and their edges for visualisation."""
    settings = get_settings()
    try:
        driver = await get_driver(settings)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j database connection unavailable: {exc}")

    # Validate document exists
    exists = await document_exists(driver=driver, doc_id=doc_id)
    if not exists:
        # Check if we have registry record to throw a better message
        doc = await get_document_by_id(doc_id)
        if doc and doc.processing_status != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Document graph is still being built (status: {doc.processing_status}).",
            )
        raise HTTPException(
            status_code=404,
            detail=f"Document with ID '{doc_id}' not found. Please ingest it first.",
        )

    try:
        return await get_subgraph(
            driver=driver,
            doc_id=doc_id,
            max_nodes=200,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph retrieval failed: {exc}")


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


@app.delete("/papers/{doc_id}", summary="Soft delete paper in document registry")
async def delete_paper(doc_id: str) -> dict:
    """Soft delete paper by changing status to archived."""
    doc = await get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    await update_document_status(doc_id, "archived")
    return {"message": f"Document '{doc_id}' soft deleted successfully.", "doc_id": doc_id}


@app.get("/documents", response_model=list[DocumentResponse], summary="List all documents for user")
async def list_documents(x_user_id: Optional[str] = Header(None, alias="X-User-ID")) -> list[DocumentResponse]:
    """Retrieve all documents associated with the given user_id."""
    if not x_user_id:
        return []

    docs = await get_documents_by_user(x_user_id)
    return [
        DocumentResponse(
            doc_id=d.doc_id,
            user_id=d.user_id,
            sha256_hash=d.sha256_hash,
            filename=d.filename,
            file_size_bytes=d.file_size_bytes,
            page_count=d.page_count,
            title=d.title,
            paper_domain=d.paper_domain,
            processing_status=d.processing_status,
            node_count=d.node_count,
            edge_count=d.edge_count,
            vector_count=d.vector_count,
            error_message=d.error_message,
            created_at=d.created_at.isoformat(),
            updated_at=d.updated_at.isoformat(),
        )
        for d in docs
    ]
