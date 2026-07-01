"""
Unit tests for the SQLite Document Registry.
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from backend.db.models import Base, Document
import backend.db.document_registry as dr


@pytest.fixture(autouse=True)
async def setup_test_db():
    """Overrides the database engine with an in-memory SQLite db for testing."""
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    test_session_maker = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    orig_engine = dr.engine
    orig_factory = dr.AsyncSessionFactory

    dr.engine = test_engine
    dr.AsyncSessionFactory = test_session_maker

    # Initialize tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    await test_engine.dispose()
    dr.engine = orig_engine
    dr.AsyncSessionFactory = orig_factory


@pytest.mark.asyncio
async def test_create_document_success():
    """Verify document registration is created correctly."""
    doc = await dr.create_document(
        doc_id="doc-123",
        user_id="user-456",
        sha256_hash="hash-123456",
        filename="test.pdf",
        file_size_bytes=1024,
        page_count=5,
        title="Title Test",
        paper_domain="computer_science",
    )

    assert doc.doc_id == "doc-123"
    assert doc.user_id == "user-456"
    assert doc.sha256_hash == "hash-123456"
    assert doc.processing_status == "pending"
    assert doc.node_count == 0

    # Retrieve from DB
    retrieved = await dr.get_document_by_id("doc-123")
    assert retrieved is not None
    assert retrieved.sha256_hash == "hash-123456"


@pytest.mark.asyncio
async def test_hash_deduplication_returns_existing():
    """Verify hash lookup correctly identifies duplicate documents."""
    await dr.create_document(
        doc_id="doc-123",
        user_id="user-456",
        sha256_hash="hash-dup",
        filename="test.pdf",
        file_size_bytes=1024,
        page_count=5,
        title="Title Test",
        paper_domain="computer_science",
    )

    doc = await dr.get_document_by_hash("hash-dup")
    assert doc is not None
    assert doc.doc_id == "doc-123"

    non_existent = await dr.get_document_by_hash("hash-nonexistent")
    assert non_existent is None


@pytest.mark.asyncio
async def test_update_status_all_transitions():
    """Verify document status transitions and metric updates work."""
    await dr.create_document(
        doc_id="doc-123",
        user_id="user-456",
        sha256_hash="hash-123",
        filename="test.pdf",
        file_size_bytes=1024,
        page_count=5,
        title="Title Test",
        paper_domain="computer_science",
    )

    # Transition to extracting
    await dr.update_document_status("doc-123", "extracting")
    doc = await dr.get_document_by_id("doc-123")
    assert doc.processing_status == "extracting"

    # Transition to completed with metadata
    await dr.update_document_status(
        "doc-123",
        "completed",
        node_count=10,
        edge_count=20,
        vector_count=10,
    )
    doc = await dr.get_document_by_id("doc-123")
    assert doc.processing_status == "completed"
    assert doc.node_count == 10
    assert doc.edge_count == 20
    assert doc.vector_count == 10


@pytest.mark.asyncio
async def test_mark_stale_jobs_failed_on_startup():
    """Verify mark_stale_jobs_failed resets all in-progress jobs to failed."""
    await dr.create_document(
        doc_id="doc-pending",
        user_id="user-1",
        sha256_hash="hash-pending",
        filename="test1.pdf",
        file_size_bytes=100,
        page_count=1,
        title="Title 1",
        paper_domain="general",
        processing_status="pending",
    )
    await dr.create_document(
        doc_id="doc-completed",
        user_id="user-1",
        sha256_hash="hash-completed",
        filename="test2.pdf",
        file_size_bytes=100,
        page_count=1,
        title="Title 2",
        paper_domain="general",
        processing_status="completed",
    )

    await dr.mark_stale_jobs_failed()

    p_doc = await dr.get_document_by_id("doc-pending")
    c_doc = await dr.get_document_by_id("doc-completed")

    assert p_doc.processing_status == "failed"
    assert p_doc.error_message == "System restarted while job was in progress."
    assert c_doc.processing_status == "completed"


@pytest.mark.asyncio
async def test_get_documents_by_user_isolation():
    """Verify document list retrieval is isolated per user."""
    await dr.create_document(
        doc_id="doc-u1",
        user_id="user-1",
        sha256_hash="hash-1",
        filename="test1.pdf",
        file_size_bytes=100,
        page_count=1,
        title="Title 1",
        paper_domain="general",
    )
    await dr.create_document(
        doc_id="doc-u2",
        user_id="user-2",
        sha256_hash="hash-2",
        filename="test2.pdf",
        file_size_bytes=100,
        page_count=1,
        title="Title 2",
        paper_domain="general",
    )

    u1_docs = await dr.get_documents_by_user("user-1")
    u2_docs = await dr.get_documents_by_user("user-2")

    assert len(u1_docs) == 1
    assert u1_docs[0].doc_id == "doc-u1"

    assert len(u2_docs) == 1
    assert u2_docs[0].doc_id == "doc-u2"
