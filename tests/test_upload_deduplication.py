"""
Integration tests for PDF upload deduplication using FastAPI TestClient.
"""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from backend.main import app
from backend.db.models import Base
import backend.db.document_registry as dr
from tests.conftest import _build_minimal_pdf


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


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.mark.asyncio
@patch("backend.main.run_ingestion")
async def test_upload_deduplication_flow(mock_run_ingestion, client):
    """
    Test first upload, second upload duplicate, zero Mistral/processing calls,
    and failed re-upload behavior.
    """
    mock_run_ingestion.return_value = None

    pdf_bytes = _build_minimal_pdf("Deduplication Test PDF Content")

    # 1. First upload (creates document, triggers run_ingestion)
    response = client.post(
        "/upload",
        files={"file": ("first_upload.pdf", pdf_bytes, "application/pdf")},
        headers={"X-User-ID": "user-test"},
    )
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "pending"
    doc_id = data["doc_id"]
    assert doc_id is not None
    assert mock_run_ingestion.call_count == 1

    # Update state in DB to 'completed' to simulate successful ingestion completion
    await dr.update_document_status(doc_id, "completed")

    # Reset mock call count
    mock_run_ingestion.reset_mock()

    # 2. Second upload (same file bytes -> already processed, skips processing)
    response_dup = client.post(
        "/upload",
        files={"file": ("first_upload.pdf", pdf_bytes, "application/pdf")},
        headers={"X-User-ID": "user-test"},
    )
    assert response_dup.status_code == 202
    data_dup = response_dup.json()
    assert data_dup["status"] == "already_exists"
    assert data_dup["doc_id"] == doc_id
    # Ensure zero new ingestion background task calls
    assert mock_run_ingestion.call_count == 0

    # 3. Simulate failure in the database
    await dr.update_document_status(doc_id, "failed")

    # 4. Uploading again when failed should re-trigger ingestion pipeline
    response_retry = client.post(
        "/upload",
        files={"file": ("first_upload.pdf", pdf_bytes, "application/pdf")},
        headers={"X-User-ID": "user-test"},
    )
    assert response_retry.status_code == 202
    data_retry = response_retry.json()
    assert data_retry["status"] == "pending"
    assert data_retry["doc_id"] == doc_id
    assert mock_run_ingestion.call_count == 1


@pytest.mark.asyncio
async def test_invalid_file_type_returns_415(client):
    """Verify non-PDF uploads return 415 Unsupported Media Type."""
    response = client.post(
        "/upload",
        files={"file": ("document.txt", b"plain text content", "text/plain")},
        headers={"X-User-ID": "user-test"},
    )
    assert response.status_code == 415
    assert "Only PDF files are accepted" in response.json()["detail"]
