"""
Asynchronous SQLite database registry for document lifecycle management.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.db.models import Base, Document

logger = logging.getLogger(__name__)

DATABASE_URL = "sqlite+aiosqlite:///./graphora.db"

# Create the async engine
engine = create_async_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite under async/multithreading
)

# Async session factory
AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Initialize database tables."""
    logger.info("Initializing SQLite database registry at %s", DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database registry initialized successfully.")


async def create_document(
    doc_id: str,
    user_id: str,
    sha256_hash: str,
    filename: str,
    file_size_bytes: int,
    page_count: int,
    title: str,
    paper_domain: str,
    processing_status: str = "pending",
) -> Document:
    """Create a new Document record in the registry."""
    async with AsyncSessionFactory() as session:
        async with session.begin():
            doc = Document(
                doc_id=doc_id,
                user_id=user_id,
                sha256_hash=sha256_hash,
                filename=filename,
                file_size_bytes=file_size_bytes,
                page_count=page_count,
                title=title,
                paper_domain=paper_domain,
                processing_status=processing_status,
                node_count=0,
                edge_count=0,
                vector_count=0,
                error_message=None,
            )
            session.add(doc)
        # Session commit is automatic when exiting the begin block
        await session.refresh(doc)
        logger.info("Registered new document: %s (hash: %s)", doc_id, sha256_hash)
        return doc


async def get_document_by_hash(sha256_hash: str) -> Optional[Document]:
    """Retrieve a document by its SHA-256 hash."""
    async with AsyncSessionFactory() as session:
        stmt = select(Document).where(Document.sha256_hash == sha256_hash)
        result = await session.execute(stmt)
        return result.scalars().first()


async def get_document_by_id(doc_id: str) -> Optional[Document]:
    """Retrieve a document by its doc_id."""
    async with AsyncSessionFactory() as session:
        stmt = select(Document).where(Document.doc_id == doc_id)
        result = await session.execute(stmt)
        return result.scalars().first()


async def update_document_status(
    doc_id: str,
    status: str,
    **kwargs: Any,
) -> None:
    """
    Update document processing status and other fields (e.g. node_count, edge_count,
    vector_count, error_message).
    """
    async with AsyncSessionFactory() as session:
        async with session.begin():
            values = {
                "processing_status": status,
                "updated_at": datetime.now(timezone.utc),
            }
            if status == "completed":
                values["last_accessed_at"] = datetime.now(timezone.utc)
            for k, v in kwargs.items():
                values[k] = v

            stmt = update(Document).where(Document.doc_id == doc_id).values(**values)
            await session.execute(stmt)
    logger.debug("Updated document %s status to %s (extra: %s)", doc_id, status, kwargs)


async def get_documents_by_user(user_id: str) -> list[Document]:
    """Retrieve all documents belonging to a user, excluding soft-deleted ones."""
    async with AsyncSessionFactory() as session:
        # Soft delete is represented by status='archived'
        stmt = (
            select(Document)
            .where(Document.user_id == user_id)
            .where(Document.processing_status != "archived")
            .order_by(Document.created_at.desc())
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def mark_stale_jobs_failed() -> None:
    """
    On startup, mark any previously active/in-flight ingestion jobs as failed
    since they were interrupted by server restart.
    """
    stale_statuses = ["pending", "extracting", "embedding", "graph_building"]
    async with AsyncSessionFactory() as session:
        async with session.begin():
            stmt = (
                update(Document)
                .where(Document.processing_status.in_(stale_statuses))
                .values(
                    processing_status="failed",
                    error_message="System restarted while job was in progress.",
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.execute(stmt)
    logger.info("Marked any stale in-flight jobs as failed.")
