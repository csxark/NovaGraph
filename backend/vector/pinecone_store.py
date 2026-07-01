"""
backend/vector/pinecone_store.py

Pinecone vector store for the Graphora Research Assistant.
Provides upsert and query operations over node embeddings.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from pinecone import Pinecone, ServerlessSpec

from backend.config import Settings
from backend.extractor.entity_extractor import NodeModel

logger = logging.getLogger(__name__)

_EMBEDDING_DIMENSION = 1024  # Must match mistral-embed output dimension
_METRIC = "cosine"
_MAX_TOP_K = 20

# Module-level singletons
_pinecone: Pinecone | None = None
_index = None  # pinecone.Index — no static type available across SDK versions


# ---------------------------------------------------------------------------
# Singleton initialisation
# ---------------------------------------------------------------------------

def get_index(settings: Settings):
    """
    Return the shared Pinecone Index, creating it if it does not yet exist.
    """
    global _pinecone, _index

    if _index is not None:
        return _index

    if _pinecone is None:
        _pinecone = Pinecone(api_key=settings.pinecone_api_key)
        logger.info("Pinecone client initialised.")

    index_name: str = settings.pinecone_index_name
    existing_indexes = [idx["name"] for idx in _pinecone.list_indexes()]

    # If index already exists, verify its dimension matches
    if index_name in existing_indexes:
        existing_desc = _pinecone.describe_index(index_name)
        existing_dim = existing_desc.dimension
        if existing_dim != _EMBEDDING_DIMENSION:
            logger.warning(
                "Pinecone index '%s' has dimension %d but expected %d. "
                "Deleting and recreating with correct dimension.",
                index_name, existing_dim, _EMBEDDING_DIMENSION,
            )
            _pinecone.delete_index(index_name)
            existing_indexes = []  # force recreation below

    if index_name not in existing_indexes:
        logger.info("Creating Pinecone index '%s' …", index_name)
        _pinecone.create_index(
            name=index_name,
            dimension=_EMBEDDING_DIMENSION,
            metric=_METRIC,
            spec=ServerlessSpec(
                cloud="aws",
                region=settings.pinecone_environment,
            ),
        )
        logger.info("Pinecone index '%s' created.", index_name)
    else:
        logger.info("Pinecone index '%s' already exists.", index_name)

    _index = _pinecone.Index(index_name)
    return _index


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _build_metadata(node: NodeModel, doc_id: str, user_id: str) -> dict[str, Any]:
    """Build a Pinecone metadata dict from a NodeModel."""
    description = node.description or ""
    return {
        "doc_id": doc_id,
        "user_id": user_id,
        "entity_id": node.entity_id,
        "name": node.name,
        "type": node.type,
        "subtype": node.subtype or "",
        "domains": node.domains or [],
        "description": description[:500],  # Pinecone metadata value size limit
        "chunk_ref": node.chunk_ref or "",
        "confidence": node.confidence,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Upsert operations
# ---------------------------------------------------------------------------

def upsert_node(
    node: NodeModel,
    embedding: list[float],
    doc_id: str,
    user_id: str,
    settings: Settings,
) -> None:
    """
    Upsert a single NodeModel vector into Pinecone.
    """
    index = get_index(settings)
    vector_id = node.entity_id
    metadata = _build_metadata(node, doc_id, user_id)

    index.upsert(
        vectors=[{"id": vector_id, "values": embedding, "metadata": metadata}],
        namespace="",
    )
    logger.debug("Upserted node %s into Pinecone", vector_id)


def upsert_batch(
    nodes: list[NodeModel],
    embeddings: list[list[float]],
    doc_id: str,
    user_id: str,
    settings: Settings,
    batch_size: int = 100,
) -> int:
    """
    Upsert a batch of NodeModel vectors into Pinecone.
    """
    if not nodes:
        return 0

    if len(nodes) != len(embeddings):
        raise ValueError(
            f"nodes ({len(nodes)}) and embeddings ({len(embeddings)}) must have the same length."
        )

    index = get_index(settings)

    vectors = [
        {
            "id": node.entity_id,
            "values": emb,
            "metadata": _build_metadata(node, doc_id, user_id),
        }
        for node, emb in zip(nodes, embeddings)
    ]

    total = 0
    for i in range(0, len(vectors), batch_size):
        chunk = vectors[i : i + batch_size]
        index.upsert(vectors=chunk, namespace="")
        total += len(chunk)
        logger.debug(
            "Upserted batch of %d vectors into default namespace (total: %d)",
            len(chunk),
            total,
        )

    logger.info(
        "upsert_batch complete: %d vectors upserted for document %s", total, doc_id
    )
    return total


# ---------------------------------------------------------------------------
# Query operations
# ---------------------------------------------------------------------------

def query_similar(
    embedding: list[float],
    doc_id: str,
    top_k: int,
    settings: Settings,
    node_type_filter: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Query within the default namespace with strict doc_id metadata filter.
    """
    top_k = min(top_k, _MAX_TOP_K)
    index = get_index(settings)

    # Every query must include filter {"doc_id": {"$eq": doc_id}}
    filter_dict: dict[str, Any] = {"doc_id": {"$eq": doc_id}}
    if node_type_filter:
        filter_dict["type"] = {"$eq": node_type_filter}

    result = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="",
        filter=filter_dict,
    )

    return [
        {"id": match["id"], "score": match["score"], "metadata": match.get("metadata", {})}
        for match in result.get("matches", [])
    ]


def query_global(
    embedding: list[float],
    top_k: int,
    settings: Settings,
) -> list[dict[str, Any]]:
    """
    Query across all documents (no restriction).
    """
    top_k = min(top_k, _MAX_TOP_K)
    index = get_index(settings)

    result = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="",
    )

    return [
        {"id": match["id"], "score": match["score"], "metadata": match.get("metadata", {})}
        for match in result.get("matches", [])
    ]


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def delete_paper_vectors(doc_id: str, settings: Settings) -> None:
    """
    Delete all vectors belonging to a document by metadata filter.
    """
    index = get_index(settings)
    index.delete(filter={"doc_id": {"$eq": doc_id}}, namespace="")
    logger.info("Deleted all vectors for doc_id %s using metadata filter", doc_id)
