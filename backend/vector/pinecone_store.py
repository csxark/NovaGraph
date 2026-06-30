"""
backend/vector/pinecone_store.py

Pinecone vector store for the GraphRAG Research Assistant.
Provides upsert and query operations over node embeddings.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pinecone import Pinecone, ServerlessSpec

from backend.config import Settings
from backend.extractor.entity_extractor import NodeModel

logger = logging.getLogger(__name__)

_EMBEDDING_DIMENSION = 384
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

    The index is created with:
        dimension = 384   (all-MiniLM-L6-v2 output size)
        metric    = cosine
        spec      = ServerlessSpec(cloud='aws', region=settings.pinecone_environment)
    """
    global _pinecone, _index

    if _index is not None:
        return _index

    if _pinecone is None:
        _pinecone = Pinecone(api_key=settings.pinecone_api_key)
        logger.info("Pinecone client initialised.")

    index_name: str = settings.pinecone_index_name
    existing_indexes = [idx["name"] for idx in _pinecone.list_indexes()]

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

def _build_metadata(node: NodeModel, paper_id: str) -> dict[str, Any]:
    """Build a Pinecone metadata dict from a NodeModel."""
    description = node.description or ""
    return {
        "entity_id": node.entity_id,
        "name": node.name,
        "type": node.type,
        "subtype": node.subtype or "",
        "domains": node.domains or [],
        "paper_id": paper_id,
        "description": description[:500],  # Pinecone metadata value size limit
        "chunk_ref": node.chunk_ref or "",
        "confidence": node.confidence,
    }


# ---------------------------------------------------------------------------
# Upsert operations
# ---------------------------------------------------------------------------

def upsert_node(
    node: NodeModel,
    embedding: list[float],
    paper_id: str,
    settings: Settings,
) -> None:
    """
    Upsert a single NodeModel vector into Pinecone.

    namespace is derived from the first 16 characters of paper_id to keep
    per-paper data logically partitioned while staying within Pinecone limits.
    """
    index = get_index(settings)
    namespace = paper_id[:16]
    vector_id = node.entity_id
    metadata = _build_metadata(node, paper_id)

    index.upsert(
        vectors=[{"id": vector_id, "values": embedding, "metadata": metadata}],
        namespace=namespace,
    )
    logger.debug("Upserted node %s into namespace '%s'", vector_id, namespace)


def upsert_batch(
    nodes: list[NodeModel],
    embeddings: list[list[float]],
    paper_id: str,
    settings: Settings,
    batch_size: int = 100,
) -> int:
    """
    Upsert a batch of NodeModel vectors into Pinecone.

    Args:
        nodes: NodeModel objects to upsert.
        embeddings: Pre-computed embeddings; must be the same length as *nodes*.
        paper_id: Used for namespace derivation and metadata.
        settings: App settings (API key, index name, region).
        batch_size: Number of vectors per Pinecone upsert call.

    Returns:
        Total number of vectors upserted.
    """
    if not nodes:
        return 0

    if len(nodes) != len(embeddings):
        raise ValueError(
            f"nodes ({len(nodes)}) and embeddings ({len(embeddings)}) must have the same length."
        )

    index = get_index(settings)
    namespace = paper_id[:16]

    vectors = [
        {
            "id": node.entity_id,
            "values": emb,
            "metadata": _build_metadata(node, paper_id),
        }
        for node, emb in zip(nodes, embeddings)
    ]

    total = 0
    for i in range(0, len(vectors), batch_size):
        chunk = vectors[i : i + batch_size]
        index.upsert(vectors=chunk, namespace=namespace)
        total += len(chunk)
        logger.debug(
            "Upserted batch of %d vectors into namespace '%s' (total: %d)",
            len(chunk),
            namespace,
            total,
        )

    logger.info(
        "upsert_batch complete: %d vectors upserted for paper %s", total, paper_id
    )
    return total


# ---------------------------------------------------------------------------
# Query operations
# ---------------------------------------------------------------------------

def query_similar(
    embedding: list[float],
    paper_id: str,
    top_k: int,
    settings: Settings,
    node_type_filter: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Query within the namespace of a specific paper.

    Args:
        embedding: Query vector.
        paper_id: Restricts results to this paper's namespace and metadata filter.
        top_k: Number of results; capped at _MAX_TOP_K (20).
        settings: App settings.
        node_type_filter: Optional Pinecone metadata filter on the 'type' field.

    Returns:
        List of {id, score, metadata} dicts ordered by descending score.
    """
    top_k = min(top_k, _MAX_TOP_K)
    index = get_index(settings)
    namespace = paper_id[:16] if paper_id else ""

    filter_dict: dict[str, Any] = {"paper_id": {"$eq": paper_id}}
    if node_type_filter:
        filter_dict["type"] = {"$eq": node_type_filter}

    result = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
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
    Query across all namespaces (no namespace restriction).

    Args:
        embedding: Query vector.
        top_k: Number of results; capped at _MAX_TOP_K (20).
        settings: App settings.

    Returns:
        List of {id, score, metadata} dicts ordered by descending score.
    """
    top_k = min(top_k, _MAX_TOP_K)
    index = get_index(settings)

    result = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
    )

    return [
        {"id": match["id"], "score": match["score"], "metadata": match.get("metadata", {})}
        for match in result.get("matches", [])
    ]


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def delete_paper_vectors(paper_id: str, settings: Settings) -> None:
    """
    Delete all vectors belonging to a paper by deleting its entire namespace.

    Uses ``delete_all=True`` within the paper's namespace, which is the
    most efficient way to bulk-remove per-paper data in Pinecone Serverless.
    """
    index = get_index(settings)
    namespace = paper_id[:16]
    index.delete(delete_all=True, namespace=namespace)
    logger.info("Deleted all vectors for paper %s (namespace '%s')", paper_id, namespace)
