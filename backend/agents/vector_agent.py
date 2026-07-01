"""
Vector retrieval agent for the GraphRAG Research Assistant.

Embeds a query string, searches Pinecone for semantically similar entity
chunks, and returns a structured :class:`VectorResult`.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Optional

from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from backend.config import Settings
from backend.vector.embedder import embed_text
from backend.vector.pinecone_store import get_index, query_similar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class VectorResult(BaseModel):
    """Result produced by the vector retrieval agent."""

    chunks: list[dict] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    total: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Core agent coroutine
# ---------------------------------------------------------------------------


async def run_vector_agent(
    query: str,
    paper_id: str,
    top_k: int,
    settings: Settings,
) -> VectorResult:
    """Embed *query* and retrieve the top-*top_k* semantically similar chunks.

    Args:
        query:    Natural-language question from the user.
        paper_id: SHA-256 identifier of the uploaded paper.
        top_k:    Maximum number of results to return.
        settings: Application settings (holds API keys, model names, etc.).

    Returns:
        A :class:`VectorResult` containing matched chunks and their scores.
        On any error the result has ``error`` set and empty lists.
    """
    if not paper_id:
        return VectorResult(
            chunks=[], scores=[], total=0,
            error="paper_id is required for vector retrieval",
        )

    try:
        # 1. Embed the user query
        query_vector: list[float] = await embed_text(query, settings)

        # 2. Retrieve similar vectors from Pinecone
        raw_matches: list[dict] = query_similar(
            embedding=query_vector,
            paper_id=paper_id,
            top_k=top_k,
            settings=settings,
        )

        # 3. Normalise each match to a consistent shape
        chunks: list[dict] = []
        scores: list[float] = []

        for match in raw_matches:
            metadata: dict = match.get("metadata", {})
            score: float = float(match.get("score", 0.0))

            chunks.append(
                {
                    "entity_id": match.get("id", metadata.get("entity_id", "")),
                    "name": metadata.get("name", ""),
                    "type": metadata.get("type", ""),
                    "description": metadata.get("description", metadata.get("text", "")),
                    "score": score,
                    "paper_id": metadata.get("paper_id", paper_id),
                }
            )
            scores.append(score)

        logger.info(
            "VectorAgent: query=%r paper_id=%s top_k=%d -> %d results",
            query[:60],
            paper_id,
            top_k,
            len(chunks),
        )
        return VectorResult(chunks=chunks, scores=scores, total=len(chunks))

    except Exception as exc:  # noqa: BLE001
        logger.error("VectorAgent error: %s", exc, exc_info=True)
        return VectorResult(chunks=[], scores=[], total=0, error=str(exc))


# ---------------------------------------------------------------------------
# LCEL chain builder
# ---------------------------------------------------------------------------


def build_vector_chain(
    paper_id: str,
    top_k: int,
    settings: Settings,
) -> RunnableLambda:
    """Return an LCEL RunnableLambda that wraps :func:`run_vector_agent`.

    The returned runnable accepts a *query* string as input and produces a
    :class:`VectorResult`.

    Args:
        paper_id: SHA-256 identifier of the paper to search within.
        top_k:    Maximum number of Pinecone matches to return.
        settings: Application settings.

    Returns:
        A RunnableLambda whose ainvoke signature is (query: str) -> VectorResult.
    """
    bound = partial(run_vector_agent, paper_id=paper_id, top_k=top_k, settings=settings)
    return RunnableLambda(bound)
