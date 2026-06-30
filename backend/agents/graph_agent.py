"""
Graph traversal agent for the GraphRAG Research Assistant.

Performs entity extraction from the query, full-text Neo4j search, and
multi-hop graph traversal. Returns a structured :class:`GraphResult`.
"""

from __future__ import annotations

import logging
import re
from functools import partial
from typing import Any, Optional

from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from backend.config import Settings
from backend.graph.neo4j_queries import search_nodes_by_label, traverse_from_entities

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple stopword set for noun-phrase heuristic
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "this", "that", "these", "those", "it", "its",
        "not", "no", "so", "as", "if", "then", "than", "how", "what", "which",
        "who", "when", "where", "why", "about", "into", "through", "between",
        "after", "before", "during", "such", "each", "both", "more", "most",
    }
)


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class GraphResult(BaseModel):
    """Result produced by the graph traversal agent."""

    nodes: list[dict] = Field(default_factory=list)
    edges: list[dict] = Field(default_factory=list)
    paths: list = Field(default_factory=list)
    entity_ids_found: list[str] = Field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_noun_phrases(query: str) -> list[str]:
    """Extract candidate entity terms from *query* using a simple heuristic.

    Keeps tokens that are:
    * longer than 4 characters, and
    * not in the stopword list.

    Returns a deduplicated list preserving first-seen order.
    """
    tokens = re.sub(r"[^\w\s]", " ", query).split()
    seen: set[str] = set()
    result: list[str] = []
    for tok in tokens:
        lower = tok.lower()
        if len(lower) > 4 and lower not in _STOPWORDS and lower not in seen:
            seen.add(lower)
            result.append(tok)
    return result


# ---------------------------------------------------------------------------
# Core agent coroutine
# ---------------------------------------------------------------------------


async def run_graph_agent(
    query: str,
    paper_id: str,
    driver: Any,
    settings: Settings,
) -> GraphResult:
    """Traverse the Neo4j knowledge graph starting from entities found in *query*.

    Steps:
    1. Extract candidate entity names from *query* via noun-phrase heuristic.
    2. Full-text search Neo4j for each candidate using ``search_nodes_by_label``.
    3. Collect the top entity IDs from the matches.
    4. Perform a 2-hop ``traverse_from_entities`` to build the subgraph.

    Args:
        query:    Natural-language question from the user.
        paper_id: SHA-256 identifier of the paper (used as a filter).
        driver:   Neo4j async driver instance.
        settings: Application settings.

    Returns:
        A :class:`GraphResult` with nodes, edges, and found entity IDs.
        On any error the result has ``error`` set and empty lists.
    """
    try:
        # Step 1: extract candidate names from query
        candidates = _extract_noun_phrases(query)
        if not candidates:
            # fall back: use every token longer than 2 chars
            candidates = [w for w in query.split() if len(w) > 2]

        logger.debug("GraphAgent candidates: %s", candidates)

        # Step 2: full-text search Neo4j for each candidate
        matched_nodes: list[dict] = []
        for term in candidates:
            try:
                results = await search_nodes_by_label(
                    driver=driver,
                    label="",
                    search_term=term,
                    paper_id=paper_id,
                )
                matched_nodes.extend(results)
            except Exception as term_exc:  # noqa: BLE001
                logger.warning("GraphAgent search failed for term %r: %s", term, term_exc)

        # Step 3: deduplicate and collect entity IDs
        seen_ids: set[str] = set()
        entity_ids: list[str] = []
        for node in matched_nodes:
            eid = node.get("entity_id", "")
            if eid and eid not in seen_ids:
                seen_ids.add(eid)
                entity_ids.append(eid)

        if not entity_ids:
            logger.info(
                "GraphAgent: no entities matched for query=%r paper_id=%s",
                query[:60],
                paper_id,
            )
            return GraphResult(
                nodes=[],
                edges=[],
                paths=[],
                entity_ids_found=[],
            )

        # Step 4: traverse from found entities (depth=2)
        subgraph: dict = await traverse_from_entities(
            driver=driver,
            entity_ids=entity_ids,
            paper_id=paper_id,
            depth=2,
            settings=settings,
        )

        nodes: list[dict] = subgraph.get("nodes", [])
        edges: list[dict] = subgraph.get("edges", [])
        paths: list = subgraph.get("paths", [])

        logger.info(
            "GraphAgent: query=%r -> %d seed ids, %d nodes, %d edges",
            query[:60],
            len(entity_ids),
            len(nodes),
            len(edges),
        )
        return GraphResult(
            nodes=nodes,
            edges=edges,
            paths=paths,
            entity_ids_found=entity_ids,
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("GraphAgent error: %s", exc, exc_info=True)
        return GraphResult(
            nodes=[],
            edges=[],
            paths=[],
            entity_ids_found=[],
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# LCEL chain builder
# ---------------------------------------------------------------------------


def build_graph_chain(
    paper_id: str,
    driver: Any,
    settings: Settings,
) -> RunnableLambda:
    """Return an LCEL RunnableLambda that wraps :func:`run_graph_agent`.

    The returned runnable accepts a *query* string and produces a
    :class:`GraphResult`.

    Args:
        paper_id: SHA-256 identifier of the paper to traverse.
        driver:   Neo4j async driver instance.
        settings: Application settings.

    Returns:
        A RunnableLambda whose ainvoke signature is (query: str) -> GraphResult.
    """
    bound = partial(run_graph_agent, paper_id=paper_id, driver=driver, settings=settings)
    return RunnableLambda(bound)
