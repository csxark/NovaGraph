"""
Graph traversal agent for the GraphRAG Research Assistant.

Performs entity anchoring via full-text search, multi-hop graph traversal,
and returns a structured :class:`GraphResult` with anchor nodes, neighbors,
relationships, and traversal depth.

All queries are strictly scoped to *doc_id* — no cross-paper data leakage.
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

    anchor_nodes: list[dict] = Field(default_factory=list)
    nodes: list[dict] = Field(default_factory=list)
    edges: list[dict] = Field(default_factory=list)
    paths: list = Field(default_factory=list)
    entity_ids_found: list[str] = Field(default_factory=list)
    traversal_depth: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_noun_phrases(query: str) -> list[str]:
    """Extract candidate entity terms from *query* using a simple heuristic.
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
# Core agent coroutine — multi-hop graph traversal
# ---------------------------------------------------------------------------


async def run_graph_agent(
    query: str,
    doc_id: str,
    driver: Any,
    settings: Settings,
) -> GraphResult:
    """Traverse the Neo4j knowledge graph using proper multi-hop graph traversal.

    Step 1 — Entity anchoring:
        Extract key terms from the query. Find matching nodes in Neo4j using
        full-text index ``ft_name_desc``, scoped to *doc_id*.

    Step 2 — Graph expansion:
        For each anchor node, traverse up to 2 hops to retrieve the connected
        subgraph, scoped to *doc_id*.

    Step 3 — Format output:
        Return a structured dict with ``anchor_nodes``, ``nodes``,
        ``edges``, ``traversal_depth`` for the synthesizer to use.
    """
    if not doc_id:
        return GraphResult(
            error="doc_id is required for graph traversal",
        )

    try:
        # ------------------------------------------------------------------
        # Step 1: Entity anchoring — full-text search scoped to doc_id
        # ------------------------------------------------------------------
        candidates = _extract_noun_phrases(query)
        if not candidates:
            # fall back: use every token longer than 2 chars
            candidates = [w for w in query.split() if len(w) > 2]

        logger.debug("GraphAgent candidates: %s", candidates)

        # Strategy 1: Try full candidate phrase
        anchor_nodes = []
        query_terms = " ".join(candidates)
        try:
            results = await search_nodes_by_label(
                driver=driver, label="", search_term=query_terms, doc_id=doc_id,
            )
            anchor_nodes = results[:5]
        except Exception as exc:
            logger.warning("GraphAgent full-phrase search failed: %s", exc)

        # Strategy 2: If no results, try each candidate individually and merge
        if not anchor_nodes:
            seen_names: set[str] = set()
            for term in candidates[:4]:  # max 4 individual lookups
                try:
                    results = await search_nodes_by_label(
                        driver=driver, label="", search_term=term, doc_id=doc_id,
                    )
                    for r in results[:2]:
                        name = r.get("name", "")
                        if name and name not in seen_names:
                            seen_names.add(name)
                            anchor_nodes.append(r)
                except Exception:
                    pass
                if len(anchor_nodes) >= 5:
                    break

        # Strategy 3: If still no results, fetch top 5 nodes by doc_id directly
        if not anchor_nodes:
            try:
                from backend.graph.neo4j_queries import get_subgraph
                fallback = await get_subgraph(driver=driver, doc_id=doc_id, max_nodes=5)
                # Map standard keys
                anchor_nodes = [
                    {
                        "entity_id": n.get("id"),
                        "name": n.get("name"),
                        "type": n.get("type"),
                        "description": n.get("description"),
                        "_labels": [n.get("type")],
                    }
                    for n in fallback.get("nodes", [])[:5]
                ]
                logger.info("GraphAgent: using fallback top-5 nodes for doc_id %s", doc_id)
            except Exception as exc:
                logger.warning("GraphAgent fallback failed: %s", exc)

        # Deduplicate and collect entity IDs from anchors
        seen_ids: set[str] = set()
        entity_ids: list[str] = []
        for node in anchor_nodes:
            eid = node.get("entity_id", "")
            if eid and eid not in seen_ids:
                seen_ids.add(eid)
                entity_ids.append(eid)

        if not entity_ids:
            logger.info(
                "GraphAgent: no entities matched for query=%r doc_id=%s",
                query[:60],
                doc_id,
            )
            return GraphResult(
                anchor_nodes=[],
                nodes=[],
                edges=[],
                paths=[],
                entity_ids_found=[],
                traversal_depth=0,
            )

        # ------------------------------------------------------------------
        # Step 2: Graph expansion — 2-hop traversal from anchor nodes
        # ------------------------------------------------------------------
        traversal_depth = 2
        subgraph: dict = await traverse_from_entities(
            driver=driver,
            entity_ids=entity_ids,
            doc_id=doc_id,
            depth=traversal_depth,
        )

        all_nodes: list[dict] = subgraph.get("nodes", [])
        all_edges: list[dict] = subgraph.get("edges", [])

        logger.info(
            "GraphAgent: query=%r -> %d anchors, %d total nodes, %d edges (depth=%d)",
            query[:60],
            len(anchor_nodes),
            len(all_nodes),
            len(all_edges),
            traversal_depth,
        )

        # ------------------------------------------------------------------
        # Step 3: Structured output
        # ------------------------------------------------------------------
        return GraphResult(
            anchor_nodes=anchor_nodes,
            nodes=all_nodes,
            edges=all_edges,
            paths=[],
            entity_ids_found=entity_ids,
            traversal_depth=traversal_depth,
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("GraphAgent error: %s", exc, exc_info=True)
        return GraphResult(
            anchor_nodes=[],
            nodes=[],
            edges=[],
            paths=[],
            entity_ids_found=[],
            traversal_depth=0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# LCEL chain builder
# ---------------------------------------------------------------------------


def build_graph_chain(
    doc_id: str,
    driver: Any,
    settings: Settings,
) -> RunnableLambda:
    """Return an LCEL RunnableLambda that wraps :func:`run_graph_agent`.
    """
    bound = partial(run_graph_agent, doc_id=doc_id, driver=driver, settings=settings)
    return RunnableLambda(bound)
