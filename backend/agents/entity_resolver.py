"""
Entity resolver for the Graphora Research Assistant.

Tokenises the user query, removes stopwords, and looks up each surviving
term in the Neo4j knowledge graph to produce a deduplicated list of resolved
entity IDs and their expanded names.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from backend.config import Settings
from backend.graph.neo4j_queries import find_similar_nodes_by_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stopword set (~50 common English stopwords)
# ---------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "also",
        "am", "an", "and", "any", "are", "as", "at", "be", "because", "been",
        "before", "being", "between", "both", "but", "by", "can", "could",
        "did", "do", "does", "doing", "during", "each", "few", "for", "from",
        "further", "get", "had", "has", "have", "having", "he", "her", "here",
        "him", "his", "how", "if", "in", "into", "is", "it", "its", "itself",
        "just", "may", "me", "might", "more", "most", "my", "no", "not", "now",
        "of", "on", "or", "other", "our", "out", "over", "same", "she", "so",
        "some", "such", "than", "that", "the", "their", "them", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "up", "use",
        "very", "was", "we", "were", "what", "when", "where", "which", "while",
        "who", "why", "will", "with", "would", "you", "your",
    }
)


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class EntityResult(BaseModel):
    """Result produced by the entity resolver."""

    resolved_ids: list[str] = Field(default_factory=list)
    expanded_terms: list[str] = Field(default_factory=list)
    matched_nodes: list[dict] = Field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Query tokeniser
# ---------------------------------------------------------------------------


def extract_query_terms(query: str) -> list[str]:
    """Tokenise *query* and return candidate entity terms.
    """
    # Remove punctuation (except hyphens inside words)
    cleaned = re.sub(r"[^\w\s\-]", " ", query)
    tokens = cleaned.lower().split()

    seen: set[str] = set()
    terms: list[str] = []
    for tok in tokens:
        # strip leading/trailing hyphens
        tok = tok.strip("-")
        if len(tok) >= 3 and tok not in STOPWORDS and tok not in seen:
            seen.add(tok)
            terms.append(tok)

    return terms


# ---------------------------------------------------------------------------
# Core resolver coroutine
# ---------------------------------------------------------------------------


async def resolve_entities(
    query: str,
    doc_id: str,
    driver: Any,
    settings: Settings,
) -> EntityResult:
    """Resolve query terms to knowledge-graph entity IDs.

    For each term extracted from *query*, calls ``find_similar_nodes_by_name``
    against Neo4j, then deduplicates results by ``entity_id``.
    """
    if not doc_id:
        return EntityResult(
            resolved_ids=[],
            expanded_terms=[],
            matched_nodes=[],
            error="doc_id is required for entity resolution",
        )

    try:
        terms = extract_query_terms(query)
        if not terms:
            logger.info("EntityResolver: no query terms extracted from %r", query[:60])
            return EntityResult(
                resolved_ids=[],
                expanded_terms=[],
                matched_nodes=[],
            )

        logger.debug("EntityResolver terms: %s", terms)

        # Collect results per term — deduplicate by entity_id
        seen_ids: set[str] = set()
        matched_nodes: list[dict] = []

        for term in terms:
            try:
                candidates: list[dict] = await find_similar_nodes_by_name(
                    driver=driver,
                    name=term,
                    doc_id=doc_id,
                )
                for node in candidates:
                    eid = node.get("entity_id", "")
                    if eid and eid not in seen_ids:
                        seen_ids.add(eid)
                        matched_nodes.append(node)
            except Exception as term_exc:  # noqa: BLE001
                logger.warning(
                    "EntityResolver: lookup failed for term %r: %s", term, term_exc
                )

        # Fallback: if nothing resolved, grab sample nodes from the paper
        if not matched_nodes:
            logger.info("EntityResolver: zero matches — using sample nodes as fallback")
            try:
                from backend.graph.neo4j_queries import get_subgraph
                fallback = await get_subgraph(driver=driver, doc_id=doc_id, max_nodes=10)
                for node in fallback.get("nodes", [])[:5]:
                    eid = node.get("id", node.get("entity_id", ""))
                    if eid and eid not in seen_ids:
                        seen_ids.add(eid)
                        # Normalize to expected format
                        matched_nodes.append({
                            "entity_id": eid,
                            "name": node.get("name", ""),
                            "description": node.get("description", ""),
                            "type": node.get("type", "Entity"),
                        })
            except Exception as exc:
                logger.warning("EntityResolver fallback failed: %s", exc)

        resolved_ids: list[str] = [n.get("entity_id", "") for n in matched_nodes if n.get("entity_id")]
        expanded_terms: list[str] = [n.get("name", "") for n in matched_nodes if n.get("name")]

        logger.info(
            "EntityResolver: query=%r -> %d terms -> %d entities",
            query[:60],
            len(terms),
            len(resolved_ids),
        )
        return EntityResult(
            resolved_ids=resolved_ids,
            expanded_terms=expanded_terms,
            matched_nodes=matched_nodes,
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("EntityResolver error: %s", exc, exc_info=True)
        return EntityResult(
            resolved_ids=[],
            expanded_terms=[],
            matched_nodes=[],
            error=str(exc),
        )
