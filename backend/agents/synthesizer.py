"""
Synthesis pipeline for the GraphRAG Research Assistant.

Runs the vector, graph, and entity agents in parallel, fetches full
knowledge-graph context, merges and ranks the retrieved context, then calls
Mistral large to generate a grounded answer.

All retrieval is strictly scoped to *paper_id*.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Optional

from langchain_core.runnables import RunnableLambda, RunnableParallel
from mistralai.client import Mistral
from pydantic import BaseModel, Field

from backend.agents.entity_resolver import EntityResult, resolve_entities
from backend.agents.graph_agent import GraphResult, run_graph_agent
from backend.agents.vector_agent import VectorResult, run_vector_agent
from backend.config import Settings
from backend.graph.neo4j_queries import get_full_graph_context

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthesis prompt template
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = """
You are an expert research assistant analyzing an academic paper.

KNOWLEDGE GRAPH CONTEXT (entities and relationships extracted from the paper):
{graph_context}

SEMANTIC SEARCH RESULTS (most relevant text chunks):
{vector_context}

ENTITY RESOLUTION RESULTS:
{entity_context}

USER QUESTION: {query}

Answer the question using the knowledge graph context and semantic search results above.
Be specific, cite entity names and relationships from the graph where relevant.
If the graph context is insufficient, say so clearly.
"""


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class SourceCitation(BaseModel):
    """A single cited entity from the knowledge graph."""

    entity_id: str
    name: str
    type: str
    score: Optional[float] = None
    source_agent: str  # 'vector' | 'graph' | 'entity'


class QueryResponse(BaseModel):
    """Full pipeline response returned to the API layer."""

    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    trace: dict = Field(default_factory=dict)
    query_id: str
    total_nodes_retrieved: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Context merging and ranking
# ---------------------------------------------------------------------------


def _node_degree(entity_id: str, edges: list[dict]) -> int:
    """Count edges connected to *entity_id* in the subgraph."""
    return sum(
        1
        for e in edges
        if e.get("source") == entity_id or e.get("target") == entity_id
    )


def merge_and_rank_context(
    results: dict,
) -> tuple[str, list[SourceCitation]]:
    """Merge agent results, deduplicate by entity_id, and rank for context.

    Ranking priority:
    1. Vector results ordered by descending similarity score.
    2. Graph nodes ordered by descending subgraph degree (connectivity).
    3. Entity resolver matches (remaining not already seen).

    The top 15 unique nodes are formatted into a context string.

    Args:
        results: Dict with keys ``'vector'``, ``'graph'``, ``'entities'`` mapping
                 to VectorResult / GraphResult / EntityResult (or Exception on
                 catastrophic failure — those are treated as empty).

    Returns:
        Tuple of ``(context_text, sources)`` where *context_text* is a
        formatted string for the LLM prompt and *sources* is a list of
        :class:`SourceCitation`.
    """
    vector_result: VectorResult = results.get("vector") or VectorResult(chunks=[], scores=[], total=0)
    graph_result: GraphResult = results.get("graph") or GraphResult(anchor_nodes=[], nodes=[], edges=[], paths=[], entity_ids_found=[], traversal_depth=0)
    entity_result: EntityResult = results.get("entities") or EntityResult(resolved_ids=[], expanded_terms=[], matched_nodes=[])

    # If agent returned an Exception object wrap it as an errored result
    if isinstance(vector_result, Exception):
        vector_result = VectorResult(chunks=[], scores=[], total=0, error=str(vector_result))
    if isinstance(graph_result, Exception):
        graph_result = GraphResult(anchor_nodes=[], nodes=[], edges=[], paths=[], entity_ids_found=[], traversal_depth=0, error=str(graph_result))
    if isinstance(entity_result, Exception):
        entity_result = EntityResult(resolved_ids=[], expanded_terms=[], matched_nodes=[], error=str(entity_result))

    # Containers: list of (entity_id, name, type, description, score, source_agent)
    ranked: list[dict] = []
    seen_ids: set[str] = set()

    # 1. Vector chunks — already score-sorted by Pinecone
    if not vector_result.error:
        for chunk in sorted(
            vector_result.chunks, key=lambda c: c.get("score", 0.0), reverse=True
        ):
            eid = chunk.get("entity_id", "")
            if not eid or eid in seen_ids:
                continue
            seen_ids.add(eid)
            ranked.append(
                {
                    "entity_id": eid,
                    "name": chunk.get("name", ""),
                    "type": chunk.get("type", ""),
                    "description": chunk.get("description", ""),
                    "score": chunk.get("score"),
                    "source_agent": "vector",
                }
            )

    # 2. Graph nodes — ranked by subgraph degree
    if not graph_result.error:
        graph_edges = graph_result.edges
        graph_nodes_sorted = sorted(
            graph_result.nodes,
            key=lambda n: _node_degree(n.get("entity_id", n.get("id", "")), graph_edges),
            reverse=True,
        )
        for node in graph_nodes_sorted:
            eid = node.get("entity_id", node.get("id", ""))
            if not eid or eid in seen_ids:
                continue
            seen_ids.add(eid)
            ranked.append(
                {
                    "entity_id": eid,
                    "name": node.get("name", ""),
                    "type": node.get("type", node.get("label", "")),
                    "description": node.get("description", node.get("properties", {}).get("description", "")),
                    "score": None,
                    "source_agent": "graph",
                }
            )

    # 3. Entity resolver matches
    if not entity_result.error:
        for node in entity_result.matched_nodes:
            eid = node.get("entity_id", "")
            if not eid or eid in seen_ids:
                continue
            seen_ids.add(eid)
            ranked.append(
                {
                    "entity_id": eid,
                    "name": node.get("name", ""),
                    "type": node.get("type", node.get("label", "")),
                    "description": node.get("description", ""),
                    "score": None,
                    "source_agent": "entity",
                }
            )

    # Take top 15
    top = ranked[:15]

    # Build context text
    context_lines: list[str] = []
    for i, item in enumerate(top, start=1):
        name = item["name"] or item["entity_id"]
        etype = item["type"] or "Unknown"
        desc = item["description"] or "(no description available)"
        score_str = f" [score={item['score']:.3f}]" if item["score"] is not None else ""
        context_lines.append(
            f"{i}. [{etype}] {name}{score_str}\n   {desc}"
        )

    context_text = "\n\n".join(context_lines) if context_lines else "(No relevant entities found in the knowledge graph.)"

    # Build sources
    sources: list[SourceCitation] = [
        SourceCitation(
            entity_id=item["entity_id"],
            name=item["name"] or item["entity_id"],
            type=item["type"] or "Unknown",
            score=item["score"],
            source_agent=item["source_agent"],
        )
        for item in top
    ]

    return context_text, sources


# ---------------------------------------------------------------------------
# LLM synthesis
# ---------------------------------------------------------------------------


async def synthesize_answer(
    query: str,
    graph_context: str,
    vector_context: str,
    entity_context: str,
    settings: Settings,
) -> str:
    """Call Mistral large to synthesize a grounded answer.

    Uses the structured :data:`SYNTHESIS_PROMPT` with separate sections for
    knowledge-graph context, vector search results, and entity resolution.

    Args:
        query:          The original user question.
        graph_context:  Formatted knowledge-graph relationship context.
        vector_context: Formatted semantic search results.
        entity_context: Formatted entity resolution results.
        settings:       Application settings (holds the Mistral API key).

    Returns:
        The model's answer as a plain string. On any error returns a fallback
        error message string (never raises).
    """
    user_prompt = SYNTHESIS_PROMPT.format(
        graph_context=graph_context or "(No graph context available.)",
        vector_context=vector_context or "(No vector results available.)",
        entity_context=entity_context or "(No entity resolution results.)",
        query=query,
    )

    try:
        client = Mistral(api_key=settings.mistral_api_key)
        response = await asyncio.wait_for(
            client.chat.complete_async(
                model=settings.mistral_large_model,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            ),
            timeout=60.0,
        )
        answer: str = response.choices[0].message.content
        logger.info(
            "Synthesizer: generated %d-char answer for query=%r",
            len(answer),
            query[:60],
        )
        return answer

    except Exception as exc:  # noqa: BLE001
        logger.error("Synthesizer LLM error: %s", exc, exc_info=True)
        return f"Unable to generate answer: {exc}"


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


async def run_pipeline(
    query: str,
    paper_id: str,
    driver: Any,
    settings: Settings,
    include_trace: bool = False,
    domain: str = "general",
) -> QueryResponse:
    """Run the full retrieval-augmented generation pipeline.

    1. Fetch full knowledge-graph context for the paper.
    2. Execute vector, graph, and entity agents concurrently.
    3. Merge and rank their context.
    4. Generate a Mistral answer using the structured synthesis prompt.

    Args:
        query:         The user's question.
        paper_id:      SHA-256 identifier of the target paper.
        driver:        Neo4j async driver.
        settings:      Application settings.
        include_trace: When ``True``, embed raw agent dumps in the response.
        domain:        Paper domain string forwarded to the LLM prompt.

    Returns:
        A :class:`QueryResponse` ready for serialisation by FastAPI.
    """
    query_id = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Step 0: Fetch the full knowledge-graph context for this paper
    # ------------------------------------------------------------------
    try:
        full_graph = await get_full_graph_context(
            driver=driver,
            paper_id=paper_id,
            max_nodes=150,
        )
        graph_context_text = full_graph.get("context_text", "")
    except Exception as exc:
        logger.error("Failed to fetch full graph context: %s", exc, exc_info=True)
        full_graph = {}
        graph_context_text = ""

    # ------------------------------------------------------------------
    # Step 1: Run all three agents concurrently
    # ------------------------------------------------------------------
    vector_coro = run_vector_agent(
        query=query,
        paper_id=paper_id,
        top_k=10,
        settings=settings,
    )
    graph_coro = run_graph_agent(
        query=query,
        paper_id=paper_id,
        driver=driver,
        settings=settings,
    )
    entity_coro = resolve_entities(
        query=query,
        paper_id=paper_id,
        driver=driver,
        settings=settings,
    )

    raw = await asyncio.gather(vector_coro, graph_coro, entity_coro, return_exceptions=True)
    vector_raw, graph_raw, entity_raw = raw

    # Wrap bare exceptions into error-state result models
    if isinstance(vector_raw, Exception):
        vector_result = VectorResult(chunks=[], scores=[], total=0, error=str(vector_raw))
    else:
        vector_result: VectorResult = vector_raw  # type: ignore[no-redef]

    if isinstance(graph_raw, Exception):
        graph_result = GraphResult(
            anchor_nodes=[], nodes=[], edges=[], paths=[],
            entity_ids_found=[], traversal_depth=0, error=str(graph_raw),
        )
    else:
        graph_result: GraphResult = graph_raw  # type: ignore[no-redef]

    if isinstance(entity_raw, Exception):
        entity_result = EntityResult(
            resolved_ids=[], expanded_terms=[], matched_nodes=[], error=str(entity_raw)
        )
    else:
        entity_result: EntityResult = entity_raw  # type: ignore[no-redef]

    # ------------------------------------------------------------------
    # Step 2: Merge and rank agent context (for vector_context section)
    # ------------------------------------------------------------------
    vector_context, sources = merge_and_rank_context(
        {
            "vector": vector_result,
            "graph": graph_result,
            "entities": entity_result,
        }
    )

    # Build entity context text from resolved entities
    entity_context_lines: list[str] = []
    if not entity_result.error:
        for node in entity_result.matched_nodes:
            name = node.get("name", "")
            etype = node.get("type", node.get("label", "Unknown"))
            desc = node.get("description", "")
            if name:
                entity_context_lines.append(f"- {name} ({etype}): {desc}" if desc else f"- {name} ({etype})")
    entity_context = "\n".join(entity_context_lines) if entity_context_lines else "(No entities resolved.)"

    # ------------------------------------------------------------------
    # Step 3: Generate LLM answer with structured prompt
    # ------------------------------------------------------------------
    answer = await synthesize_answer(
        query=query,
        graph_context=graph_context_text,
        vector_context=vector_context,
        entity_context=entity_context,
        settings=settings,
    )

    # Optionally embed raw agent dumps in trace
    trace: dict = {}
    if include_trace:
        trace = {
            "vector_agent": vector_result.model_dump(),
            "graph_agent": graph_result.model_dump(),
            "entity_resolver": entity_result.model_dump(),
            "full_graph_context": {
                "node_count": full_graph.get("node_count", 0),
                "edge_count": full_graph.get("edge_count", 0),
            },
        }

    total_nodes = (
        vector_result.total
        + len(graph_result.nodes)
        + len(entity_result.resolved_ids)
    )

    # Determine top-level error flag (only if ALL agents failed)
    all_failed = bool(vector_result.error and graph_result.error and entity_result.error)
    pipeline_error: Optional[str] = None
    if all_failed:
        pipeline_error = (
            f"All agents failed — vector: {vector_result.error}; "
            f"graph: {graph_result.error}; entity: {entity_result.error}"
        )

    return QueryResponse(
        answer=answer,
        sources=sources,
        trace=trace,
        query_id=query_id,
        total_nodes_retrieved=total_nodes,
        error=pipeline_error,
    )
