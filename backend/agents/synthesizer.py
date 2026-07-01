"""
Synthesis pipeline for the Graphora GraphRAG Research Assistant.

Runs the vector, graph, and entity agents in parallel, fetches targeted
knowledge-graph context, merges and ranks the retrieved context, then calls
Mistral large to generate a grounded answer.

All retrieval is strictly scoped to *doc_id*.
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
from backend.graph.neo4j_queries import search_nodes_by_label, traverse_from_entities
from backend.agents.prompt_architect import classify_intent, generate_prompt_architect_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthesis prompt template
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = """
You are an expert research assistant for Graphora, analyzing a specific academic paper.

STRICT RULES — follow these without exception:
1. You ONLY answer questions about the paper whose context is provided. If the user asks anything unrelated to this paper's content, methodology, findings, or contributions, respond: 'I can only answer questions about the uploaded research paper.'
2. If the user asks anything unrelated to the paper (coding help, general knowledge, other topics), respond with: "I can only answer questions about the uploaded research paper. Please ask something related to its content, methodology, findings, or contributions."
3. Never generate code, tutorials, or explanations unrelated to the paper's content.
4. If the knowledge graph context is empty or insufficient, say so clearly and ask the user to rephrase their question about the paper.
5. Always ground your answer in the provided context. Do not hallucinate facts not present in the graph or semantic results.

KNOWLEDGE GRAPH CONTEXT (entities and relationships from the paper):
{graph_context}

SEMANTIC SEARCH RESULTS (most relevant passages from the paper):
{vector_context}

ENTITY RESOLUTION RESULTS:
{entity_context}

USER QUESTION: {query}

Answer using ONLY the paper context above. Be specific, cite entity names and relationship types from the graph. If the question is not about this paper, apply Rule 1.
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
    response_type: str = Field(default='paper_answer')
    prompts: list[dict] = Field(default_factory=list)
    domain: str = Field(default='')


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
    """
    vector_result: VectorResult = results.get("vector") or VectorResult(chunks=[], scores=[], total=0)
    graph_result: GraphResult = results.get("graph") or GraphResult(anchor_nodes=[], nodes=[], edges=[], paths=[], entity_ids_found=[], traversal_depth=0)
    entity_result: EntityResult = results.get("entities") or EntityResult(resolved_ids=[], expanded_terms=[], matched_nodes=[])

    if isinstance(vector_result, Exception):
        vector_result = VectorResult(chunks=[], scores=[], total=0, error=str(vector_result))
    if isinstance(graph_result, Exception):
        graph_result = GraphResult(anchor_nodes=[], nodes=[], edges=[], paths=[], entity_ids_found=[], traversal_depth=0, error=str(graph_result))
    if isinstance(entity_result, Exception):
        entity_result = EntityResult(resolved_ids=[], expanded_terms=[], matched_nodes=[], error=str(entity_result))

    ranked: list[dict] = []
    seen_ids: set[str] = set()

    # 1. Vector chunks
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

    # 2. Graph nodes
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

    top = ranked[:15]

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
    """
    if (
        (not graph_context or graph_context.strip() == "(No graph context available.)")
        and (not vector_context or "No relevant entities" in vector_context)
    ):
        logger.warning("Synthesizer: both contexts empty for query=%r — returning guardrail response", query[:60])
        return (
            "I don't have enough context from the paper to answer this question. "
            "Please try rephrasing your question about the paper's content, methods, or findings."
        )

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
# Targeted Retrieval Pipeline
# ---------------------------------------------------------------------------

async def run_targeted_retrieval(
    query: str,
    doc_id: str,
    driver: Any,
    settings: Settings,
) -> tuple[str, list[dict], list[dict]]:
    """
    Executes the targeted retrieval pipeline:
      1. Vector search filtered by doc_id (top 10 results).
      2. Extract unique candidate entity names.
      3. Neo4j full-text search scoped to doc_id for up to 3 names (top 5 nodes per name).
      4. 2-hop Neo4j traversal from these anchors (max 20 nodes, max 40 edges).
      5. Rank nodes by a combined score: semantic similarity + degree centrality in traversal graph.
      6. Compress to top 15 context nodes and their relationships.
    """
    try:
        # Step 1 & 2: Vector search and name extraction
        vector_result = await run_vector_agent(query, doc_id, top_k=10, settings=settings)
        if vector_result.error:
            logger.warning("Targeted retrieval: Vector search error: %s", vector_result.error)
        
        entity_names: list[str] = []
        seen_names: set[str] = set()
        # Keep track of vector scores for ranking
        vector_scores: dict[str, float] = {}
        for chunk in vector_result.chunks:
            name = chunk.get("name")
            eid = chunk.get("entity_id")
            if name and name.lower() not in seen_names:
                seen_names.add(name.lower())
                entity_names.append(name)
            if eid:
                vector_scores[eid] = chunk.get("score", 0.0)

        # Step 3: Scoped Neo4j full-text search (top 3 terms, top 5 nodes per term)
        search_terms = entity_names[:3]
        anchor_node_ids: list[str] = []
        anchor_nodes_dict: dict[str, dict] = {}
        ft_scores: dict[str, float] = {}
        
        for term in search_terms:
            nodes = await search_nodes_by_label(driver, label="", doc_id=doc_id, search_term=term)
            for node in nodes[:5]:
                eid = node.get("entity_id")
                if eid:
                    anchor_node_ids.append(eid)
                    anchor_nodes_dict[eid] = node
                    # Keep full-text score
                    ft_scores[eid] = node.get("_score", 0.0)

        # Fallback: if no anchor nodes, retrieve up to 5 sample nodes for doc_id
        if not anchor_node_ids:
            from backend.graph.neo4j_queries import get_subgraph
            fallback = await get_subgraph(driver, doc_id=doc_id, max_nodes=5)
            for node in fallback.get("nodes", [])[:5]:
                eid = node.get("id")
                if eid:
                    anchor_node_ids.append(eid)
                    anchor_nodes_dict[eid] = {
                        "entity_id": eid,
                        "name": node.get("name"),
                        "type": node.get("type"),
                        "description": node.get("description"),
                        "_labels": [node.get("type")]
                    }

        # Step 4: 2-hop traversal capped at 20 nodes, 40 edges
        subgraph = {"nodes": [], "edges": []}
        if anchor_node_ids:
            subgraph = await traverse_from_entities(driver, anchor_node_ids, doc_id, depth=2)
        
        nodes: list[dict] = subgraph.get("nodes", [])[:20]
        edges: list[dict] = subgraph.get("edges", [])[:40]

        if not nodes:
            # If traversal returns nothing, just use the anchor nodes
            nodes = list(anchor_nodes_dict.values())
            edges = []

        # Step 5: Score and rank by semantic similarity + degree centrality
        # Degree centrality count
        node_degrees: dict[str, int] = {}
        for n in nodes:
            n_eid = n.get("_element_id") or n.get("entity_id")
            node_degrees[n_eid] = 0
            for e in edges:
                if e.get("_start_node_id") == n_eid or e.get("_end_node_id") == n_eid:
                    node_degrees[n_eid] += 1

        # Combine scores
        # Formula: Final = max(vector_score, ft_score / 10) + 0.5 * degree
        scored_nodes: list[tuple[dict, float]] = []
        for n in nodes:
            eid = n.get("entity_id")
            n_eid = n.get("_element_id") or eid
            degree = node_degrees.get(n_eid, 0)
            
            v_score = vector_scores.get(eid, 0.0)
            ft_score = ft_scores.get(eid, 0.0) / 10.0
            semantic_score = max(v_score, ft_score)
            
            final_score = semantic_score + 0.5 * degree
            scored_nodes.append((n, final_score))

        # Sort and get top 15
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        top_15_nodes = [item[0] for item in scored_nodes[:15]]
        top_15_ids = {n.get("entity_id") for n in top_15_nodes if n.get("entity_id")}
        top_15_element_ids = {n.get("_element_id") for n in top_15_nodes if n.get("_element_id")}

        # Filter edges to only include relationships between top 15 nodes
        filtered_edges = []
        for e in edges:
            start = e.get("_start_node_id")
            end = e.get("_end_node_id")
            if start in top_15_element_ids and end in top_15_element_ids:
                filtered_edges.append(e)

        # Format context text
        context_lines: list[str] = []
        for edge in filtered_edges:
            # Try to resolve names
            s_node = next((n for n in top_15_nodes if n.get("_element_id") == edge.get("_start_node_id")), None)
            t_node = next((n for n in top_15_nodes if n.get("_element_id") == edge.get("_end_node_id")), None)
            s_name = s_node.get("name") if s_node else "Unknown"
            t_name = t_node.get("name") if t_node else "Unknown"
            context_lines.append(
                f"{s_name} --[{edge.get('_type', 'REL')}]--> {t_name}"
            )
            
        for node in top_15_nodes:
            desc = node.get("description")
            if desc:
                labels = node.get("_labels") or [node.get("type", "Entity")]
                context_lines.append(
                    f"{node.get('name')} ({labels[0]}): {desc}"
                )

        context_text = "\n".join(context_lines) if context_lines else "(No graph context found.)"
        return context_text, top_15_nodes, filtered_edges

    except Exception as exc:
        logger.error("Targeted retrieval pipeline error: %s", exc, exc_info=True)
        return "(No graph context found due to pipeline error.)", [], []


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


async def run_pipeline(
    query: str,
    doc_id: str,
    driver: Any,
    settings: Settings,
    include_trace: bool = False,
    domain: str = "general",
) -> QueryResponse:
    """Run the full retrieval-augmented generation pipeline.
    """
    intent = classify_intent(query)
    if intent == 'PROMPT_ARCHITECT':
        pa_response = await generate_prompt_architect_response(query, settings)
        return QueryResponse(
            answer='',
            sources=[],
            trace={},
            query_id=pa_response.query_id,
            total_nodes_retrieved=0,
            error=None,
            response_type='prompt_architect',
            prompts=[p.model_dump() for p in pa_response.prompts],
            domain=pa_response.domain,
        )

    query_id = str(uuid.uuid4())

    # Step 0: Targeted retrieval graph context scoped to doc_id
    graph_context_text, top_nodes, filtered_edges = await run_targeted_retrieval(
        query=query,
        doc_id=doc_id,
        driver=driver,
        settings=settings,
    )

    # Step 1: Run all three agents concurrently (collecting vector/entity details)
    vector_coro = run_vector_agent(
        query=query,
        doc_id=doc_id,
        top_k=10,
        settings=settings,
    )
    graph_coro = run_graph_agent(
        query=query,
        doc_id=doc_id,
        driver=driver,
        settings=settings,
    )
    entity_coro = resolve_entities(
        query=query,
        doc_id=doc_id,
        driver=driver,
        settings=settings,
    )

    raw = await asyncio.gather(vector_coro, graph_coro, entity_coro, return_exceptions=True)
    vector_raw, graph_raw, entity_raw = raw

    if isinstance(vector_raw, Exception):
        vector_result = VectorResult(chunks=[], scores=[], total=0, error=str(vector_raw))
    else:
        vector_result: VectorResult = vector_raw

    if isinstance(graph_raw, Exception):
        graph_result = GraphResult(
            anchor_nodes=[], nodes=[], edges=[], paths=[],
            entity_ids_found=[], traversal_depth=0, error=str(graph_raw),
        )
    else:
        graph_result: GraphResult = graph_raw

    if isinstance(entity_raw, Exception):
        entity_result = EntityResult(
            resolved_ids=[], expanded_terms=[], matched_nodes=[], error=str(entity_raw)
        )
    else:
        entity_result: EntityResult = entity_raw

    # Merge and rank agent context (for semantic results section)
    vector_context, sources = merge_and_rank_context(
        {
            "vector": vector_result,
            "graph": graph_result,
            "entities": entity_result,
        }
    )

    entity_context_lines: list[str] = []
    if not entity_result.error:
        for node in entity_result.matched_nodes:
            name = node.get("name", "")
            etype = node.get("type", node.get("label", "Unknown"))
            desc = node.get("description", "")
            if name:
                entity_context_lines.append(f"- {name} ({etype}): {desc}" if desc else f"- {name} ({etype})")
    entity_context = "\n".join(entity_context_lines) if entity_context_lines else "(No entities resolved.)"

    # Step 3: Generate LLM answer with structured prompt
    answer = await synthesize_answer(
        query=query,
        graph_context=graph_context_text,
        vector_context=vector_context,
        entity_context=entity_context,
        settings=settings,
    )

    trace: dict = {}
    if include_trace:
        trace = {
            "vector_agent": vector_result.model_dump(),
            "graph_agent": graph_result.model_dump(),
            "entity_resolver": entity_result.model_dump(),
            "targeted_graph_context": {
                "node_count": len(top_nodes),
                "edge_count": len(filtered_edges),
            },
        }

    total_nodes = (
        vector_result.total
        + len(graph_result.nodes)
        + len(entity_result.resolved_ids)
    )

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
