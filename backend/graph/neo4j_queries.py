"""
backend/graph/neo4j_queries.py

Read-only Neo4j query functions for the Graphora Research Assistant.
All queries use $parameter syntax — zero string interpolation of user data.
Every query is scoped to a specific doc_id — no cross-paper data leakage.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from neo4j import AsyncDriver

logger = logging.getLogger(__name__)

_MAX_DEPTH = 3  # hard cap for traversal depth


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record_to_node(node_obj: Any) -> dict[str, Any]:
    """Convert a neo4j Node object to a plain dict including labels."""
    props = dict(node_obj.items())
    props["_labels"] = list(node_obj.labels)
    props["_element_id"] = node_obj.element_id
    return props


def _record_to_rel(rel_obj: Any) -> dict[str, Any]:
    """Convert a neo4j Relationship object to a plain dict."""
    props = dict(rel_obj.items())
    props["_type"] = rel_obj.type
    props["_start_node_id"] = rel_obj.start_node.element_id
    props["_end_node_id"] = rel_obj.end_node.element_id
    return props


# ---------------------------------------------------------------------------
# Subgraph
# ---------------------------------------------------------------------------

async def get_subgraph(
    driver: AsyncDriver,
    doc_id: str,
    max_nodes: int = 200,
) -> dict[str, Any]:
    """
    Return a subgraph dict with nodes/edges reachable for the doc_id.
    """
    query = """
        MATCH (n {doc_id: $doc_id})
        OPTIONAL MATCH (n)-[r]->(m {doc_id: $doc_id})
        RETURN
            n.entity_id AS source_id,
            n.name AS source_name,
            labels(n)[0] AS source_type,
            n.description AS source_desc,
            n.domains AS source_domains,
            type(r) AS rel_type,
            r.evidence AS rel_evidence,
            m.entity_id AS target_id,
            m.name AS target_name,
            labels(m)[0] AS target_type
        LIMIT $limit
    """
    async with driver.session() as session:
        result = await session.run(query, doc_id=doc_id, limit=max_nodes)
        records = await result.data()

    nodes = {}
    edges = []

    for rec in records:
        if rec["source_id"] and rec["source_name"]:
            nodes[rec["source_id"]] = {
                "id": rec["source_id"],
                "name": rec["source_name"],
                "type": rec["source_type"] or "Entity",
                "description": rec["source_desc"] or "",
                "domains": rec["source_domains"] or [],
            }
        if rec["target_id"] and rec["target_name"]:
            nodes[rec["target_id"]] = {
                "id": rec["target_id"],
                "name": rec["target_name"],
                "type": rec["target_type"] or "Entity",
                "description": "",
                "domains": [],
            }
        if rec["rel_type"] and rec["source_id"] and rec["target_id"]:
            edges.append({
                "source": rec["source_id"],
                "target": rec["target_id"],
                "type": rec["rel_type"],
                "evidence": rec["rel_evidence"] or "",
            })

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "doc_id": doc_id,
    }


# ---------------------------------------------------------------------------
# Entity traversal (doc_id-scoped)
# ---------------------------------------------------------------------------

async def traverse_from_entities(
    driver: AsyncDriver,
    entity_ids: list[str],
    doc_id: str,
    depth: int = 2,
) -> dict[str, Any]:
    """
    Return a subgraph reachable within *depth* hops from the given entity_ids,
    strictly scoped to *doc_id*.
    """
    depth = min(max(depth, 1), _MAX_DEPTH)

    cypher = (
        f"MATCH (start) WHERE start.entity_id IN $ids AND start.doc_id = $doc_id "
        f"MATCH p = (start)-[r*1..{depth}]-(neighbor) "
        "WHERE neighbor.doc_id = $doc_id "
        "RETURN nodes(p) AS path_nodes, relationships(p) AS path_rels"
    )
    params: dict[str, Any] = {"ids": entity_ids, "doc_id": doc_id}

    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    records, _, _ = await driver.execute_query(cypher, params)
    for record in records:
        for node_obj in record["path_nodes"]:
            if node_obj.element_id not in nodes:
                nodes[node_obj.element_id] = _record_to_node(node_obj)
        for rel_obj in record["path_rels"]:
            edges.append(_record_to_rel(rel_obj))

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# Full-text search
# ---------------------------------------------------------------------------

async def search_nodes_by_label(
    driver: AsyncDriver,
    label: str,
    doc_id: str,
    search_term: str,
) -> list[dict[str, Any]]:
    """
    Full-text search via the ft_name_desc index, filtered by doc_id.
    """
    cypher = (
        "CALL db.index.fulltext.queryNodes('ft_name_desc', $term) "
        "YIELD node, score "
        "WHERE node.doc_id = $doc_id "
        "RETURN node, score "
        "ORDER BY score DESC "
        "LIMIT 20"
    )
    params: dict[str, Any] = {"term": search_term, "doc_id": doc_id}

    records, _, _ = await driver.execute_query(cypher, params)
    results: list[dict[str, Any]] = []
    for record in records:
        node_dict = _record_to_node(record["node"])
        node_dict["_score"] = record["score"]
        # Post-filter by label if caller requested a specific type
        if label and label not in node_dict.get("_labels", []):
            continue
        results.append(node_dict)
    return results


# ---------------------------------------------------------------------------
# Lookup by entity_ids (doc_id-scoped)
# ---------------------------------------------------------------------------

async def find_nodes_by_entity_ids(
    driver: AsyncDriver,
    entity_ids: list[str],
    doc_id: str,
) -> list[dict[str, Any]]:
    """MATCH any node whose entity_id is in the given list, scoped to doc_id."""
    cypher = (
        "MATCH (n) WHERE n.entity_id IN $ids AND n.doc_id = $doc_id "
        "RETURN n"
    )
    records, _, _ = await driver.execute_query(
        cypher, {"ids": entity_ids, "doc_id": doc_id}
    )
    return [_record_to_node(record["n"]) for record in records]


# ---------------------------------------------------------------------------
# Paper/Document helpers
# ---------------------------------------------------------------------------

async def document_exists(driver: AsyncDriver, doc_id: str) -> bool:
    """Check if the document exists in the graph database."""
    query = """
        MATCH (p:Paper {doc_id: $doc_id})
        RETURN count(p) > 0 AS exists
        LIMIT 1
    """
    async with driver.session() as session:
        result = await session.run(query, doc_id=doc_id)
        record = await result.single()
        return bool(record["exists"]) if record else False


async def get_document_node_count(driver: AsyncDriver, doc_id: str) -> int:
    """Return the number of nodes tagged with doc_id."""
    cypher = "MATCH (n {doc_id: $doc_id}) RETURN count(n) AS cnt"
    records, _, _ = await driver.execute_query(cypher, {"doc_id": doc_id})
    if records:
        return int(records[0]["cnt"])
    return 0


# ---------------------------------------------------------------------------
# Similarity by name (doc_id always required)
# ---------------------------------------------------------------------------

async def find_similar_nodes_by_name(
    driver: AsyncDriver,
    name: str,
    doc_id: str,
) -> list[dict[str, Any]]:
    """
    Full-text search by name, always scoped to *doc_id*.

    Returns up to 10 results ordered by score.
    """
    cypher = (
        "CALL db.index.fulltext.queryNodes('ft_name_desc', $name) "
        "YIELD node, score "
        "WHERE node.doc_id = $doc_id "
        "RETURN node, score "
        "ORDER BY score DESC "
        "LIMIT 10"
    )
    params: dict[str, Any] = {"name": name, "doc_id": doc_id}

    records, _, _ = await driver.execute_query(cypher, params)
    results: list[dict[str, Any]] = []
    for record in records:
        node_dict = _record_to_node(record["node"])
        node_dict["_score"] = record["score"]
        results.append(node_dict)
    return results


# ---------------------------------------------------------------------------
# Full knowledge-graph context for a paper (DEPRECATED: replaced by targeted retrieval)
# ---------------------------------------------------------------------------

async def get_full_graph_context(
    driver: AsyncDriver,
    doc_id: str,
    max_nodes: int = 150,
) -> dict:
    """
    Retrieve the complete knowledge graph for a paper as structured context.
    (Kept for compatibility or trace fallback, but Synthesizer now uses targeted retrieval).
    """
    query = """
        MATCH (n {doc_id: $doc_id})
        OPTIONAL MATCH (n)-[r]->(m {doc_id: $doc_id})
        RETURN
            n.name AS source_name,
            labels(n)[0] AS source_type,
            n.description AS source_desc,
            type(r) AS rel_type,
            m.name AS target_name,
            labels(m)[0] AS target_type,
            m.description AS target_desc
        LIMIT $limit
    """
    async with driver.session() as session:
        result = await session.run(query, doc_id=doc_id, limit=max_nodes)
        records = await result.data()

    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    for rec in records:
        if rec["source_name"]:
            nodes[rec["source_name"]] = {
                "name": rec["source_name"],
                "type": rec["source_type"],
                "description": rec["source_desc"],
            }
        if rec["target_name"]:
            nodes[rec["target_name"]] = {
                "name": rec["target_name"],
                "type": rec["target_type"],
                "description": rec["target_desc"],
            }
        if rec["rel_type"] and rec["source_name"] and rec["target_name"]:
            edges.append({
                "source": rec["source_name"],
                "relation": rec["rel_type"],
                "target": rec["target_name"],
            })

    context_lines: list[str] = []
    for edge in edges:
        context_lines.append(
            f"{edge['source']} --[{edge['relation']}]--> {edge['target']}"
        )
    for node in nodes.values():
        if node["description"]:
            context_lines.append(
                f"{node['name']} ({node['type']}): {node['description']}"
            )

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "context_text": "\n".join(context_lines),
        "node_count": len(nodes),
        "edge_count": len(edges),
    }
