"""
backend/graph/neo4j_queries.py

Read-only Neo4j query functions for the GraphRAG Research Assistant.
All queries use $parameter syntax — zero string interpolation of user data.
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
    paper_id: str,
    max_nodes: int = 200,
    depth: int = 2,
) -> dict[str, Any]:
    """
    Return a subgraph dict with nodes/edges reachable from a Paper via CONTAINS.

    depth is capped internally; max_nodes limits result size.
    Returns: {nodes, edges, node_count, edge_count, truncated}
    """
    depth = min(depth, _MAX_DEPTH)
    max_nodes = min(max_nodes, 500)

    # Variable-length CONTAINS path from Paper, then collect inter-node edges
    cypher = (
        "MATCH (p:Paper {paper_id: $paper_id})-[:CONTAINS*1..$depth]->(n) "
        "WITH n LIMIT $max_nodes "
        "OPTIONAL MATCH (n)-[r]-(m) WHERE m.paper_id = $paper_id "
        "RETURN n, r, m"
    )
    params: dict[str, Any] = {
        "paper_id": paper_id,
        "depth": depth,
        "max_nodes": max_nodes,
    }

    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    records, _, _ = await driver.execute_query(cypher, params)
    for record in records:
        n_obj = record["n"]
        r_obj = record["r"]
        m_obj = record["m"]

        if n_obj is not None and n_obj.element_id not in nodes:
            nodes[n_obj.element_id] = _record_to_node(n_obj)

        if m_obj is not None and m_obj.element_id not in nodes:
            nodes[m_obj.element_id] = _record_to_node(m_obj)

        if r_obj is not None:
            edges.append(_record_to_rel(r_obj))

    node_list = list(nodes.values())
    return {
        "nodes": node_list,
        "edges": edges,
        "node_count": len(node_list),
        "edge_count": len(edges),
        "truncated": len(node_list) >= max_nodes,
    }


# ---------------------------------------------------------------------------
# Entity traversal
# ---------------------------------------------------------------------------

async def traverse_from_entities(
    driver: AsyncDriver,
    entity_ids: list[str],
    depth: int = 2,
) -> dict[str, Any]:
    """
    Return a subgraph reachable within *depth* hops from the given entity_ids.

    depth is capped at _MAX_DEPTH (3).  Because depth is a bounded integer
    (never user-supplied free text), it is safe to embed in the Cypher template
    for variable-length path syntax which does not support runtime parameters.
    """
    depth = min(max(depth, 1), _MAX_DEPTH)

    # Variable-length path — depth must be a literal integer in Cypher syntax
    cypher = (
        f"MATCH (start) WHERE start.entity_id IN $ids "
        f"MATCH p = (start)-[r*1..{depth}]-(neighbor) "
        "RETURN nodes(p) AS path_nodes, relationships(p) AS path_rels"
    )
    params: dict[str, Any] = {"ids": entity_ids}

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
    paper_id: str,
    search_term: str,
) -> list[dict[str, Any]]:
    """
    Full-text search via the ft_name_desc index, filtered by paper_id.
    *label* is accepted as a parameter but filtering is done via paper_id
    and node properties — the label is NOT interpolated into Cypher.
    """
    cypher = (
        "CALL db.index.fulltext.queryNodes('ft_name_desc', $term) "
        "YIELD node, score "
        "WHERE node.paper_id = $paper_id "
        "RETURN node, score "
        "ORDER BY score DESC "
        "LIMIT 20"
    )
    params: dict[str, Any] = {"term": search_term, "paper_id": paper_id}

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
# Lookup by entity_ids
# ---------------------------------------------------------------------------

async def find_nodes_by_entity_ids(
    driver: AsyncDriver,
    entity_ids: list[str],
) -> list[dict[str, Any]]:
    """MATCH any node whose entity_id is in the given list."""
    cypher = "MATCH (n) WHERE n.entity_id IN $ids RETURN n"
    records, _, _ = await driver.execute_query(cypher, {"ids": entity_ids})
    return [_record_to_node(record["n"]) for record in records]


# ---------------------------------------------------------------------------
# Paper helpers
# ---------------------------------------------------------------------------

async def paper_exists(driver: AsyncDriver, paper_id: str) -> bool:
    """Return True if any node with this paper_id exists in the graph."""
    query = """
        MATCH (n {paper_id: $paper_id})
        RETURN count(n) > 0 AS exists
        LIMIT 1
    """
    async with driver.session() as session:
        result = await session.run(query, paper_id=paper_id)
        record = await result.single()
        if record is None:
            return False
        return bool(record["exists"])


async def get_paper_node_count(driver: AsyncDriver, paper_id: str) -> int:
    """Return the number of nodes tagged with paper_id."""
    cypher = "MATCH (n {paper_id: $paper_id}) RETURN count(n) AS cnt"
    records, _, _ = await driver.execute_query(cypher, {"paper_id": paper_id})
    if records:
        return int(records[0]["cnt"])
    return 0


# ---------------------------------------------------------------------------
# Similarity by name
# ---------------------------------------------------------------------------

async def find_similar_nodes_by_name(
    driver: AsyncDriver,
    name: str,
    paper_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Full-text search by name across all nodes.

    If paper_id is provided the results are filtered to that paper.
    Returns up to 10 results ordered by score.
    """
    if paper_id is not None:
        cypher = (
            "CALL db.index.fulltext.queryNodes('ft_name_desc', $name) "
            "YIELD node, score "
            "WHERE node.paper_id = $paper_id "
            "RETURN node, score "
            "ORDER BY score DESC "
            "LIMIT 10"
        )
        params: dict[str, Any] = {"name": name, "paper_id": paper_id}
    else:
        cypher = (
            "CALL db.index.fulltext.queryNodes('ft_name_desc', $name) "
            "YIELD node, score "
            "RETURN node, score "
            "ORDER BY score DESC "
            "LIMIT 10"
        )
        params = {"name": name}

    records, _, _ = await driver.execute_query(cypher, params)
    results: list[dict[str, Any]] = []
    for record in records:
        node_dict = _record_to_node(record["node"])
        node_dict["_score"] = record["score"]
        results.append(node_dict)
    return results
