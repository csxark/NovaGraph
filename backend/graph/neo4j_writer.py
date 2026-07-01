"""
backend/graph/neo4j_writer.py

Neo4j write operations for the Graphora Research Assistant.
All Cypher queries use $parameter syntax — zero string interpolation of user data.
Dynamic labels/rel-types come exclusively from Pydantic Literal-validated fields.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from neo4j import AsyncDriver

from backend.extractor.entity_extractor import EdgeModel, NodeModel

logger = logging.getLogger(__name__)

_BATCH_SIZE = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _node_props(node: NodeModel, doc_id: str) -> dict[str, Any]:
    """Return a plain dict of all serialisable node properties."""
    return {
        "entity_id": node.entity_id,
        "name": node.name,
        "description": node.description,
        "subtype": node.subtype or "",
        "domains": node.domains or [],
        "confidence": node.confidence,
        "chunk_ref": node.chunk_ref or "",
        "doc_id": doc_id,
        "created_at": _now_iso(),
    }


# ---------------------------------------------------------------------------
# Paper-level helpers
# ---------------------------------------------------------------------------

async def create_paper_node(
    driver: AsyncDriver,
    doc_id: str,
    title: str,
    filename: str,
    page_count: int,
    domains: list[str],
) -> None:
    """MERGE a Paper node and set all metadata properties."""
    cypher = (
        "MERGE (p:Paper {doc_id: $doc_id}) "
        "SET p += {title: $title, filename: $filename, page_count: $page_count, "
        "domains: $domains, created_at: $created_at, status: $status}"
    )
    params: dict[str, Any] = {
        "doc_id": doc_id,
        "title": title,
        "filename": filename,
        "page_count": page_count,
        "domains": domains,
        "created_at": _now_iso(),
        "status": "pending",
    }
    await driver.execute_query(cypher, params)
    logger.debug("Paper node created/merged: %s", doc_id)


async def update_paper_status(driver: AsyncDriver, doc_id: str, status: str) -> None:
    """MERGE Paper node and update its status field."""
    cypher = (
        "MERGE (p:Paper {doc_id: $doc_id}) "
        "SET p.status = $status, p.updated_at = $updated_at"
    )
    params: dict[str, Any] = {
        "doc_id": doc_id,
        "status": status,
        "updated_at": _now_iso(),
    }
    await driver.execute_query(cypher, params)
    logger.debug("Paper %s status → %s", doc_id, status)


# ---------------------------------------------------------------------------
# Single-node / single-edge writes
# ---------------------------------------------------------------------------

async def create_node(driver: AsyncDriver, node: NodeModel, doc_id: str) -> None:
    """
    MERGE a typed node and link it to its Paper.

    ``node.type`` is a Pydantic Literal-validated string.
    All other data is passed via parameters.
    """
    label = node.type
    props = _node_props(node, doc_id)

    # MERGE the typed node
    node_cypher = (
        f"MERGE (n:{label} {{entity_id: $entity_id}}) "
        "SET n += $props"
    )
    await driver.execute_query(
        node_cypher,
        {"entity_id": node.entity_id, "props": props},
    )

    # Link to Paper
    link_cypher = (
        "MATCH (p:Paper {doc_id: $doc_id}) "
        f"MATCH (n:{label} {{entity_id: $entity_id}}) "
        "MERGE (p)-[:CONTAINS]->(n)"
    )
    await driver.execute_query(
        link_cypher,
        {"doc_id": doc_id, "entity_id": node.entity_id},
    )
    logger.debug("Node created: %s (%s)", node.entity_id, label)


async def create_edge(
    driver: AsyncDriver,
    edge: EdgeModel,
    source_id: str,
    target_id: str,
) -> None:
    """
    MERGE a directed relationship between two nodes.

    ``edge.type`` is a Pydantic Literal-validated string.
    """
    rel_type = edge.type
    props: dict[str, Any] = {
        "weight": edge.weight,
        "evidence": edge.evidence or "",
        "doc_id": edge.paper_id,  # Uses the doc_id passed as paper_id inside the EdgeModel
        "created_at": _now_iso(),
    }
    cypher = (
        f"MATCH (a {{entity_id: $sid}}) "
        f"MATCH (b {{entity_id: $tid}}) "
        f"MERGE (a)-[r:{rel_type}]->(b) "
        "SET r += $props"
    )
    await driver.execute_query(
        cypher,
        {"sid": source_id, "tid": target_id, "props": props},
    )
    logger.debug("Edge created: %s -[%s]-> %s", source_id, rel_type, target_id)


# ---------------------------------------------------------------------------
# Batch writes
# ---------------------------------------------------------------------------

async def _create_paper_anchor(driver: AsyncDriver, doc_id: str, filename: str = "") -> None:
    """Create or update the Paper anchor node for this doc_id."""
    query = """
        MERGE (p:Paper {doc_id: $doc_id})
        ON CREATE SET p.created_at = timestamp(), p.filename = $filename
        ON MATCH SET p.updated_at = timestamp()
    """
    async with driver.session() as session:
        await session.run(query, doc_id=doc_id, filename=filename)


async def batch_write_nodes(
    driver: AsyncDriver,
    nodes: list[NodeModel],
    doc_id: str,
) -> int:
    """
    Write nodes in batches of 50 using UNWIND, grouped by node type.
    Returns the total number of nodes written.
    """
    await _create_paper_anchor(driver, doc_id)

    if not nodes:
        return 0

    # Group by type so we can use a single label per UNWIND batch
    by_type: dict[str, list[NodeModel]] = {}
    for node in nodes:
        by_type.setdefault(node.type, []).append(node)

    total = 0
    for label, typed_nodes in by_type.items():
        node_cypher = (
            f"UNWIND $batch AS row "
            f"MERGE (n:{label} {{entity_id: row.entity_id}}) "
            "SET n += row.props"
        )
        link_cypher = (
            "UNWIND $batch AS row "
            "MATCH (p:Paper {doc_id: $doc_id}) "
            f"MATCH (n:{label} {{entity_id: row.entity_id}}) "
            "MERGE (p)-[:CONTAINS]->(n)"
        )

        # Chunk into batches of _BATCH_SIZE
        for i in range(0, len(typed_nodes), _BATCH_SIZE):
            chunk = typed_nodes[i : i + _BATCH_SIZE]
            batch_data = [
                {"entity_id": n.entity_id, "props": _node_props(n, doc_id)}
                for n in chunk
            ]
            async with driver.session() as session:
                await session.run(node_cypher, {"batch": batch_data})
                await session.run(link_cypher, {"batch": batch_data, "doc_id": doc_id})
            total += len(chunk)
            logger.debug(
                "Wrote batch of %d %s nodes (total so far: %d)", len(chunk), label, total
            )

    logger.info("batch_write_nodes complete: %d nodes written for doc_id %s", total, doc_id)
    return total


async def batch_write_edges(
    driver: AsyncDriver,
    edges: list[EdgeModel],
    name_to_id: dict[str, str],
    doc_id: str,
) -> int:
    """
    Resolve source/target names → entity_ids, then write edges in batches of 50.
    Skips any edge where either endpoint cannot be resolved.
    Returns the total number of edges written.
    """
    if not edges:
        return 0

    # Resolve endpoints
    resolved: list[tuple[EdgeModel, str, str]] = []
    for edge in edges:
        source_id = name_to_id.get(edge.source_name)
        target_id = name_to_id.get(edge.target_name)
        if source_id is None:
            logger.warning(
                "Skipping edge — source '%s' not in name_to_id map", edge.source_name
            )
            continue
        if target_id is None:
            logger.warning(
                "Skipping edge — target '%s' not in name_to_id map", edge.target_name
            )
            continue
        resolved.append((edge, source_id, target_id))

    if not resolved:
        logger.info("No resolvable edges to write for doc_id %s", doc_id)
        return 0

    # Group by rel type so we can keep labels in the Cypher template
    by_type: dict[str, list[tuple[EdgeModel, str, str]]] = {}
    for item in resolved:
        by_type.setdefault(item[0].type, []).append(item)

    total = 0
    for rel_type, items in by_type.items():
        cypher = (
            f"UNWIND $batch AS row "
            f"MATCH (a {{entity_id: row.sid}}) "
            f"MATCH (b {{entity_id: row.tid}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "SET r += row.props"
        )
        for i in range(0, len(items), _BATCH_SIZE):
            chunk = items[i : i + _BATCH_SIZE]
            batch_data = [
                {
                    "sid": sid,
                    "tid": tid,
                    "props": {
                        "weight": e.weight,
                        "evidence": e.evidence or "",
                        "doc_id": doc_id,
                        "created_at": _now_iso(),
                    },
                }
                for e, sid, tid in chunk
            ]
            async with driver.session() as session:
                await session.run(cypher, {"batch": batch_data})
            total += len(chunk)
            logger.debug(
                "Wrote batch of %d %s edges (total so far: %d)", len(chunk), rel_type, total
            )

    logger.info("batch_write_edges complete: %d edges written for doc_id %s", total, doc_id)
    return total


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

async def delete_paper_graph(driver: AsyncDriver, doc_id: str) -> None:
    """Remove all nodes tagged with doc_id and the Paper node itself."""
    # Delete all content nodes for this paper
    content_cypher = "MATCH (n {doc_id: $doc_id}) DETACH DELETE n"
    await driver.execute_query(content_cypher, {"doc_id": doc_id})

    # Delete the Paper node itself
    paper_cypher = "MATCH (p:Paper {doc_id: $doc_id}) DETACH DELETE p"
    await driver.execute_query(paper_cypher, {"doc_id": doc_id})

    logger.info("Deleted graph for doc_id %s", doc_id)
