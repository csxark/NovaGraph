"""
Session cleanup — wipes Neo4j nodes and Pinecone vectors for a paper_id.
Called when the user resets the session or explicitly deletes a paper.
"""

from __future__ import annotations

import logging
from backend.config import Settings

logger = logging.getLogger(__name__)


async def delete_paper_data(
    paper_id: str,
    driver,
    settings: Settings,
) -> dict:
    """
    Delete all Neo4j nodes and Pinecone vectors associated with *paper_id*.

    1. Delete ALL Neo4j nodes where paper_id = $paper_id using DETACH DELETE.
    2. Delete the entire Pinecone namespace for paper_id using index.delete(delete_all=True, namespace=paper_id).
    3. Return a result dict with counts of what was deleted.
    4. Log every step — success and failure both.
    """
    logger.info("Starting deletion of paper data for paper_id=%s", paper_id)
    neo4j_deleted = 0
    pinecone_deleted = False

    # 1. Neo4j - DETACH DELETE
    try:
        async with driver.session() as session:
            res1 = await session.run(
                "MATCH (n {paper_id: $paper_id}) DETACH DELETE n",
                paper_id=paper_id,
            )
            summary1 = await res1.consume()
            del1 = summary1.counters.nodes_deleted
            
            res2 = await session.run(
                "MATCH (p:Paper {paper_id: $paper_id}) DETACH DELETE p",
                paper_id=paper_id,
            )
            summary2 = await res2.consume()
            del2 = summary2.counters.nodes_deleted
            
            neo4j_deleted = del1 + del2
            logger.info("Successfully deleted %d Neo4j nodes/papers for paper_id=%s", neo4j_deleted, paper_id)
    except Exception as exc:
        logger.error("Failed to delete Neo4j nodes for paper_id=%s: %s", paper_id, exc)

    # 2. Pinecone - delete namespace
    try:
        from backend.vector.pinecone_store import get_index
        index = get_index(settings)
        index.delete(delete_all=True, namespace=paper_id)
        pinecone_deleted = True
        logger.info("Deleted Pinecone namespace for paper_id=%s", paper_id)
    except Exception as exc:
        if '404' in str(exc) or 'Namespace not found' in str(exc):
            logger.debug("Pinecone namespace '%s' did not exist — nothing to delete", paper_id)
            pinecone_deleted = True
        else:
            logger.error("Failed to delete Pinecone namespace '%s': %s", paper_id, exc)

    return {
        "paper_id": paper_id,
        "neo4j_nodes_deleted": neo4j_deleted,
        "pinecone_namespace_deleted": pinecone_deleted,
    }


async def delete_all_papers(driver, settings: Settings) -> dict:
    """
    Delete all papers.
    1. Queries Neo4j for all distinct paper_id values: MATCH (n) WHERE n.paper_id IS NOT NULL RETURN DISTINCT n.paper_id AS pid
    2. Calls delete_paper_data for each one
    3. Clears the Pinecone index entirely as a fallback: index.delete(delete_all=True, namespace="") — only if per-namespace delete fails.
    """
    logger.info("Starting deletion of all papers")
    paper_ids = []
    
    # 1. Query distinct paper IDs
    try:
        async with driver.session() as session:
            result = await session.run(
                "MATCH (n) WHERE n.paper_id IS NOT NULL RETURN DISTINCT n.paper_id AS pid"
            )
            records = await result.data()
            paper_ids = [rec["pid"] for rec in records if rec.get("pid")]
            logger.info("Found distinct paper IDs to delete: %s", paper_ids)
    except Exception as exc:
        logger.error("Failed to query distinct paper IDs from Neo4j: %s", exc)

    # 2. Call delete_paper_data for each paper_id
    results = []
    any_failed = False
    for pid in paper_ids:
        res = await delete_paper_data(pid, driver, settings)
        results.append(res)
        if not res["pinecone_namespace_deleted"]:
            any_failed = True

    # 3. Fallback: Clear Pinecone index entirely if namespace deletion failed or we want to be safe
    if any_failed or not paper_ids:
        try:
            from backend.vector.pinecone_store import get_index
            index = get_index(settings)
            index.delete(delete_all=True, namespace="")
            logger.info("Fallback: Cleared default Pinecone namespace completely")
        except Exception as exc:
            if '404' in str(exc) or 'Namespace not found' in str(exc):
                logger.debug("Default Pinecone namespace did not exist — nothing to delete")
            else:
                logger.error("Failed fallback global Pinecone cleanup: %s", exc)

    return {
        "papers_deleted": results,
        "total_deleted": len(paper_ids)
    }
