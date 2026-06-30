"""
backend/graph/schema_init.py

Creates all Neo4j constraints and indexes for the GraphRAG Research Assistant.
Run directly:  python -m backend.graph.schema_init
"""

from __future__ import annotations

import asyncio
import logging

from neo4j import AsyncDriver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# All DDL statements
# ---------------------------------------------------------------------------

_UNIQUENESS_CONSTRAINTS: list[tuple[str, str]] = [
    (
        "constraint_paper_paper_id",
        "CREATE CONSTRAINT constraint_paper_paper_id IF NOT EXISTS "
        "FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
    ),
    (
        "constraint_concept_entity_id",
        "CREATE CONSTRAINT constraint_concept_entity_id IF NOT EXISTS "
        "FOR (n:Concept) REQUIRE n.entity_id IS UNIQUE",
    ),
    (
        "constraint_method_entity_id",
        "CREATE CONSTRAINT constraint_method_entity_id IF NOT EXISTS "
        "FOR (n:Method) REQUIRE n.entity_id IS UNIQUE",
    ),
    (
        "constraint_evidence_entity_id",
        "CREATE CONSTRAINT constraint_evidence_entity_id IF NOT EXISTS "
        "FOR (n:Evidence) REQUIRE n.entity_id IS UNIQUE",
    ),
    (
        "constraint_finding_entity_id",
        "CREATE CONSTRAINT constraint_finding_entity_id IF NOT EXISTS "
        "FOR (n:Finding) REQUIRE n.entity_id IS UNIQUE",
    ),
    (
        "constraint_entity_entity_id",
        "CREATE CONSTRAINT constraint_entity_entity_id IF NOT EXISTS "
        "FOR (n:Entity) REQUIRE n.entity_id IS UNIQUE",
    ),
    (
        "constraint_reference_entity_id",
        "CREATE CONSTRAINT constraint_reference_entity_id IF NOT EXISTS "
        "FOR (n:Reference) REQUIRE n.entity_id IS UNIQUE",
    ),
    (
        "constraint_proposition_entity_id",
        "CREATE CONSTRAINT constraint_proposition_entity_id IF NOT EXISTS "
        "FOR (n:Proposition) REQUIRE n.entity_id IS UNIQUE",
    ),
    (
        "constraint_assumption_entity_id",
        "CREATE CONSTRAINT constraint_assumption_entity_id IF NOT EXISTS "
        "FOR (n:Assumption) REQUIRE n.entity_id IS UNIQUE",
    ),
    (
        "constraint_reference_doi",
        "CREATE CONSTRAINT constraint_reference_doi IF NOT EXISTS "
        "FOR (n:Reference) REQUIRE n.doi IS UNIQUE",
    ),
]

_RANGE_INDEXES: list[tuple[str, str]] = [
    (
        "idx_concept_paper_id",
        "CREATE INDEX idx_concept_paper_id IF NOT EXISTS "
        "FOR (n:Concept) ON (n.paper_id)",
    ),
    (
        "idx_method_paper_id",
        "CREATE INDEX idx_method_paper_id IF NOT EXISTS "
        "FOR (n:Method) ON (n.paper_id)",
    ),
    (
        "idx_entity_paper_id",
        "CREATE INDEX idx_entity_paper_id IF NOT EXISTS "
        "FOR (n:Entity) ON (n.paper_id)",
    ),
    (
        "idx_finding_paper_id",
        "CREATE INDEX idx_finding_paper_id IF NOT EXISTS "
        "FOR (n:Finding) ON (n.paper_id)",
    ),
]

_COMPOSITE_INDEXES: list[tuple[str, str]] = [
    (
        "idx_entity_entity_id_paper_id",
        "CREATE INDEX idx_entity_entity_id_paper_id IF NOT EXISTS "
        "FOR (n:Entity) ON (n.entity_id, n.paper_id)",
    ),
]

_FULLTEXT_INDEXES: list[tuple[str, str]] = [
    (
        "ft_name_desc",
        "CREATE FULLTEXT INDEX ft_name_desc IF NOT EXISTS "
        "FOR (n:Concept|Method|Evidence|Finding|Entity|Reference|Proposition|Assumption) "
        "ON EACH [n.name, n.description]",
    ),
    (
        "ft_reference",
        "CREATE FULLTEXT INDEX ft_reference IF NOT EXISTS "
        "FOR (n:Reference) "
        "ON EACH [n.title, n.authors]",
    ),
]


async def _run_statement(driver: AsyncDriver, name: str, cypher: str) -> None:
    """Execute a single DDL statement, warning (not raising) on failure."""
    try:
        async with driver.session() as session:
            await session.run(cypher)
        logger.info("Schema statement OK: %s", name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Schema statement FAILED (%s): %s — %s", name, cypher, exc)


async def initialize_schema(driver: AsyncDriver) -> None:
    """Create all constraints and indexes, each in its own try/except."""
    logger.info("Initialising Neo4j schema …")

    for name, cypher in _UNIQUENESS_CONSTRAINTS:
        await _run_statement(driver, name, cypher)

    for name, cypher in _RANGE_INDEXES:
        await _run_statement(driver, name, cypher)

    for name, cypher in _COMPOSITE_INDEXES:
        await _run_statement(driver, name, cypher)

    for name, cypher in _FULLTEXT_INDEXES:
        await _run_statement(driver, name, cypher)

    logger.info("Neo4j schema initialisation complete.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s – %(message)s",
        stream=sys.stdout,
    )

    async def _main() -> None:
        from backend.graph.neo4j_client import get_driver, close_driver

        driver = await get_driver()
        try:
            await initialize_schema(driver)
        finally:
            await close_driver()

    asyncio.run(_main())
