"""
backend/graph/neo4j_client.py

Singleton async Neo4j driver for the GraphRAG Research Assistant.
"""

from __future__ import annotations

import logging

from neo4j import AsyncGraphDatabase, AsyncDriver

from backend.config import get_settings, Settings  # noqa: F401 – re-exported for callers

logger = logging.getLogger(__name__)

_driver: AsyncDriver | None = None


async def get_driver(settings: Settings | None = None) -> AsyncDriver:
    """Return the module-level singleton AsyncDriver, initialising it on first call."""
    global _driver
    if _driver is None:
        if settings is None:
            settings = get_settings()
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
            connection_acquisition_timeout=60,
        )
        logger.info("Neo4j async driver initialised (URI=%s)", settings.neo4j_uri)
    return _driver


async def close_driver() -> None:
    """Close the singleton driver if it is open."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
        logger.info("Neo4j async driver closed.")


async def verify_connectivity() -> None:
    """Verify that the driver can reach the Neo4j instance."""
    driver = await get_driver()
    await driver.verify_connectivity()
    logger.info("Neo4j connectivity verified.")
