"""
Unit tests for multi-tenant data isolation checking Neo4j and Pinecone filter scoping.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from backend.vector.pinecone_store import query_similar, delete_paper_vectors
from backend.graph.neo4j_queries import get_subgraph, traverse_from_entities, find_similar_nodes_by_name
from backend.agents.graph_agent import run_graph_agent
from backend.agents.vector_agent import run_vector_agent


@pytest.fixture
def neo4j_mock():
    """Mock Neo4j driver matching conftest.py pattern."""
    session = AsyncMock()
    session.run = AsyncMock()
    session.execute_query = AsyncMock(return_value=([], None, None))
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    driver.execute_query = AsyncMock(return_value=([], None, None))
    driver.close = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    return driver


def test_query_similar_always_includes_doc_id_filter():
    """Verify that query_similar always adds the doc_id $eq filter in index.query."""
    mock_index = MagicMock()
    mock_settings = MagicMock()
    mock_settings.pinecone_index_name = "test-index"

    # Default query response
    mock_index.query.return_value = {"matches": []}

    with patch("backend.vector.pinecone_store.get_index", return_value=mock_index):
        query_similar(
            embedding=[0.1] * 1024,
            doc_id="doc-A",
            top_k=5,
            settings=mock_settings,
        )

        # Assert query is called with correct namespace and filter dict
        mock_index.query.assert_called_once()
        kwargs = mock_index.query.call_args[1]
        assert kwargs["namespace"] == ""
        assert kwargs["filter"] == {"doc_id": {"$eq": "doc-A"}}


def test_delete_paper_vectors_uses_metadata_filter():
    """Verify delete_paper_vectors deletes via metadata filter, not namespace."""
    mock_index = MagicMock()
    mock_settings = MagicMock()
    mock_settings.pinecone_index_name = "test-index"

    with patch("backend.vector.pinecone_store.get_index", return_value=mock_index):
        delete_paper_vectors(doc_id="doc-A", settings=mock_settings)

        # Assert delete is called with filter and empty namespace
        mock_index.delete.assert_called_once_with(
            filter={"doc_id": {"$eq": "doc-A"}},
            namespace="",
        )


@pytest.mark.asyncio
async def test_get_subgraph_scoped_to_doc_id(neo4j_mock):
    """Verify that get_subgraph uses doc_id in its Cypher queries."""
    mock_result = AsyncMock()
    neo4j_mock.session().run.return_value = mock_result
    mock_result.data.return_value = []

    await get_subgraph(driver=neo4j_mock, doc_id="doc-A", max_nodes=200)

    # Check Cypher parameter query
    neo4j_mock.session().run.assert_called_once()
    args, kwargs = neo4j_mock.session().run.call_args
    query = args[0]
    assert kwargs["doc_id"] == "doc-A"
    assert "doc_id" in query


@pytest.mark.asyncio
async def test_traverse_from_entities_scoped_to_doc_id(neo4j_mock):
    """Verify traverse_from_entities filters strictly by doc_id."""
    await traverse_from_entities(
        driver=neo4j_mock,
        entity_ids=["entity-1"],
        doc_id="doc-A",
        depth=2,
    )

    neo4j_mock.execute_query.assert_called_once()
    args, kwargs = neo4j_mock.execute_query.call_args
    query = args[0]
    params = args[1]
    assert "doc_id" in params
    assert params["doc_id"] == "doc-A"
    assert "doc_id = $doc_id" in query


@pytest.mark.asyncio
async def test_graph_agent_fallback_scoped_to_doc_id(neo4j_mock):
    """Verify GraphAgent fallback retrieves sample nodes scoped to the doc_id."""
    mock_settings = MagicMock()
    
    # We patch search_nodes_by_label to return empty, forcing fallback
    with patch("backend.agents.graph_agent.search_nodes_by_label", return_value=[]), \
         patch("backend.graph.neo4j_queries.get_subgraph", new_callable=AsyncMock) as mock_get_subgraph, \
         patch("backend.agents.graph_agent.traverse_from_entities", return_value={"nodes": [], "edges": []}):
        
        mock_get_subgraph.return_value = {"nodes": [], "edges": []}
        
        await run_graph_agent(
            query="test query",
            doc_id="doc-A",
            driver=neo4j_mock,
            settings=mock_settings,
        )
        
        # Verify fallback get_subgraph called with doc_id="doc-A"
        mock_get_subgraph.assert_called_once_with(driver=neo4j_mock, doc_id="doc-A", max_nodes=5)
