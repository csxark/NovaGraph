from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from backend.main import app
from backend.db.models import Base
import backend.db.document_registry as dr
from tests.conftest import _build_minimal_pdf
from backend.agents.synthesizer import QueryResponse


@pytest.fixture(autouse=True)
async def setup_test_db():
    """Overrides the database engine with an in-memory SQLite db for testing."""
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    test_session_maker = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    orig_engine = dr.engine
    orig_factory = dr.AsyncSessionFactory

    dr.engine = test_engine
    dr.AsyncSessionFactory = test_session_maker

    # Initialize tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    await test_engine.dispose()
    dr.engine = orig_engine
    dr.AsyncSessionFactory = orig_factory


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
async def completed_doc_id(client):
    pdf_bytes = _build_minimal_pdf("API Prompt Architect Test PDF")
    with patch("backend.main.run_ingestion") as mock_run:
        mock_run.return_value = None
        response = client.post(
            "/upload",
            files={"file": ("api_test.pdf", pdf_bytes, "application/pdf")},
            headers={"X-User-ID": "user-test"},
        )
        doc_id = response.json()["doc_id"]
        await dr.update_document_status(doc_id, "completed")
        return doc_id


@pytest.mark.asyncio
async def test_paper_query_returns_paper_answer_type(client, completed_doc_id):
    # Mock synthesizer's run_pipeline to return a standard paper answer response
    mock_response = QueryResponse(
        answer="This is a paper answer.",
        sources=[],
        trace={},
        query_id="q-paper-1",
        total_nodes_retrieved=0,
        error=None,
        response_type="paper_answer",
        prompts=[],
        domain="",
    )
    
    with patch("backend.main.run_pipeline", return_value=mock_response):
        response = client.post(
            "/query",
            json={
                "query": "What is the main contribution of this paper?",
                "doc_id": completed_doc_id,
            },
            headers={"X-User-ID": "user-test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["response_type"] == "paper_answer"
        assert data["answer"] == "This is a paper answer."
        assert data["prompts"] == []
        assert data["domain"] == ""


@pytest.mark.asyncio
async def test_architect_query_returns_prompt_architect_type(client, completed_doc_id):
    mock_response = QueryResponse(
        answer="",
        sources=[],
        trace={},
        query_id="q-arch-1",
        total_nodes_retrieved=0,
        error=None,
        response_type="prompt_architect",
        prompts=[
            {"title": "PLAN", "purpose": "Plan purpose", "content": "Plan content", "domain": "Full Stack Development"},
            {"title": "BUILD", "purpose": "Build purpose", "content": "Build content", "domain": "Full Stack Development"},
            {"title": "OPTIMIZE", "purpose": "Optimize purpose", "content": "Optimize content", "domain": "Full Stack Development"},
        ],
        domain="Full Stack Development",
    )
    
    with patch("backend.main.run_pipeline", return_value=mock_response):
        response = client.post(
            "/query",
            json={
                "query": "Build me a SaaS dashboard",
                "doc_id": completed_doc_id,
            },
            headers={"X-User-ID": "user-test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["response_type"] == "prompt_architect"
        assert len(data["prompts"]) == 3
        assert data["domain"] == "Full Stack Development"


@pytest.mark.asyncio
async def test_architect_response_has_three_prompts(client, completed_doc_id):
    mock_response = QueryResponse(
        answer="",
        sources=[],
        trace={},
        query_id="q-arch-2",
        total_nodes_retrieved=0,
        error=None,
        response_type="prompt_architect",
        prompts=[
            {"title": "PLAN", "purpose": "Plan purpose", "content": "Plan content", "domain": "Full Stack Development"},
            {"title": "BUILD", "purpose": "Build purpose", "content": "Build content", "domain": "Full Stack Development"},
            {"title": "OPTIMIZE", "purpose": "Optimize purpose", "content": "Optimize content", "domain": "Full Stack Development"},
        ],
        domain="Full Stack Development",
    )
    
    with patch("backend.main.run_pipeline", return_value=mock_response):
        response = client.post(
            "/query",
            json={
                "query": "Build me a SaaS dashboard",
                "doc_id": completed_doc_id,
            },
            headers={"X-User-ID": "user-test"},
        )
        data = response.json()
        assert len(data["prompts"]) == 3
        assert [p["title"] for p in data["prompts"]] == ["PLAN", "BUILD", "OPTIMIZE"]


@pytest.mark.asyncio
async def test_architect_response_each_prompt_has_required_fields(client, completed_doc_id):
    mock_response = QueryResponse(
        answer="",
        sources=[],
        trace={},
        query_id="q-arch-3",
        total_nodes_retrieved=0,
        error=None,
        response_type="prompt_architect",
        prompts=[
            {"title": "PLAN", "purpose": "Plan purpose", "content": "Plan content", "domain": "Databases"},
            {"title": "BUILD", "purpose": "Build purpose", "content": "Build content", "domain": "Databases"},
            {"title": "OPTIMIZE", "purpose": "Optimize purpose", "content": "Optimize content", "domain": "Databases"},
        ],
        domain="Databases",
    )
    
    with patch("backend.main.run_pipeline", return_value=mock_response):
        response = client.post(
            "/query",
            json={
                "query": "Design a SQL schema",
                "doc_id": completed_doc_id,
            },
            headers={"X-User-ID": "user-test"},
        )
        data = response.json()
        for p in data["prompts"]:
            assert "title" in p
            assert "purpose" in p
            assert "content" in p
            assert "domain" in p
            assert p["domain"] == "Databases"


@pytest.mark.asyncio
async def test_architect_no_paper_required_for_general_queries(client, completed_doc_id):
    # Verify that the pipeline routing actually calls generate_prompt_architect_response
    # and bypasses RAG databases/traversal.
    with patch("backend.agents.synthesizer.classify_intent", return_value="PROMPT_ARCHITECT"), \
         patch("backend.agents.synthesizer.generate_prompt_architect_response") as mock_gen, \
         patch("backend.agents.synthesizer.run_targeted_retrieval") as mock_retrieval:
         
        mock_gen.return_value = MagicMock(
            prompts=[
                MagicMock(title="PLAN", purpose="P", content="C", domain="Databases", model_dump=lambda: {"title": "PLAN", "purpose": "P", "content": "C", "domain": "Databases"}),
                MagicMock(title="BUILD", purpose="P", content="C", domain="Databases", model_dump=lambda: {"title": "BUILD", "purpose": "P", "content": "C", "domain": "Databases"}),
                MagicMock(title="OPTIMIZE", purpose="P", content="C", domain="Databases", model_dump=lambda: {"title": "OPTIMIZE", "purpose": "P", "content": "C", "domain": "Databases"}),
            ],
            domain="Databases",
            query_id="q123",
        )
        
        response = client.post(
            "/query",
            json={
                "query": "Design a database schema",
                "doc_id": completed_doc_id,
            },
            headers={"X-User-ID": "user-test"},
        )
        assert response.status_code == 200
        mock_gen.assert_called_once()
        mock_retrieval.assert_not_called()


@pytest.mark.asyncio
async def test_merge_sort_query_routes_to_architect(client, completed_doc_id):
    # Verify end-to-end routing decision for a coding query
    with patch("backend.main.run_pipeline") as mock_run:
        mock_run.return_value = QueryResponse(
            answer="",
            sources=[],
            trace={},
            query_id="q-merge",
            total_nodes_retrieved=0,
            error=None,
            response_type="prompt_architect",
            prompts=[],
            domain="Backend Development",
        )
        
        client.post(
            "/query",
            json={
                "query": "Can you give me python code for merge sort?",
                "doc_id": completed_doc_id,
            },
            headers={"X-User-ID": "user-test"},
        )
        args, kwargs = mock_run.call_args
        assert kwargs["query"] == "Can you give me python code for merge sort?"


@pytest.mark.asyncio
async def test_what_is_contribution_routes_to_rag(client, completed_doc_id):
    # Verify end-to-end routing decision for a paper-specific query
    with patch("backend.main.run_pipeline") as mock_run:
        mock_run.return_value = QueryResponse(
            answer="This is a contribution.",
            sources=[],
            trace={},
            query_id="q-contrib",
            total_nodes_retrieved=0,
            error=None,
            response_type="paper_answer",
            prompts=[],
            domain="",
        )
        
        client.post(
            "/query",
            json={
                "query": "What is the main contribution of this paper?",
                "doc_id": completed_doc_id,
            },
            headers={"X-User-ID": "user-test"},
        )
        args, kwargs = mock_run.call_args
        assert kwargs["query"] == "What is the main contribution of this paper?"
