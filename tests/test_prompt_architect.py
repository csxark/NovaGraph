from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from backend.agents.prompt_architect import (
    classify_intent,
    detect_domain,
    generate_prompt_architect_response,
    PromptBlock,
    PromptArchitectResponse,
    _parse_prompt_blocks,
)
from backend.agents.synthesizer import run_pipeline, QueryResponse
from backend.main import QueryResponseModel

# Classification tests
def test_paper_queries_route_correctly():
    assert classify_intent("What is the main contribution of this paper?") == 'PAPER_QUERY'
    assert classify_intent("What datasets were used?") == 'PAPER_QUERY'
    assert classify_intent("Summarize the methodology") == 'PAPER_QUERY'
    assert classify_intent("What are the key results?") == 'PAPER_QUERY'
    assert classify_intent("What are the limitations?") == 'PAPER_QUERY'

def test_architect_queries_route_correctly():
    assert classify_intent("Build me a SaaS dashboard") == 'PROMPT_ARCHITECT'
    assert classify_intent("Can you give me python code for merge sort") == 'PROMPT_ARCHITECT'
    assert classify_intent("Design a microservices architecture") == 'PROMPT_ARCHITECT'
    assert classify_intent("I want to build a mobile app") == 'PROMPT_ARCHITECT'
    assert classify_intent("Create a REST API with auth") == 'PROMPT_ARCHITECT'
    assert classify_intent("write a javascript function for debounce") == 'PROMPT_ARCHITECT'
    assert classify_intent("implement a binary search algorithm") == 'PROMPT_ARCHITECT'

# Parser tests
def test_parser_handles_standard_headers():
    raw = "PROMPT 1\nPlan content here\nPROMPT 2\nBuild content here\nPROMPT 3\nOptimize content here"
    blocks = _parse_prompt_blocks(raw, "Backend Development")
    assert len(blocks) == 3
    assert blocks[0].title == 'PLAN'
    assert 'Plan content' in blocks[0].content

def test_parser_handles_markdown_headers():
    raw = "## PROMPT 1\nPlan\n## PROMPT 2\nBuild\n## PROMPT 3\nOptimize"
    blocks = _parse_prompt_blocks(raw, "AI Engineering")
    assert len(blocks) == 3

def test_parser_handles_bold_headers():
    raw = "**PROMPT 1**\nPlan\n**PROMPT 2**\nBuild\n**PROMPT 3**\nOptimize"
    blocks = _parse_prompt_blocks(raw, "Frontend Development")
    assert len(blocks) == 3

def test_parser_handles_title_case_headers():
    raw = "Prompt 1:\nPlan\nPrompt 2:\nBuild\nPrompt 3:\nOptimize"
    blocks = _parse_prompt_blocks(raw, "SaaS Products")
    assert len(blocks) == 3

def test_parser_falls_back_on_missing_headers():
    raw = "Here is some content without headers that is quite long and has no structure"
    blocks = _parse_prompt_blocks(raw, "General")
    assert len(blocks) == 3

# Schema tests
def test_query_response_defaults():
    from backend.agents.synthesizer import QueryResponse
    r = QueryResponse(answer="test", query_id="abc")
    assert r.response_type == 'paper_answer'
    assert r.prompts == []
    assert r.domain == ''

# Pipeline routing & generation tests
@pytest.mark.asyncio
async def test_generate_response_returns_exactly_three_prompts(settings):
    with patch("backend.agents.prompt_architect.Mistral") as mock_class:
        mock_choices = [MagicMock()]
        mock_choices[0].message.content = """PROMPT 1
Purpose: Plan
---
Content 1
PROMPT 2
Purpose: Build
---
Content 2
PROMPT 3
Purpose: Optimize
---
Content 3"""
        mock_mistral = MagicMock()
        mock_mistral.chat.complete_async = AsyncMock(return_value=MagicMock(choices=mock_choices))
        mock_class.return_value = mock_mistral
        
        response = await generate_prompt_architect_response("Create an API", settings)
        assert isinstance(response, PromptArchitectResponse)
        assert len(response.prompts) == 3
        assert response.domain == "APIs"

@pytest.mark.asyncio
async def test_generate_response_structure_validity(settings):
    with patch("backend.agents.prompt_architect.Mistral") as mock_class:
        mock_choices = [MagicMock()]
        mock_choices[0].message.content = """PROMPT 1
Purpose: Architectural design details.
---
Plan details
PROMPT 2
Purpose: Setup and run instructions.
---
Build details
PROMPT 3
Purpose: Security guidelines.
---
Optimize details"""
        mock_mistral = MagicMock()
        mock_mistral.chat.complete_async = AsyncMock(return_value=MagicMock(choices=mock_choices))
        mock_class.return_value = mock_mistral
        
        response = await generate_prompt_architect_response("saas multi-tenant app", settings)
        assert response.response_type == "prompt_architect"
        assert len(response.prompts) == 3
        for p in response.prompts:
            assert isinstance(p, PromptBlock)
            assert p.title in ["PLAN", "BUILD", "OPTIMIZE"]
            assert p.purpose != ""
            assert p.content != ""

@pytest.mark.asyncio
async def test_routing_paper_query_bypasses_architect(neo4j_mock, settings):
    with patch("backend.agents.synthesizer.classify_intent", return_value="PAPER_QUERY") as mock_classify, \
         patch("backend.agents.synthesizer.run_targeted_retrieval", return_value=("", [], [])) as mock_retrieval, \
         patch("backend.agents.synthesizer.run_vector_agent", return_value=MagicMock(chunks=[], scores=[], total=0, error=None)), \
         patch("backend.agents.synthesizer.run_graph_agent", return_value=MagicMock(nodes=[], edges=[], error=None)), \
         patch("backend.agents.synthesizer.resolve_entities", return_value=MagicMock(resolved_ids=[], matched_nodes=[], error=None)), \
         patch("backend.agents.synthesizer.synthesize_answer", return_value="Sample Answer"):
        
        res = await run_pipeline(
            query="What is the contribution?",
            doc_id="doc1",
            driver=neo4j_mock,
            settings=settings,
        )
        assert res.response_type == "paper_answer"
        assert res.answer == "Sample Answer"
        mock_classify.assert_called_once()
        mock_retrieval.assert_called_once()

@pytest.mark.asyncio
async def test_routing_architect_query_bypasses_rag(neo4j_mock, settings):
    mock_resp = PromptArchitectResponse(
        prompts=[
            PromptBlock(title="PLAN", purpose="Plan", content="C1", domain="SaaS Products"),
            PromptBlock(title="BUILD", purpose="Build", content="C2", domain="SaaS Products"),
            PromptBlock(title="OPTIMIZE", purpose="Optimize", content="C3", domain="SaaS Products"),
        ],
        domain="SaaS Products",
        query_id="uuid123",
    )
    with patch("backend.agents.synthesizer.classify_intent", return_value="PROMPT_ARCHITECT") as mock_classify, \
         patch("backend.agents.synthesizer.generate_prompt_architect_response", return_value=mock_resp) as mock_gen, \
         patch("backend.agents.synthesizer.run_targeted_retrieval") as mock_retrieval:
         
        res = await run_pipeline(
            query="Build me a SaaS app",
            doc_id="doc1",
            driver=neo4j_mock,
            settings=settings,
        )
        assert res.response_type == "prompt_architect"
        assert len(res.prompts) == 3
        assert res.domain == "SaaS Products"
        mock_classify.assert_called_once()
        mock_gen.assert_called_once()
        mock_retrieval.assert_not_called()

def test_query_response_model_backward_compat():
    model = QueryResponseModel(
        answer="Hello",
        query_id="q1",
    )
    assert model.response_type == "paper_answer"
    assert model.prompts == []
    assert model.domain == ""

def test_query_response_model_new_fields_default():
    model = QueryResponseModel(
        answer="",
        query_id="q2",
    )
    assert model.response_type == "paper_answer"
    assert model.prompts == []
    assert model.domain == ""

@pytest.mark.asyncio
async def test_architect_response_no_doc_id_required(settings):
    mock_resp = PromptArchitectResponse(
        prompts=[
            PromptBlock(title="PLAN", purpose="Plan", content="C1", domain="DevOps"),
            PromptBlock(title="BUILD", purpose="Build", content="C2", domain="DevOps"),
            PromptBlock(title="OPTIMIZE", purpose="Optimize", content="C3", domain="DevOps"),
        ],
        domain="DevOps",
        query_id="uuid_devops",
    )
    with patch("backend.agents.synthesizer.classify_intent", return_value="PROMPT_ARCHITECT"), \
         patch("backend.agents.synthesizer.generate_prompt_architect_response", return_value=mock_resp):
        res = await run_pipeline(
            query="Setup a docker compose for deployment",
            doc_id="dummy_doc",
            driver=None,
            settings=settings,
        )
        assert res.response_type == "prompt_architect"
        assert res.prompts[0]["title"] == "PLAN"
        assert res.domain == "DevOps"

def test_detect_domain_all_20_domains():
    assert detect_domain("next.js fullstack application") == "Full Stack Development"
    assert detect_domain("tailwind css frontend component styling") == "Frontend Development"
    assert detect_domain("fastapi backend server with node and java") == "Backend Development"
    assert detect_domain("flutter mobile app for android and ios") == "Mobile Development"
    assert detect_domain("llm embeddings prompt and pinecone vector db") == "AI Engineering"
    assert detect_domain("pytorch model classification and dataset training") == "Machine Learning"
    assert detect_domain("langgraph multi-agent tool calling swarm") == "Agent Systems"
    assert detect_domain("saas multi-tenant billing subscription stripe") == "SaaS Products"
    assert detect_domain("automation selenium playwright scraping cron job") == "Automation"
    assert detect_domain("rest api graphql endpoint grpc") == "APIs"
    assert detect_domain("postgres database mysql redis prisma migrations") == "Databases"
    assert detect_domain("devops docker kubernetes k8s ci/cd github actions") == "DevOps"
    assert detect_domain("aws cloud terraform serverless lambda s3") == "Cloud Infrastructure"
    assert detect_domain("security auth jwt oauth encryption vulnerability") == "Cybersecurity"
    assert detect_domain("data engineering spark kafka etl airflow pipeline") == "Data Engineering"
    assert detect_domain("ui/ux design figma wireframe typography color palette") == "UI/UX Design"
    assert detect_domain("product design user flow user journey onboarding roadmap") == "Product Design"
    assert detect_domain("business strategy monetization pricing model roi competitor") == "Business Strategy"
    assert detect_domain("academic research literature citation latex thesis") == "Research"
    assert detect_domain("headless blog cms markdown seo content editor") == "Content Systems"
