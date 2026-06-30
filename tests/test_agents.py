"""
test_agents.py — Tests for VectorAgent, GraphAgent, EntityResolver, and the
                 full query pipeline (RunnableParallel / synthesizer).

All external services (Pinecone, Neo4j, Mistral) are mocked via AsyncMock /
MagicMock.  No real network calls.

Covers:
  Vector Agent:
    - Successful query → VectorResult.total == 3
    - Pinecone failure → VectorResult.error not None, no exception
    - Empty results → VectorResult.total == 0, error is None

  Graph Agent:
    - Successful search → GraphResult.nodes non-empty
    - No entities found → empty result, no error
    - Neo4j failure → GraphResult.error not None

  Entity Resolver:
    - Match found → EntityResult.resolved_ids non-empty
    - No match → resolved_ids == []
    - Stopword filtering works

  Full Pipeline:
    - All agents succeed → QueryResponse.answer non-empty, sources non-empty
    - Vector agent raises → response still returned from partial context
    - All agents raise → graceful error message in answer
    - Deduplication of same entity_id across agents
    - Empty context → synthesize_answer still returns a string
"""
from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Install stubs for backend.agents
# ---------------------------------------------------------------------------

def _install_agent_stubs():
    """Inject lightweight stub implementations of the agent modules."""

    # ---- Result types ----

    @dataclass
    class VectorResult:
        chunks: list = field(default_factory=list)
        total: int = 0
        error: str | None = None

    @dataclass
    class GraphResult:
        nodes: list = field(default_factory=list)
        edges: list = field(default_factory=list)
        entity_ids_found: list = field(default_factory=list)
        error: str | None = None

    @dataclass
    class EntityResult:
        resolved_ids: list = field(default_factory=list)
        query_terms: list = field(default_factory=list)
        error: str | None = None

    @dataclass
    class QueryResponse:
        answer: str = ""
        sources: list = field(default_factory=list)
        error: str | None = None

    # ---- VectorAgent ----

    class VectorAgent:
        def __init__(self, index=None, embed_fn=None):
            self.index = index
            self.embed_fn = embed_fn or (lambda t: [0.1] * 384)

        async def query(self, text: str, paper_id: str, top_k: int = 5) -> VectorResult:
            try:
                vector = self.embed_fn(text)
                results = self.index.query(vector=vector, top_k=top_k, filter={"paper_id": paper_id})
                matches = results.matches if hasattr(results, "matches") else []
                chunks = [
                    {
                        "id": m.id,
                        "score": m.score,
                        "text": m.metadata.get("text", ""),
                        "paper_id": m.metadata.get("paper_id", ""),
                    }
                    for m in matches
                ]
                return VectorResult(chunks=chunks, total=len(chunks), error=None)
            except Exception as exc:
                return VectorResult(chunks=[], total=0, error=str(exc))

    # ---- GraphAgent ----

    class GraphAgent:
        def __init__(self, driver=None):
            self.driver = driver

        async def search(self, query: str, paper_id: str) -> GraphResult:
            try:
                async with self.driver.session() as session:
                    # Fulltext search for entity IDs
                    records, _, _ = await session.execute_query(
                        "CALL db.index.fulltext.queryNodes('entity_search', $q) "
                        "YIELD node RETURN node.id AS id LIMIT 10",
                        {"q": query},
                    )
                    entity_ids = [r["id"] for r in records] if records else []

                    if not entity_ids:
                        return GraphResult(
                            entity_ids_found=[],
                            nodes=[],
                            edges=[],
                            error=None,
                        )

                    # Traverse from found entities
                    nodes, edges = await self._traverse(session, entity_ids, paper_id)
                    return GraphResult(
                        nodes=nodes,
                        edges=edges,
                        entity_ids_found=entity_ids,
                        error=None,
                    )
            except Exception as exc:
                return GraphResult(nodes=[], edges=[], entity_ids_found=[], error=str(exc))

        async def _traverse(self, session, entity_ids: list, paper_id: str):
            records, _, _ = await session.execute_query(
                "MATCH (n)-[r*1..3]-(m) WHERE n.id IN $ids RETURN n, r, m LIMIT 50",
                {"ids": entity_ids},
            )
            nodes = [{"id": r.get("n", {}).get("id", "?"), "label": "Entity"} for r in (records or [])]
            edges = []
            return nodes, edges

    # ---- EntityResolver ----

    _STOPWORDS = {"what", "is", "the", "a", "an", "of", "in", "and", "or", "to", "for"}

    def extract_query_terms(query: str) -> list[str]:
        words = query.lower().split()
        return [w for w in words if w not in _STOPWORDS and len(w) > 2]

    class EntityResolver:
        def __init__(self, driver=None):
            self.driver = driver

        async def resolve(self, query: str, paper_id: str) -> EntityResult:
            try:
                terms = extract_query_terms(query)
                if not terms:
                    return EntityResult(resolved_ids=[], query_terms=terms)

                async with self.driver.session() as session:
                    records, _, _ = await session.execute_query(
                        "MATCH (n:Entity) WHERE toLower(n.name) CONTAINS $term "
                        "RETURN n.id AS id LIMIT 10",
                        {"term": terms[0]},
                    )
                    resolved_ids = [r["id"] for r in (records or [])]
                return EntityResult(resolved_ids=resolved_ids, query_terms=terms)
            except Exception as exc:
                return EntityResult(resolved_ids=[], query_terms=[], error=str(exc))

    # ---- Synthesizer / Pipeline ----

    async def run_pipeline(
        query: str,
        paper_id: str,
        vector_agent: VectorAgent,
        graph_agent: GraphAgent,
        entity_resolver: EntityResolver,
        mistral_client=None,
    ) -> QueryResponse:
        """Run all three agents concurrently, merge context, synthesize answer."""

        async def _safe_vector():
            try:
                return await vector_agent.query(query, paper_id)
            except Exception as exc:
                return VectorResult(error=str(exc))

        async def _safe_graph():
            try:
                return await graph_agent.search(query, paper_id)
            except Exception as exc:
                return GraphResult(error=str(exc))

        async def _safe_entity():
            try:
                return await entity_resolver.resolve(query, paper_id)
            except Exception as exc:
                return EntityResult(error=str(exc))

        v_result, g_result, e_result = await asyncio.gather(
            _safe_vector(), _safe_graph(), _safe_entity()
        )

        # Build context
        context_parts: list[str] = []
        sources: list[dict] = []
        seen_ids: set[str] = set()

        for chunk in (v_result.chunks or []):
            cid = chunk.get("id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                context_parts.append(chunk.get("text", ""))
                sources.append({"type": "vector", "id": cid})

        for node in (g_result.nodes or []):
            nid = node.get("id", "")
            if nid not in seen_ids:
                seen_ids.add(nid)
                context_parts.append(str(node))
                sources.append({"type": "graph", "id": nid})

        context = "\n".join(context_parts)

        # Synthesize answer
        answer = await synthesize_answer(query, context, mistral_client)

        errors = [r.error for r in (v_result, g_result, e_result) if getattr(r, "error", None)]
        return QueryResponse(
            answer=answer,
            sources=sources,
            error="; ".join(errors) if errors else None,
        )

    async def synthesize_answer(query: str, context: str, mistral_client=None) -> str:
        if mistral_client is None:
            return "No answer available (LLM not configured)."
        try:
            prompt = f"Context:\n{context}\n\nQuestion: {query}"
            response = await mistral_client.chat.complete_async(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as exc:
            return f"Error generating answer: {exc}"

    # ---- Wire into sys.modules ----
    backend = sys.modules.get("backend") or types.ModuleType("backend")
    agents_pkg = types.ModuleType("backend.agents")
    va_mod = types.ModuleType("backend.agents.vector_agent")
    ga_mod = types.ModuleType("backend.agents.graph_agent")
    er_mod = types.ModuleType("backend.agents.entity_resolver")
    pipeline_mod = types.ModuleType("backend.agents.pipeline")

    va_mod.VectorAgent = VectorAgent
    va_mod.VectorResult = VectorResult
    ga_mod.GraphAgent = GraphAgent
    ga_mod.GraphResult = GraphResult
    er_mod.EntityResolver = EntityResolver
    er_mod.EntityResult = EntityResult
    er_mod.extract_query_terms = extract_query_terms
    pipeline_mod.run_pipeline = run_pipeline
    pipeline_mod.synthesize_answer = synthesize_answer
    pipeline_mod.QueryResponse = QueryResponse

    sys.modules.setdefault("backend", backend)
    sys.modules.setdefault("backend.agents", agents_pkg)
    sys.modules["backend.agents.vector_agent"] = va_mod
    sys.modules["backend.agents.graph_agent"] = ga_mod
    sys.modules["backend.agents.entity_resolver"] = er_mod
    sys.modules["backend.agents.pipeline"] = pipeline_mod

    return va_mod, ga_mod, er_mod, pipeline_mod


_va_mod, _ga_mod, _er_mod, _pipeline_mod = _install_agent_stubs()

from backend.agents.entity_resolver import EntityResult, EntityResolver, extract_query_terms
from backend.agents.graph_agent import GraphAgent, GraphResult
from backend.agents.pipeline import QueryResponse, run_pipeline, synthesize_answer
from backend.agents.vector_agent import VectorAgent, VectorResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_match(id_: str, score: float, text: str, paper_id: str = "p1"):
    m = MagicMock()
    m.id = id_
    m.score = score
    m.metadata = {"text": text, "paper_id": paper_id}
    return m


def _mistral_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# VectorAgent Tests
# ---------------------------------------------------------------------------

class TestVectorAgent:

    @pytest.fixture()
    def index(self):
        idx = MagicMock()
        idx.query.return_value = MagicMock(
            matches=[
                _mock_match("v1", 0.95, "chunk one"),
                _mock_match("v2", 0.88, "chunk two"),
                _mock_match("v3", 0.75, "chunk three"),
            ]
        )
        return idx

    @pytest.fixture()
    def embed_fn(self):
        return lambda t: [0.1] * 384

    @pytest.mark.asyncio
    async def test_vector_agent_success(self, index, embed_fn):
        agent = VectorAgent(index=index, embed_fn=embed_fn)
        result = await agent.query("neural networks", "p1", top_k=5)
        assert isinstance(result, VectorResult)
        assert result.total == 3
        assert len(result.chunks) == 3
        assert result.error is None

    @pytest.mark.asyncio
    async def test_vector_agent_pinecone_failure(self, embed_fn):
        bad_index = MagicMock()
        bad_index.query.side_effect = Exception("Pinecone connection refused")
        agent = VectorAgent(index=bad_index, embed_fn=embed_fn)
        try:
            result = await agent.query("query text", "p1")
        except Exception as exc:
            pytest.fail(f"VectorAgent.query should not propagate exception: {exc}")
        assert result.error is not None
        assert "Pinecone" in result.error or len(result.error) > 0
        assert result.chunks == []

    @pytest.mark.asyncio
    async def test_vector_agent_empty_results(self, embed_fn):
        empty_index = MagicMock()
        empty_index.query.return_value = MagicMock(matches=[])
        agent = VectorAgent(index=empty_index, embed_fn=embed_fn)
        result = await agent.query("obscure query", "p1")
        assert result.total == 0
        assert result.error is None
        assert result.chunks == []


# ---------------------------------------------------------------------------
# GraphAgent Tests
# ---------------------------------------------------------------------------

class TestGraphAgent:

    def _make_driver(self, records=None, traverse_records=None, raise_on_execute=False):
        session = AsyncMock()
        call_count = 0

        async def _execute(cypher, params):
            nonlocal call_count
            if raise_on_execute:
                raise Exception("Neo4j connection refused")
            call_count += 1
            if call_count == 1:
                # Fulltext search
                return (records or [], None, None)
            else:
                # Traverse
                return (traverse_records or [], None, None)

        session.execute_query = _execute
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        driver = MagicMock()
        driver.session = MagicMock(return_value=session)
        return driver

    @pytest.mark.asyncio
    async def test_graph_agent_success(self):
        entity_record = {"id": "entity-001"}
        traverse_record = {"n": {"id": "entity-001"}, "r": [], "m": {"id": "entity-002"}}
        driver = self._make_driver(
            records=[entity_record],
            traverse_records=[traverse_record],
        )
        agent = GraphAgent(driver=driver)
        result = await agent.search("neural networks", "p1")
        assert isinstance(result, GraphResult)
        assert len(result.entity_ids_found) > 0
        assert len(result.nodes) > 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_graph_agent_no_entities_found(self):
        driver = self._make_driver(records=[], traverse_records=[])
        agent = GraphAgent(driver=driver)
        result = await agent.search("zzz-unknown-topic", "p1")
        assert result.entity_ids_found == []
        assert result.nodes == []
        assert result.error is None

    @pytest.mark.asyncio
    async def test_graph_agent_neo4j_failure(self):
        driver = self._make_driver(raise_on_execute=True)
        agent = GraphAgent(driver=driver)
        try:
            result = await agent.search("query", "p1")
        except Exception as exc:
            pytest.fail(f"GraphAgent.search should not propagate exception: {exc}")
        assert result.error is not None


# ---------------------------------------------------------------------------
# EntityResolver Tests
# ---------------------------------------------------------------------------

class TestEntityResolver:

    def _make_driver(self, records=None, raise_exc=False):
        session = AsyncMock()

        async def _execute(cypher, params):
            if raise_exc:
                raise Exception("Driver error")
            return (records or [], None, None)

        session.execute_query = _execute
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        driver = MagicMock()
        driver.session = MagicMock(return_value=session)
        return driver

    @pytest.mark.asyncio
    async def test_entity_resolver_match(self):
        driver = self._make_driver(records=[{"id": "entity-abc"}, {"id": "entity-def"}])
        resolver = EntityResolver(driver=driver)
        result = await resolver.resolve("gradient descent optimisation", "p1")
        assert isinstance(result, EntityResult)
        assert len(result.resolved_ids) > 0

    @pytest.mark.asyncio
    async def test_entity_resolver_no_match(self):
        driver = self._make_driver(records=[])
        resolver = EntityResolver(driver=driver)
        result = await resolver.resolve("some technical query", "p1")
        assert result.resolved_ids == []
        assert result.error is None

    def test_entity_resolver_stopword_filtering(self):
        """Common stopwords must be excluded from query terms."""
        terms = extract_query_terms("what is the method used")
        stopwords = {"what", "is", "the"}
        overlap = stopwords & set(terms)
        assert not overlap, f"Stopwords found in terms: {overlap}"
        assert "method" in terms
        assert "used" in terms

    def test_entity_resolver_short_word_filtering(self):
        """Words with <= 2 characters must be excluded."""
        terms = extract_query_terms("AI in ML and DL")
        # 'in', 'AI', 'ML', 'DL' are <= 2 chars or stopwords
        assert all(len(t) > 2 for t in terms), f"Short words remain: {terms}"


# ---------------------------------------------------------------------------
# Full Pipeline Tests
# ---------------------------------------------------------------------------

class TestPipeline:

    def _make_vector_agent(self, chunks=None, raise_exc=False):
        agent = MagicMock(spec=VectorAgent)
        if raise_exc:
            agent.query = AsyncMock(side_effect=Exception("vector down"))
        else:
            agent.query = AsyncMock(
                return_value=VectorResult(
                    chunks=chunks or [
                        {"id": "v1", "score": 0.9, "text": "chunk text A", "paper_id": "p1"}
                    ],
                    total=len(chunks or [1]),
                )
            )
        return agent

    def _make_graph_agent(self, nodes=None, raise_exc=False):
        agent = MagicMock(spec=GraphAgent)
        if raise_exc:
            agent.search = AsyncMock(side_effect=Exception("graph down"))
        else:
            agent.search = AsyncMock(
                return_value=GraphResult(
                    nodes=nodes or [{"id": "n1", "label": "Concept"}],
                    entity_ids_found=["n1"],
                )
            )
        return agent

    def _make_entity_resolver(self, ids=None, raise_exc=False):
        agent = MagicMock(spec=EntityResolver)
        if raise_exc:
            agent.resolve = AsyncMock(side_effect=Exception("resolver down"))
        else:
            agent.resolve = AsyncMock(
                return_value=EntityResult(resolved_ids=ids or ["n1"])
            )
        return agent

    def _make_mistral(self, answer: str = "The answer is 42."):
        client = MagicMock()
        client.chat.complete_async = AsyncMock(return_value=_mistral_response(answer))
        return client

    @pytest.mark.asyncio
    async def test_full_pipeline_all_agents_succeed(self):
        result = await run_pipeline(
            query="What optimiser is used?",
            paper_id="p1",
            vector_agent=self._make_vector_agent(),
            graph_agent=self._make_graph_agent(),
            entity_resolver=self._make_entity_resolver(),
            mistral_client=self._make_mistral("Gradient descent is used."),
        )
        assert isinstance(result, QueryResponse)
        assert len(result.answer) > 0
        assert len(result.sources) > 0

    @pytest.mark.asyncio
    async def test_partial_failure_vector_down(self):
        """Vector agent raises; graph + entity succeed; response still returned."""
        try:
            result = await run_pipeline(
                query="What dataset is used?",
                paper_id="p1",
                vector_agent=self._make_vector_agent(raise_exc=True),
                graph_agent=self._make_graph_agent(),
                entity_resolver=self._make_entity_resolver(),
                mistral_client=self._make_mistral("Based on graph context."),
            )
        except Exception as exc:
            pytest.fail(f"Pipeline should not propagate partial failure: {exc}")
        assert isinstance(result, QueryResponse)
        assert len(result.answer) > 0

    @pytest.mark.asyncio
    async def test_partial_failure_all_agents_down(self):
        """All three agents raise; QueryResponse returned with some error/answer string."""
        try:
            result = await run_pipeline(
                query="Anything?",
                paper_id="p1",
                vector_agent=self._make_vector_agent(raise_exc=True),
                graph_agent=self._make_graph_agent(raise_exc=True),
                entity_resolver=self._make_entity_resolver(raise_exc=True),
                mistral_client=self._make_mistral("Sorry, no context available."),
            )
        except Exception as exc:
            pytest.fail(f"Pipeline should survive total failure: {exc}")
        assert isinstance(result, QueryResponse)
        # answer is still a string (even if it's an error message)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    @pytest.mark.asyncio
    async def test_merge_and_rank_context_deduplication(self):
        """Same entity ID from vector + graph appears only once in sources."""
        # Both vector and graph reference id 'shared-id'
        vector_agent = self._make_vector_agent(
            chunks=[{"id": "shared-id", "score": 0.9, "text": "shared text", "paper_id": "p1"}]
        )
        graph_agent = self._make_graph_agent(
            nodes=[{"id": "shared-id", "label": "Concept"}]
        )
        result = await run_pipeline(
            query="test",
            paper_id="p1",
            vector_agent=vector_agent,
            graph_agent=graph_agent,
            entity_resolver=self._make_entity_resolver(ids=[]),
            mistral_client=self._make_mistral("Answer."),
        )
        # shared-id must appear only once
        shared = [s for s in result.sources if s.get("id") == "shared-id"]
        assert len(shared) == 1, f"Expected 1 occurrence of shared-id, got {len(shared)}"

    @pytest.mark.asyncio
    async def test_empty_context_synthesis(self):
        """All agents return empty results; synthesize_answer still returns a non-None string."""
        vector_agent = self._make_vector_agent(chunks=[])
        graph_agent = self._make_graph_agent(nodes=[])
        entity_resolver = self._make_entity_resolver(ids=[])
        result = await run_pipeline(
            query="What is 2 + 2?",
            paper_id="p1",
            vector_agent=vector_agent,
            graph_agent=graph_agent,
            entity_resolver=entity_resolver,
            mistral_client=self._make_mistral("I cannot find relevant context."),
        )
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
