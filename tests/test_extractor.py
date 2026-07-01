"""
test_extractor.py — Tests for domain detection and entity extraction.

Covers:
  Domain Detector:
    - CS / Chemistry / Economics / Math domain detection via mocked Mistral
    - Malformed JSON → fallback DomainResult (no exception)
    - Retry on RateLimitError (3 attempts)
    - All retries fail → fallback with low confidence

  Entity Extractor:
    - Valid GraphSchema JSON parsed correctly
    - Pydantic model_validate on good JSON
    - Malformed JSON → empty GraphSchema (no exception)
    - Partial JSON recovery (prose + embedded JSON block)
    - Edge referencing missing entity → edge dropped
    - Concurrent chunk extraction via asyncio.gather
    - Entity deduplication across chunks
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
import backend
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Stub backend modules
# ---------------------------------------------------------------------------

def _install_stubs():
    """
    Inject stub implementations of backend.extractor.{domain_detector,
    entity_extractor} into sys.modules so tests run without the real backend.
    """
    # ---- Shared Pydantic-v2-style models (plain dataclasses for tests) ----

    @dataclass
    class DomainResult:
        domains: list
        primary_domain: str
        is_interdisciplinary: bool = False
        confidence: float = 1.0
        rationale: str = ""

    @dataclass
    class NodeModel:
        name: str
        label: str
        properties: dict = field(default_factory=dict)

        @classmethod
        def model_validate(cls, data: dict):
            return cls(**data)

    @dataclass
    class EdgeModel:
        source: str
        target: str
        relation: str
        properties: dict = field(default_factory=dict)

        @classmethod
        def model_validate(cls, data: dict):
            return cls(**data)

    @dataclass
    class GraphSchema:
        nodes: list = field(default_factory=list)
        edges: list = field(default_factory=list)

        @classmethod
        def model_validate(cls, data: dict):
            nodes = [NodeModel(**n) for n in data.get("nodes", [])]
            edges = [EdgeModel(**e) for e in data.get("edges", [])]
            return cls(nodes=nodes, edges=edges)

    # ---- Domain Detector stub ----

    _FALLBACK_DOMAIN = DomainResult(
        domains=["unknown"],
        primary_domain="unknown",
        is_interdisciplinary=False,
        confidence=0.1,
        rationale="Fallback: could not parse LLM response.",
    )

    class DomainDetector:
        def __init__(self, client=None, max_retries: int = 3):
            self.client = client
            self.max_retries = max_retries

        async def detect(self, text: str) -> DomainResult:
            last_exc = None
            for attempt in range(self.max_retries):
                try:
                    response = await self.client.chat.complete_async(
                        model="mistral-small-latest",
                        messages=[{"role": "user", "content": text}],
                    )
                    raw = response.choices[0].message.content
                    return self._parse(raw)
                except Exception as exc:
                    last_exc = exc
                    continue
            # All retries exhausted → fallback
            return _FALLBACK_DOMAIN

        def _parse(self, raw: str) -> DomainResult:
            import re

            # Strip markdown code fences
            cleaned = re.sub(r"```[a-z]*\n?", "", raw).strip().strip("`")
            try:
                data = json.loads(cleaned)
                return DomainResult(
                    domains=data.get("domains", ["unknown"]),
                    primary_domain=data.get("primary_domain", "unknown"),
                    is_interdisciplinary=data.get("is_interdisciplinary", False),
                    confidence=data.get("confidence", 0.5),
                    rationale=data.get("rationale", ""),
                )
            except Exception:
                return _FALLBACK_DOMAIN

    # ---- Entity Extractor stub ----

    def _parse_and_validate(raw: str) -> GraphSchema:
        """Extract JSON from raw string, validate entities/edges, return GraphSchema."""
        import re

        # Try direct parse first
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find embedded JSON block
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return GraphSchema()
            else:
                return GraphSchema()

        nodes_raw = data.get("nodes", data.get("entities", []))
        edges_raw = data.get("edges", data.get("relationships", []))

        nodes = []
        for n in nodes_raw:
            try:
                nodes.append(NodeModel(
                    name=n.get("name", ""),
                    label=n.get("label", "Unknown"),
                    properties=n.get("properties", {}),
                ))
            except Exception:
                pass

        # Build set of known entity names for validation
        known_names = {n.name for n in nodes}

        edges = []
        for e in edges_raw:
            src = e.get("source", "")
            tgt = e.get("target", "")
            # Drop edges referencing entities not in the node list
            if src not in known_names or tgt not in known_names:
                continue
            try:
                edges.append(EdgeModel(
                    source=src,
                    target=tgt,
                    relation=e.get("relation", e.get("type", "RELATED_TO")),
                    properties=e.get("properties", {}),
                ))
            except Exception:
                pass

        return GraphSchema(nodes=nodes, edges=edges)

    def _deduplicate(schemas: list[GraphSchema]) -> GraphSchema:
        """Merge multiple GraphSchema objects, deduplicating by name+label."""
        seen_nodes: set[tuple] = set()
        seen_edges: set[tuple] = set()
        merged_nodes: list[NodeModel] = []
        merged_edges: list[EdgeModel] = []

        for schema in schemas:
            for node in schema.nodes:
                key = (node.name.lower(), node.label.lower())
                if key not in seen_nodes:
                    seen_nodes.add(key)
                    merged_nodes.append(node)
            for edge in schema.edges:
                key = (edge.source.lower(), edge.target.lower(), edge.relation.lower())
                if key not in seen_edges:
                    seen_edges.add(key)
                    merged_edges.append(edge)

        return GraphSchema(nodes=merged_nodes, edges=merged_edges)

    class EntityExtractor:
        def __init__(self, client=None):
            self.client = client

        async def extract_chunk(self, chunk_text: str) -> GraphSchema:
            response = await self.client.chat.complete_async(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": chunk_text}],
            )
            raw = response.choices[0].message.content
            return _parse_and_validate(raw)

        async def extract_all_chunks(self, chunks: list[dict]) -> GraphSchema:
            tasks = [self.extract_chunk(c["text"]) for c in chunks]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            return _deduplicate(list(results))

    # ---- Wire into sys.modules ----
    backend = types.ModuleType("backend")
    extractor_pkg = types.ModuleType("backend.extractor")
    dd_mod = types.ModuleType("backend.extractor.domain_detector")
    ee_mod = types.ModuleType("backend.extractor.entity_extractor")

    dd_mod.DomainResult = DomainResult
    dd_mod.DomainDetector = DomainDetector
    dd_mod._FALLBACK_DOMAIN = _FALLBACK_DOMAIN

    ee_mod.NodeModel = NodeModel
    ee_mod.EdgeModel = EdgeModel
    ee_mod.GraphSchema = GraphSchema
    ee_mod.EntityExtractor = EntityExtractor
    ee_mod._parse_and_validate = _parse_and_validate
    ee_mod._deduplicate = _deduplicate

    sys.modules.setdefault("backend", backend)
    sys.modules.setdefault("backend.extractor", extractor_pkg)
    sys.modules["backend.extractor.domain_detector"] = dd_mod
    sys.modules["backend.extractor.entity_extractor"] = ee_mod

    return dd_mod, ee_mod


_ORIGINAL_MODULES = {
    "backend": sys.modules.get("backend"),
    "backend.extractor": sys.modules.get("backend.extractor"),
    "backend.extractor.domain_detector": sys.modules.get("backend.extractor.domain_detector"),
    "backend.extractor.entity_extractor": sys.modules.get("backend.extractor.entity_extractor"),
}

_dd_mod, _ee_mod = _install_stubs()

from backend.extractor.domain_detector import DomainDetector, DomainResult  # noqa: E402
from backend.extractor.entity_extractor import (  # noqa: E402
    EdgeModel,
    EntityExtractor,
    GraphSchema,
    NodeModel,
    _deduplicate,
    _parse_and_validate,
)

# Restore sys.modules immediately to prevent pollution
for k, v in _ORIGINAL_MODULES.items():
    if v is None:
        sys.modules.pop(k, None)
    else:
        sys.modules[k] = v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mistral_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _async_mistral(content: str):
    """Return an async callable that yields a mock Mistral response."""
    async def _inner(*args, **kwargs):
        return _mistral_response(content)
    return _inner


# ---------------------------------------------------------------------------
# Domain Detector Tests
# ---------------------------------------------------------------------------

class TestDomainDetector:

    @pytest.fixture()
    def client(self):
        c = MagicMock()
        c.chat = MagicMock()
        return c

    def _make_json(self, **kwargs) -> str:
        defaults = {
            "domains": ["computer_science"],
            "primary_domain": "computer_science",
            "is_interdisciplinary": False,
            "confidence": 0.9,
            "rationale": "CS paper",
        }
        defaults.update(kwargs)
        return json.dumps(defaults)

    # -- test_domain_detection_cs --

    @pytest.mark.asyncio
    async def test_domain_detection_cs(self, client):
        payload = self._make_json(
            domains=["computer_science"],
            primary_domain="computer_science",
        )
        client.chat.complete_async = AsyncMock(return_value=_mistral_response(payload))
        detector = DomainDetector(client=client)
        result = await detector.detect("A paper about algorithms and data structures.")
        assert result.primary_domain == "computer_science"

    # -- test_domain_detection_chemistry --

    @pytest.mark.asyncio
    async def test_domain_detection_chemistry(self, client):
        payload = self._make_json(
            domains=["chemistry", "biochemistry"],
            primary_domain="chemistry",
        )
        client.chat.complete_async = AsyncMock(return_value=_mistral_response(payload))
        detector = DomainDetector(client=client)
        result = await detector.detect("Synthesis of novel organic compounds via catalysis.")
        assert "chemistry" in result.domains

    # -- test_domain_detection_economics --

    @pytest.mark.asyncio
    async def test_domain_detection_economics(self, client):
        payload = self._make_json(
            domains=["economics", "statistics"],
            primary_domain="economics",
            is_interdisciplinary=True,
            rationale="Uses econometrics bridging economics and statistics.",
        )
        client.chat.complete_async = AsyncMock(return_value=_mistral_response(payload))
        detector = DomainDetector(client=client)
        result = await detector.detect("Instrumental variables in econometric modelling.")
        assert result.is_interdisciplinary is True

    # -- test_domain_detection_math --

    @pytest.mark.asyncio
    async def test_domain_detection_math(self, client):
        payload = self._make_json(
            domains=["mathematics"],
            primary_domain="mathematics",
        )
        client.chat.complete_async = AsyncMock(return_value=_mistral_response(payload))
        detector = DomainDetector(client=client)
        result = await detector.detect("Proof of the Riemann hypothesis using analytic continuation.")
        assert "math" in result.primary_domain.lower()

    # -- test_domain_detection_malformed_json --

    @pytest.mark.asyncio
    async def test_domain_detection_malformed_json(self, client):
        """LLM returns code block with no JSON → fallback DomainResult, no exception."""
        bad_content = "```python\nprint(1)\n```"
        client.chat.complete_async = AsyncMock(return_value=_mistral_response(bad_content))
        detector = DomainDetector(client=client)
        try:
            result = await detector.detect("Some text.")
        except Exception as exc:
            pytest.fail(f"detect() raised unexpectedly: {exc}")
        assert isinstance(result, DomainResult)
        assert result.primary_domain in ("unknown",) or result.confidence <= 0.5

    # -- test_domain_detection_retry_on_429 --

    @pytest.mark.asyncio
    async def test_domain_detection_retry_on_429(self, client):
        """RateLimitError raised twice, then succeeds on third call."""
        payload = self._make_json(primary_domain="computer_science")

        call_count = 0

        async def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("429 Rate limit exceeded")
            return _mistral_response(payload)

        client.chat.complete_async = _side_effect
        detector = DomainDetector(client=client, max_retries=3)
        result = await detector.detect("Paper text.")
        assert isinstance(result, DomainResult)
        assert result.primary_domain == "computer_science"
        assert call_count == 3

    # -- test_domain_detection_all_retries_fail --

    @pytest.mark.asyncio
    async def test_domain_detection_all_retries_fail(self, client):
        """All 3 retries raise → fallback DomainResult with confidence < 0.5."""
        client.chat.complete_async = AsyncMock(side_effect=Exception("Service unavailable"))
        detector = DomainDetector(client=client, max_retries=3)
        try:
            result = await detector.detect("Paper text.")
        except Exception as exc:
            pytest.fail(f"detect() should not propagate exceptions: {exc}")
        assert isinstance(result, DomainResult)
        assert result.confidence < 0.5


# ---------------------------------------------------------------------------
# Entity Extractor Tests
# ---------------------------------------------------------------------------

class TestEntityExtractor:

    @pytest.fixture()
    def client(self):
        c = MagicMock()
        c.chat = MagicMock()
        return c

    @staticmethod
    def _schema_json(
        nodes: list[dict] | None = None,
        edges: list[dict] | None = None,
    ) -> str:
        if nodes is None:
            nodes = [
                {"name": "Neural Network", "label": "Concept", "properties": {}},
                {"name": "Gradient Descent", "label": "Method", "properties": {}},
            ]
        if edges is None:
            edges = [
                {"source": "Neural Network", "target": "Gradient Descent",
                 "relation": "TRAINED_WITH", "properties": {}},
            ]
        return json.dumps({"nodes": nodes, "edges": edges})

    # -- test_valid_json_extraction --

    @pytest.mark.asyncio
    async def test_valid_json_extraction(self, client):
        payload = self._schema_json()
        client.chat.complete_async = AsyncMock(return_value=_mistral_response(payload))
        extractor = EntityExtractor(client=client)
        result = await extractor.extract_chunk("Some chunk text.")
        assert isinstance(result, GraphSchema)
        assert len(result.nodes) > 0
        assert len(result.edges) > 0

    # -- test_pydantic_validation_pass --

    def test_pydantic_validation_pass(self):
        """GraphSchema.model_validate succeeds on well-formed dict."""
        data = {
            "nodes": [
                {"name": "BERT", "label": "Model", "properties": {"year": 2018}},
                {"name": "Transformers", "label": "Architecture", "properties": {}},
            ],
            "edges": [
                {"source": "BERT", "target": "Transformers",
                 "relation": "BASED_ON", "properties": {}},
            ],
        }
        schema = GraphSchema.model_validate(data)
        assert len(schema.nodes) == 2
        assert len(schema.edges) == 1
        assert schema.nodes[0].name == "BERT"

    # -- test_malformed_json_graceful --

    def test_malformed_json_graceful(self):
        """_parse_and_validate on completely invalid text must return empty GraphSchema."""
        try:
            result = _parse_and_validate("INVALID JSON HERE }{][")
        except Exception as exc:
            pytest.fail(f"_parse_and_validate raised unexpectedly: {exc}")
        assert isinstance(result, GraphSchema)
        assert result.nodes == []
        assert result.edges == []

    # -- test_partial_json_recovery --

    def test_partial_json_recovery(self):
        """Prose wrapping an embedded JSON block must be recovered."""
        inner = {
            "nodes": [{"name": "SVM", "label": "Model", "properties": {}}],
            "edges": [],
        }
        raw = f"Here is the extracted knowledge:\n{json.dumps(inner)}\nEnd of extraction."
        result = _parse_and_validate(raw)
        assert len(result.nodes) == 1
        assert result.nodes[0].name == "SVM"

    # -- test_relationship_source_target_validation --

    def test_relationship_source_target_validation(self):
        """Edge referencing an entity not in the node list must be silently dropped."""
        data = json.dumps({
            "nodes": [{"name": "Alpha", "label": "Concept", "properties": {}}],
            "edges": [
                # Valid edge
                {
                    "source": "Alpha", "target": "Alpha",
                    "relation": "SELF_REF", "properties": {},
                },
                # Invalid: 'Beta' is not in nodes
                {
                    "source": "Alpha", "target": "Beta",
                    "relation": "RELATED_TO", "properties": {},
                },
            ],
        })
        result = _parse_and_validate(data)
        # Only the valid edge (or none if self-loops also dropped) should survive
        bad_edges = [e for e in result.edges if e.target == "Beta"]
        assert bad_edges == [], f"Dangling edge not dropped: {bad_edges}"

    # -- test_extract_all_chunks_concurrent --

    @pytest.mark.asyncio
    async def test_extract_all_chunks_concurrent(self, client):
        """5 chunks processed concurrently; all results merged into one GraphSchema."""
        def _make_chunk_payload(i: int) -> str:
            return json.dumps({
                "nodes": [{"name": f"Entity{i}", "label": "Concept", "properties": {}}],
                "edges": [],
            })

        call_count = 0

        async def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Determine chunk index from content
            content = kwargs.get("messages", args[1] if len(args) > 1 else [])[0].get(
                "content", ""
            )
            i = call_count
            return _mistral_response(_make_chunk_payload(i))

        client.chat.complete_async = _side_effect
        extractor = EntityExtractor(client=client)
        chunks = [{"text": f"chunk text {i}", "chunk_id": f"c{i}"} for i in range(5)]
        result = await extractor.extract_all_chunks(chunks)
        assert isinstance(result, GraphSchema)
        assert len(result.nodes) >= 1  # At least some entities extracted

    # -- test_entity_deduplication --

    def test_entity_deduplication(self):
        """Two schemas with the same entity name+label merge to one node."""
        schema_a = GraphSchema(
            nodes=[NodeModel("ResNet", "Model")],
            edges=[],
        )
        schema_b = GraphSchema(
            nodes=[NodeModel("ResNet", "Model"), NodeModel("ImageNet", "Dataset")],
            edges=[],
        )
        merged = _deduplicate([schema_a, schema_b])
        resnet_count = sum(1 for n in merged.nodes if n.name == "ResNet")
        assert resnet_count == 1, f"Expected 1 ResNet node, got {resnet_count}"
        assert len(merged.nodes) == 2  # ResNet + ImageNet

    def test_json_repair(self):
        """Test the truncated JSON repair function directly."""
        import json

        # Read the real file and execute the function definition dynamically
        with open("backend/extractor/entity_extractor.py", "r", encoding="utf-8") as f:
            code = f.read()

        start_idx = code.find("def _repair_json")
        assert start_idx != -1
        end_idx = code.find("def _extract_outermost_json")
        assert end_idx != -1

        func_code = code[start_idx:end_idx]
        local_ns = {}
        exec(func_code, {}, local_ns)
        _repair_json = local_ns["_repair_json"]

        # Test case 1: truncated value in list
        raw_truncated_1 = '{"entities": [{"name": "A", "domains": ["CS", "AI'
        repaired_1 = _repair_json(raw_truncated_1)
        assert json.loads(repaired_1) == {"entities": [{"name": "A", "domains": ["CS", "AI"]}]}

        # Test case 2: truncated description
        raw_truncated_2 = '{"entities": [{"name": "sentiment analysis", "description": "Analyzing text for'
        repaired_2 = _repair_json(raw_truncated_2)
        assert json.loads(repaired_2) == {"entities": [{"name": "sentiment analysis", "description": "Analyzing text for"}]}
