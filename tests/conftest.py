"""
conftest.py — shared pytest fixtures for the GraphRAG Research Assistant test suite.
All external services (Mistral, Neo4j, Pinecone) are mocked; no real network calls.
"""
from __future__ import annotations

import io
import struct
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal raw PDF builder (no external dependency)
# ---------------------------------------------------------------------------

def _build_minimal_pdf(text: str = "GraphRAG test content") -> bytes:
    """
    Construct a minimal but structurally valid PDF-1.4 document in memory.
    Sufficient for pdfminer / PyMuPDF to open without errors.
    """
    lines: list[str] = []

    def emit(s: str) -> int:
        """Append line, return byte offset of its start."""
        offset = sum(len(l.encode()) + 1 for l in lines)
        lines.append(s)
        return offset

    # Header
    emit("%PDF-1.4")
    emit("%\xe2\xe3\xcf\xd3")  # binary comment so readers treat as binary

    # Object 1 — catalog
    obj1_offset = sum(len(l.encode()) + 1 for l in lines)
    emit("1 0 obj")
    emit("<< /Type /Catalog /Pages 2 0 R >>")
    emit("endobj")

    # Object 2 — pages
    obj2_offset = sum(len(l.encode()) + 1 for l in lines)
    emit("2 0 obj")
    emit("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    emit("endobj")

    # Object 4 — font
    obj4_offset = sum(len(l.encode()) + 1 for l in lines)
    emit("4 0 obj")
    emit("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    emit("endobj")

    # Object 5 — content stream
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    obj5_offset = sum(len(l.encode()) + 1 for l in lines)
    emit("5 0 obj")
    emit(f"<< /Length {len(content)} >>")
    emit("stream")
    emit(content)
    emit("endstream")
    emit("endobj")

    # Object 3 — page
    obj3_offset = sum(len(l.encode()) + 1 for l in lines)
    emit("3 0 obj")
    emit(
        "<< /Type /Page /Parent 2 0 R "
        "/MediaBox [0 0 612 792] "
        "/Contents 5 0 R "
        "/Resources << /Font << /F1 4 0 R >> >> >>"
    )
    emit("endobj")

    # Cross-reference table
    xref_offset = sum(len(l.encode()) + 1 for l in lines)
    emit("xref")
    emit("0 6")
    emit("0000000000 65535 f ")
    for off in (obj1_offset, obj2_offset, obj3_offset, obj4_offset, obj5_offset):
        emit(f"{off:010d} 00000 n ")

    emit("trailer")
    emit("<< /Size 6 /Root 1 0 R >>")
    emit("startxref")
    emit(str(xref_offset))
    emit("%%EOF")

    return "\n".join(lines).encode("latin-1")


def _build_sectioned_pdf() -> bytes:
    """PDF whose text body contains common academic section headings."""
    body = (
        "Abstract\n"
        "This paper studies neural networks and deep learning.\n\n"
        "Introduction\n"
        "Neural networks are universal function approximators.\n\n"
        "Methods\n"
        "We use gradient descent with Adam optimiser.\n\n"
        "Results\n"
        "We achieve 95% accuracy on the benchmark dataset.\n"
    )
    return _build_minimal_pdf(body)


def _build_arxiv_pdf() -> bytes:
    """Mimics an arXiv-style structure with Abstract + References."""
    body = (
        "Abstract\n"
        "We present a novel approach to entity extraction using large language models.\n\n"
        "1 Introduction\n"
        "Knowledge graphs have been widely studied.\n\n"
        "References\n"
        "[1] LeCun et al. (1998). Gradient-based learning applied to document recognition.\n"
    )
    return _build_minimal_pdf(body)


# ---------------------------------------------------------------------------
# Settings fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def settings():
    """Return a mock Settings object with safe test-only values."""
    mock_settings = MagicMock()
    mock_settings.mistral_api_key = "test-mistral-key-00000000"
    mock_settings.mistral_small_model = "mistral-small-latest"
    mock_settings.mistral_large_model = "mistral-large-latest"
    mock_settings.mistral_timeout = 30
    mock_settings.mistral_max_retries = 3
    mock_settings.neo4j_uri = "bolt://localhost:7687"
    mock_settings.neo4j_username = "neo4j"
    mock_settings.neo4j_password = "test-password"
    mock_settings.neo4j_database = "neo4j"
    mock_settings.pinecone_api_key = "test-pinecone-key-00000000"
    mock_settings.pinecone_index_name = "graphrag-test"
    mock_settings.pinecone_environment = "us-east-1"
    mock_settings.embedding_model = "all-MiniLM-L6-v2"
    mock_settings.embedding_dim = 384
    mock_settings.embedding_batch_size = 32
    mock_settings.max_upload_size_mb = 50
    mock_settings.cors_origins = ["http://localhost:5173"]
    mock_settings.job_ttl_seconds = 86400
    return mock_settings


# ---------------------------------------------------------------------------
# PDF / paper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_pdf_bytes() -> bytes:
    """Minimal valid PDF with generic text content."""
    return _build_minimal_pdf("GraphRAG Research Assistant test document page one.")


@pytest.fixture()
def sectioned_pdf_bytes() -> bytes:
    """PDF containing Abstract / Introduction / Methods / Results headings."""
    return _build_sectioned_pdf()


@pytest.fixture()
def arxiv_pdf_bytes() -> bytes:
    """Minimal arXiv-style PDF with Abstract and References sections."""
    return _build_arxiv_pdf()


@pytest.fixture()
def sample_parsed_paper():
    """
    Pre-built ParsedPaper-like object using a plain dataclass so tests run
    even if backend imports are unavailable in isolation.
    """
    from dataclasses import dataclass, field

    @dataclass
    class _ParsedPaper:
        paper_id: str = "paper-test-001"
        title: str = "Neural Networks for Knowledge Graphs"
        full_text: str = (
            "This paper studies neural networks. "
            "We use gradient descent. "
            "We achieve 95% accuracy."
        )
        sections: dict = field(default_factory=dict)
        page_count: int = 3
        word_count: int = 150
        metadata: dict = field(default_factory=dict)

    paper = _ParsedPaper()
    paper.sections = {
        "abstract": "This paper studies neural networks and deep learning architectures.",
        "methods": "We use gradient descent with momentum and adaptive learning rates.",
        "results": "We achieve 95% accuracy on the MNIST benchmark dataset.",
    }
    paper.metadata = {"source": "test", "authors": ["Alice", "Bob"]}
    return paper


# ---------------------------------------------------------------------------
# Domain / extractor fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_domain_result():
    """DomainResult for a computer-science / machine-learning paper."""
    try:
        from backend.extractor.domain_detector import DomainResult

        return DomainResult(
            domains=["machine_learning", "computer_science"],
            primary_domain="machine_learning",
            is_interdisciplinary=False,
            confidence=0.95,
            rationale="Deep learning paper focused on neural network optimisation.",
        )
    except ImportError:
        # Fallback plain object for isolated test environments
        from dataclasses import dataclass

        @dataclass
        class _DomainResult:
            domains: list
            primary_domain: str
            is_interdisciplinary: bool
            confidence: float
            rationale: str

        return _DomainResult(
            domains=["machine_learning", "computer_science"],
            primary_domain="machine_learning",
            is_interdisciplinary=False,
            confidence=0.95,
            rationale="Deep learning paper focused on neural network optimisation.",
        )


@pytest.fixture()
def mock_graph_schema():
    """GraphSchema with 3 NodeModels and 2 EdgeModels."""
    try:
        from backend.extractor.entity_extractor import EdgeModel, GraphSchema, NodeModel

        nodes = [
            NodeModel(
                name="Neural Network",
                label="Concept",
                properties={"description": "A computational model inspired by the brain"},
            ),
            NodeModel(
                name="Gradient Descent",
                label="Method",
                properties={"description": "Optimisation algorithm"},
            ),
            NodeModel(
                name="MNIST",
                label="Dataset",
                properties={"description": "Handwritten digit benchmark"},
            ),
        ]
        edges = [
            EdgeModel(
                source="Neural Network",
                target="MNIST",
                relation="EVALUATED_ON",
                properties={"metric": "accuracy"},
            ),
            EdgeModel(
                source="Neural Network",
                target="Gradient Descent",
                relation="TRAINED_WITH",
                properties={},
            ),
        ]
        return GraphSchema(nodes=nodes, edges=edges)
    except ImportError:
        from dataclasses import dataclass, field

        @dataclass
        class _Node:
            name: str
            label: str
            properties: dict = field(default_factory=dict)

        @dataclass
        class _Edge:
            source: str
            target: str
            relation: str
            properties: dict = field(default_factory=dict)

        @dataclass
        class _GraphSchema:
            nodes: list
            edges: list

        return _GraphSchema(
            nodes=[
                _Node("Neural Network", "Concept"),
                _Node("Gradient Descent", "Method"),
                _Node("MNIST", "Dataset"),
            ],
            edges=[
                _Edge("Neural Network", "MNIST", "EVALUATED_ON"),
                _Edge("Neural Network", "Gradient Descent", "TRAINED_WITH"),
            ],
        )


# ---------------------------------------------------------------------------
# Mistral mock fixture
# ---------------------------------------------------------------------------

def _make_mistral_message(content: str):
    """Build a minimal mock Mistral chat completion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture()
def mock_mistral_response():
    """
    Factory fixture: call mock_mistral_response(content) to get a mock
    Mistral ChatCompletion with the given string as message content.
    """
    return _make_mistral_message


# ---------------------------------------------------------------------------
# Neo4j mock fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def neo4j_mock():
    """
    AsyncMock Neo4j driver that supports `async with driver.session() as s`.
    The session exposes `run`, `execute_query`, and async iteration on records.
    """
    session = AsyncMock()
    session.run = AsyncMock()
    session.execute_query = AsyncMock(return_value=([], None, None))
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    driver = AsyncMock()
    driver.session = MagicMock(return_value=session)
    driver.close = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    return driver


# ---------------------------------------------------------------------------
# Pinecone mock fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def pinecone_mock():
    """MagicMock Pinecone index with query / upsert / delete methods."""
    index = MagicMock()

    # Default query response: 3 matching vectors
    index.query.return_value = MagicMock(
        matches=[
            MagicMock(id="vec-1", score=0.95, metadata={"text": "chunk one", "paper_id": "p1"}),
            MagicMock(id="vec-2", score=0.88, metadata={"text": "chunk two", "paper_id": "p1"}),
            MagicMock(id="vec-3", score=0.75, metadata={"text": "chunk three", "paper_id": "p1"}),
        ]
    )
    index.upsert = MagicMock(return_value={"upserted_count": 1})
    index.delete = MagicMock()
    index.describe_index_stats = MagicMock(return_value={"total_vector_count": 42})
    return index


# ---------------------------------------------------------------------------
# Helpers re-exported for test modules
# ---------------------------------------------------------------------------

__all__ = [
    "settings",
    "sample_pdf_bytes",
    "sectioned_pdf_bytes",
    "arxiv_pdf_bytes",
    "sample_parsed_paper",
    "mock_domain_result",
    "mock_graph_schema",
    "mock_mistral_response",
    "neo4j_mock",
    "pinecone_mock",
    "_build_minimal_pdf",
    "_build_sectioned_pdf",
    "_build_arxiv_pdf",
    "_make_mistral_message",
]
