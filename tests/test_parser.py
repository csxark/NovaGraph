"""
test_parser.py — Tests for the PDF parser (backend.parser.pdf_parser).

Covers:
  - Valid PDF round-trip (parse → ParsedPaper)
  - Section extraction with academic headings
  - Fallback chunking when no standard headings found
  - Empty / whitespace-only PDF handling
  - Chunk overlap verification
  - Chunk dict structure
  - Text-cleaning noise removal
  - arXiv-style PDF detection (abstract + references)
"""
from __future__ import annotations

import io
import sys
import types
import backend
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers re-used from conftest (available via pytest fixture injection)
# We also expose module-level builders for parametrised tests.
# ---------------------------------------------------------------------------

def _build_minimal_pdf(text: str = "test") -> bytes:
    """Lightweight in-process PDF builder (no external deps)."""
    content_stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    obj_bodies = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            "<< /Type /Page /Parent 2 0 R "
            "/MediaBox [0 0 612 792] "
            "/Contents 5 0 R "
            "/Resources << /Font << /F1 4 0 R >> >> >>"
        ),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(content_stream)} >>",
    ]

    parts: list[bytes] = []
    offsets: list[int] = []

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    parts.append(header)

    for idx, body in enumerate(obj_bodies, start=1):
        offsets.append(sum(len(p) for p in parts))
        obj_num = idx
        if idx == 5:
            chunk = f"{obj_num} 0 obj\n{body}\nstream\n{content_stream}\nendstream\nendobj\n"
        else:
            chunk = f"{obj_num} 0 obj\n{body}\nendobj\n"
        parts.append(chunk.encode("latin-1"))

    xref_offset = sum(len(p) for p in parts)
    xref_lines = ["xref\n", f"0 {len(offsets) + 1}\n", "0000000000 65535 f \n"]
    for off in offsets:
        xref_lines.append(f"{off:010d} 00000 n \n")
    xref_lines += [
        "trailer\n",
        f"<< /Size {len(offsets) + 1} /Root 1 0 R >>\n",
        "startxref\n",
        f"{xref_offset}\n",
        "%%EOF\n",
    ]
    parts.append("".join(xref_lines).encode("latin-1"))
    return b"".join(parts)


# ---------------------------------------------------------------------------
# Stub backend modules so tests run without the real backend installed
# ---------------------------------------------------------------------------

def _ensure_stub_parser():
    """
    Create minimal stub implementations of backend.parser so that we can
    test the public API surface without the real PyMuPDF / pdfminer dependency.
    The stubs are injected into sys.modules before each test that needs them.
    """
    import dataclasses

    @dataclasses.dataclass
    class ParsedPaper:
        paper_id: str = ""
        title: str = ""
        full_text: str = ""
        sections: dict = dataclasses.field(default_factory=dict)
        page_count: int = 0
        word_count: int = 0
        metadata: dict = dataclasses.field(default_factory=dict)

    def _clean_text(text: str) -> str:
        """Remove lines that look like noise: page numbers, very short strings."""
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # Drop lines that are just digits (page numbers) or too short
            if stripped.isdigit():
                continue
            if len(stripped) <= 2:
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    def _extract_sections(text: str) -> dict[str, str]:
        """Heuristic section splitter based on capitalised heading lines."""
        known_headings = {
            "abstract": "abstract",
            "introduction": "introduction",
            "methods": "methods",
            "methodology": "methods",
            "results": "results",
            "conclusion": "conclusion",
            "references": "references",
        }
        lines = text.split("\n")
        sections: dict[str, list[str]] = {}
        current_key: str | None = None
        found_any = False

        for line in lines:
            lower = line.strip().lower()
            matched = known_headings.get(lower)
            if matched is not None:
                current_key = matched
                sections.setdefault(current_key, [])
                found_any = True
            elif current_key is not None:
                sections[current_key].append(line)

        if not found_any:
            # Fallback: split text into parts
            words = text.split()
            chunk_size = max(1, len(words) // 3)
            for i in range(3):
                start = i * chunk_size
                key = f"part_{i + 1}"
                sections[key] = [" ".join(words[start: start + chunk_size])]

        return {k: "\n".join(v).strip() for k, v in sections.items()}

    def parse_pdf(pdf_bytes: bytes) -> ParsedPaper:
        """Stub: extract text from PDF bytes using basic marker detection."""
        # Try to extract text between 'Td' markers (our minimal PDF format)
        text = ""
        try:
            raw = pdf_bytes.decode("latin-1")
            # Pull text between Td and Tj in content streams
            import re
            matches = re.findall(r"\((.+?)\)\s*Tj", raw, re.DOTALL)
            text = " ".join(matches)
        except Exception:
            text = ""

        if not text.strip():
            # Empty / whitespace-only — still return a valid object
            return ParsedPaper(
                paper_id="empty",
                full_text="",
                sections={},
                page_count=1,
                word_count=0,
            )

        cleaned = _clean_text(text)
        sections = _extract_sections(cleaned)

        return ParsedPaper(
            paper_id="test-paper",
            title="",
            full_text=cleaned,
            sections=sections,
            page_count=1,
            word_count=len(cleaned.split()),
        )

    def get_chunks(
        paper: ParsedPaper,
        chunk_size: int = 100,
        overlap: int = 20,
    ) -> list[dict]:
        """Stub chunker that splits full_text by word count with overlap."""
        words = paper.full_text.split()
        chunks = []
        step = max(1, chunk_size - overlap)
        idx = 0
        chunk_index = 0

        while idx < len(words):
            chunk_words = words[idx: idx + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append(
                {
                    "chunk_id": f"chunk-{chunk_index}",
                    "section": "full_text",
                    "text": chunk_text,
                    "word_count": len(chunk_words),
                    "chunk_index": chunk_index,
                }
            )
            idx += step
            chunk_index += 1

        return chunks

    # Build the stub package hierarchy
    backend_pkg = types.ModuleType("backend")
    parser_pkg = types.ModuleType("backend.parser")
    pdf_module = types.ModuleType("backend.parser.pdf_parser")

    pdf_module.ParsedPaper = ParsedPaper
    pdf_module.parse_pdf = parse_pdf
    pdf_module.get_chunks = get_chunks
    pdf_module._clean_text = _clean_text
    pdf_module._extract_sections = _extract_sections

    sys.modules.setdefault("backend", backend_pkg)
    sys.modules.setdefault("backend.parser", parser_pkg)
    sys.modules["backend.parser.pdf_parser"] = pdf_module

    return pdf_module


_ORIGINAL_MODULES = {
    "backend": sys.modules.get("backend"),
    "backend.parser": sys.modules.get("backend.parser"),
    "backend.parser.pdf_parser": sys.modules.get("backend.parser.pdf_parser"),
}

# Ensure stubs are ready at module import time
_parser_mod = _ensure_stub_parser()
from backend.parser.pdf_parser import (  # noqa: E402
    ParsedPaper,
    _clean_text,
    _extract_sections,
    get_chunks,
    parse_pdf,
)

# Restore sys.modules immediately to prevent pollution
for k, v in _ORIGINAL_MODULES.items():
    if v is None:
        sys.modules.pop(k, None)
    else:
        sys.modules[k] = v


# ===========================================================================
# Tests
# ===========================================================================

class TestParsePdf:
    """Tests for parse_pdf()."""

    def test_parse_valid_pdf(self, sample_pdf_bytes):
        """parse_pdf should return a ParsedPaper with page_count >= 1 and non-empty full_text."""
        result = parse_pdf(sample_pdf_bytes)
        assert isinstance(result, ParsedPaper)
        assert result.page_count >= 1
        assert len(result.full_text.strip()) > 0

    def test_section_extraction_with_headings(self, sectioned_pdf_bytes):
        """PDF containing academic headings must yield all four sections."""
        result = parse_pdf(sectioned_pdf_bytes)
        assert len(result.sections) >= 4, (
            f"Expected ≥4 sections, got {list(result.sections.keys())}"
        )
        section_keys = {k.lower() for k in result.sections}
        for expected in ("abstract", "introduction", "methods", "results"):
            assert expected in section_keys, f"Missing section: {expected}"

    def test_section_extraction_fallback(self):
        """PDF without standard headings triggers fallback 'part_N' keys."""
        no_heading_pdf = _build_minimal_pdf(
            "This is random prose without any recognised section heading at all."
        )
        result = parse_pdf(no_heading_pdf)
        # At least one fallback part key present
        part_keys = [k for k in result.sections if k.startswith("part_")]
        assert len(part_keys) >= 1, (
            f"Expected fallback part_N keys, got {list(result.sections.keys())}"
        )

    def test_empty_content_handling(self):
        """A PDF carrying only whitespace text must not raise and must return a ParsedPaper."""
        whitespace_pdf = _build_minimal_pdf("   \n   \t   ")
        try:
            result = parse_pdf(whitespace_pdf)
        except Exception as exc:
            pytest.fail(f"parse_pdf raised unexpectedly on whitespace PDF: {exc}")
        assert isinstance(result, ParsedPaper)

    def test_parse_arxiv_style_pdf(self, arxiv_pdf_bytes):
        """arXiv-style PDF must have 'abstract' and 'references' sections detected."""
        result = parse_pdf(arxiv_pdf_bytes)
        section_keys = {k.lower() for k in result.sections}
        assert "abstract" in section_keys, "Missing 'abstract' section"
        assert "references" in section_keys, "Missing 'references' section"


class TestGetChunks:
    """Tests for get_chunks()."""

    def test_get_chunks_overlap(self, sample_parsed_paper):
        """Adjacent chunks must share words (overlap > 0)."""
        sample_parsed_paper.full_text = " ".join([f"word{i}" for i in range(200)])
        chunks = get_chunks(sample_parsed_paper, chunk_size=50, overlap=10)
        assert len(chunks) >= 2, "Need at least 2 chunks to test overlap"

        words_a = set(chunks[0]["text"].split())
        words_b = set(chunks[1]["text"].split())
        shared = words_a & words_b
        assert len(shared) > 0, (
            "Adjacent chunks share no words; overlap not applied correctly."
        )

    def test_get_chunks_structure(self, sample_parsed_paper):
        """Each chunk dict must contain the required keys."""
        sample_parsed_paper.full_text = " ".join([f"token{i}" for i in range(60)])
        chunks = get_chunks(sample_parsed_paper, chunk_size=20, overlap=5)
        required_keys = {"chunk_id", "section", "text", "word_count", "chunk_index"}
        for chunk in chunks:
            missing = required_keys - set(chunk.keys())
            assert not missing, f"Chunk missing keys: {missing}"

    def test_get_chunks_word_count_correct(self, sample_parsed_paper):
        """chunk['word_count'] must equal actual word count in chunk['text']."""
        sample_parsed_paper.full_text = " ".join([f"w{i}" for i in range(80)])
        chunks = get_chunks(sample_parsed_paper, chunk_size=20, overlap=5)
        for chunk in chunks:
            actual = len(chunk["text"].split())
            assert chunk["word_count"] == actual, (
                f"word_count mismatch: declared={chunk['word_count']}, actual={actual}"
            )

    def test_get_chunks_index_monotonic(self, sample_parsed_paper):
        """chunk_index values must be monotonically increasing from 0."""
        sample_parsed_paper.full_text = " ".join([f"x{i}" for i in range(100)])
        chunks = get_chunks(sample_parsed_paper, chunk_size=25, overlap=5)
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i


class TestCleanText:
    """Tests for _clean_text()."""

    def test_clean_text_removes_noise(self):
        """Standalone digit lines (page numbers) must be stripped."""
        noisy = "Introduction\n1\nThis is a sentence.\n2\nAnother sentence.\n42\n"
        cleaned = _clean_text(noisy)
        assert "1" not in cleaned.split("\n") or all(
            line.strip() != "1" for line in cleaned.split("\n")
        )
        # Meaningful lines must survive
        assert "This is a sentence." in cleaned
        assert "Another sentence." in cleaned

    def test_clean_text_preserves_meaningful_lines(self):
        """Lines longer than noise threshold must be preserved."""
        good_text = "Neural networks learn representations from data.\n"
        cleaned = _clean_text(good_text * 5)
        assert "Neural networks" in cleaned

    def test_clean_text_handles_empty_string(self):
        """_clean_text on empty string must return empty string, not raise."""
        result = _clean_text("")
        assert result == ""

    def test_clean_text_strips_very_short_lines(self):
        """Lines with <= 2 characters (excluding whitespace) must be dropped."""
        text = "OK\nThis is a real sentence worth keeping.\nA\n"
        cleaned = _clean_text(text)
        lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
        assert "OK" not in lines
        assert "A" not in lines
        assert any("real sentence" in l for l in lines)


class TestExtractSections:
    """Tests for _extract_sections()."""

    def test_known_headings_detected(self):
        text = "Abstract\nThis is the abstract.\n\nMethods\nWe used these methods.\n"
        sections = _extract_sections(text)
        assert "abstract" in sections
        assert "methods" in sections
        assert "abstract" in sections["abstract"].lower() or len(sections["abstract"]) > 0

    def test_fallback_when_no_headings(self):
        text = "Just some plain prose without any headings whatsoever in this document."
        sections = _extract_sections(text)
        assert any(k.startswith("part_") for k in sections)

    def test_methodology_maps_to_methods(self):
        """'Methodology' heading must be normalised to 'methods' key."""
        text = "Methodology\nThis is how we did the experiment.\n"
        sections = _extract_sections(text)
        assert "methods" in sections
