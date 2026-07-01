"""
PDF parser module using PyMuPDF (fitz).

Extracts structured sections, full text, metadata, and overlapping chunks
from academic PDF files supplied as raw bytes.
"""

from __future__ import annotations

import fitz  # PyMuPDF
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_HEADINGS: list[str] = [
    'abstract',
    'introduction',
    'background',
    'related work',
    'methodology',
    'methods',
    'materials and methods',
    'experimental setup',
    'results',
    'experiments',
    'discussion',
    'conclusion',
    'conclusions',
    'references',
    'bibliography',
    'acknowledgments',
    'appendix',
]

# Pre-compiled pattern to match numbered or un-numbered section headings.
# Matches lines like:
#   "Abstract", "1. Introduction", "2.1 Methods", "III. Results"
_HEADING_PATTERN = re.compile(
    r'^\s*(?:'
    r'(?:[IVX]+\.?\s+)'                 # Roman numerals (I. II. III.)
    r'|(?:\d+(?:\.\d+)*\.?\s+)'         # Arabic numbers (1. 2.1 3.2.1)
    r'|(?:[A-Z]\.\s+)'                  # Letter headings (A. B.)
    r')?'
    r'('
    + '|'.join(re.escape(h) for h in STANDARD_HEADINGS)
    + r')'
    r'[\s:]*$',
    re.IGNORECASE | re.MULTILINE,
)

# Ligature normalisation table
_LIGATURE_MAP: dict[str, str] = {
    '\ufb00': 'ff',
    '\ufb01': 'fi',
    '\ufb02': 'fl',
    '\ufb03': 'ffi',
    '\ufb04': 'ffl',
    '\ufb05': 'st',
    '\ufb06': 'st',
}
_LIGATURE_RE = re.compile('|'.join(re.escape(k) for k in _LIGATURE_MAP))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ParsedPaper:
    """Structured representation of a parsed academic PDF."""

    sections: dict[str, str] = field(default_factory=dict)
    full_text: str = ''
    page_count: int = 0
    title: str = ''
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf(file_bytes: bytes) -> ParsedPaper:
    """Parse a PDF supplied as *file_bytes* and return a :class:`ParsedPaper`.

    Args:
        file_bytes: Raw PDF file content.

    Returns:
        A :class:`ParsedPaper` instance with extracted text, sections, and
        metadata.
    """
    try:
        doc: fitz.Document = fitz.open(stream=file_bytes, filetype='pdf')
    except Exception as exc:
        logger.error('Failed to open PDF: %s', exc)
        raise ValueError(f'Cannot open PDF: {exc}') from exc

    page_count = doc.page_count

    # ---- Extract raw text page-by-page ----
    raw_pages: list[str] = []
    for page_index in range(page_count):
        try:
            page: fitz.Page = doc[page_index]
            raw_pages.append(page.get_text())
        except Exception as exc:
            logger.warning('Failed to read page %d: %s', page_index, exc)
            raw_pages.append('')

    full_raw = '\n'.join(raw_pages)

    # ---- Extract metadata ----
    meta = doc.metadata or {}
    title = _extract_title(meta, full_raw)

    # ---- Clean full text ----
    full_text = _clean_text(full_raw)

    # ---- Detect sections ----
    sections = _extract_sections(full_text)

    doc.close()

    return ParsedPaper(
        sections=sections,
        full_text=full_text,
        page_count=page_count,
        title=title,
        metadata={
            'author': meta.get('author', ''),
            'creator': meta.get('creator', ''),
            'producer': meta.get('producer', ''),
            'subject': meta.get('subject', ''),
            'keywords': meta.get('keywords', ''),
            'creation_date': meta.get('creationDate', ''),
            'mod_date': meta.get('modDate', ''),
        },
    )


def get_chunks(
    parsed: ParsedPaper,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[dict]:
    """Split every section of *parsed* into overlapping word-window chunks.

    Args:
        parsed:     A :class:`ParsedPaper` returned by :func:`parse_pdf`.
        chunk_size: Target number of words per chunk.
        overlap:    Number of words to overlap between consecutive chunks.

    Returns:
        A flat list of chunk dicts, each with keys:
        ``chunk_id``, ``section``, ``text``, ``word_count``, ``chunk_index``.
    """
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    if overlap < 0:
        raise ValueError('overlap must be non-negative')
    if overlap >= chunk_size:
        raise ValueError('overlap must be less than chunk_size')

    all_chunks: list[dict] = []

    for section_name, section_text in parsed.sections.items():
        words = section_text.split()
        if not words:
            continue

        step = chunk_size - overlap
        chunk_index = 0

        for start in range(0, len(words), step):
            window = words[start: start + chunk_size]
            if not window:
                break

            text = ' '.join(window)
            chunk_id = f'{section_name}_{chunk_index}'

            all_chunks.append(
                {
                    'chunk_id': chunk_id,
                    'section': section_name,
                    'text': text,
                    'word_count': len(window),
                    'chunk_index': chunk_index,
                }
            )
            chunk_index += 1

            # If the last window is shorter than chunk_size the loop ends
            if start + chunk_size >= len(words):
                break

    logger.debug(
        'Produced %d chunks across %d sections.',
        len(all_chunks),
        len(parsed.sections),
    )
    return all_chunks


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_title(meta: dict, raw_text: str) -> str:
    """Return the paper title from PDF metadata or heuristically from text."""
    title = (meta.get('title') or '').strip()
    if title:
        return title

    # Fallback: first non-empty line that is neither a single word nor
    # too short (avoids grabbing page numbers / running headers).
    for line in raw_text.splitlines():
        line = line.strip()
        if 10 < len(line) < 300 and not line.isdigit():
            return line

    return 'Unknown Title'


def _extract_sections(full_text: str) -> dict[str, str]:
    """Split *full_text* into a dict mapping canonical section names to text.

    If fewer than 2 sections are detected the text is chunked into three
    equal parts labelled ``part_1``, ``part_2``, ``part_3``.
    """
    lines = full_text.splitlines(keepends=True)
    sections: dict[str, str] = {}
    current_heading: Optional[str] = None
    buffer: list[str] = []

    for line in lines:
        match = _HEADING_PATTERN.match(line)
        if match:
            # Flush buffer into the previous section
            if current_heading is not None:
                content = ''.join(buffer).strip()
                if content:
                    if current_heading in sections:
                        # Append if heading appears more than once (rare)
                        sections[current_heading] += '\n' + content
                    else:
                        sections[current_heading] = content
            # Normalise to canonical heading
            detected = match.group(1).strip().lower()
            current_heading = _canonical_heading(detected)
            buffer = []
        else:
            buffer.append(line)

    # Flush the last section
    if current_heading is not None:
        content = ''.join(buffer).strip()
        if content:
            if current_heading in sections:
                sections[current_heading] += '\n' + content
            else:
                sections[current_heading] = content
    elif buffer:
        # Text before any detected heading
        content = ''.join(buffer).strip()
        if content:
            sections['preamble'] = content

    # Fallback: too few sections detected
    if len(sections) < 2:
        logger.warning(
            'Fewer than 2 sections detected (%d). '
            'Falling back to three-part chunking.',
            len(sections),
        )
        sections = _fallback_sections(full_text)

    logger.info('Sections detected: %s', list(sections.keys()))
    return sections


def _canonical_heading(detected: str) -> str:
    """Map a detected heading string to the closest canonical heading name."""
    detected_lower = detected.lower().strip()
    if detected_lower in STANDARD_HEADINGS:
        return detected_lower
    # Partial / substring match
    for canonical in STANDARD_HEADINGS:
        if canonical in detected_lower or detected_lower in canonical:
            return canonical
    return detected_lower


def _fallback_sections(full_text: str) -> dict[str, str]:
    """Divide *full_text* into three roughly equal parts."""
    words = full_text.split()
    n = len(words)
    if n == 0:
        return {'part_1': '', 'part_2': '', 'part_3': ''}

    third = max(1, n // 3)
    return {
        'part_1': ' '.join(words[:third]),
        'part_2': ' '.join(words[third: 2 * third]),
        'part_3': ' '.join(words[2 * third:]),
    }


def _clean_text(text: str) -> str:
    """Normalise *text* extracted from a PDF.

    Steps:
    1. Replace Unicode ligatures (ﬁ → fi, etc.).
    2. Remove lines shorter than 5 characters (headers, footers, page numbers).
    3. Collapse runs of whitespace / blank lines.
    """
    # 1. Ligatures
    text = _LIGATURE_RE.sub(lambda m: _LIGATURE_MAP[m.group()], text)

    # 2. Remove very short lines (page numbers, running headers/footers)
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        # Keep empty lines (paragraph breaks) but drop noise lines
        if stripped == '' or (len(stripped) >= 5 and not _is_page_number(stripped)):
            cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # 3. Collapse more than two consecutive newlines into exactly two
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 4. Normalise horizontal whitespace within lines
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


def _is_page_number(line: str) -> bool:
    """Return True if *line* looks like a standalone page number."""
    return bool(re.fullmatch(r'\d{1,4}', line.strip()))
