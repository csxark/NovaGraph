"""
Entity and relationship extractor module.

Uses the Mistral small model to extract a knowledge-graph schema (entities +
relationships) from individual text chunks, then merges results across all
chunks with deduplication.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Literal, Optional, Any

from mistralai.client import Mistral
from pydantic import BaseModel, Field, field_validator, model_validator

from backend.config import Settings
from backend.extractor.domain_detector import DomainResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

NodeType = Literal[
    'Concept',
    'Method',
    'Evidence',
    'Finding',
    'Entity',
    'Reference',
    'Proposition',
    'Assumption',
]

EdgeType = Literal[
    'USES',
    'SUPPORTS',
    'CONTRADICTS',
    'ESTABLISHES',
    'DERIVES_FROM',
    'COMPARES',
    'APPLIES_TO',
    'REFERENCES',
    'CAUSES',
    'EQUIVALENT_TO',
    'SCOPES',
    'INCLUDES',
    'PART_OF',
    'RELATED_TO',
]


class NodeModel(BaseModel):
    """A single entity node in the knowledge graph."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description='Canonical entity name.')
    type: NodeType = Field(..., description='Ontological type of the entity.')
    subtype: Optional[str] = Field(default=None, description='Fine-grained sub-type.')
    description: str = Field(..., description='Short description of the entity.')
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description='Extraction confidence.',
    )
    domains: list[str] = Field(default_factory=list)
    chunk_ref: str = Field(default='', description='chunk_id this entity was found in.')
    paper_id: str = Field(default='', description='Parent paper identifier.')

    @model_validator(mode='before')
    @classmethod
    def preprocess_fields(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            allowed = set(NodeType.__args__)  # type: ignore[attr-defined]
            if 'type' in obj and obj['type'] not in allowed:
                logger.debug(
                    'Unknown node type %r — remapping to Entity.', obj['type']
                )
                obj = {**obj, 'type': 'Entity'}
        return obj

    @property
    def entity_id(self) -> str:
        return self.id


class EdgeModel(BaseModel):
    """A directed relationship between two entities."""

    source_name: str = Field(..., description='Name of the source entity.')
    target_name: str = Field(..., description='Name of the target entity.')
    type: EdgeType = Field(..., description='Relationship type.')
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: str = Field(default='', description='Supporting text snippet.')
    paper_id: str = Field(default='', description='Parent paper identifier.')

    @model_validator(mode='before')
    @classmethod
    def preprocess_fields(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            # Coerce 'name' → 'source_name' if LLM uses wrong field name
            if 'source_name' not in obj and 'name' in obj:
                obj = {**obj, 'source_name': obj.pop('name')}
            # Map unknown edge types to RELATED_TO instead of failing
            allowed = set(EdgeType.__args__)  # type: ignore[attr-defined]
            if 'type' in obj and obj['type'] not in allowed:
                logger.debug(
                    'Unknown edge type %r — remapping to RELATED_TO.', obj['type']
                )
                obj = {**obj, 'type': 'RELATED_TO'}
        return obj


class GraphSchema(BaseModel):
    """Container for a set of entities and relationships."""

    entities: list[NodeModel] = Field(default_factory=list)
    relationships: list[EdgeModel] = Field(default_factory=list)

    @field_validator('relationships', mode='before')
    @classmethod
    def filter_bad_relationships(cls, v: Any) -> Any:
        if isinstance(v, list):
            valid_rels = []
            for item in v:
                if isinstance(item, dict):
                    # Keep if it has either source_name or name
                    if 'source_name' in item or 'name' in item:
                        valid_rels.append(item)
                    else:
                        logger.warning(
                            'Filtering out relationship dict missing both source_name and name: %r',
                            item,
                        )
                else:
                    valid_rels.append(item)
            return valid_rels
        return v


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT: str = (
    'You are an expert knowledge-graph builder specialising in academic research. '
    'Extract entities (concepts, methods, findings, evidence, propositions, '
    'assumptions, references) and their relationships from the provided text. '
    'Be precise and conservative — only extract entities that are clearly stated. '
    'Return ONLY valid JSON with no additional commentary.'
)

EXTRACTION_USER_PROMPT_TEMPLATE: str = (
    'Domain: {domain}\n'
    'Chunk {chunk_index} of {total} — Section: {section}\n\n'
    'TEXT:\n'
    '"""\n'
    '{text}\n'
    '"""\n\n'
    'Extract a knowledge graph from the text above.\n'
    'Return ONLY this JSON structure (no markdown, no explanation):\n'
    '{{\n'
    '  "entities": [\n'
    '    {{\n'
    '      "name": "<entity name>",\n'
    '      "type": "<Concept|Method|Evidence|Finding|Entity|Reference|Proposition|Assumption>",\n'
    '      "subtype": "<optional fine-grained type or null>",\n'
    '      "description": "<one-sentence description>",\n'
    '      "confidence": <float 0.0-1.0>,\n'
    '      "domains": ["<domain1>"]\n'
    '    }}\n'
    '  ],\n'
    '  "relationships": [\n'
    '    {{\n'
    '      "source_name": "<entity name>",\n'
    '      "target_name": "<entity name>",\n'
    '      "type": "<USES|SUPPORTS|CONTRADICTS|ESTABLISHES|DERIVES_FROM|COMPARES|APPLIES_TO|REFERENCES|CAUSES|EQUIVALENT_TO|SCOPES>",\n'
    '      "weight": <float 0.0-1.0>,\n'
    '      "evidence": "<short verbatim quote or empty string>"\n'
    '    }}\n'
    '  ]\n'
    '}}'
)

# Regex that captures the outermost JSON object (non-greedy is intentionally
# avoided; we use a brace-counting approach via the helper below).
_JSON_FENCE_RE = re.compile(r'```(?:json)?', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Token-bucket rate limiter — enforces N requests/sec with no extra delay."""

    def __init__(self, rate: float = 1.0) -> None:
        self._min_interval = 1.0 / rate
        self._lock = asyncio.Lock()
        self._last_call: float = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            wait = self._min_interval - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_entities(
    chunk: dict,
    domain_result: DomainResult,
    settings: Settings,
) -> GraphSchema:
    """Extract entities and relationships from a single text *chunk*.

    Args:
        chunk:         A chunk dict as produced by
                       :func:`~backend.parser.pdf_parser.get_chunks`.
        domain_result: Domain classification for the parent paper.
        settings:      Application settings.

    Returns:
        A :class:`GraphSchema` — always, even on failure (may be empty).
    """
    text: str = chunk.get('text', '').strip()
    if not text:
        logger.debug('Empty chunk %s; skipping extraction.', chunk.get('chunk_id'))
        return GraphSchema()

    section: str = chunk.get('section', 'unknown')
    chunk_index: int = chunk.get('chunk_index', 0)
    chunk_id: str = chunk.get('chunk_id', str(uuid.uuid4()))

    user_prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(
        domain=domain_result.primary_domain,
        chunk_index=chunk_index,
        total='?',          # total is unknown at single-chunk level
        section=section,
        text=text,
    )

    # Rough token estimate: 1 token ≈ 4 characters
    estimated_input_tokens = len(user_prompt) // 4
    if estimated_input_tokens > 6000:
        logger.warning(
            'Chunk %s estimated input tokens %d > 6000 — truncating text to 3000 chars',
            chunk_id, estimated_input_tokens,
        )
        text = text[:3000]
        # Rebuild prompt with truncated text
        user_prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(
            domain=domain_result.primary_domain,
            chunk_index=chunk_index,
            total='?',
            section=section,
            text=text,
        )

    messages: list[dict[str, str]] = [
        {'role': 'system', 'content': EXTRACTION_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    client = Mistral(api_key=settings.mistral_api_key)

    try:
        raw = await _call_mistral_with_retry_after(
            client=client,
            messages=messages,
            model=settings.mistral_small_model,  # Switch to mistral-small
            temperature=0.1,
            max_tokens=4096,  # was 8192 — smaller output fits within limits
            max_retries=5,    # was settings.mistral_max_retries — more retries
            base_delay=2.0,
        )
    except Exception as exc:
        logger.error(
            'Entity extraction failed for chunk %s after all retries: %s',
            chunk_id,
            exc,
        )
        return GraphSchema()

    graph = _parse_and_validate(raw)

    logger.info(
        'Chunk %s parsed: %d entities, %d relationships (raw_len=%d)',
        chunk_id,
        len(graph.entities),
        len(graph.relationships),
        len(raw),
    )
    if not graph.entities:
        logger.warning('Zero entities from chunk %s. Raw response preview: %r', chunk_id, raw[:500])

    # ---- Post-processing ----
    # Attach chunk_ref to every entity
    for entity in graph.entities:
        entity.chunk_ref = chunk_id

    # Filter relationships whose endpoints are not in the extracted entity set
    entity_names: set[str] = {e.name for e in graph.entities}
    valid_relationships: list[EdgeModel] = []
    for rel in graph.relationships:
        if rel.source_name in entity_names and rel.target_name in entity_names:
            valid_relationships.append(rel)
        else:
            logger.debug(
                'Dropping relationship %s→%s: endpoint not in entity set.',
                rel.source_name,
                rel.target_name,
            )
    graph.relationships = valid_relationships

    return graph


async def extract_all_chunks(
    chunks: list[dict],
    domain_result: DomainResult,
    settings: Settings,
    concurrency: int = 3,  # Optimisation 4: Concurrency level 3
) -> GraphSchema:
    """Extract entities from *all* chunks concurrently and merge results.

    Args:
        chunks:         List of chunk dicts from
                       :func:`~backend.parser.pdf_parser.get_chunks`.
        domain_result: Domain classification for the paper.
        settings:      Application settings.
        concurrency:   Maximum number of concurrent Mistral API calls.

    Returns:
        A single merged :class:`GraphSchema` with deduplicated entities and
        combined relationships.
    """
    if not chunks:
        return GraphSchema()

    # Optimisation 1: Skip references & aggressively merge chunks
    batched = _merge_chunks_into_batches(chunks, max_chars_per_batch=3000)

    logger.info(
        'Processing %d chunks merged into %d batches for extraction',
        len(chunks),
        len(batched),
    )
    if not batched:
        logger.error('All chunks were filtered out during batching — check section names in PDF')
        return GraphSchema()

    # Optimisation 3: Token-bucket rate limiter (0.8 req/sec)
    rate_limiter = _RateLimiter(rate=0.8)
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded_extract(chunk: dict) -> GraphSchema:
        async with semaphore:
            await rate_limiter.acquire()
            return await extract_entities(chunk, domain_result, settings)

    results: list[GraphSchema] = await asyncio.gather(
        *[_bounded_extract(c) for c in batched],
        return_exceptions=False,
    )

    return _merge_schemas(results)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _merge_chunks_into_batches(
    chunks: list[dict],
    max_chars_per_batch: int = 3000,
) -> list[dict]:
    """Merge consecutive chunks into batches, skipping reference sections."""
    batches: list[dict] = []
    current_text = ""
    current_sections: list[str] = []
    current_indices: list[int] = []

    for chunk in chunks:
        section = str(chunk.get("section", "") or "")
        section_lower = section.lower()
        skip_sections = {"references", "bibliography", "citations", "reference"}
        if section_lower in skip_sections:
            continue

        text = chunk.get("text", "").strip()
        if not text:
            continue

        if current_text and len(current_text) + len(text) > max_chars_per_batch:
            batches.append({
                "text": current_text,
                "section": ", ".join(dict.fromkeys(current_sections)),
                "chunk_index": current_indices[0],
                "chunk_id": f"batch_{current_indices[0]}_{current_indices[-1]}",
            })
            current_text = text
            current_sections = [chunk.get("section", "unknown")]
            current_indices = [chunk.get("chunk_index", 0)]
        else:
            current_text = (current_text + "\n\n" + text).strip()
            current_sections.append(chunk.get("section", "unknown"))
            current_indices.append(chunk.get("chunk_index", 0))

    if current_text:
        batches.append({
            "text": current_text,
            "section": ", ".join(dict.fromkeys(current_sections)),
            "chunk_index": current_indices[0],
            "chunk_id": f"batch_{current_indices[0]}_{current_indices[-1]}",
        })

    return batches


async def _call_mistral_with_retry_after(
    client: Mistral,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    base_delay: float,
) -> str:
    """Call the Mistral completions endpoint, using Retry-After headers if available."""
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = await client.chat.complete_async(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ''
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break

            error_str = str(exc).lower()
            # Extract Retry-After if present in the error body or string
            retry_after: float | None = None
            try:
                import re as _re
                match = _re.search(r'retry.after["\s:]+(\d+\.?\d*)', str(exc), _re.IGNORECASE)
                if match:
                    retry_after = float(match.group(1))
            except Exception:
                pass

            # Try to retrieve retry-after directly from raw response headers if httpx/SDKError
            if retry_after is None:
                try:
                    raw_response = getattr(exc, 'raw_response', None)
                    if raw_response and raw_response.headers:
                        val = raw_response.headers.get('retry-after')
                        if val:
                            retry_after = float(val)
                except Exception:
                    pass

            # Optimisation 5: Smarter 429 backoff
            delay = retry_after if retry_after else min((2 ** (attempt - 1)) * base_delay, 30.0)
            logger.warning(
                'Mistral call attempt %d/%d failed (%s). Retrying in %.2fs.',
                attempt,
                max_retries,
                type(exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)

    raise last_exc  # type: ignore[misc]


def _parse_and_validate(raw: str) -> GraphSchema:
    """Parse *raw* LLM output into a validated :class:`GraphSchema`."""
    if not raw:
        logger.warning('Empty LLM response in entity extraction.')
        return GraphSchema()

    cleaned = _JSON_FENCE_RE.sub('', raw).replace('`', '').strip()

    json_str = _extract_outermost_json(cleaned)
    if json_str is None:
        logger.warning('Outermost JSON object not closed. Attempting truncation repair.')
        start_idx = cleaned.find('{')
        if start_idx != -1:
            json_str = _repair_json(cleaned[start_idx:])
        else:
            return GraphSchema()

    try:
        data: Any = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning('JSON decode error in extraction (%s), attempting repair...', exc)
        try:
            start_idx = cleaned.find('{')
            if start_idx != -1:
                repaired = _repair_json(cleaned[start_idx:])
                data = json.loads(repaired)
            else:
                return GraphSchema()
        except Exception as repair_exc:
            logger.warning('JSON repair failed (%s).', repair_exc)
            return GraphSchema()

    try:
        return GraphSchema.model_validate(data)
    except Exception as exc:
        logger.warning('GraphSchema validation error (%s). data=%r', exc, data)
        return GraphSchema()


def _repair_json(json_str: str) -> str:
    """Repair truncated JSON by finding the last complete entity object."""
    import re
    import json
    
    json_str = json_str.strip()
    if not json_str:
        return '{"entities": [], "relationships": []}'

    # First try: Stack-based repair (very good for simple end-of-string truncations)
    in_quote = False
    escape = False
    stack = []
    for ch in json_str:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch in ('}', ']'):
            if stack:
                stack.pop()

    candidate = json_str
    if in_quote:
        candidate += '"'

    temp_stack = list(stack)
    while temp_stack:
        top = temp_stack.pop()
        if top == '{':
            candidate = candidate.rstrip(':, \n\r\t')
            candidate += '}'
        elif top == '[':
            candidate = candidate.rstrip(':, \n\r\t')
            candidate += ']'

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict) and ('entities' in parsed or 'relationships' in parsed):
            return json.dumps(parsed)
    except Exception:
        pass

    # Strategy 1: Find last complete entity/relationship object by scanning for '}' boundaries
    # and attempting to parse up to that point
    entities_start = json_str.find('"entities"')
    if entities_start == -1:
        return '{"entities": [], "relationships": []}'

    best_result = None
    brace_positions = [i for i, ch in enumerate(json_str) if ch == '}']

    for pos in reversed(brace_positions):
        candidate = json_str[:pos + 1]
        # Try to close open structures
        open_braces = candidate.count('{') - candidate.count('}')
        open_brackets = candidate.count('[') - candidate.count(']')
        
        # Remove unclosed string by stripping to last valid '"' boundary
        closed = candidate
        for _ in range(open_brackets):
            closed = closed.rstrip(', \n\r\t') + ']'
        for _ in range(open_braces):
            closed = closed.rstrip(', \n\r\t') + '}'
        try:
            parsed = json.loads(closed)
            if isinstance(parsed, dict) and 'entities' in parsed:
                best_result = parsed
                break
        except (json.JSONDecodeError, ValueError):
            continue

    if best_result:
        return json.dumps(best_result)

    # Strategy 2: Extract only complete entity objects using regex
    entity_pattern = re.compile(
        r'\{\s*"name"\s*:\s*"[^"]*"\s*,\s*"type"\s*:\s*"[^"]*"[^}]*\}',
        re.DOTALL
    )
    matches = entity_pattern.findall(json_str)
    if matches:
        entities = []
        for m in matches:
            try:
                entity = json.loads(m)
                entities.append(entity)
            except (json.JSONDecodeError, ValueError):
                continue
        if entities:
            return json.dumps({'entities': entities, 'relationships': []})

    return '{"entities": [], "relationships": []}'


def _extract_outermost_json(text: str) -> str | None:
    """Return the first outermost ``{...}`` block using brace counting."""
    depth = 0
    start: int | None = None

    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                return text[start: i + 1]

    return None


def _merge_schemas(schemas: list[GraphSchema]) -> GraphSchema:
    """Merge multiple :class:`GraphSchema` instances into one with entity deduplication."""
    merged_entities: list[NodeModel] = []
    seen_entity_keys: set[tuple[str, str]] = set()

    merged_relationships: list[EdgeModel] = []

    for schema in schemas:
        for entity in schema.entities:
            key = (entity.name.lower().strip(), entity.type)
            if key not in seen_entity_keys:
                seen_entity_keys.add(key)
                merged_entities.append(entity)

        merged_relationships.extend(schema.relationships)

    logger.debug(
        'Merged %d schemas → %d unique entities, %d relationships.',
        len(schemas),
        len(merged_entities),
        len(merged_relationships),
    )

    return GraphSchema(
        entities=merged_entities,
        relationships=merged_relationships,
    )
