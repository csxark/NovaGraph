"""
backend/vector/embedder.py

Cloud-based embedding via the Mistral Embeddings API (mistral-embed model).

Uses the same Mistral API key already configured for entity extraction.
Produces 1024-dim vectors — richer than the previous 384-dim HF model.

Public API is identical to the previous HF Inference API version —
all call-sites (vector_agent, main.py pipeline) work without changes.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from backend.config import Settings
    from backend.extractor.entity_extractor import NodeModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mistral Embeddings API constants
# ---------------------------------------------------------------------------

MISTRAL_EMBED_URL = "https://api.mistral.ai/v1/embeddings"
MISTRAL_EMBED_MODEL = "mistral-embed"

# ---------------------------------------------------------------------------
# Module-level async HTTP client singleton
# ---------------------------------------------------------------------------

_client: httpx.AsyncClient | None = None


def _get_client(settings: "Settings") -> httpx.AsyncClient:
    """Return (or create) a persistent async HTTP client for Mistral API calls."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.mistral_timeout, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _client


async def close_client() -> None:
    """Cleanly close the HTTP client. Called during app shutdown."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None


# ---------------------------------------------------------------------------
# Mistral Embeddings API core call
# ---------------------------------------------------------------------------

async def _call_mistral_embed(
    texts: list[str],
    settings: "Settings",
) -> list[list[float]]:
    """Call the Mistral Embeddings API with exponential backoff.

    Args:
        texts:    List of strings to embed (max 16384 tokens per string).
        settings: Application settings (holds Mistral API key, timeout, retries).

    Returns:
        List of 1024-dim float lists, one per input text.

    Raises:
        RuntimeError: After exhausting all retries.
    """
    headers = {
        "Authorization": f"Bearer {settings.mistral_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MISTRAL_EMBED_MODEL,
        "input": texts,
        "encoding_format": "float",
    }

    client = _get_client(settings)
    last_exc: Exception | None = None

    for attempt in range(settings.mistral_max_retries):
        try:
            response = await client.post(
                MISTRAL_EMBED_URL,
                json=payload,
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                # Response: { "data": [{ "embedding": [float,...], "index": int }, ...] }
                # Sort by index to guarantee order matches input
                items = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in items]

            if response.status_code == 429:
                delay = min(2 ** attempt + random.uniform(0, 1), 30.0)
                logger.warning(
                    "Mistral embed rate limit — retrying in %.1fs (attempt %d/%d)",
                    delay, attempt + 1, settings.mistral_max_retries,
                )
                last_exc = RuntimeError(f"Mistral embed rate limit (429) on attempt {attempt + 1}")
                await asyncio.sleep(delay)
                continue

            if response.status_code in (500, 502, 503, 504):
                delay = min(2 ** attempt + random.uniform(0, 1), 30.0)
                logger.warning(
                    "Mistral embed server error %d — retrying in %.1fs (attempt %d/%d)",
                    response.status_code, delay, attempt + 1, settings.mistral_max_retries,
                )
                last_exc = RuntimeError(f"Mistral embed server error {response.status_code}")
                await asyncio.sleep(delay)
                continue

            last_exc = RuntimeError(
                f"Mistral embed API error {response.status_code}: {response.text[:500]}"
            )
            logger.error("Mistral embed API error: %s", last_exc)
            break

        except httpx.TimeoutException as exc:
            last_exc = exc
            delay = min(2 ** attempt + random.uniform(0, 1), 30.0)
            logger.warning(
                "Mistral embed timeout — retrying in %.1fs (attempt %d/%d)",
                delay, attempt + 1, settings.mistral_max_retries,
            )
            await asyncio.sleep(delay)

        except Exception as exc:
            last_exc = exc
            logger.error("Mistral embed unexpected error: %s", exc, exc_info=True)
            break

    raise RuntimeError(
        f"Mistral Embeddings API failed after {settings.mistral_max_retries} "
        f"attempts: {last_exc}"
    )


# ---------------------------------------------------------------------------
# L2 normalisation utility
# ---------------------------------------------------------------------------

def _normalize(vec: list[float]) -> list[float]:
    """L2-normalise a vector (returns new list)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        return [x / norm for x in vec]
    return vec


# ---------------------------------------------------------------------------
# Public API — identical signatures to the previous HF API version
# ---------------------------------------------------------------------------

async def embed_text(text: str, settings: "Settings") -> list[float]:
    """Embed a single string via the Mistral Embeddings API.

    Returns a 1024-dim float list, L2-normalised.
    """
    results = await _call_mistral_embed([text], settings)
    return _normalize(results[0])


async def embed_batch(
    texts: list[str],
    settings: "Settings",
    batch_size: int | None = None,
) -> list[list[float]]:
    """Embed a list of strings via the Mistral Embeddings API.

    Chunks into sub-batches of batch_size to stay within token limits.
    Returns list of 1024-dim float lists, L2-normalised.
    """
    if not texts:
        return []

    bs = batch_size or settings.embedding_batch_size
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), bs):
        batch = texts[i: i + bs]
        results = await _call_mistral_embed(batch, settings)
        all_embeddings.extend([_normalize(v) for v in results])

    return all_embeddings


async def embed_node(node: "NodeModel", settings: "Settings") -> list[float]:
    """Embed a single NodeModel using its name and description."""
    text = f"{node.name}: {node.description}"
    return await embed_text(text, settings)


async def embed_nodes_batch(
    nodes: list["NodeModel"],
    settings: "Settings",
    batch_size: int | None = None,
) -> list[list[float]]:
    """Batch-embed a list of NodeModels. Returns embeddings in same order as nodes."""
    texts = [f"{n.name}: {n.description}" for n in nodes]
    return await embed_batch(texts, settings, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Startup verification probe
# ---------------------------------------------------------------------------

async def verify_hf_api(settings: "Settings") -> bool:
    """Verify the Mistral Embeddings API is reachable and returns correct dimensions.

    Named verify_hf_api to keep the call-site in main.py unchanged.
    """
    try:
        result = await embed_text("connectivity test", settings)
        if len(result) == settings.embedding_dim:
            logger.info(
                "Mistral Embeddings API verified: model=%s, dim=%d",
                MISTRAL_EMBED_MODEL,
                len(result),
            )
            return True
        logger.error(
            "Mistral embed returned unexpected dimension: expected %d, got %d",
            settings.embedding_dim,
            len(result),
        )
        return False
    except Exception as exc:
        logger.error("Mistral Embeddings API verification failed: %s", exc)
        return False