"""
Domain detector module.

Uses the Mistral small model to classify the scientific domain of an academic
text sample with exponential back-off retry logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from mistralai.client import Mistral
from pydantic import BaseModel, Field

from backend.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class DomainResult(BaseModel):
    """Structured result from domain classification."""

    domains: list[str] = Field(
        ...,
        description='List of detected domains, primary first.',
    )
    primary_domain: str = Field(
        ...,
        description='Single primary domain string.',
    )
    is_interdisciplinary: bool = Field(
        ...,
        description='True when the paper spans multiple domains.',
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description='Classifier confidence in [0, 1].',
    )
    rationale: str = Field(
        ...,
        description='Short rationale (max 50 words).',
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    'You are a scientific domain classifier. '
    'Classify the research domain of the given academic text. '
    'Return ONLY valid JSON.'
)

_USER_PROMPT_TEMPLATE = (
    'Classify the primary domain of this academic text:\n\n'
    '{text_sample}\n\n'
    'Return ONLY this JSON:\n'
    '{{\n'
    '  "domains": ["<primary>", "<secondary if interdisciplinary>"],\n'
    '  "primary_domain": "<string>",\n'
    '  "is_interdisciplinary": <bool>,\n'
    '  "confidence": <float 0.0-1.0>,\n'
    '  "rationale": "<max 50 words>"\n'
    '}}'
)

# Regex to capture the first complete JSON object in a string
_JSON_OBJECT_RE = re.compile(r'\{[^{}]*\}', re.DOTALL)

# Fallback result returned when all retries are exhausted
_FALLBACK_RESULT = DomainResult(
    domains=['general'],
    primary_domain='general',
    is_interdisciplinary=False,
    confidence=0.1,
    rationale='Classification failed',
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def detect_domain(text_sample: str, settings: Settings) -> DomainResult:
    """Detect the scientific domain of *text_sample* using the Mistral API.

    Only the first 2 000 characters of *text_sample* are sent to the model.

    Args:
        text_sample: Raw text from the academic paper (any length).
        settings:    Application :class:`~backend.config.Settings`.

    Returns:
        A :class:`DomainResult` — always, even on failure.
    """
    sample = text_sample[:2000].strip()
    if not sample:
        logger.warning('Empty text sample supplied to detect_domain; using fallback.')
        return _FALLBACK_RESULT

    messages: list[dict[str, str]] = [
        {'role': 'system', 'content': _SYSTEM_PROMPT},
        {
            'role': 'user',
            'content': _USER_PROMPT_TEMPLATE.format(text_sample=sample),
        },
    ]

    # Instantiate the async Mistral client (key is never logged)
    client = Mistral(api_key=settings.mistral_api_key)

    try:
        raw = await _call_with_backoff(
            client=client,
            messages=messages,
            model=settings.mistral_small_model,
            temperature=0.0,
            max_tokens=200,
            max_retries=settings.mistral_max_retries,
            base_delay=2.0,
        )
    except Exception as exc:
        logger.error('Domain detection failed after all retries: %s', exc)
        return _FALLBACK_RESULT

    return _parse_domain_result(raw)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _call_with_backoff(
    client: Mistral,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    base_delay: float,
) -> str:
    """Call the Mistral chat completion endpoint with exponential back-off.

    Args:
        client:      Async :class:`Mistral` client instance.
        messages:    Chat message list.
        model:       Mistral model identifier.
        temperature: Sampling temperature.
        max_tokens:  Maximum tokens in the response.
        max_retries: Total number of attempts before raising.
        base_delay:  Initial delay in seconds (doubles on each retry).

    Returns:
        The raw text content of the first choice.

    Raises:
        Exception: Propagated from the last failed attempt.
    """
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = await client.chat.complete_async(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content: str = response.choices[0].message.content or ''
            return content

        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break

            # Exponential back-off with ±20 % jitter
            import random  # local import to keep module-level imports clean

            delay = base_delay * (2 ** (attempt - 1))
            jitter = delay * 0.2 * (random.random() * 2 - 1)  # ±20 %
            wait = max(0.0, delay + jitter)

            logger.warning(
                'Mistral call attempt %d/%d failed (%s). Retrying in %.2fs.',
                attempt,
                max_retries,
                type(exc).__name__,
                wait,
            )
            await asyncio.sleep(wait)

    raise last_exc  # type: ignore[misc]


def _parse_domain_result(raw: str) -> DomainResult:
    """Extract and validate a :class:`DomainResult` from *raw* LLM output.

    Falls back to :data:`_FALLBACK_RESULT` on any parse / validation error.
    """
    if not raw:
        logger.warning('Empty LLM response; returning fallback DomainResult.')
        return _FALLBACK_RESULT

    # Strip markdown code fences if present
    cleaned = re.sub(r'```(?:json)?', '', raw).strip()

    match = _JSON_OBJECT_RE.search(cleaned)
    if not match:
        logger.warning(
            'No JSON object found in domain-detection response. raw=%r', raw[:200]
        )
        return _FALLBACK_RESULT

    try:
        data: Any = json.loads(match.group())
        return DomainResult.model_validate(data)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            'Failed to parse DomainResult JSON (%s). raw=%r', exc, raw[:200]
        )
        return _FALLBACK_RESULT
