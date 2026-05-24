"""
cache.py — Hybrid Redis cache layer for LangGraph thread state and meta-registry.

L1 (Redis):  hot, short-lived state with TTL
L2 (PostgreSQL):  handled in database.py; called on cache miss in main.py

Silent fallback: all public functions catch Redis connectivity errors, log a
warning, and return a safe default (None / empty list) so the application
continues operating from L2.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# A single shared connection pool reused across requests.
_redis_client: Optional[aioredis.Redis] = None


def get_redis() -> aioredis.Redis:
    """Return (or lazily create) the shared async Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _redis_client


async def close_redis() -> None:
    """Gracefully close the Redis connection pool on shutdown."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

_THREAD_STATE_PREFIX = "thread:state:"
_META_REGISTRY_KEY = "meta:registry"
_DEFAULT_TTL = 1800  # 30 minutes


def _thread_state_key(thread_id: str) -> str:
    return f"{_THREAD_STATE_PREFIX}{thread_id}"


# ---------------------------------------------------------------------------
# Thread state cache  (L1 for LangGraph state dicts)
# ---------------------------------------------------------------------------


async def cache_thread_state(
    thread_id: str,
    state_dict: dict[str, Any],
    ttl: int = _DEFAULT_TTL,
) -> None:
    """
    JSON-serialize *state_dict* and store it in Redis with *ttl* seconds TTL.

    Silent fallback on Redis errors.
    """
    try:
        client = get_redis()
        payload = json.dumps(state_dict, default=str)
        await client.setex(_thread_state_key(thread_id), ttl, payload)
        logger.debug("Cached thread state for '%s' (TTL=%ds)", thread_id, ttl)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Redis unavailable — could not cache thread state for '%s': %s",
            thread_id,
            exc,
        )


async def get_cached_thread_state(
    thread_id: str,
) -> Optional[dict[str, Any]]:
    """
    Return the cached LangGraph state dict for *thread_id*, or ``None`` on a
    cache miss or Redis error.
    """
    try:
        client = get_redis()
        raw = await client.get(_thread_state_key(thread_id))
        if raw is None:
            logger.debug("Cache miss for thread '%s'", thread_id)
            return None
        state: dict[str, Any] = json.loads(raw)
        logger.debug("Cache hit for thread '%s'", thread_id)
        return state
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Redis unavailable — falling back to L2 for thread '%s': %s",
            thread_id,
            exc,
        )
        return None


async def invalidate_thread_state(thread_id: str) -> None:
    """Remove the cached state for *thread_id* (e.g., after a hard reset)."""
    try:
        client = get_redis()
        await client.delete(_thread_state_key(thread_id))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Redis unavailable — could not invalidate thread '%s': %s",
            thread_id,
            exc,
        )


# ---------------------------------------------------------------------------
# Meta-registry cache  (high-level directory of all active threads)
# ---------------------------------------------------------------------------


async def cache_meta_registry(registry_list: list[dict[str, Any]]) -> None:
    """
    Serialise and store the full list of thread summaries in Redis so the
    GlobalRouter can do semantic routing without a DB round-trip.

    No TTL is set here — the registry is considered permanently warm and is
    refreshed explicitly after every write in main.py.

    Silent fallback on Redis errors.
    """
    try:
        client = get_redis()
        payload = json.dumps(registry_list, default=str)
        await client.set(_META_REGISTRY_KEY, payload)
        logger.debug("Meta-registry cached (%d entries)", len(registry_list))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Redis unavailable — meta-registry not cached: %s", exc
        )


async def get_meta_registry() -> list[dict[str, Any]]:
    """
    Return the cached meta-registry list, or an empty list on a miss / error.
    """
    try:
        client = get_redis()
        raw = await client.get(_META_REGISTRY_KEY)
        if raw is None:
            return []
        registry: list[dict[str, Any]] = json.loads(raw)
        return registry
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Redis unavailable — returning empty meta-registry: %s", exc
        )
        return []
