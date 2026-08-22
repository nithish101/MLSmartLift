"""
Redis-backed cache with hit/miss accounting.

The cache is an optimisation, never a dependency: if Redis is unreachable every
lookup counts as a miss and writes are dropped, so requests still succeed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import redis

from app.metrics import CACHE_ERRORS, CACHE_HIT_RATIO, CACHE_REQUESTS

logger = logging.getLogger(__name__)

KEY_PREFIX = "smartlift"


def feature_hash(row: Mapping[str, Any]) -> str:
    """Stable hash of a feature vector (key order and whitespace independent)."""
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def prediction_key(model_version: int, row: Mapping[str, Any]) -> str:
    # The model version is part of the key so a promotion never serves stale predictions.
    return f"{KEY_PREFIX}:pred:v{model_version}:{feature_hash(row)}"


def features_key(exercise: str) -> str:
    return f"{KEY_PREFIX}:features:{exercise}"


@dataclass
class CacheStats:
    hits: dict[str, int] = field(default_factory=dict)
    misses: dict[str, int] = field(default_factory=dict)
    errors: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, cache: str, hit: bool) -> None:
        with self._lock:
            bucket = self.hits if hit else self.misses
            bucket[cache] = bucket.get(cache, 0) + 1
            h, m = self.hits.get(cache, 0), self.misses.get(cache, 0)
        CACHE_REQUESTS.labels(cache, "hit" if hit else "miss").inc()
        CACHE_HIT_RATIO.labels(cache).set(h / (h + m))

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1
        CACHE_ERRORS.inc()

    def snapshot(self) -> dict[str, dict[str, int | float]]:
        with self._lock:
            names = sorted(set(self.hits) | set(self.misses))
            out: dict[str, dict[str, int | float]] = {}
            for name in names:
                h, m = self.hits.get(name, 0), self.misses.get(name, 0)
                out[name] = {"hits": h, "misses": m, "hit_ratio": round(h / (h + m), 4) if h + m else 0.0}
            return out


class Cache:
    def __init__(self, url: str, socket_timeout: float = 0.25) -> None:
        self._redis = redis.Redis.from_url(
            url,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
            decode_responses=True,
        )
        self.stats = CacheStats()

    def ping(self) -> bool:
        try:
            return bool(self._redis.ping())
        except redis.RedisError:
            return False

    def get_json(self, cache: str, key: str) -> Any | None:
        """Return the cached value, recording a hit or miss against ``cache``."""
        try:
            raw = self._redis.get(key)
        except redis.RedisError as exc:
            self.stats.record_error()
            logger.warning("cache read failed: %s", exc)
            raw = None
        value = json.loads(raw) if isinstance(raw, str) else None
        self.stats.record(cache, hit=value is not None)
        return value

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        try:
            self._redis.set(key, json.dumps(value), ex=ttl_seconds)
        except redis.RedisError as exc:
            self.stats.record_error()
            logger.warning("cache write failed: %s", exc)

    def delete(self, key: str) -> None:
        try:
            self._redis.delete(key)
        except redis.RedisError as exc:
            self.stats.record_error()
            logger.warning("cache delete failed: %s", exc)

    def ttl(self, key: str) -> int:
        return int(self._redis.ttl(key))  # type: ignore[arg-type]

    def close(self) -> None:
        self._redis.close()
