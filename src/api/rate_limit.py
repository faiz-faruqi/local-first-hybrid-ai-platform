"""
Per-IP rate limiting for the public demo.

This is a cost guard, not a security feature — it exists so a public demo
with a real, spendable OPENROUTER_API_KEY behind it can't be run up by one
visitor. Reuses the response cache's Redis connection under a separate key
namespace, following the same reuse pattern as BudgetTracker
(src/routing/budget.py).
"""

import logging
import os
from datetime import datetime, timezone

from fastapi import HTTPException, Request

from src.cache.redis_cache import ResponseCache

logger = logging.getLogger(__name__)

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "5") or "5")
RATE_LIMIT_PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", "30") or "30")
RATE_LIMIT_KEY_PREFIX = "ratelimit:"


def get_client_ip(request: Request) -> str:
    """
    Prefer X-Forwarded-For's first entry — the real client IP, appended by
    Railway's edge proxy in this deployment's single-hop topology — falling
    back to the direct connection's IP for local/Docker Compose dev, where
    there's no proxy in front and the header is absent.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """
    Fixed-window per-IP rate limiter: N requests/minute and M requests/day.

    Checks Redis INCR's atomic return value directly rather than reading
    the count, checking it, then incrementing — that read-then-act sequence
    has a race window under concurrent requests from the same IP; INCR's
    return value doesn't. Fails open on any Redis error, matching
    ResponseCache/BudgetTracker's convention — a cost-guard outage must
    never block the request path it's protecting.
    """

    def __init__(
        self,
        cache: ResponseCache,
        per_minute: int = RATE_LIMIT_PER_MINUTE,
        per_day: int = RATE_LIMIT_PER_DAY,
    ) -> None:
        self._client = cache._client
        self._per_minute = per_minute
        self._per_day = per_day

    async def check(self, client_ip: str) -> None:
        """Raise HTTPException(429) if either window is exceeded for this IP."""
        try:
            now = datetime.now(timezone.utc)
            minute_key = f"{RATE_LIMIT_KEY_PREFIX}{client_ip}:min:{int(now.timestamp() // 60)}"
            day_key = f"{RATE_LIMIT_KEY_PREFIX}{client_ip}:day:{now.strftime('%Y-%m-%d')}"

            minute_count = await self._client.incr(minute_key)
            if minute_count == 1:
                await self._client.expire(minute_key, 70)
            if minute_count > self._per_minute:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Rate limit exceeded: max {self._per_minute} requests/minute "
                        "on this public demo. Try again in under a minute."
                    ),
                )

            day_count = await self._client.incr(day_key)
            if day_count == 1:
                await self._client.expire(day_key, 90000)
            if day_count > self._per_day:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Daily demo limit reached: max {self._per_day} requests/day "
                        "per visitor. Please come back tomorrow."
                    ),
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Rate limit check failed — failing open: %s", exc)
            return
