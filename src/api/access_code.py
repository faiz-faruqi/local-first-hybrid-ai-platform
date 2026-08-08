"""
Time-limited demo access code.

Generating a code overwrites whatever code was previously active — there
is only ever one valid code at a time. Reuses the response cache's Redis
connection under a separate key namespace, following the same reuse
pattern as BudgetTracker/RateLimiter/TraceStore.
"""

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone

from src.cache.redis_cache import ResponseCache

logger = logging.getLogger(__name__)

ACCESS_CODE_KEY = "auth:access_code"
ACCESS_CODE_DEFAULT_TTL_HOURS = float(os.getenv("ACCESS_CODE_DEFAULT_TTL_HOURS", "168") or "168")


class AccessCodeStore:
    """
    Tracks the single currently-valid demo access code in Redis, using
    Redis's own TTL as the expiry mechanism — an expired code simply
    stops being found, no separate cleanup step needed.
    """

    def __init__(self, cache: ResponseCache) -> None:
        self._client = cache._client

    async def generate(self, ttl_hours: float = ACCESS_CODE_DEFAULT_TTL_HOURS) -> tuple[str, str]:
        """
        Create a new code, replacing any previously active one.

        Returns (code, iso8601_expires_at). Unlike verify(), this does not
        degrade gracefully on a Redis error — an admin explicitly asking
        for a new code should see the failure, not a silent no-op.
        """
        code = secrets.token_urlsafe(9)
        ttl_seconds = max(1, int(ttl_hours * 3600))
        await self._client.set(ACCESS_CODE_KEY, code, ex=ttl_seconds)
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()
        logger.info("Generated new access code, expires_at=%s", expires_at)
        return code, expires_at

    async def verify(self, code: str) -> bool:
        """
        Return True only if `code` matches the currently active, unexpired
        code.

        Deliberately fails *closed*: RateLimiter/BudgetTracker degrade
        gracefully on a Redis error because an outage there must never
        block real requests over a cost guard. An auth check is the
        opposite — failing open would let everyone in during an outage,
        which is worse than denying sign-in.
        """
        if not code:
            return False
        try:
            current = await self._client.get(ACCESS_CODE_KEY)
        except Exception as exc:
            logger.warning("Access code verification failed — denying (fail closed): %s", exc)
            return False
        return current is not None and code == current
