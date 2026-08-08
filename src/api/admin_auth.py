"""
Shared admin-key check for protected endpoints.

Used as a FastAPI dependency on any route that should only be callable by
the deployment owner (e.g. DELETE /ingest/flush-cache, POST /auth/generate-code).
"""

import os

from fastapi import Header, HTTPException, status

ADMIN_KEY = os.getenv("ADMIN_KEY", "")


def verify_admin_key(x_admin_key: str = Header(default="")) -> None:
    """Dependency: verify the X-Admin-Key header for admin endpoints."""
    if not ADMIN_KEY:
        # If ADMIN_KEY is not configured, block all admin operations.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin key not configured on this deployment.",
        )
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin key.",
        )
