"""
Generate a new time-limited demo access code.

Calls the deployed backend's admin-protected POST /auth/generate-code
endpoint. Generating a new code retires whatever code was previously
active — there is only ever one valid code at a time. Prints the code and
its expiry to stdout for pasting into a message to whoever you're sharing
demo access with.

The admin key is never required as a bare CLI argument by default — pass
--admin-key explicitly only if you don't want to rely on the ADMIN_KEY
environment variable (keeping it off the shell history).

Usage:
    # Against local dev backend, default TTL (server default, e.g. 7 days)
    ADMIN_KEY=... python scripts/generate_access_code.py

    # Against a deployed backend, custom TTL
    python scripts/generate_access_code.py \\
        --api-url https://api.yourdomain.com \\
        --admin-key your-admin-key \\
        --ttl-hours 48
"""

import argparse
import json
import logging
import os
import sys
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def generate_code(base_url: str, admin_key: str, ttl_hours: float | None) -> dict:
    url = f"{base_url}/auth/generate-code"
    payload: dict = {}
    if ttl_hours is not None:
        payload["ttl_hours"] = ttl_hours

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-Admin-Key": admin_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode() if hasattr(exc, "read") else ""
        logger.error("HTTP %d from %s: %s", exc.code, url, body)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a new time-limited demo access code.")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the FastAPI backend (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--admin-key",
        default=None,
        help="Admin key (X-Admin-Key). Falls back to the ADMIN_KEY environment variable if omitted.",
    )
    parser.add_argument(
        "--ttl-hours",
        type=float,
        default=None,
        help="How long the code stays valid, in hours. Omit to use the server default.",
    )
    args = parser.parse_args()

    admin_key = args.admin_key or os.getenv("ADMIN_KEY", "")
    if not admin_key:
        logger.error("No admin key provided — pass --admin-key or set the ADMIN_KEY environment variable.")
        sys.exit(1)

    base_url = args.api_url.rstrip("/")

    try:
        result = generate_code(base_url, admin_key, args.ttl_hours)
    except Exception as exc:
        logger.error("Failed to generate access code: %s", exc)
        sys.exit(1)

    print()
    print(f"Access code:  {result['code']}")
    print(f"Expires at:   {result['expires_at']} (UTC)")
    print(f"Valid for:    {result['ttl_hours']} hours")
    print()


if __name__ == "__main__":
    main()
