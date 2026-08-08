"""
Demo access-code endpoints.

POST /auth/generate-code — admin-only. Creates a new time-limited access
    code, replacing whatever was previously active.
POST /auth/verify-code   — open. Called by the frontend's NextAuth sign-in
    flow to check whether an entered code is currently valid, before any
    session exists — so this cannot be admin-protected like generate-code.
"""

import logging

from fastapi import APIRouter, Depends

from src.api.access_code import ACCESS_CODE_DEFAULT_TTL_HOURS, AccessCodeStore
from src.api.admin_auth import verify_admin_key
from src.api.dependencies import get_access_code_store
from src.models.auth import (
    GenerateCodeRequest,
    GenerateCodeResponse,
    VerifyCodeRequest,
    VerifyCodeResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/generate-code",
    response_model=GenerateCodeResponse,
    dependencies=[Depends(verify_admin_key)],
)
async def generate_code(
    request: GenerateCodeRequest,
    store: AccessCodeStore = Depends(get_access_code_store),
) -> GenerateCodeResponse:
    """
    Generate a new demo access code. Protected by X-Admin-Key header.

    Retires any previously active code — there is only ever one valid
    code at a time.
    """
    ttl_hours = request.ttl_hours if request.ttl_hours is not None else ACCESS_CODE_DEFAULT_TTL_HOURS
    code, expires_at = await store.generate(ttl_hours=ttl_hours)
    return GenerateCodeResponse(code=code, expires_at=expires_at, ttl_hours=ttl_hours)


@router.post("/verify-code", response_model=VerifyCodeResponse)
async def verify_code(
    request: VerifyCodeRequest,
    store: AccessCodeStore = Depends(get_access_code_store),
) -> VerifyCodeResponse:
    """
    Check whether a code is currently valid. Never echoes the actual
    stored code back — only a yes/no answer.
    """
    valid = await store.verify(request.code)
    return VerifyCodeResponse(valid=valid)
