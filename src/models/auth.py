"""Access-code generation/verification schemas."""

from pydantic import BaseModel, Field


class GenerateCodeRequest(BaseModel):
    ttl_hours: float | None = Field(
        default=None,
        gt=0,
        description=(
            "How long the generated code stays valid. Omit to use the server "
            "default (ACCESS_CODE_DEFAULT_TTL_HOURS). Generating a new code "
            "retires any previous one."
        ),
    )


class GenerateCodeResponse(BaseModel):
    code: str
    expires_at: str = Field(description="ISO 8601 timestamp (UTC) when this code stops working.")
    ttl_hours: float


class VerifyCodeRequest(BaseModel):
    code: str


class VerifyCodeResponse(BaseModel):
    valid: bool
