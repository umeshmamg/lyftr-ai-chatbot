"""Pydantic request and response schemas matching the API contract."""

from typing import Literal
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Response payload for GET /health."""

    status: str
    keys_configured: int
    cache_enabled: bool


class ChatRequest(BaseModel):
    """Request payload for POST /chat."""

    session_id: str
    message: str
    system_prompt: str | None = Field(default=None, max_length=4000)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("session_id must not be empty")
        if len(trimmed) > 128:
            raise ValueError("session_id must be at most 128 characters")
        return trimmed

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("message must not be empty")
        if len(trimmed) > 4000:
            raise ValueError("message must be at most 4000 characters after trimming")
        return trimmed

    @field_validator("system_prompt")
    @classmethod
    def normalize_system_prompt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None


class ChatMetadata(BaseModel):
    """Per-request execution metadata attached to assistant replies."""

    resolution_source: Literal["LIVE_INFERENCE", "LLM_CACHE"]
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int


class ChatResponse(BaseModel):
    """Response payload for POST /chat."""

    session_id: str
    reply: str
    metadata: ChatMetadata


class SessionMessageItem(BaseModel):
    """Serialized message returned by GET /sessions/{session_id}/messages."""

    role: Literal["user", "assistant"]
    content: str
    metadata: ChatMetadata | None = None


class SessionMessagesResponse(BaseModel):
    """Response payload for GET /sessions/{session_id}/messages."""

    session_id: str
    messages: list[SessionMessageItem]
