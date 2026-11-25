"""LiteLLM gateway with multi-key failover, response caching, and failure simulation."""

import asyncio
from dataclasses import dataclass
import logging
import os
import time
import json
from typing import Any

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
from litellm.caching.caching import Cache, LiteLLMCacheType

from app.config import Settings


logger = logging.getLogger(__name__)

_SIMULATABLE_STATUSES = {429, 500, 502, 503}


class AllProvidersUnavailableError(Exception):
    """Raised when every configured key/provider fails to produce a reply."""


class SimulatedKeyFailureError(Exception):
    """Internal exception raised when X-Simulate-Failure targets a key."""

    def __init__(self, key_id: str, status_code: int) -> None:
        self.key_id = key_id
        self.status_code = status_code
        super().__init__(f"Simulated HTTP {status_code} failure for {key_id}")


@dataclass(frozen=True, slots=True)
class KeySlot:
    """A single API key + model entry in the ordered failover pool."""

    key_id: str
    api_key: str
    model: str


@dataclass(slots=True)
class LLMReply:
    """Resolved assistant response and per-request telemetry."""

    reply: str
    resolution_source: str
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int


def parse_simulate_failure_header(header_value: str | None) -> dict[str, int]:
    """Parse X-Simulate-Failure header (e.g. 'key_1=429,key_2=500' or '*=429')."""
    if not header_value:
        return {}

    directives: dict[str, int] = {}
    for raw_pair in header_value.split(","):
        pair = raw_pair.strip()
        if not pair or "=" not in pair:
            continue
        target, status_raw = pair.split("=", 1)
        key_target = target.strip()
        try:
            status_code = int(status_raw.strip())
        except ValueError:
            continue
        if key_target and status_code in _SIMULATABLE_STATUSES:
            directives[key_target] = status_code
    return directives


def _is_failover_error(exc: Exception) -> bool:
    """Return True for rate-limit, quota, 5xx, connection, or auth key failures."""
    if isinstance(exc, SimulatedKeyFailureError):
        return True

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or status_code >= 500 or status_code in {401, 403}):
        return True

    failover_types = (
        litellm.exceptions.RateLimitError,
        litellm.exceptions.InternalServerError,
        litellm.exceptions.ServiceUnavailableError,
        litellm.exceptions.BadGatewayError,
        litellm.exceptions.APIConnectionError,
        litellm.exceptions.Timeout,
        litellm.exceptions.BudgetExceededError,
        litellm.exceptions.AuthenticationError,
        litellm.exceptions.PermissionDeniedError,
    )
    if isinstance(exc, failover_types):
        return True

    message = str(exc).lower()
    keywords = ("429", "rate limit", "quota", "resource_exhausted", "500", "502", "503", "504", "unavailable", "timeout")
    return any(word in message for word in keywords)


class LLMService:
    """Executes LLM completions via LiteLLM with built-in caching and key failover."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.key_pool = self._build_key_pool(settings)
        self.cache_enabled = settings.cache_enabled
        self._mock_cache: dict[str, tuple[str, int, int, str]] = {}

        litellm.suppress_debug_info = True
        if self.cache_enabled:
            litellm.cache = None
            litellm.enable_cache(type=LiteLLMCacheType.LOCAL, max_messages=None)
        else:
            litellm.cache = None

    @staticmethod
    def _build_key_pool(settings: Settings) -> list[KeySlot]:
        keys = settings.parsed_api_keys
        models = settings.parsed_models
        pool: list[KeySlot] = []
        for idx, api_key in enumerate(keys):
            model = models[idx] if idx < len(models) else settings.llm_model
            pool.append(KeySlot(key_id=f"key_{idx + 1}", api_key=api_key, model=model))
        return pool

    @property
    def keys_configured(self) -> int:
        """Number of API keys loaded in the pool."""
        return len(self.key_pool)

    async def generate_reply(
        self,
        messages: list[dict[str, str]],
        simulate_failure_header: str | None = None,
    ) -> LLMReply:
        """Generate a reply using LiteLLM cache and ordered key failover."""
        if not self.key_pool:
            simulated_map: dict[str, int] = {}
            if self.settings.enable_failure_simulation:
                simulated_map = parse_simulate_failure_header(simulate_failure_header)
            if "*" in simulated_map:
                raise AllProvidersUnavailableError("all providers unavailable")

            model = "lyftr/mock-v1"
            cache_key = json.dumps(
                {"model": model, "messages": messages},
                ensure_ascii=False,
                sort_keys=True,
            )
            bypass_cache = bool(simulated_map)
            start_time = time.perf_counter()
            cached = None if bypass_cache else self._mock_cache.get(cache_key)
            if cached is not None and self.cache_enabled:
                reply, input_tokens, output_tokens, cached_model = cached
                return LLMReply(
                    reply=reply,
                    resolution_source="LLM_CACHE",
                    model=cached_model,
                    latency_ms=max(1, int(round((time.perf_counter() - start_time) * 1000))),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    key_id=None,
                    fallback_attempts=0,
                )

            await asyncio.sleep(0.03)
            reply = self._generate_mock_reply(messages)
            input_tokens = sum(max(1, len(str(message.get("content", ""))) // 4) for message in messages)
            output_tokens = max(1, len(reply) // 4)
            if self.cache_enabled and not bypass_cache:
                self._mock_cache[cache_key] = (reply, input_tokens, output_tokens, model)
            return LLMReply(
                reply=reply,
                resolution_source="LIVE_INFERENCE",
                model=model,
                latency_ms=max(1, int(round((time.perf_counter() - start_time) * 1000))),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                key_id=None,
                fallback_attempts=0,
            )

        simulated_map: dict[str, int] = {}
        if self.settings.enable_failure_simulation:
            simulated_map = parse_simulate_failure_header(simulate_failure_header)
        bypass_cache = bool(simulated_map)
        start_time = time.perf_counter()
        fallback_attempts = 0

        for slot in self.key_pool:
            simulated_status = simulated_map.get("*", simulated_map.get(slot.key_id))
            try:
                if simulated_status is not None:
                    raise SimulatedKeyFailureError(slot.key_id, simulated_status)

                completion_kwargs: dict[str, Any] = {
                    "model": slot.model,
                    "messages": messages,
                    "api_key": slot.api_key,
                    "temperature": self.settings.llm_temperature,
                    "max_tokens": self.settings.llm_max_tokens,
                    "timeout": self.settings.llm_timeout_seconds,
                    "num_retries": 0,
                    "drop_params": True,
                }
                if self.settings.llm_base_url.strip():
                    completion_kwargs["api_base"] = self.settings.llm_base_url.strip()
                if bypass_cache:
                    completion_kwargs["cache"] = {"no-cache": True}

                response = await litellm.acompletion(**completion_kwargs)

                choices = getattr(response, "choices", None) or []
                if not choices:
                    raise ValueError("Empty choices returned by provider")
                content = (getattr(choices[0].message, "content", None) or "").strip()
                if not content:
                    raise ValueError("Empty assistant reply returned by provider")

                hidden_params = getattr(response, "_hidden_params", None) or {}
                is_cache_hit = bool(hidden_params.get("cache_hit"))

                usage = getattr(response, "usage", None)
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                latency_ms = max(1, int(round((time.perf_counter() - start_time) * 1000)))

                if is_cache_hit:
                    return LLMReply(
                        reply=content,
                        resolution_source="LLM_CACHE",
                        model=slot.model,
                        latency_ms=latency_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        key_id=None,
                        fallback_attempts=0,
                    )

                return LLMReply(
                    reply=content,
                    resolution_source="LIVE_INFERENCE",
                    model=slot.model,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    key_id=slot.key_id,
                    fallback_attempts=fallback_attempts,
                )

            except Exception as exc:
                if _is_failover_error(exc) or isinstance(exc, ValueError):
                    fallback_attempts += 1
                    logger.warning(
                        "Key %s failed (%s); attempting failover (%d/%d).",
                        slot.key_id,
                        type(exc).__name__,
                        fallback_attempts,
                        len(self.key_pool),
                    )
                    continue
                fallback_attempts += 1
                logger.warning(
                    "Unexpected error on %s (%s: %s); failing over.",
                    slot.key_id,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                continue

        raise AllProvidersUnavailableError("all providers unavailable")

    def _generate_mock_reply(self, messages: list[dict[str, str]]) -> str:
        """Create a useful context-aware local reply when no API key is configured."""
        user_messages = [item["content"].strip() for item in messages if item["role"] == "user"]
        prompt = user_messages[-1] if user_messages else ""
        lowered = prompt.lower()
        turn_count = len(user_messages)

        if any(word in lowered for word in ("hello", "hi ", "hey", "greetings")) or lowered in {"hi", "hello", "hey"}:
            return (
                "Hello! I am Lyftr AI running in local demo mode. "
                "No API key is configured, so this response uses the built-in mock provider."
            )
        if "key failover" in lowered or "failover" in lowered:
            return (
                "API key failover improves reliability by trying keys or providers in sequence. "
                "In this app, a 429, quota, 401/403, or 5xx failure moves to the next key. "
                "The response reports the successful key and fallback count, while simulated "
                "all-key failure returns a clean 503. Cache hits use no key."
            )
        if "cache" in lowered:
            return (
                "LiteLLM response caching keeps a completed answer for the full message context. "
                "An identical request can therefore be marked LLM_CACHE and return the same reply "
                "with lower latency and stored token usage."
            )
        if "sqlite" in lowered:
            return (
                "SQLite stores each session and ordered message turn. "
                "The backend restores conversation history before generating a new reply."
            )
        if "bullet" in lowered:
            return (
                "- API key failover tries the next provider after a failure.\n"
                "- Response caching returns identical replies faster.\n"
                "- SQLite preserves sessions and per-message metadata."
            )
        return (
            f"I received your message: \"{prompt}\". "
            f"This is turn {turn_count} of the current session, answered in local mock mode."
        )
