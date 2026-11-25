"""Tests verifying API contract, LiteLLM caching, key failover, and SQLite persistence."""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from fastapi.testclient import TestClient
import litellm

from app.config import Settings
from app.main import create_app


def _make_fake_response(
    content: str,
    model: str = "gemini/gemini-2.0-flash",
    prompt_tokens: int = 42,
    completion_tokens: int = 9,
    cache_hit: bool = False,
):
    """Build a LiteLLM-like ModelResponse object."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        model=model,
        _hidden_params={"cache_hit": cache_hit},
    )


def _make_app(tmp_path: Path, **overrides):
    settings = Settings(
        DATABASE_URL=f"sqlite:///{tmp_path / 'test_app.db'}",
        LLM_API_KEYS=overrides.get("LLM_API_KEYS", "secret-key-one,secret-key-two"),
        LLM_MODEL=overrides.get("LLM_MODEL", "gemini/gemini-2.0-flash"),
        ENABLE_FAILURE_SIMULATION=overrides.get("ENABLE_FAILURE_SIMULATION", True),
        CACHE_ENABLED=overrides.get("CACHE_ENABLED", True),
    )
    return create_app(settings=settings)


def test_health_endpoint_contract(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "keys_configured": 2,
        "cache_enabled": True,
    }


def test_only_three_contract_routes_exist(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    user_paths = {
        route.path
        for route in app.routes
        if route.path in {"/health", "/chat", "/sessions/{session_id}/messages"}
    }
    assert user_paths == {"/health", "/chat", "/sessions/{session_id}/messages"}


def test_chat_validation_errors_not_stored_in_history(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        empty_msg = client.post("/chat", json={"session_id": "s-1", "message": "   "})
        assert empty_msg.status_code == 422
        assert "detail" in empty_msg.json()

        too_long = client.post(
            "/chat",
            json={"session_id": "s-1", "message": "x" * 4001},
        )
        assert too_long.status_code == 422

        missing_session = client.post("/chat", json={"session_id": " ", "message": "hi"})
        assert missing_session.status_code == 422

        history = client.get("/sessions/s-1/messages")
        assert history.status_code == 404
        assert history.json() == {"detail": "session not found"}


def test_multi_turn_history_and_cache_hit_across_sessions(
    tmp_path: Path, monkeypatch
) -> None:
    in_memory_store: dict[str, object] = {}
    provider_calls: list[dict] = []

    async def fake_acompletion(**kwargs):
        cache_controls = kwargs.get("cache") or {}
        cache_key = litellm.cache.get_cache_key(**kwargs)
        if not cache_controls.get("no-cache") and cache_key in in_memory_store:
            cached = in_memory_store[cache_key]
            return _make_fake_response(
                content=cached["content"],
                model=cached["model"],
                prompt_tokens=cached["prompt_tokens"],
                completion_tokens=cached["completion_tokens"],
                cache_hit=True,
            )

        await asyncio.sleep(0.05)
        provider_calls.append({"api_key": kwargs["api_key"], "messages": kwargs["messages"]})
        user_text = kwargs["messages"][-1]["content"]
        reply_text = f"Answer to: {user_text}"
        prompt_tokens = sum(len(m["content"]) for m in kwargs["messages"])
        completion_tokens = len(reply_text)
        in_memory_store[cache_key] = {
            "content": reply_text,
            "model": kwargs["model"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        return _make_fake_response(
            content=reply_text,
            model=kwargs["model"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_hit=False,
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    app = _make_app(tmp_path)

    with TestClient(app) as client:
        r1 = client.post(
            "/chat",
            json={"session_id": "s-100", "message": "What is the capital of France?"},
        )
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["session_id"] == "s-100"
        assert d1["reply"] == "Answer to: What is the capital of France?"
        assert d1["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert d1["metadata"]["key_id"] == "key_1"
        assert d1["metadata"]["fallback_attempts"] == 0
        assert d1["metadata"]["latency_ms"] >= 15

        # Same first message in a brand new session must hit LLM_CACHE
        r2 = client.post(
            "/chat",
            json={"session_id": "s-200", "message": "What is the capital of France?"},
        )
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["reply"] == d1["reply"]
        assert d2["metadata"]["resolution_source"] == "LLM_CACHE"
        assert d2["metadata"]["key_id"] is None
        assert d2["metadata"]["fallback_attempts"] == 0
        assert d2["metadata"]["input_tokens"] == d1["metadata"]["input_tokens"]
        assert d2["metadata"]["output_tokens"] == d1["metadata"]["output_tokens"]
        assert d2["metadata"]["latency_ms"] < d1["metadata"]["latency_ms"]

        # Follow-up turn in s-100 sends prior history and is NOT served from cache
        r3 = client.post(
            "/chat",
            json={"session_id": "s-100", "message": "What river flows through it?"},
        )
        assert r3.status_code == 200
        d3 = r3.json()
        assert d3["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert len(provider_calls) == 2
        # Turn 2 messages: system + user1 + assistant1 + user2
        assert [m["role"] for m in provider_calls[1]["messages"]] == [
            "system",
            "user",
            "assistant",
            "user",
        ]

        # Session history endpoint excludes system prompt and preserves order
        hist = client.get("/sessions/s-100/messages")
        assert hist.status_code == 200
        msgs = hist.json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
        assert "metadata" not in msgs[0]
        assert msgs[1]["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_real_key_failover_and_simulated_failure_hooks(
    tmp_path: Path, monkeypatch
) -> None:
    attempted_keys: list[str] = []

    async def fake_acompletion(**kwargs):
        attempted_keys.append(kwargs["api_key"])
        if kwargs["api_key"] == "secret-key-one":
            raise litellm.exceptions.RateLimitError(
                message="429 Too Many Requests",
                llm_provider="gemini",
                model=kwargs["model"],
            )
        return _make_fake_response("Fallback reply from key 2")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    app = _make_app(tmp_path)

    with TestClient(app) as client:
        # Real failover when key_1 raises RateLimitError
        resp = client.post(
            "/chat",
            json={"session_id": "s-failover", "message": "Test real failover"},
        )
        assert resp.status_code == 200
        meta = resp.json()["metadata"]
        assert meta["key_id"] == "key_2"
        assert meta["fallback_attempts"] == 1
        assert attempted_keys == ["secret-key-one", "secret-key-two"]

        # Simulated failure: key_1=429 -> key_2 succeeds
        attempted_keys.clear()
        sim1 = client.post(
            "/chat",
            headers={"X-Simulate-Failure": "key_1=429"},
            json={"session_id": "s-sim-1", "message": "Sim 429"},
        )
        assert sim1.status_code == 200
        assert sim1.json()["metadata"]["key_id"] == "key_2"
        assert sim1.json()["metadata"]["fallback_attempts"] == 1
        assert attempted_keys == ["secret-key-two"]

        # Simulated failure: key_1=500 -> key_2 succeeds
        attempted_keys.clear()
        sim500 = client.post(
            "/chat",
            headers={"X-Simulate-Failure": "key_1=500"},
            json={"session_id": "s-sim-500", "message": "Sim 500"},
        )
        assert sim500.status_code == 200
        assert sim500.json()["metadata"]["key_id"] == "key_2"
        assert sim500.json()["metadata"]["fallback_attempts"] == 1
        assert attempted_keys == ["secret-key-two"]

        # Simulated failure: key_1=429,key_2=500 -> 503 and nothing saved
        sim_both = client.post(
            "/chat",
            headers={"X-Simulate-Failure": "key_1=429,key_2=500"},
            json={"session_id": "s-sim-both", "message": "Both keys fail"},
        )
        assert sim_both.status_code == 503
        assert sim_both.json() == {"detail": "all providers unavailable"}
        assert client.get("/sessions/s-sim-both/messages").status_code == 404

        # Simulated failure: *=429 -> 503 and nothing saved in SQLite
        sim_all = client.post(
            "/chat",
            headers={"X-Simulate-Failure": "*=429"},
            json={"session_id": "s-sim-all", "message": "All keys down"},
        )
        assert sim_all.status_code == 503
        assert sim_all.json() == {"detail": "all providers unavailable"}

        missing = client.get("/sessions/s-sim-all/messages")
        assert missing.status_code == 404
        assert missing.json() == {"detail": "session not found"}


def test_session_system_prompt_persists_across_turns(
    tmp_path: Path, monkeypatch
) -> None:
    captured_messages: list[list[dict[str, str]]] = []

    async def fake_acompletion(**kwargs):
        captured_messages.append(kwargs["messages"])
        return _make_fake_response("Aye, captain!")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    app = _make_app(tmp_path)

    with TestClient(app) as client:
        r1 = client.post(
            "/chat",
            json={
                "session_id": "s-pirate",
                "message": "Greet me",
                "system_prompt": "Respond like a pirate.",
            },
        )
        assert r1.status_code == 200

        r2 = client.post(
            "/chat",
            json={"session_id": "s-pirate", "message": "Say goodbye"},
        )
        assert r2.status_code == 200

        history = client.get("/sessions/s-pirate/messages")
        assert history.status_code == 200
        roles = [m["role"] for m in history.json()["messages"]]
        assert roles == ["user", "assistant", "user", "assistant"]

    assert captured_messages[0][0] == {"role": "system", "content": "Respond like a pirate."}
    assert captured_messages[1][0] == {"role": "system", "content": "Respond like a pirate."}


def test_simulate_failure_ignored_when_disabled(tmp_path: Path, monkeypatch) -> None:
    async def fake_acompletion(**kwargs):
        return _make_fake_response("Key 1 active")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    app = _make_app(tmp_path, ENABLE_FAILURE_SIMULATION=False)

    with TestClient(app) as client:
        resp = client.post(
            "/chat",
            headers={"X-Simulate-Failure": "*=429"},
            json={"session_id": "s-disabled", "message": "Should ignore header"},
        )
        assert resp.status_code == 200
        assert resp.json()["metadata"]["key_id"] == "key_1"


def test_mock_mode_placeholder_keys_cache_and_simulated_503(tmp_path: Path) -> None:
    app = _make_app(tmp_path, LLM_API_KEYS="your-first-api-key,your-second-api-key")

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["keys_configured"] == 0

        r1 = client.post(
            "/chat",
            json={"session_id": "s-mock-1", "message": "Explain caching"},
        )
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert d1["metadata"]["model"] == "lyftr/mock-v1"

        r2 = client.post(
            "/chat",
            json={"session_id": "s-mock-2", "message": "Explain caching"},
        )
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["reply"] == d1["reply"]
        assert d2["metadata"]["resolution_source"] == "LLM_CACHE"
        assert d2["metadata"]["latency_ms"] < d1["metadata"]["latency_ms"]

        r_fail = client.post(
            "/chat",
            headers={"X-Simulate-Failure": "*=429"},
            json={"session_id": "s-mock-fail", "message": "Explain caching"},
        )
        assert r_fail.status_code == 503
        assert r_fail.json() == {"detail": "all providers unavailable"}
        assert client.get("/sessions/s-mock-fail/messages").status_code == 404
