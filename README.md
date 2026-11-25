# Lyftr AI Chatbot — LLM Chatbot with Key Failover and Response Caching

A full-stack AI chatbot built with **React + TypeScript (Vite)**, **FastAPI**, **LiteLLM**, and **SQLite**, featuring ordered multi-key failover, response caching, per-request token tracking, and on-demand failure simulation via `X-Simulate-Failure`.

## 1. Live URLs

- **Frontend URL**: _Pending deployment (e.g. `https://<your-app>.vercel.app`)_
- **Backend URL**: _Pending deployment (e.g. `https://<your-api>.onrender.com`)_
- **Local Docker URLs**: Frontend at `http://localhost:3000`, Backend at `http://localhost:8000`

## 2. Run locally: with Docker and without

### With Docker (`make up`)

1. Create a `.env` file from the template and add at least two keys in `LLM_API_KEYS`:

```bash
cp .env.example .env
```

2. Start both services:

```bash
make up
```

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000` (SQLite persisted on Docker volume mounted at `/data`)
- Stop containers: `make down`

### Without Docker (Local Development)

**Backend:**

```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend (separate terminal):**

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/health`, `/chat`, and `/sessions/*` to `http://127.0.0.1:8000`.

## 3. Design decisions: key fallback, caching, SQLite session storage

- **Key fallback (`backend/app/llm_service.py`)**: `LLM_API_KEYS` is parsed in order into `key_1`, `key_2`, ... (optionally paired with per-key models from `LLM_MODELS`, defaulting to `LLM_MODEL`). Each `POST /chat` request iterates through the key pool and catches rate-limit (`429`), quota, `5xx`, timeout, and provider/connection errors, incrementing `fallback_attempts` and moving seamlessly to the next key. If all keys fail, FastAPI returns HTTP `503` with `{"detail": "all providers unavailable"}` and does not write the failed turn to SQLite.
- **Response caching (`backend/app/llm_service.py`)**: LiteLLM's built-in in-memory cache (`Cache(type=LiteLLMCacheType.LOCAL, max_messages=None)`) is enabled at startup so multi-turn conversations of any length can be cached by `model + full message list`. Cache hits inspect LiteLLM's `_hidden_params["cache_hit"]` flag and return `resolution_source: "LLM_CACHE"`, `key_id: null`, `fallback_attempts: 0`, and the stored prompt/completion token counts. When `ENABLE_FAILURE_SIMULATION=true` and `X-Simulate-Failure` is present, cache lookup is bypassed (`cache={"no-cache": True}`) so simulated key failover always exercises the key pool.
- **SQLite session storage (`backend/app/database.py`)**: Uses SQLite (`DATABASE_URL`, e.g. `sqlite:////data/app.db` in Docker) with `sessions` and `messages` tables created automatically at startup. `save_turn()` commits both the `user` and `assistant` rows (including serialized JSON `metadata`) atomically only after a valid reply is received, ensuring `422` and `503` failures never pollute session history. `GET /sessions/{session_id}/messages` returns messages oldest-first without the system prompt.

## 4. Verification: how to see failover, cache hits and per-message token usage

### Automated Tests

```bash
make test
```

Runs `pytest -v` for backend contract, caching, failover, and SQLite tests, plus `tsc && vite build` for the frontend.

### Manual API Checks (`curl`)

1. **Health check**:

```bash
curl http://localhost:8000/health
```

2. **Live inference + cache hit across sessions**:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-demo-1","message":"What is the capital of France?"}'

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-demo-2","message":"What is the capital of France?"}'
```

The first returns `resolution_source: "LIVE_INFERENCE"` with `key_id: "key_1"`; the second returns `resolution_source: "LLM_CACHE"`, `key_id: null`, identical `reply` and token counts, and lower `latency_ms`.

3. **Failover test hook (`X-Simulate-Failure`)**:

```bash
# Fails key_1 with 429 and serves via key_2 (fallback_attempts: 1)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=429" \
  -d '{"session_id":"s-fail-1","message":"Hello from key 2"}'

# Fails all keys and returns 503 without persisting to SQLite
curl -i -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: *=429" \
  -d '{"session_id":"s-fail-all","message":"Should fail"}'
```

4. **Session history & per-message tokens**:

```bash
curl http://localhost:8000/sessions/s-demo-1/messages
```

## 5. Setup Used: your editor and AI tools

- **Editor / Environment**: VS Code / Antigravity IDE on Windows with Python 3.11 and Node.js 24
- **AI Tools**: Antigravity AI coding assistant used to scaffold, refactor to contract, and verify the FastAPI + LiteLLM + React codebase

## 6. Bonus features (if any): what you added beyond the requirements and why

- **Per-key model mapping (`LLM_MODELS`)**: Allows pairing `key_1`, `key_2`, ... with either the same model or distinct fallback models across providers without code changes.
- **Session-scoped `system_prompt` persistence**: Stores an optional `system_prompt` per `session_id` in SQLite (`sessions` table) so multi-turn chats keep their custom instructions even if subsequent turns omit `system_prompt`, while keeping `GET /sessions/{session_id}/messages` clean of system messages.
- **Markdown & code block formatting in UI**: Cleanly formats code fences and bullet lists in assistant replies without external markdown dependencies.
