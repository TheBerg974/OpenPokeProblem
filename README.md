# OpenPokeProblem

A FastAPI + LangGraph orchestration server that routes user messages to versioned AI sub-agents managed by the [Microsoft APM CLI](https://microsoft.github.io/apm/). Conversations are persisted in PostgreSQL and cached in Redis for low-latency thread resumption.

---

## What Was Built

### Architecture

```
POST /api/chat
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  FastAPI  (main.py)                                  │
│  • Hydrates thread state: Redis → Postgres → empty  │
│  • Invokes LangGraph StateGraph                     │
│  • Persists history async via BackgroundTasks       │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
        ┌──────────────────┐
        │  GlobalRouter    │  LLM picks best agent from apm_modules/
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ SubAgentExecutor │  LLM generates tool args → executor runs
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  ContextPruner   │  Formats output, trims state (no LLM call)
        └──────────────────┘
```

### Components

| File | Purpose |
|------|---------|
| `main.py` | FastAPI entry point; startup, `/health`, `POST /api/chat` |
| `graph.py` | LangGraph StateGraph with 3 nodes; lazy LLM init |
| `agent_loader.py` | Runtime loader for APM-installed sub-agent packages |
| `database.py` | SQLAlchemy 2 async ORM; `ThreadMeta` + `ThreadHistory` tables |
| `cache.py` | Redis L1 cache; thread state (30 min TTL) + meta-registry |
| `docker-compose.yml` | PostgreSQL 15 + Redis Stack (local infra) |
| `apm.yml` | APM manifest; pins `TheBerg974/open-poke-agents` package |

### Sub-Agent Package

Versioned sub-agents live in a separate GitHub repo: **[TheBerg974/open-poke-agents](https://github.com/TheBerg974/open-poke-agents)**

Installed locally via:
```bash
apm install TheBerg974/open-poke-agents --target copilot
```

| Agent | Capability |
|-------|-----------|
| `web-search` | Web search via Tavily / Brave / Google CSE |
| `gmail-draft` | Draft emails via Gmail API |

Each agent ships an `agent.json` (OpenAI function schema) and an `executor.py` (`async def execute(tool_input: dict) -> dict`). The APM CLI also deploys `.agent.md` files to `.github/agents/` for GitHub Copilot.

### Thread Caching — Hydration Path

```
Request with thread_id
  → Redis hit?   YES → use cached state
  → Redis miss?  YES thread_id exists in Postgres → rehydrate → re-cache
  → New thread?  empty state
```

After each response, `_sync_to_db()` runs as a background task:
- Appends user + assistant turns to `ThreadHistory`
- Upserts `ThreadMeta` (title, semantic summary)
- Re-caches state to Redis
- Refreshes the in-memory meta-registry

---

## Stack

- **Python 3.14** · **FastAPI 0.136** · **uvicorn**
- **LangGraph 1.2** · **langchain-google-genai** (Gemini) / **langchain-openai**
- **SQLAlchemy 2 + asyncpg** (PostgreSQL 15)
- **Redis 7 + hiredis**
- **Microsoft APM CLI v0.14**
- **Docker Compose** (local infra)

---

## Quick Start

### 1. Prerequisites

- Docker Desktop running
- Python 3.14 with a `.venv/`
- Microsoft APM CLI: `brew install microsoft/apm/apm`

### 2. Start infrastructure

```bash
docker compose up -d
```

### 3. Install dependencies

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 4. Install sub-agent packages

```bash
apm install TheBerg974/open-poke-agents --target copilot
```

### 5. Configure environment

Create a `.env` file (see `.env.example` for all variables):

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=<your-key-from-aistudio.google.com/apikey>
GEMINI_MODEL=gemini-2.0-flash

DATABASE_URL=postgresql+asyncpg://admin:password@localhost:5432/open_poke
REDIS_URL=redis://localhost:6379/0
```

> **Important:** Generate the Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) using **"Create API key in new project"** to get free-tier quotas.

### 6. Run the server

```bash
.venv/bin/uvicorn main:app --port 8000
```

### 7. Test

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user1","message":"Search for the latest AI news"}'

# Resume a thread
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user1","thread_id":"<id-from-above>","message":"Tell me more about the first result"}'
```

---

## LLM Provider

Switch between Gemini and OpenAI via `.env`:

```env
# Gemini (default)
LLM_PROVIDER=gemini
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

---

## Wiring Real Sub-Agent Credentials

Add to `.env` to enable live executor calls:

```env
# Web search (pick one)
TAVILY_API_KEY=...
BRAVE_SEARCH_API_KEY=...

# Gmail
GMAIL_CREDENTIALS_PATH=/path/to/credentials.json
```
