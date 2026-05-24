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

Switch between local Ollama and cloud providers via `.env`:

```env
# Local (default) — no API key, no rate limits
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:14b          # or llama3.2:3b, llama3.1:8b, llama3.3:70b …
OLLAMA_BASE_URL=http://localhost:11434

# Gemini
LLM_PROVIDER=gemini
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

### Recommended local models (M4 Pro / Apple Silicon)

| Model | Size | RAM | Quality |
|-------|------|-----|---------|
| `llama3.2:3b` | 2 GB | ~2 GB | Fast, basic routing |
| `qwen2.5:7b` | 5 GB | ~5 GB | Good instruction following |
| `qwen2.5:14b` ✓ | 9 GB | ~9 GB | **Recommended** — accurate routing + clean JSON |
| `llama3.1:8b` | 5 GB | ~5 GB | Strong general purpose |
| `llama3.3:70b` | 42 GB | ~42 GB | Near GPT-4 quality |

Install Ollama and pull a model:
```bash
brew install ollama
brew services start ollama
ollama pull qwen2.5:14b
```

---

## Prompt Budget

Each `POST /api/chat` makes **2 LLM calls** (ContextPruner is deterministic — no LLM):

| Call | System | User | Total |
|------|--------|------|-------|
| GlobalRouter | ~73 tokens | ~74 tokens | **~147 tokens** |
| SubAgentExecutor | ~152 tokens | ~13 tokens | **~165 tokens** |
| **Per request** | | | **~312 tokens** |

- GlobalRouter sends the agent capability catalog as JSON (~74-token user message that grows linearly with number of installed agents)
- SubAgentExecutor sends the full OpenAI function schema for the chosen agent + the user's last message only (not the full history)
- ContextPruner formats output deterministically — zero LLM tokens

---

## Thread Resumption

Every response includes a `thread_id`. Pass it back to continue the same conversation:

```bash
# Turn 1
RESP=$(curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user1","message":"Search for the best local LLMs in 2026"}')
echo $RESP | python3 -m json.tool
THREAD=$(echo $RESP | python3 -c "import sys,json; print(json.load(sys.stdin)['thread_id'])")

# Turn 2 — resumes the same thread
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"user1\",\"thread_id\":\"$THREAD\",\"message\":\"Which runs best on Apple Silicon?\"}" \
  | python3 -m json.tool
```

**Hydration path on resumption:**

```
thread_id provided
  → Redis hit (< 30 min since last turn)?  YES  → use cached state (< 1 ms)
  → Redis miss, thread exists in Postgres?  YES  → rehydrate → re-cache to Redis
  → Neither?                                     → 404 / empty state
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
