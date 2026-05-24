"""
main.py — FastAPI entry point for the Open Poke Dynamic APM system.

Lifecycle
---------
startup  → initialise DB tables, warm Redis meta-registry
shutdown → close Redis connection pool

POST /api/chat
--------------
1. Pre-process : resolve / rehydrate thread state (Redis → Postgres fallback)
2. Execute     : run the LangGraph workflow
3. Respond     : return assistant reply immediately
4. Background  : async DB sync (history + meta-registry) without blocking
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from cache import (
    cache_meta_registry,
    cache_thread_state,
    close_redis,
    get_cached_thread_state,
)
from database import (
    append_thread_history,
    fetch_all_thread_metas,
    fetch_thread_history,
    init_db,
    upsert_thread_meta,
)
from graph import AgentState, agent_graph

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Open Poke — Dynamic APM",
    description="Hierarchical thread caching + local APM agent orchestration.",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup() -> None:
    logger.info("Initialising database tables…")
    await init_db()

    logger.info("Warming Redis meta-registry…")
    metas = await fetch_all_thread_metas()
    registry_list = [m.model_dump(mode="json") for m in metas]
    await cache_meta_registry(registry_list)
    logger.info("Startup complete (%d threads in registry).", len(registry_list))


@app.on_event("shutdown")
async def shutdown() -> None:
    await close_redis()
    logger.info("Redis connection closed.")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    user_id: str
    thread_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    thread_id: str
    reply: str


# ---------------------------------------------------------------------------
# Background task: persist state & refresh registry
# ---------------------------------------------------------------------------


async def _sync_to_db(
    thread_id: str,
    user_id: str,
    user_message: str,
    assistant_reply: str,
    final_state: AgentState,
) -> None:
    """
    Runs after the HTTP response is sent.

    1. Append the new user + assistant turns to ThreadHistory.
    2. Upsert ThreadMeta with a fresh title / summary placeholder.
    3. Write the pruned state back to Redis.
    4. Refresh the in-memory meta-registry in Redis.
    """
    try:
        await append_thread_history(thread_id, "user", user_message)
        await append_thread_history(thread_id, "assistant", assistant_reply)

        # Use the last assistant message as a lightweight semantic summary.
        await upsert_thread_meta(
            thread_id=thread_id,
            title=f"Thread {thread_id[:8]}",
            summary=assistant_reply[:200],
        )

        # Write the pruned state back to L1.
        await cache_thread_state(thread_id, dict(final_state))

        # Refresh the full meta-registry in Redis.
        metas = await fetch_all_thread_metas()
        await cache_meta_registry([m.model_dump(mode="json") for m in metas])

        logger.debug("Background DB sync complete for thread '%s'.", thread_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Background DB sync failed for thread '%s': %s", thread_id, exc)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    # ------------------------------------------------------------------
    # 1. Resolve thread ID
    # ------------------------------------------------------------------
    thread_id = request.thread_id or str(uuid.uuid4())

    # ------------------------------------------------------------------
    # 2. Pre-process: hydrate state (L1 Redis → L2 Postgres fallback)
    # ------------------------------------------------------------------
    state: Optional[AgentState] = await get_cached_thread_state(thread_id)

    if state is None and request.thread_id:
        # Cache miss — rehydrate from Postgres.
        logger.info("Cache miss for thread '%s'. Rehydrating from DB.", thread_id)
        history_rows = await fetch_thread_history(thread_id)
        messages = [{"role": r.role, "content": r.content} for r in history_rows]

        state = AgentState(
            messages=messages,
            current_thread_id=thread_id,
            active_tools=[],
            _raw_tool_output=None,
            _active_tool_schema=None,
        )
        # Write back to L1 so subsequent requests are fast.
        await cache_thread_state(thread_id, dict(state))
    elif state is None:
        # Brand new thread.
        state = AgentState(
            messages=[],
            current_thread_id=thread_id,
            active_tools=[],
            _raw_tool_output=None,
            _active_tool_schema=None,
        )

    # ------------------------------------------------------------------
    # 3. Append the new user message and run the graph
    # ------------------------------------------------------------------
    state["messages"] = state["messages"] + [
        {"role": "user", "content": request.message}
    ]
    state["current_thread_id"] = thread_id

    final_state: AgentState = await agent_graph.ainvoke(state)

    # ------------------------------------------------------------------
    # 4. Extract assistant reply
    # ------------------------------------------------------------------
    assistant_reply = next(
        (
            m["content"]
            for m in reversed(final_state["messages"])
            if m["role"] == "assistant"
        ),
        "I'm sorry, I couldn't generate a response.",
    )

    # ------------------------------------------------------------------
    # 5. Schedule async DB sync (does NOT block the HTTP response)
    # ------------------------------------------------------------------
    background_tasks.add_task(
        _sync_to_db,
        thread_id,
        request.user_id,
        request.message,
        assistant_reply,
        final_state,
    )

    return ChatResponse(thread_id=thread_id, reply=assistant_reply)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
