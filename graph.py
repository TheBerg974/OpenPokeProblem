"""
graph.py — LangGraph StateGraph orchestration for Open Poke.

Nodes
-----
GlobalRouter      — Reads the user prompt + Redis meta-registry to choose
                    which thread / tool package to activate.
SubAgentExecutor  — Fetches the tool schema from the APM, runs the tool,
                    captures raw output.
ContextPruner     — Summarises raw output to one sentence, appends the
                    summary to messages, and strips heavy data from state.

Flow
----
    GlobalRouter → SubAgentExecutor → ContextPruner → END
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from agent_loader import get_agent_manifest, list_installed_agents, load_agent_executor
from cache import get_meta_registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM client — provider selected by LLM_PROVIDER env var
#   LLM_PROVIDER=gemini  -> uses GOOGLE_API_KEY  (default)
#   LLM_PROVIDER=openai  -> uses OPENAI_API_KEY
# ---------------------------------------------------------------------------

def _build_llm() -> Any:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
    # Default: Gemini
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
    )

_llm_instance: Any = None

def get_llm() -> Any:
    """Lazily build the LLM client so .env is loaded before first use."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = _build_llm()
    return _llm_instance

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """Shared state dict threaded through every node."""

    messages: list[dict[str, str]]          # trimmed history (role + content)
    current_thread_id: str                  # active thread identifier
    active_tools: list[str]                 # resolved APM package namespaces
    # Ephemeral fields — written by SubAgentExecutor, deleted by ContextPruner
    _raw_tool_output: Optional[dict[str, Any]]
    _active_tool_schema: Optional[dict[str, Any]]


# ---------------------------------------------------------------------------
# Node: GlobalRouter
# ---------------------------------------------------------------------------


async def global_router(state: AgentState) -> AgentState:
    """
    Read the latest user message and the installed APM agent descriptions to
    decide which sub-agent should handle this turn.

    Agent capability descriptions come from each agent's ``agent.json``
    (the ``function.description`` field), loaded directly from ``apm_modules/``.
    The Redis meta-registry is consulted for thread-level context but is NOT
    used to select the agent — that would mix thread history with capability
    routing.

    Returns ``active_tools=[]`` (plain-text LLM fallback) when no agent fits.
    """
    user_message = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        "",
    )

    # ------------------------------------------------------------------ #
    # 1. Build an agent capability map from apm_modules/                  #
    # ------------------------------------------------------------------ #
    agent_names = await list_installed_agents()
    if not agent_names:
        logger.debug("GlobalRouter: no agents installed in apm_modules/, falling back to plain LLM.")
        return {**state, "active_tools": []}

    agent_catalog: dict[str, str] = {}
    for name in agent_names:
        try:
            manifest = await get_agent_manifest(name)
            description = (
                manifest.get("function", {}).get("description")
                or manifest.get("description")
                or name
            )
            agent_catalog[name] = description
        except Exception as exc:
            logger.warning("GlobalRouter: could not load manifest for '%s': %s", name, exc)

    if not agent_catalog:
        return {**state, "active_tools": []}

    catalog_json = json.dumps(agent_catalog, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ #
    # 2. Thread context from Redis (used as extra signal, not for routing) #
    # ------------------------------------------------------------------ #
    thread_context = ""
    registry = await get_meta_registry()
    thread_entry = next(
        (t for t in registry if t.get("thread_id") == state.get("current_thread_id")),
        None,
    )
    if thread_entry and thread_entry.get("semantic_summary"):
        thread_context = f"\nThread context: {thread_entry['semantic_summary']}"

    # ------------------------------------------------------------------ #
    # 3. Ask the LLM to pick the right agent                              #
    # ------------------------------------------------------------------ #
    prompt = [
        SystemMessage(
            content=(
                "You are a tool-routing agent. Given the user's message and a JSON "
                "map of available agent names to their descriptions, return ONLY the "
                "single agent name (key) that best handles the request. "
                "If no agent is relevant, return the string 'none'. "
                "Return only the agent name — no explanation, no quotes."
            )
        ),
        HumanMessage(
            content=(
                f"Available agents:\n{catalog_json}"
                f"{thread_context}\n\n"
                f"User message:\n{user_message}"
            )
        ),
    ]

    response = await get_llm().ainvoke(prompt)
    chosen = response.content.strip().strip('"').strip("'")

    if chosen.lower() == "none" or chosen not in agent_catalog:
        logger.debug("GlobalRouter: no matching agent for this turn (LLM chose '%s').", chosen)
        return {**state, "active_tools": []}

    logger.info("GlobalRouter: routing to agent '%s'", chosen)
    return {**state, "active_tools": [chosen]}


# ---------------------------------------------------------------------------
# Node: SubAgentExecutor
# ---------------------------------------------------------------------------


async def sub_agent_executor(state: AgentState) -> AgentState:
    """
    For each tool in active_tools, fetch its schema, inject it into the LLM,
    and run the executor.  Raw output is stored ephemerally in state.
    """
    if not state.get("active_tools"):
        # Plain-text fallback: let the LLM respond directly.
        lc_messages = [
            (HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
            for m in state["messages"]
        ]
        response = await get_llm().ainvoke(lc_messages)
        updated_messages = state["messages"] + [
            {"role": "assistant", "content": response.content}
        ]
        return {
            **state,
            "messages": updated_messages,
            "_raw_tool_output": None,
            "_active_tool_schema": None,
        }

    agent_name = state["active_tools"][0]

    try:
        schema = await get_agent_manifest(agent_name)
        executor = load_agent_executor(agent_name)
    except (FileNotFoundError, AttributeError) as exc:
        logger.error("SubAgentExecutor: could not load '%s': %s", agent_name, exc)
        error_msg = f"Tool '{agent_name}' is not available: {exc}"
        return {
            **state,
            "messages": state["messages"] + [{"role": "assistant", "content": error_msg}],
            "_raw_tool_output": None,
            "_active_tool_schema": None,
        }

    # Ask the LLM to generate the tool call arguments.
    tool_call_prompt = [
        SystemMessage(
            content=(
                "You are a tool-call generation agent. "
                "Given the conversation and the tool schema below, "
                "output ONLY a valid JSON object matching the tool's input parameters. "
                "Do not add any explanation.\n\n"
                f"Tool schema:\n{json.dumps(schema, ensure_ascii=False)}"
            )
        ),
        *[
            (HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
            for m in state["messages"]
        ],
    ]

    args_response = await get_llm().ainvoke(tool_call_prompt)

    try:
        tool_input: dict[str, Any] = json.loads(args_response.content)
    except json.JSONDecodeError:
        tool_input = {"raw_prompt": args_response.content}

    raw_output: dict[str, Any] = await executor(tool_input)
    logger.debug("SubAgentExecutor: raw output keys=%s", list(raw_output.keys()))

    return {
        **state,
        "_raw_tool_output": raw_output,
        "_active_tool_schema": schema,
    }


# ---------------------------------------------------------------------------
# Node: ContextPruner
# ---------------------------------------------------------------------------


def _format_tool_output(raw_output: dict[str, Any]) -> str:
    """
    Convert raw tool output to a concise assistant message without an LLM call.
    Prefers common 'message'/'result'/'content'/'summary' keys; falls back to
    a truncated JSON dump.
    """
    for key in ("message", "result", "content", "summary", "text", "output"):
        if key in raw_output and isinstance(raw_output[key], str):
            return raw_output[key]
    serialised = json.dumps(raw_output, default=str, ensure_ascii=False)
    return serialised[:500] + ("…" if len(serialised) > 500 else "")


async def context_pruner(state: AgentState) -> AgentState:
    """
    Format the raw tool output into a readable assistant message and drop the
    heavy ephemeral fields from state to keep the context window lean.
    No LLM call — deterministic formatting keeps us well inside rate limits.
    """
    raw_output = state.get("_raw_tool_output")

    if raw_output is None:
        # SubAgentExecutor already wrote the assistant message; just clean up.
        return {
            **state,
            "_raw_tool_output": None,
            "_active_tool_schema": None,
        }

    summary_sentence: str = _format_tool_output(raw_output)

    updated_messages = state["messages"] + [
        {"role": "assistant", "content": summary_sentence}
    ]

    # Explicitly drop heavy ephemeral keys.
    pruned_state: AgentState = {
        **state,
        "messages": updated_messages,
        "_raw_tool_output": None,
        "_active_tool_schema": None,
    }

    logger.debug("ContextPruner: pruned state, summary='%s'", summary_sentence)
    return pruned_state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    """Compile and return the LangGraph StateGraph."""
    builder: StateGraph = StateGraph(AgentState)

    builder.add_node("GlobalRouter", global_router)
    builder.add_node("SubAgentExecutor", sub_agent_executor)
    builder.add_node("ContextPruner", context_pruner)

    builder.set_entry_point("GlobalRouter")
    builder.add_edge("GlobalRouter", "SubAgentExecutor")
    builder.add_edge("SubAgentExecutor", "ContextPruner")
    builder.add_edge("ContextPruner", END)

    return builder.compile()


# Module-level compiled graph — imported by main.py
agent_graph = build_graph()
