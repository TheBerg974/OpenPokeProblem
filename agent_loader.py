"""
agent_loader.py — Loads sub-agent tool schemas and executors from APM-installed packages.

Packages are installed by the Microsoft APM CLI (`apm install <owner>/<repo>`)
into the `apm_modules/` directory.  This module provides a thin runtime layer
that reads the compiled output so the LangGraph orchestrator can dynamically
inject tool schemas and run executors without ever hardcoding agent logic.

Package layout (after `apm install TheBerg974/open-poke-agents`):
    apm_modules/
        TheBerg974/open-poke-agents/
            agents/
                gmail-draft/
                    agent.json      <- OpenAI-compatible tool schema
                    executor.py     <- async def execute(tool_input: dict) -> dict
                web-search/
                    agent.json
                    executor.py

Agent names passed to this module match the directory name, e.g. "gmail-draft".
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of installed APM packages — resolved relative to this file so it works
# regardless of the working directory.
APM_MODULES_PATH: Path = Path(
    os.getenv(
        "APM_MODULES_PATH",
        str(Path(__file__).parent / "apm_modules"),
    )
)


# ---------------------------------------------------------------------------
# Internal: resolve an agent directory across all installed packages
# ---------------------------------------------------------------------------


def _find_agent_path(agent_name: str) -> Path:
    """
    Walk every installed package under ``apm_modules/`` and return the first
    ``agents/<agent_name>/`` directory that contains an ``agent.json``.

    Raises
    ------
    FileNotFoundError
        If no installed package exposes the requested agent.
    """
    if not APM_MODULES_PATH.exists():
        raise FileNotFoundError(
            f"apm_modules/ not found at {APM_MODULES_PATH}. "
            "Run `apm install` to install agent packages first."
        )

    # Glob pattern: apm_modules/<owner>/<repo>/agents/<agent_name>/agent.json
    matches = list(APM_MODULES_PATH.glob(f"*/*/agents/{agent_name}/agent.json"))

    if not matches:
        raise FileNotFoundError(
            f"Agent '{agent_name}' not found in any installed APM package under "
            f"{APM_MODULES_PATH}. "
            f"Install a package that provides it, e.g.: "
            f"apm install TheBerg974/open-poke-agents"
        )

    return matches[0].parent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def get_agent_manifest(agent_name: str) -> dict[str, Any]:
    """
    Load and return the OpenAI-compatible tool schema for *agent_name*.

    Raises
    ------
    FileNotFoundError
        If no installed package provides the agent.
    ValueError
        If ``agent.json`` is not valid JSON.
    """
    agent_path = _find_agent_path(agent_name)
    manifest_path = agent_path / "agent.json"

    try:
        text = manifest_path.read_text(encoding="utf-8")
        manifest: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"agent.json for '{agent_name}' is not valid JSON: {exc}"
        ) from exc

    logger.debug("Loaded manifest for '%s' from %s", agent_name, manifest_path)
    return manifest


def load_agent_executor(agent_name: str) -> Callable[..., Any]:
    """
    Dynamically import ``executor.py`` from the installed package directory
    and return the module-level ``execute`` callable.

    The returned function must have the signature:
        async def execute(tool_input: dict) -> dict

    Raises
    ------
    FileNotFoundError
        If no installed package provides the agent or its executor.
    AttributeError
        If ``executor.py`` does not expose an ``execute`` function.
    """
    agent_path = _find_agent_path(agent_name)
    executor_path = agent_path / "executor.py"

    if not executor_path.exists():
        raise FileNotFoundError(
            f"executor.py missing for agent '{agent_name}' at {executor_path}."
        )

    # Build a stable, importable module name so repeated calls reuse the cache.
    module_name = f"apm_modules.agents.{agent_name.replace('-', '_')}"

    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, executor_path)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not create module spec for agent '{agent_name}' "
                f"at {executor_path}."
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]

    if not hasattr(module, "execute"):
        raise AttributeError(
            f"executor.py for '{agent_name}' must define "
            "`async def execute(tool_input: dict) -> dict`."
        )

    logger.debug("Loaded executor for '%s' from %s", agent_name, executor_path)
    return module.execute  # type: ignore[return-value]


async def list_installed_agents() -> list[str]:
    """
    Return the names of every agent exposed by all installed APM packages.
    Used by the GlobalRouter to populate the meta-registry.
    """
    if not APM_MODULES_PATH.exists():
        logger.warning("apm_modules/ not found at %s", APM_MODULES_PATH)
        return []

    agents: list[str] = []
    for manifest_path in sorted(APM_MODULES_PATH.glob("*/*/agents/*/agent.json")):
        agents.append(manifest_path.parent.name)

    return agents
