"""
CrewAI auto-instrumentation.

Usage::

    from crewai import Crew, Agent, Task
    from execlave import Execlave
    from execlave.integrations.crewai import instrument_crew

    exe = Execlave(api_key="...")
    crew = Crew(agents=[...], tasks=[...])
    instrument_crew(crew, exe, agent_id="my-crew")
    result = crew.kickoff()

Implementation: CrewAI exposes ``step_callback`` and ``task_callback``
hooks on ``Agent`` and ``Crew`` objects. We chain our callbacks in front
of any existing user callbacks so attaching the instrumentation never
overrides user-supplied hooks.

A ``_execlave_instrumented`` marker is set on the crew the first time
``instrument_crew`` runs so the helper is idempotent.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ..client import Execlave
from ..errors import (
    AgentPausedError,
    ApprovalTimeoutError,
    EnforcementUnavailableError,
    PolicyBlockedError,
    PolicyDeniedError,
)
from ..instrumentation.spans import (
    SPAN_KIND_AGENT,
    SPAN_KIND_TOOL,
    Span,
    get_span_tree,
)

logger = logging.getLogger("Execlave.crewai")

_MARKER = "_execlave_instrumented"

_ENFORCEMENT_ERRORS: tuple[type[BaseException], ...] = (
    PolicyBlockedError,
    PolicyDeniedError,
    ApprovalTimeoutError,
    EnforcementUnavailableError,
    AgentPausedError,
)


def instrument_crew(
    crew: Any,
    exe: Execlave,
    *,
    agent_id: str,
    enforce: bool = True,
    session_id: str | None = None,
    user_id: str | None = None,
) -> Any:
    """Attach Execlave instrumentation to a CrewAI ``Crew`` instance.

    Idempotent: calling twice on the same crew is a no-op.

    Returns the same crew for fluent chaining.
    """
    if crew is None:
        raise ValueError("instrument_crew: crew must not be None")
    if exe is None:
        raise ValueError("instrument_crew: exe must not be None")
    if not agent_id:
        raise ValueError("instrument_crew: agent_id is required")

    if getattr(crew, _MARKER, False):
        logger.debug("crew already instrumented — skipping")
        return crew

    tree = get_span_tree(exe)

    # --- step_callback: fires after every step in an agent's thought loop.
    def _step_callback(step: Any) -> None:
        try:
            _record_step(tree, exe, step, agent_id, enforce, session_id, user_id)
        except _ENFORCEMENT_ERRORS:
            raise
        except Exception:
            logger.warning("crewai step_callback failed", exc_info=True)

    # --- task_callback: fires after each task completes.
    def _task_callback(task_output: Any) -> None:
        try:
            _record_task(tree, exe, task_output, agent_id, session_id, user_id)
        except Exception:
            logger.warning("crewai task_callback failed", exc_info=True)

    # Chain rather than replace — preserve user-supplied callbacks.
    _chain_callback(crew, "step_callback", _step_callback)
    _chain_callback(crew, "task_callback", _task_callback)

    # Crew also exposes a list of Agent objects each with their own
    # step_callback. Instrument them too so tool calls from sub-agents
    # are captured.
    for inner in getattr(crew, "agents", []) or []:
        _chain_callback(inner, "step_callback", _step_callback)

    try:
        setattr(crew, _MARKER, True)
    except Exception:  # pragma: no cover — pydantic frozen?
        logger.debug("could not set instrumented marker; double-wrap guard inactive")

    return crew


# ----------------------------------------------------------------------
# Internal: chaining + recording
# ----------------------------------------------------------------------
def _chain_callback(obj: Any, attr: str, new_cb: Callable[[Any], None]) -> None:
    existing = getattr(obj, attr, None)
    if existing is None:
        try:
            setattr(obj, attr, new_cb)
        except Exception:
            logger.debug("could not set %s on %r", attr, obj, exc_info=True)
        return

    def chained(payload: Any) -> None:
        try:
            new_cb(payload)
        except _ENFORCEMENT_ERRORS:
            raise
        except Exception:
            logger.debug("execlave %s failed", attr, exc_info=True)
        try:
            existing(payload)
        except Exception:
            logger.debug("user-provided %s raised", attr, exc_info=True)

    try:
        setattr(obj, attr, chained)
    except Exception:
        logger.debug("could not chain %s on %r", attr, obj, exc_info=True)


def _record_step(
    tree: Any,
    exe: Execlave,
    step: Any,
    agent_id: str,
    enforce: bool,
    session_id: str | None,
    user_id: str | None,
) -> None:
    """Record a single step in the agent loop.

    CrewAI step objects vary in shape; we do attribute detection rather
    than structural assumptions."""
    tool = getattr(step, "tool", None)
    tool_input = getattr(step, "tool_input", None)
    kind = SPAN_KIND_TOOL if tool else SPAN_KIND_AGENT
    name = str(tool) if tool else "step"

    if kind == SPAN_KIND_TOOL and enforce and tool:
        try:
            exe.enforce_policy(
                agent_id,
                str(tool_input) if tool_input is not None else f"tool:{tool}",
                tools=[str(tool)],
            )
        except _ENFORCEMENT_ERRORS:
            raise
        except Exception:
            logger.warning("enforce_policy failed (non-fatal)", exc_info=True)

    span: Span = tree.start(
        kind=kind,
        name=name,
        agent_id=agent_id,
        session_id=session_id,
        user_id=user_id,
        metadata={"crewaiStep": True},
    )
    if tool_input is not None:
        span.set_input(tool_input)
    output = getattr(step, "output", None) or getattr(step, "result", None)
    if output is not None:
        span.set_output(output)
    span.finish("success")


def _record_task(
    tree: Any,
    exe: Execlave,
    task_output: Any,
    agent_id: str,
    session_id: str | None,
    user_id: str | None,
) -> None:
    """Record a completed task as an agent-level span."""
    description = (
        getattr(task_output, "description", None)
        or getattr(task_output, "task", None)
        or "task"
    )
    span: Span = tree.start(
        kind=SPAN_KIND_AGENT,
        name=str(description)[:80],
        agent_id=agent_id,
        session_id=session_id,
        user_id=user_id,
        metadata={"crewaiTask": True},
    )
    raw = getattr(task_output, "raw", None) or getattr(task_output, "output", None)
    if raw is not None:
        span.set_output(raw)
    span.finish("success")
