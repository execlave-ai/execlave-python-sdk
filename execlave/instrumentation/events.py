"""
High-level event helpers used by framework adapters.

These wrap :class:`Span` lifecycle + :meth:`Execlave.enforce_policy`
in a few lines so per-framework adapters stay small.

All helpers are *fail-safe*: any unexpected exception from the
instrumentation side is swallowed and logged. Policy-blocked errors
(``PolicyBlockedError``) are intentionally *re-raised* so the host
application actually halts — that is the whole point of enforcement.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from ..errors import (
    AgentPausedError,
    ApprovalTimeoutError,
    EnforcementUnavailableError,
    ExeclaveError,
    PolicyBlockedError,
    PolicyDeniedError,
)
from .spans import (
    SPAN_KIND_AGENT,
    SPAN_KIND_LLM,
    SPAN_KIND_RETRIEVER,
    SPAN_KIND_TOOL,
    Span,
    get_span_tree,
)

logger = logging.getLogger("Execlave.instrumentation")

# Errors that SHOULD propagate to the caller. Anything else gets logged.
_ENFORCEMENT_ERRORS: tuple[type[BaseException], ...] = (
    PolicyBlockedError,
    PolicyDeniedError,
    ApprovalTimeoutError,
    EnforcementUnavailableError,
    AgentPausedError,
)


def _maybe_enforce(
    exe: Any,
    *,
    agent_id: str | None,
    input_text: str | None,
    tools: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    estimated_cost: float | None = None,
) -> dict | None:
    """Run ``enforce_policy`` if we have enough info. Returns the decision
    dict, or None if enforcement was skipped (no agent_id or no input).
    Re-raises enforcement errors; swallows everything else with a log."""
    if not agent_id or input_text is None:
        return None
    try:
        return exe.enforce_policy(
            agent_id,
            input_text,
            metadata=metadata,
            estimated_cost=estimated_cost,
            tools=tools,
        )
    except _ENFORCEMENT_ERRORS:
        raise
    except ExeclaveError:
        # Auth / quota / transient — log, do not crash the host app.
        logger.warning("enforce_policy failed (non-fatal)", exc_info=True)
        return None
    except Exception:  # pragma: no cover — defensive
        logger.warning("enforce_policy raised unexpected error", exc_info=True)
        return None


def _summarise_decision(decision: dict) -> dict:
    """Strip volatile/verbose fields from an enforcement response so we
    don't balloon span metadata. Keep only the outcome summary."""
    keys = ("allowed", "mode", "source", "warnings", "requiresApproval")
    return {k: decision[k] for k in keys if k in decision}


@contextmanager
def record_llm_call(
    exe: Any,
    *,
    agent_id: str | None,
    model: str | None = None,
    input_text: str | None = None,
    metadata: dict[str, Any] | None = None,
    estimated_cost: float | None = None,
) -> Iterator[Span]:
    """Context manager for an LLM invocation.

    Runs policy enforcement pre-execution and opens a child LLM span
    under whatever span is currently active. On exit, the span is
    finished; exceptions mark it as ``error``.

    Example::

        with record_llm_call(exe, agent_id="bot", model="gpt-4o",
                             input_text=prompt) as span:
            result = llm.invoke(prompt)
            span.set_output(result)
    """
    decision = _maybe_enforce(
        exe,
        agent_id=agent_id,
        input_text=input_text,
        metadata=metadata,
        estimated_cost=estimated_cost,
    )

    tree = get_span_tree(exe)
    span = tree.start(kind=SPAN_KIND_LLM, name=model or "llm", agent_id=agent_id)
    if input_text is not None:
        span.set_input(input_text)
    if model:
        span.set_model(model)
    if decision is not None:
        span.add_metadata({"enforcement": _summarise_decision(decision)})

    try:
        yield span
    except Exception as exc:
        span.finish("error", str(exc), exc.__class__.__name__)
        raise
    else:
        span.finish("success")


@contextmanager
def record_tool_call(
    exe: Any,
    *,
    agent_id: str | None,
    tool_name: str,
    input_text: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Span]:
    """Context manager for a tool/function invocation.

    Enforcement is called with ``tools=[tool_name]`` so tool-allowlist
    policies can block the call before it runs.
    """
    decision = _maybe_enforce(
        exe,
        agent_id=agent_id,
        input_text=input_text if input_text is not None else f"tool:{tool_name}",
        metadata={**(metadata or {}), "toolName": tool_name},
        tools=[tool_name],
    )

    tree = get_span_tree(exe)
    span = tree.start(kind=SPAN_KIND_TOOL, name=tool_name, agent_id=agent_id)
    if input_text is not None:
        span.set_input(input_text)
    span.add_metadata({"toolName": tool_name})
    if decision is not None:
        span.add_metadata({"enforcement": _summarise_decision(decision)})

    try:
        yield span
    except Exception as exc:
        span.finish("error", str(exc), exc.__class__.__name__)
        raise
    else:
        span.finish("success")


def record_agent_action(
    exe: Any,
    *,
    agent_id: str | None,
    action: str,
    metadata: dict[str, Any] | None = None,
) -> Span:
    """Open an agent-level span representing a decision/step.

    Returned span must be ``.finish()``-ed by the caller — this is NOT
    a context manager because many frameworks emit action-started and
    action-finished callbacks separately.
    """
    tree = get_span_tree(exe)
    return tree.start(
        kind=SPAN_KIND_AGENT,
        name=action,
        agent_id=agent_id,
        metadata=metadata,
    )


@contextmanager
def record_retrieval(
    exe: Any,
    *,
    agent_id: str | None,
    query: str,
    retriever_name: str = "retriever",
    metadata: dict[str, Any] | None = None,
) -> Iterator[Span]:
    """Context manager for a retrieval/RAG step."""
    tree = get_span_tree(exe)
    span = tree.start(
        kind=SPAN_KIND_RETRIEVER,
        name=retriever_name,
        agent_id=agent_id,
        metadata=metadata,
    )
    span.set_input(query)
    try:
        yield span
    except Exception as exc:
        span.finish("error", str(exc), exc.__class__.__name__)
        raise
    else:
        span.finish("success")
