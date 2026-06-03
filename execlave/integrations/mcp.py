"""
Model Context Protocol (MCP) client auto-instrumentation.

Wraps an ``mcp.ClientSession`` so every ``call_tool`` invocation is
enforced through Execlave (tool allowlist) and recorded as a span.

Usage::

    from mcp import ClientSession
    from execlave import Execlave
    from execlave.integrations.mcp import instrument_mcp_session

    exe = Execlave(api_key="...")
    session = ClientSession(...)
    instrument_mcp_session(session, exe, agent_id="my-bot")

    # Now every ``await session.call_tool(...)`` is governed.
    result = await session.call_tool("search", {"q": "x"})

The wrap is idempotent — a marker attribute prevents double-wrapping.
"""

from __future__ import annotations

import logging
from typing import Any

from ..client import Execlave
from ..errors import (
    AgentPausedError,
    ApprovalTimeoutError,
    EnforcementUnavailableError,
    PolicyBlockedError,
    PolicyDeniedError,
)
from ..instrumentation.spans import SPAN_KIND_TOOL, get_span_tree

logger = logging.getLogger("Execlave.mcp")

_MARKER = "_execlave_instrumented"

_ENFORCEMENT_ERRORS: tuple[type[BaseException], ...] = (
    PolicyBlockedError,
    PolicyDeniedError,
    ApprovalTimeoutError,
    EnforcementUnavailableError,
    AgentPausedError,
)


def instrument_mcp_session(
    session: Any,
    exe: Execlave,
    *,
    agent_id: str,
    enforce: bool = True,
    session_id: str | None = None,
    user_id: str | None = None,
) -> Any:
    """Wrap an MCP ``ClientSession`` so ``call_tool`` is governed by Execlave.

    Idempotent. Returns the same session for fluent chaining.
    """
    if session is None:
        raise ValueError("instrument_mcp_session: session must not be None")
    if exe is None:
        raise ValueError("instrument_mcp_session: exe must not be None")
    if not agent_id:
        raise ValueError("instrument_mcp_session: agent_id is required")

    if getattr(session, _MARKER, False):
        logger.debug("MCP session already instrumented — skipping")
        return session

    original_call_tool = getattr(session, "call_tool", None)
    if original_call_tool is None or not callable(original_call_tool):
        raise TypeError(
            "instrument_mcp_session: session has no callable 'call_tool' attribute"
        )

    tree = get_span_tree(exe)

    async def wrapped_call_tool(name: str, arguments: dict | None = None, **kwargs: Any) -> Any:
        if enforce:
            try:
                exe.enforce_policy(
                    agent_id,
                    _safe_str(arguments) or f"tool:{name}",
                    tools=[str(name)],
                )
            except _ENFORCEMENT_ERRORS:
                raise
            except Exception:
                logger.warning("MCP enforce_policy failed (non-fatal)", exc_info=True)

        span = tree.start(
            kind=SPAN_KIND_TOOL,
            name=str(name),
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            metadata={"mcpTool": True},
        )
        if arguments is not None:
            try:
                span.set_input(arguments)
            except Exception:  # pragma: no cover
                logger.debug("span.set_input failed", exc_info=True)
        try:
            result = await original_call_tool(name, arguments, **kwargs)
        except Exception as exc:
            span.finish(
                status="error",
                error_message=_safe_str(exc),
                error_type=type(exc).__name__,
            )
            raise
        try:
            output = _extract_result(result)
            if output is not None:
                span.set_output(output)
        except Exception:  # pragma: no cover
            logger.debug("output extraction failed", exc_info=True)
        # MCP results carry an isError flag in CallToolResult.
        is_err = getattr(result, "isError", False)
        span.finish(status="error" if is_err else "success")
        return result

    try:
        setattr(session, "call_tool", wrapped_call_tool)
        setattr(session, _MARKER, True)
    except Exception as exc:  # pragma: no cover — frozen pydantic model
        logger.warning("could not patch MCP session: %s", exc)
        return session

    return session


# Lower-cased alias for naming consistency with other Python instrumenters.
instrument_mcp_client = instrument_mcp_session


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _extract_result(result: Any) -> Any:
    """Pull a meaningful output out of an MCP CallToolResult."""
    if result is None:
        return None
    content = getattr(result, "content", None)
    if content is None:
        return _safe_str(result)
    out: list[str] = []
    try:
        for item in content:
            text = getattr(item, "text", None)
            if text is not None:
                out.append(str(text))
            else:
                out.append(_safe_str(item) or "")
    except TypeError:
        return _safe_str(content)
    return "\n".join(s for s in out if s) or None


def _safe_str(value: Any, limit: int = 4000) -> str | None:
    if value is None:
        return None
    try:
        s = value if isinstance(value, str) else str(value)
    except Exception:  # pragma: no cover
        return None
    return s[:limit]
