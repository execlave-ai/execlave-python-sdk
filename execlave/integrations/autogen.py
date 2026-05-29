"""
Microsoft AutoGen (autogen-agentchat / pyautogen) auto-instrumentation.

Wrap a ``ConversableAgent`` (or any agent exposing
``register_reply``/``register_hook``) so each reply round is recorded as
an Execlave span and tool calls in the reply payload are pre-enforced.

Usage::

    from autogen import ConversableAgent
    from execlave import Execlave
    from execlave.integrations.autogen import instrument_autogen_agent

    exe = Execlave(api_key="...")
    bot = ConversableAgent("assistant", llm_config={...})
    instrument_autogen_agent(bot, exe, agent_id="my-bot")

Policy enforcement runs:
* before each ``generate_reply`` on the incoming user message,
* before each tool call extracted from the reply (function_call /
  tool_calls).

Idempotent — marker attribute prevents double-wrapping.
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
    get_span_tree,
)

logger = logging.getLogger("Execlave.autogen")

_MARKER = "_execlave_instrumented"

_ENFORCEMENT_ERRORS: tuple[type[BaseException], ...] = (
    PolicyBlockedError,
    PolicyDeniedError,
    ApprovalTimeoutError,
    EnforcementUnavailableError,
    AgentPausedError,
)


def instrument_autogen_agent(
    agent: Any,
    exe: Execlave,
    *,
    agent_id: str,
    enforce: bool = True,
    session_id: str | None = None,
    user_id: str | None = None,
) -> Any:
    """Attach Execlave instrumentation to an AutoGen agent.

    Idempotent. Returns the same agent for fluent chaining.
    """
    if agent is None:
        raise ValueError("instrument_autogen_agent: agent must not be None")
    if exe is None:
        raise ValueError("instrument_autogen_agent: exe must not be None")
    if not agent_id:
        raise ValueError("instrument_autogen_agent: agent_id is required")

    if getattr(agent, _MARKER, False):
        logger.debug("autogen agent already instrumented — skipping")
        return agent

    generate_reply = getattr(agent, "generate_reply", None)
    if generate_reply is None or not callable(generate_reply):
        raise TypeError(
            "instrument_autogen_agent: agent has no callable 'generate_reply' method"
        )

    tree = get_span_tree(exe)

    def _enforce_message(messages: Any) -> None:
        if not enforce:
            return
        text = _last_user_text(messages) or "autogen.generate_reply"
        try:
            exe.enforce_policy(agent_id, text)
        except _ENFORCEMENT_ERRORS:
            raise
        except Exception:
            logger.warning("autogen enforce_policy(msg) failed", exc_info=True)

    def _enforce_tool_calls(reply: Any) -> None:
        if not enforce or reply is None:
            return
        for name, args in _extract_tool_calls(reply):
            try:
                exe.enforce_policy(
                    agent_id,
                    _safe_str(args) or f"tool:{name}",
                    tools=[name],
                )
            except _ENFORCEMENT_ERRORS:
                raise
            except Exception:
                logger.warning("autogen enforce_policy(tool=%s) failed", name, exc_info=True)

    def wrapped_generate_reply(*args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages") or (args[0] if args else None)
        _enforce_message(messages)
        span = tree.start(
            kind=SPAN_KIND_AGENT,
            name=getattr(agent, "name", "autogen-agent"),
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            metadata={"framework": "autogen"},
        )
        try:
            span.set_input(messages)
        except Exception:  # pragma: no cover
            logger.debug("set_input failed", exc_info=True)
        try:
            reply = generate_reply(*args, **kwargs)
        except Exception as exc:
            span.finish(status="error", error_message=_safe_str(exc), error_type=type(exc).__name__)
            raise
        _enforce_tool_calls(reply)
        # Tool-call spans (logged as children for visibility).
        for name, payload in _extract_tool_calls(reply):
            child = tree.start(
                kind=SPAN_KIND_TOOL,
                name=str(name),
                parent=span,
                agent_id=agent_id,
                session_id=session_id,
                user_id=user_id,
                metadata={"autogenToolCall": True},
            )
            if payload is not None:
                try:
                    child.set_input(payload)
                except Exception:  # pragma: no cover
                    pass
            child.finish(status="success")
        try:
            span.set_output(reply)
        except Exception:  # pragma: no cover
            pass
        span.finish(status="success")
        return reply

    try:
        setattr(agent, "generate_reply", wrapped_generate_reply)
        setattr(agent, _MARKER, True)
    except Exception as exc:  # pragma: no cover
        logger.warning("could not patch autogen agent: %s", exc)
        return agent

    return agent


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _last_user_text(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return _safe_str(messages)
    for msg in reversed(messages):
        role = _get(msg, "role")
        if role in ("user", "human"):
            content = _get(msg, "content")
            return _safe_str(content)
    # Fallback: last message regardless of role.
    if messages:
        return _safe_str(_get(messages[-1], "content"))
    return None


def _extract_tool_calls(reply: Any) -> list[tuple[str, Any]]:
    """Pull (name, arguments) tuples from an AutoGen reply payload.

    Handles both the legacy ``function_call`` shape and the modern
    ``tool_calls`` list shape that mirrors OpenAI's API.
    """
    out: list[tuple[str, Any]] = []
    if reply is None or isinstance(reply, (str, bytes)):
        return out

    fc = _get(reply, "function_call")
    if isinstance(fc, dict) and fc.get("name"):
        out.append((str(fc["name"]), fc.get("arguments")))

    tcs = _get(reply, "tool_calls")
    if isinstance(tcs, list):
        for tc in tcs:
            fn = _get(tc, "function") or {}
            name = _get(fn, "name") or _get(tc, "name")
            args = _get(fn, "arguments") or _get(tc, "arguments")
            if name:
                out.append((str(name), args))
    return out


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    val = getattr(obj, name, None)
    if val is not None:
        return val
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _safe_str(value: Any, limit: int = 4000) -> str | None:
    if value is None:
        return None
    try:
        s = value if isinstance(value, str) else str(value)
    except Exception:  # pragma: no cover
        return None
    return s[:limit]


# Re-export for type-checkers
__all__ = ["instrument_autogen_agent"]
_Callable = Callable  # keep import alive
