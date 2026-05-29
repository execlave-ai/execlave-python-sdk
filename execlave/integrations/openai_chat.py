"""
OpenAI Chat Completions auto-instrumentation.

Wrap an ``openai.OpenAI`` (or ``AsyncOpenAI``) client so every
``chat.completions.create(...)`` call is governed by Execlave: the user
prompt is enforced before the request leaves, and the call is recorded
as an ``llm`` span with model + token usage + cost-friendly metadata.

Usage::

    from openai import OpenAI
    from execlave import Execlave
    from execlave.integrations.openai_chat import instrument_openai

    exe = Execlave(api_key="...")
    openai = instrument_openai(OpenAI(), exe, agent_id="my-bot")
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )

Idempotent — a marker attribute guards against double-wrapping.

The wrap is *additive*: any other instrumentation already applied to the
``create`` callable is preserved by chaining.
"""

from __future__ import annotations

import asyncio
import inspect
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
from ..instrumentation.spans import SPAN_KIND_LLM, get_span_tree

logger = logging.getLogger("Execlave.openai_chat")

_MARKER = "_execlave_instrumented"

_ENFORCEMENT_ERRORS: tuple[type[BaseException], ...] = (
    PolicyBlockedError,
    PolicyDeniedError,
    ApprovalTimeoutError,
    EnforcementUnavailableError,
    AgentPausedError,
)


def instrument_openai(
    client: Any,
    exe: Execlave,
    *,
    agent_id: str,
    enforce: bool = True,
    session_id: str | None = None,
    user_id: str | None = None,
) -> Any:
    """Wrap ``client.chat.completions.create`` (sync or async).

    Returns the same client for fluent chaining.
    """
    if client is None:
        raise ValueError("instrument_openai: client must not be None")
    if exe is None:
        raise ValueError("instrument_openai: exe must not be None")
    if not agent_id:
        raise ValueError("instrument_openai: agent_id is required")

    completions = _resolve(client, "chat.completions")
    if completions is None:
        raise TypeError("instrument_openai: client.chat.completions is missing")

    if getattr(completions, _MARKER, False):
        logger.debug("OpenAI client already instrumented — skipping")
        return client

    original_create = getattr(completions, "create", None)
    if original_create is None or not callable(original_create):
        raise TypeError("instrument_openai: chat.completions.create is missing")

    tree = get_span_tree(exe)
    is_async = inspect.iscoroutinefunction(original_create)

    def _extract_user_input(messages: Any) -> str | None:
        if not isinstance(messages, list):
            return _safe_str(messages)
        for msg in reversed(messages):
            role = _get(msg, "role")
            if role == "user":
                content = _get(msg, "content")
                if isinstance(content, list):
                    parts = [_get(p, "text") for p in content if _get(p, "text")]
                    if parts:
                        return _safe_str("\n".join(parts))
                return _safe_str(content)
        return None

    def _before(model: str | None, messages: Any) -> Any:
        if enforce:
            user_text = _extract_user_input(messages) or "chat.completions"
            try:
                exe.enforce_policy(agent_id, user_text)
            except _ENFORCEMENT_ERRORS:
                raise
            except Exception:
                logger.warning("enforce_policy failed (non-fatal)", exc_info=True)
        span = tree.start(
            kind=SPAN_KIND_LLM,
            name=str(model or "chat.completions"),
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            metadata={"provider": "openai", "endpoint": "chat.completions"},
        )
        if model:
            span.set_model(str(model))
        try:
            span.set_input(messages)
        except Exception:  # pragma: no cover
            logger.debug("set_input failed", exc_info=True)
        return span

    def _after(span: Any, response: Any) -> None:
        try:
            choices = _get(response, "choices") or []
            output = _get(choices[0], "message") if choices else None
            content = _get(output, "content") if output is not None else None
            if content is not None:
                span.set_output(content)
            usage = _get(response, "usage")
            p = _get(usage, "prompt_tokens")
            c = _get(usage, "completion_tokens")
            if isinstance(p, int) and isinstance(c, int):
                span.set_tokens(p, c)
            model = _get(response, "model")
            if isinstance(model, str) and model:
                span.set_model(model)
        except Exception:  # pragma: no cover
            logger.debug("after() failed", exc_info=True)
        span.finish(status="success")

    if is_async:

        async def wrapped_create_async(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            messages = kwargs.get("messages")
            span = _before(model, messages)
            try:
                resp = await original_create(*args, **kwargs)
            except Exception as exc:
                span.finish(status="error", error_message=_safe_str(exc), error_type=type(exc).__name__)
                raise
            _after(span, resp)
            return resp

        wrapped = wrapped_create_async
    else:

        def wrapped_create_sync(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            messages = kwargs.get("messages")
            span = _before(model, messages)
            try:
                resp = original_create(*args, **kwargs)
            except Exception as exc:
                span.finish(status="error", error_message=_safe_str(exc), error_type=type(exc).__name__)
                raise
            _after(span, resp)
            return resp

        wrapped = wrapped_create_sync

    try:
        setattr(completions, "create", wrapped)
        setattr(completions, _MARKER, True)
    except Exception as exc:  # pragma: no cover
        logger.warning("could not patch openai client: %s", exc)
        return client

    return client


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _resolve(root: Any, dotted: str) -> Any:
    obj: Any = root
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


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


# Re-export so the symbol exists even when no asyncio loop is set.
_ = asyncio  # keep import (used by type-checkers for awaitable inference)
