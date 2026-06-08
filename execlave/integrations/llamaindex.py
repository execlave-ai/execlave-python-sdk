"""
LlamaIndex auto-instrumentation — :class:`ExeclaveLlamaIndexHandler`.

Usage::

    from llama_index.core.instrumentation import get_dispatcher
    from execlave import Execlave
    from execlave.integrations.llamaindex import ExeclaveLlamaIndexHandler

    exe = Execlave(api_key="...")
    handler = ExeclaveLlamaIndexHandler(exe, agent_id="my-bot")
    get_dispatcher().add_event_handler(handler)
    get_dispatcher().add_span_handler(handler.span_handler())

The handler subscribes to LlamaIndex's dispatcher events. Each
``QueryStart``/``LLMChat*Start``/``RetrievalStart``/``AgentToolCallEvent``
opens an Execlave span; matching ``*End`` events close it.

Policy enforcement runs at:
* ``QueryStartEvent`` — on the query string (re-raises
  ``PolicyBlockedError`` so the LlamaIndex pipeline aborts).
* ``AgentToolCallEvent`` — on the tool name + tool input (allowlist).
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
from ..instrumentation.spans import (
    SPAN_KIND_AGENT,
    SPAN_KIND_CHAIN,
    SPAN_KIND_LLM,
    SPAN_KIND_RETRIEVER,
    SPAN_KIND_TOOL,
    Span,
    get_span_tree,
)

logger = logging.getLogger("Execlave.llamaindex")

_ENFORCEMENT_ERRORS: tuple[type[BaseException], ...] = (
    PolicyBlockedError,
    PolicyDeniedError,
    ApprovalTimeoutError,
    EnforcementUnavailableError,
    AgentPausedError,
)


def _import_base_event_handler() -> type:
    """Soft-import LlamaIndex BaseEventHandler."""
    try:
        from llama_index.core.instrumentation.event_handlers import (  # type: ignore
            BaseEventHandler,
        )
        return BaseEventHandler
    except ImportError as exc:
        raise ImportError(
            "llama-index-core is not installed. Install it with: "
            "pip install 'execlave-sdk[llamaindex]' "
            "(or: pip install 'llama-index-core>=0.11')"
        ) from exc


_BaseEventHandler = _import_base_event_handler()


# Event class name → (Execlave span kind, is_start)
_START_EVENTS: dict[str, str] = {
    "QueryStartEvent": SPAN_KIND_CHAIN,
    "LLMChatStartEvent": SPAN_KIND_LLM,
    "LLMCompletionStartEvent": SPAN_KIND_LLM,
    "LLMPredictStartEvent": SPAN_KIND_LLM,
    "RetrievalStartEvent": SPAN_KIND_RETRIEVER,
    "EmbeddingStartEvent": SPAN_KIND_RETRIEVER,
    "AgentToolCallEvent": SPAN_KIND_TOOL,
    "AgentRunStepStartEvent": SPAN_KIND_AGENT,
    "AgentChatWithStepStartEvent": SPAN_KIND_AGENT,
}

_END_EVENTS: dict[str, None] = {
    "QueryEndEvent": None,
    "LLMChatEndEvent": None,
    "LLMCompletionEndEvent": None,
    "LLMPredictEndEvent": None,
    "RetrievalEndEvent": None,
    "EmbeddingEndEvent": None,
    "AgentRunStepEndEvent": None,
    "AgentChatWithStepEndEvent": None,
}


class ExeclaveLlamaIndexHandler(_BaseEventHandler):  # type: ignore[misc,valid-type]
    """LlamaIndex dispatcher event handler that streams into Execlave."""

    @classmethod
    def class_name(cls) -> str:
        return "ExeclaveLlamaIndexHandler"

    def __init__(
        self,
        exe: Execlave,
        *,
        agent_id: str,
        enforce: bool = True,
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        super().__init__()
        if exe is None:
            raise ValueError("ExeclaveLlamaIndexHandler requires an Execlave client")
        if not agent_id:
            raise ValueError("ExeclaveLlamaIndexHandler requires an agent_id")
        self._exe = exe
        self._agent_id = agent_id
        self._enforce = enforce
        self._session_id = session_id
        self._user_id = user_id
        self._tree = get_span_tree(exe)
        # span_id (LlamaIndex's str) → our Span
        self._spans: dict[str, Span] = {}

    # ------------------------------------------------------------------
    # BaseEventHandler interface
    # ------------------------------------------------------------------
    def handle(self, event: Any) -> Any:
        try:
            return self._handle(event)
        except _ENFORCEMENT_ERRORS:
            raise
        except Exception:
            logger.warning("llamaindex handle() failed", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _handle(self, event: Any) -> None:
        cls_name = type(event).__name__
        span_id = _get(event, "span_id") or _get(event, "id_")
        if span_id is None:
            return

        if cls_name in _START_EVENTS:
            kind = _START_EVENTS[cls_name]
            self._open(event, cls_name, kind, str(span_id))
        elif cls_name in _END_EVENTS:
            self._close(event, str(span_id), status="success")
        elif cls_name in ("AgentToolCallEvent",):
            # Already opened above; tool calls without an end event get
            # auto-closed when we see the next end-event for the same span.
            pass

    def _open(self, event: Any, cls_name: str, kind: str, span_id: str) -> None:
        # Tool enforcement before span open so block halts pipeline.
        if cls_name == "AgentToolCallEvent" and self._enforce:
            tool_name = _get(event, "tool_name") or _get(event, "name") or "tool"
            tool_input = _get(event, "arguments") or _get(event, "tool_kwargs")
            try:
                self._exe.enforce_policy(
                    self._agent_id,
                    _safe_str(tool_input) or f"tool:{tool_name}",
                    tools=[str(tool_name)],
                )
            except _ENFORCEMENT_ERRORS:
                raise
            except Exception:
                logger.warning("enforce_policy(tool) failed", exc_info=True)
        elif cls_name == "QueryStartEvent" and self._enforce:
            query = _get(event, "query") or _get(event, "str_or_query_bundle")
            try:
                self._exe.enforce_policy(self._agent_id, _safe_str(query) or "query")
            except _ENFORCEMENT_ERRORS:
                raise
            except Exception:
                logger.warning("enforce_policy(query) failed", exc_info=True)

        name = (
            _get(event, "tool_name")
            or _get(event, "model_dict", {}).get("model")
            if isinstance(_get(event, "model_dict"), dict)
            else None
        ) or kind

        span = self._tree.start(
            kind=kind,
            name=str(name),
            agent_id=self._agent_id,
            session_id=self._session_id,
            user_id=self._user_id,
            metadata={"llamaindexEventClass": cls_name, "llamaindexSpanId": span_id},
        )
        input_ = (
            _get(event, "query")
            or _get(event, "messages")
            or _get(event, "prompt")
            or _get(event, "arguments")
            or _get(event, "tool_kwargs")
        )
        if input_ is not None:
            span.set_input(input_)
        self._spans[span_id] = span

    def _close(self, event: Any, span_id: str, status: str) -> None:
        span = self._spans.pop(span_id, None)
        if span is None:
            return
        output = (
            _get(event, "response")
            or _get(event, "result")
            or _get(event, "nodes")
            or _get(event, "embedding")
        )
        if output is not None:
            try:
                span.set_output(output)
            except Exception:  # pragma: no cover
                logger.debug("set_output failed", exc_info=True)
        # Token usage on LLMChatEndEvent: response.raw.usage
        resp = _get(event, "response")
        usage = _get(resp, "raw", {}).get("usage") if isinstance(_get(resp, "raw"), dict) else None
        if isinstance(usage, dict):
            p = usage.get("prompt_tokens") or usage.get("input_tokens")
            c = usage.get("completion_tokens") or usage.get("output_tokens")
            if isinstance(p, int) and isinstance(c, int):
                span.set_tokens(p, c)
        span.finish(status=status)

    # ------------------------------------------------------------------
    # Span-handler bridge (optional — for tree shape parity)
    # ------------------------------------------------------------------
    def span_handler(self) -> "_NoopSpanHandler":
        """Return a no-op span handler so users can register both with
        ``add_event_handler`` and ``add_span_handler`` in one call. The
        Execlave span tree is built from events, not LlamaIndex span hooks,
        so this handler intentionally does nothing."""
        return _NoopSpanHandler()


class _NoopSpanHandler:
    """Stub matching LlamaIndex's BaseSpanHandler shape for registration."""

    class_name = "ExeclaveNoopSpanHandler"

    def new_span(self, *args: Any, **kwargs: Any) -> None:
        return None

    def prepare_to_exit_span(self, *args: Any, **kwargs: Any) -> None:
        return None

    def prepare_to_drop_span(self, *args: Any, **kwargs: Any) -> None:
        return None


# ----------------------------------------------------------------------
# Defensive accessors
# ----------------------------------------------------------------------
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
