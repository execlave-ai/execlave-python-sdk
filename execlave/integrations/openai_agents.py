"""
OpenAI Agents SDK auto-instrumentation.

Plug into the Agents SDK tracing pipeline via ``set_trace_processors``::

    from agents import set_trace_processors
    from execlave import Execlave
    from execlave.integrations.openai_agents import ExeclaveTracingProcessor

    exe = Execlave(api_key="...")
    set_trace_processors([ExeclaveTracingProcessor(exe, agent_id="my-bot")])

The processor listens for span lifecycle events from the Agents SDK and
maps each ``agent_span``, ``generation_span``, ``function_span``,
``handoff_span``, and ``guardrail_span`` into an Execlave span via the
shared :mod:`execlave.instrumentation` layer.

Policy enforcement is invoked on ``function_span`` (tool call) starts
with the tool name — this lets tool-allowlist policies block a tool
before the SDK executes it.
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
    SPAN_KIND_GUARDRAIL,
    SPAN_KIND_HANDOFF,
    SPAN_KIND_LLM,
    SPAN_KIND_TOOL,
    Span,
    get_span_tree,
)

logger = logging.getLogger("Execlave.openai_agents")

_ENFORCEMENT_ERRORS: tuple[type[BaseException], ...] = (
    PolicyBlockedError,
    PolicyDeniedError,
    ApprovalTimeoutError,
    EnforcementUnavailableError,
    AgentPausedError,
)


def _import_base_processor() -> type:
    """Soft-import ``TracingProcessor`` from the openai-agents SDK."""
    try:
        from agents.tracing.processor_interface import TracingProcessor  # type: ignore
        return TracingProcessor
    except ImportError:
        try:
            from agents.tracing import TracingProcessor  # type: ignore
            return TracingProcessor
        except ImportError as exc:
            raise ImportError(
                "openai-agents is not installed. Install it with: "
                "pip install 'execlave-sdk[openai-agents]' "
                "(or: pip install openai-agents)"
            ) from exc


_BaseTracingProcessor = _import_base_processor()


# Mapping from Agents-SDK span type names to Execlave span kinds.
# Names come from the SDK's public span-data classes; we key on
# ``type(span_data).__name__`` to stay version-tolerant.
_KIND_MAP: dict[str, str] = {
    "AgentSpanData": SPAN_KIND_AGENT,
    "GenerationSpanData": SPAN_KIND_LLM,
    "ResponseSpanData": SPAN_KIND_LLM,
    "FunctionSpanData": SPAN_KIND_TOOL,
    "HandoffSpanData": SPAN_KIND_HANDOFF,
    "GuardrailSpanData": SPAN_KIND_GUARDRAIL,
    "CustomSpanData": SPAN_KIND_AGENT,
}


class ExeclaveTracingProcessor(_BaseTracingProcessor):  # type: ignore[misc,valid-type]
    """Bridges OpenAI Agents SDK spans into the Execlave span tree."""

    def __init__(
        self,
        exe: Execlave,
        *,
        agent_id: str,
        enforce: bool = True,
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        if exe is None:
            raise ValueError("ExeclaveTracingProcessor requires an Execlave client")
        if not agent_id:
            raise ValueError("ExeclaveTracingProcessor requires an agent_id")
        self._exe = exe
        self._agent_id = agent_id
        self._enforce = enforce
        self._session_id = session_id
        self._user_id = user_id
        self._tree = get_span_tree(exe)
        # openai-agents span-id (str) -> our Span
        self._spans: dict[str, Span] = {}

    # ------------------------------------------------------------------
    # Enforcement
    # ------------------------------------------------------------------
    def _enforce_tool(self, tool_name: str, args: Any) -> None:
        if not self._enforce:
            return
        try:
            self._exe.enforce_policy(
                self._agent_id,
                _safe_str(args) or f"tool:{tool_name}",
                tools=[tool_name],
            )
        except _ENFORCEMENT_ERRORS:
            raise
        except Exception:
            logger.warning("enforce_policy failed (non-fatal)", exc_info=True)

    # ------------------------------------------------------------------
    # TracingProcessor interface
    # ------------------------------------------------------------------
    def on_trace_start(self, trace: Any) -> None:  # noqa: D401 — interface
        return None

    def on_trace_end(self, trace: Any) -> None:  # noqa: D401 — interface
        return None

    def on_span_start(self, span: Any) -> None:
        try:
            self._start_span(span)
        except _ENFORCEMENT_ERRORS:
            raise
        except Exception:
            logger.warning("on_span_start failed", exc_info=True)

    def on_span_end(self, span: Any) -> None:
        try:
            self._end_span(span)
        except Exception:
            logger.warning("on_span_end failed", exc_info=True)

    def shutdown(self) -> None:  # noqa: D401 — interface
        try:
            self._exe.flush()
        except Exception:  # pragma: no cover
            logger.debug("flush on shutdown failed", exc_info=True)

    def force_flush(self) -> None:  # noqa: D401 — interface
        try:
            self._exe.flush()
        except Exception:  # pragma: no cover
            logger.debug("force_flush failed", exc_info=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _start_span(self, span: Any) -> None:
        span_id = _get_attr(span, "span_id") or _get_attr(span, "id")
        if span_id is None:
            return
        parent_id = _get_attr(span, "parent_id")
        span_data = _get_attr(span, "span_data")
        kind = _KIND_MAP.get(type(span_data).__name__ if span_data else "", SPAN_KIND_AGENT)
        name = _span_name(span_data) or kind

        # Tool-call enforcement happens before we open the span so a
        # block aborts the SDK run without producing an orphaned span.
        if kind == SPAN_KIND_TOOL:
            tool_name = name
            args = _get_attr(span_data, "input") or _get_attr(span_data, "arguments")
            self._enforce_tool(tool_name, args)

        parent_span = self._spans.get(parent_id) if parent_id else None
        our_span = self._tree.start(
            kind=kind,
            name=name,
            parent=parent_span,
            agent_id=self._agent_id,
            session_id=self._session_id,
            user_id=self._user_id,
            metadata={"openaiAgentsSpanId": span_id},
        )
        input_ = _get_attr(span_data, "input")
        if input_ is not None:
            our_span.set_input(input_)
        model = _get_attr(span_data, "model")
        if model:
            our_span.set_model(model)
        self._spans[span_id] = our_span

    def _end_span(self, span: Any) -> None:
        span_id = _get_attr(span, "span_id") or _get_attr(span, "id")
        if span_id is None:
            return
        our_span = self._spans.pop(span_id, None)
        if our_span is None:
            return
        span_data = _get_attr(span, "span_data")
        output = _get_attr(span_data, "output")
        if output is not None:
            try:
                our_span.set_output(output)
            except Exception:  # pragma: no cover
                logger.debug("set_output failed", exc_info=True)
        usage = _get_attr(span_data, "usage")
        if isinstance(usage, dict):
            p = usage.get("prompt_tokens") or usage.get("input_tokens")
            c = usage.get("completion_tokens") or usage.get("output_tokens")
            if isinstance(p, int) and isinstance(c, int):
                our_span.set_tokens(p, c)
        error = _get_attr(span, "error")
        if error:
            our_span.finish(
                status="error",
                error_message=_safe_str(error),
                error_type=type(error).__name__,
            )
        else:
            our_span.finish(status="success")


# ----------------------------------------------------------------------
# Defensive getters — span-data shape varies across SDK versions.
# ----------------------------------------------------------------------
def _get_attr(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    # Pydantic model / dataclass / plain attribute
    val = getattr(obj, name, None)
    if val is not None:
        return val
    # Dict-like
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def _span_name(span_data: Any) -> str | None:
    if span_data is None:
        return None
    for attr in ("name", "tool_name", "function_name", "agent_name", "model"):
        val = _get_attr(span_data, attr)
        if isinstance(val, str) and val:
            return val
    return None


def _safe_str(value: Any, limit: int = 4000) -> str | None:
    if value is None:
        return None
    try:
        s = value if isinstance(value, str) else str(value)
    except Exception:  # pragma: no cover
        return None
    return s[:limit]
