"""
LangChain auto-instrumentation — :class:`ExeclaveCallbackHandler`.

Usage::

    from execlave import Execlave
    from execlave.integrations.langchain import ExeclaveCallbackHandler

    exe = Execlave(api_key="...")
    handler = ExeclaveCallbackHandler(exe, agent_id="my-bot")

    chain.invoke({"q": q}, config={"callbacks": [handler]})

The handler mirrors every LangChain callback we care about onto the
shared :mod:`execlave.instrumentation` span tree.

Policy enforcement hooks:
* ``on_chain_start`` — enforce on the user input (re-raises
  ``PolicyBlockedError`` which halts chain execution because LangChain
  propagates callback exceptions to the caller).
* ``on_tool_start`` — enforce on the tool call (allowlist policies).

Fail-open: unexpected instrumentation errors are swallowed. Enforcement
errors (block/deny/approval-timeout/agent-paused) are re-raised.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

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

logger = logging.getLogger("Execlave.langchain")

_ENFORCEMENT_ERRORS: tuple[type[BaseException], ...] = (
    PolicyBlockedError,
    PolicyDeniedError,
    ApprovalTimeoutError,
    EnforcementUnavailableError,
    AgentPausedError,
)


def _import_base_callback_handler() -> type:
    """Soft-import the LangChain base handler. We support both package
    layouts: the new ``langchain-core`` (preferred) and the legacy
    ``langchain`` package."""
    try:
        from langchain_core.callbacks.base import BaseCallbackHandler  # type: ignore
        return BaseCallbackHandler
    except ImportError:
        try:
            from langchain.callbacks.base import BaseCallbackHandler  # type: ignore
            return BaseCallbackHandler
        except ImportError as exc:
            raise ImportError(
                "LangChain is not installed. Install it with: "
                "pip install 'execlave-sdk[langchain]' "
                "(or: pip install langchain-core>=0.3,<0.4)"
            ) from exc


_BaseCallbackHandler = _import_base_callback_handler()


class ExeclavePolicyError(PolicyBlockedError):
    """Alias for ``PolicyBlockedError`` surfaced through the LangChain
    callback pipeline. Existing catches for ``PolicyBlockedError`` keep
    working because this subclasses it."""


class ExeclaveCallbackHandler(_BaseCallbackHandler):  # type: ignore[misc,valid-type]
    """LangChain callback handler that streams into Execlave.

    :param exe: An :class:`Execlave` client. Required.
    :param agent_id: The agent id registered with Execlave. Used for
        enforcement and audit attribution.
    :param enforce: If False, skip pre-execution enforcement (tracing
        only). Default True.
    :param session_id, user_id: Optional attribution hints forwarded to
        every span.
    """

    # LangChain resolves handler capability by attribute inspection.
    raise_error: bool = True
    run_inline: bool = True

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
            raise ValueError("ExeclaveCallbackHandler requires an Execlave client")
        if not agent_id:
            raise ValueError("ExeclaveCallbackHandler requires an agent_id")
        self._exe = exe
        self._agent_id = agent_id
        self._enforce = enforce
        self._session_id = session_id
        self._user_id = user_id
        self._tree = get_span_tree(exe)
        # run_id (UUID) -> Span. LangChain passes the same run_id for
        # paired start/end callbacks.
        self._spans: dict[UUID, Span] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _open(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        *,
        kind: str,
        name: str,
        input_: Any = None,
        metadata: dict | None = None,
    ) -> Span:
        parent_span = self._spans.get(parent_run_id) if parent_run_id else None
        span = self._tree.start(
            kind=kind,
            name=name,
            parent=parent_span,
            agent_id=self._agent_id,
            session_id=self._session_id,
            user_id=self._user_id,
            metadata=metadata,
        )
        if input_ is not None:
            span.set_input(input_)
        self._spans[run_id] = span
        return span

    def _close(
        self,
        run_id: UUID,
        *,
        output: Any = None,
        status: str = "success",
        error_message: str | None = None,
        error_type: str | None = None,
    ) -> None:
        span = self._spans.pop(run_id, None)
        if span is None:
            return
        if output is not None:
            try:
                span.set_output(output)
            except Exception:  # pragma: no cover
                logger.debug("failed to set span output", exc_info=True)
        span.finish(status=status, error_message=error_message, error_type=error_type)

    def _enforce_call(self, input_text: str, tools: list[str] | None = None) -> dict | None:
        if not self._enforce:
            return None
        try:
            return self._exe.enforce_policy(
                self._agent_id,
                input_text,
                tools=tools,
            )
        except _ENFORCEMENT_ERRORS:
            raise
        except Exception:
            logger.warning("enforce_policy raised (non-fatal)", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Chain callbacks
    # ------------------------------------------------------------------
    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or "chain"
        # LangChain chain "input" is a dict; pick the most likely user-text field.
        input_text = _extract_input_text(inputs)
        if input_text and parent_run_id is None:
            # Only enforce on top-level chain entry to avoid N x enforcement.
            self._enforce_call(input_text)
        self._open(run_id, parent_run_id, kind=SPAN_KIND_CHAIN, name=name, input_=inputs)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._close(run_id, output=outputs, status="success")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._close(
            run_id,
            status="error",
            error_message=str(error),
            error_type=error.__class__.__name__,
        )

    # ------------------------------------------------------------------
    # LLM callbacks
    # ------------------------------------------------------------------
    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or "llm"
        invocation_params = kwargs.get("invocation_params") or {}
        model = invocation_params.get("model") or invocation_params.get("model_name")
        span = self._open(
            run_id,
            parent_run_id,
            kind=SPAN_KIND_LLM,
            name=name,
            input_=prompts,
        )
        if model:
            span.set_model(model)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # Delegate to on_llm_start semantics for consistency.
        name = (serialized or {}).get("name") or "chat_model"
        invocation_params = kwargs.get("invocation_params") or {}
        model = invocation_params.get("model") or invocation_params.get("model_name")
        span = self._open(
            run_id,
            parent_run_id,
            kind=SPAN_KIND_LLM,
            name=name,
            input_=_stringify_messages(messages),
        )
        if model:
            span.set_model(model)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        span = self._spans.get(run_id)
        if span is not None:
            usage = _extract_token_usage(response)
            if usage is not None:
                span.set_tokens(usage[0], usage[1])
            output = _extract_llm_output(response)
            if output is not None:
                span.set_output(output)
        self._close(run_id, status="success")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._close(
            run_id,
            status="error",
            error_message=str(error),
            error_type=error.__class__.__name__,
        )

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------
    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = (serialized or {}).get("name") or "tool"
        self._enforce_call(input_str or f"tool:{tool_name}", tools=[tool_name])
        self._open(
            run_id,
            parent_run_id,
            kind=SPAN_KIND_TOOL,
            name=tool_name,
            input_=input_str,
            metadata={"toolName": tool_name},
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, output=output, status="success")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._close(
            run_id,
            status="error",
            error_message=str(error),
            error_type=error.__class__.__name__,
        )

    # ------------------------------------------------------------------
    # Agent callbacks
    # ------------------------------------------------------------------
    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        tool = getattr(action, "tool", None) or "agent_action"
        tool_input = getattr(action, "tool_input", None)
        self._open(
            run_id,
            parent_run_id,
            kind=SPAN_KIND_AGENT,
            name=str(tool),
            input_=tool_input,
            metadata={"agentAction": str(tool)},
        )

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        output = getattr(finish, "return_values", None) or getattr(finish, "log", None)
        self._close(run_id, output=output, status="success")

    # ------------------------------------------------------------------
    # Retriever callbacks
    # ------------------------------------------------------------------
    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or "retriever"
        self._open(
            run_id,
            parent_run_id,
            kind=SPAN_KIND_RETRIEVER,
            name=name,
            input_=query,
        )

    def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        # Record only document count + ids — never page_content (PII risk).
        summary = _summarise_docs(documents)
        span = self._spans.get(run_id)
        if span is not None and summary:
            span.add_metadata({"retrievedDocs": summary})
        self._close(run_id, status="success")

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._close(
            run_id,
            status="error",
            error_message=str(error),
            error_type=error.__class__.__name__,
        )


# ----------------------------------------------------------------------
# Internal helpers (pure functions, easy to unit-test)
# ----------------------------------------------------------------------
def _extract_input_text(inputs: dict[str, Any]) -> str | None:
    """Pick a reasonable 'user prompt' string out of a LangChain chain
    input dict for enforcement purposes."""
    if not isinstance(inputs, dict):
        try:
            return str(inputs)
        except Exception:
            return None
    for key in ("input", "question", "query", "q", "prompt", "text"):
        val = inputs.get(key)
        if isinstance(val, str) and val:
            return val
    # Fallback: concatenate string values.
    parts = [v for v in inputs.values() if isinstance(v, str)]
    return " ".join(parts) if parts else None


def _stringify_messages(messages: list[list[Any]]) -> list[dict[str, str]]:
    """Convert LangChain message objects to a plain list for span input."""
    out: list[dict[str, str]] = []
    for batch in messages or []:
        for msg in batch or []:
            role = getattr(msg, "type", None) or getattr(msg, "role", None) or "message"
            content = getattr(msg, "content", None)
            if content is None:
                content = str(msg)
            out.append({"role": str(role), "content": str(content)})
    return out


def _extract_token_usage(response: Any) -> tuple[int, int] | None:
    """LangChain LLMResult exposes token usage under llm_output on
    OpenAI-family LLMs. Return (prompt, completion) or None."""
    try:
        output = getattr(response, "llm_output", None) or {}
        usage = output.get("token_usage") or output.get("usage") or {}
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion = usage.get("completion_tokens") or usage.get("output_tokens")
        if isinstance(prompt, int) and isinstance(completion, int):
            return prompt, completion
    except Exception:  # pragma: no cover
        pass
    return None


def _extract_llm_output(response: Any) -> Any:
    """Extract a concise text representation of an LLM response."""
    try:
        generations = getattr(response, "generations", None)
        if generations:
            out: list[str] = []
            for batch in generations:
                for gen in batch:
                    text = getattr(gen, "text", None)
                    if text is None:
                        msg = getattr(gen, "message", None)
                        text = getattr(msg, "content", None) if msg is not None else None
                    if text is not None:
                        out.append(str(text))
            return out if len(out) > 1 else (out[0] if out else None)
    except Exception:  # pragma: no cover
        pass
    return None


def _summarise_docs(documents: Any) -> list[dict[str, Any]] | None:
    """Return only id/metadata summaries of retrieved docs — never
    ``page_content`` (potential PII)."""
    if not documents:
        return None
    summary: list[dict[str, Any]] = []
    try:
        for doc in documents:
            meta = getattr(doc, "metadata", None) or {}
            summary.append(
                {
                    "id": meta.get("id") or meta.get("source"),
                    "length": len(getattr(doc, "page_content", "") or ""),
                }
            )
    except Exception:  # pragma: no cover
        return None
    return summary or None
