"""
Nested span helper.

A ``Span`` wraps a ``Trace`` and carries parent/child links via metadata
(``parentSpanId``, ``spanId``, ``spanKind``) so the backend can
reconstruct the execution tree.

Thread-safety: each ``Execlave`` client gets a single ``SpanTree`` which
uses ``threading.local`` for the active-span stack. Cross-thread span
parenting is intentionally not supported — the framework callbacks we
target (LangChain, OpenAI Agents, CrewAI) all invoke hooks on the same
thread as the originating call.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
import weakref
from typing import Any, Literal

from ..trace import Trace

logger = logging.getLogger("Execlave.instrumentation")

# spanKind values mirror OTel semantic conventions where possible.
SPAN_KIND_AGENT = "agent"
SPAN_KIND_CHAIN = "chain"
SPAN_KIND_LLM = "llm"
SPAN_KIND_TOOL = "tool"
SPAN_KIND_RETRIEVER = "retriever"
SPAN_KIND_GUARDRAIL = "guardrail"
SPAN_KIND_HANDOFF = "handoff"


class Span:
    """A single span in the execution tree.

    Lifecycle::

        tree.start(kind=..., name=...) -> Span
        span.set_input(...), .set_output(...), .set_model(...), etc.
        span.finish(status='success'|'error'|'timeout', error=...)

    ``Span`` delegates payload construction to ``Trace`` so we get the
    same privacy/PII scrub + injection scan + batching semantics.
    """

    __slots__ = (
        "span_id",
        "parent_span_id",
        "kind",
        "name",
        "_trace",
        "_tree_ref",
        "_finished",
        "_start_ns",
    )

    def __init__(
        self,
        trace: Trace,
        *,
        span_id: str,
        parent_span_id: str | None,
        kind: str,
        name: str,
        tree: "SpanTree",
    ):
        self._trace = trace
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.kind = kind
        self.name = name
        # Avoid circular refcount between tree and span stack.
        self._tree_ref = weakref.ref(tree)
        self._finished = False
        self._start_ns = time.monotonic_ns()

        # Seed the metadata with span identity so _buffer_trace forwards it.
        trace.add_metadata(
            {
                "spanId": span_id,
                "parentSpanId": parent_span_id,
                "spanKind": kind,
                "spanName": name,
            }
        )

    # ------------------------------------------------------------------
    # Pass-through setters (kept minimal to avoid duplicating Trace API)
    # ------------------------------------------------------------------
    def set_input(self, value: Any) -> "Span":
        self._trace.set_input(value)
        return self

    def set_output(self, value: Any) -> "Span":
        self._trace.set_output(value)
        return self

    def set_model(self, model: str) -> "Span":
        self._trace.set_model(model)
        return self

    def set_tokens(self, prompt: int, completion: int) -> "Span":
        self._trace.set_tokens(prompt, completion)
        return self

    def set_cost(self, cost_usd: float) -> "Span":
        self._trace.set_cost(cost_usd)
        return self

    def add_metadata(self, metadata: dict[str, Any]) -> "Span":
        self._trace.add_metadata(metadata)
        return self

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def child(self, *, kind: str, name: str) -> "Span":
        """Open a nested child span. The current span remains active
        until ``finish`` is called on this span."""
        tree = self._tree_ref()
        if tree is None:  # pragma: no cover — defensive
            raise RuntimeError("SpanTree has been garbage-collected")
        return tree.start(kind=kind, name=name, parent=self)

    def finish(
        self,
        status: str = "success",
        error_message: str | None = None,
        error_type: str | None = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        tree = self._tree_ref()
        if tree is not None:
            tree._pop(self)
        try:
            self._trace.finish(
                status=status,
                error_message=error_message,
                error_type=error_type,
            )
        except Exception:  # pragma: no cover — transport handles its own errors
            logger.warning("Span.finish — Trace.finish raised", exc_info=True)

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        if exc_type is not None:
            self.finish(
                status="error",
                error_message=str(exc_val),
                error_type=exc_type.__name__ if exc_type else None,
            )
        else:
            self.finish(status="success")
        return False  # never suppress


class SpanTree:
    """Per-client stack of active spans.

    Held on ``Execlave`` via ``_span_tree`` (lazy-initialised). Each
    thread has an independent stack, so concurrent agent runs do not
    collide.
    """

    def __init__(self, exe: Any):
        self._exe = exe
        self._local = threading.local()

    def _stack(self) -> list[Span]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def current(self) -> Span | None:
        stack = self._stack()
        return stack[-1] if stack else None

    def start(
        self,
        *,
        kind: str,
        name: str,
        parent: Span | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new span. The span becomes the current span for this
        thread until ``finish`` is called."""
        stack = self._stack()
        effective_parent = parent or (stack[-1] if stack else None)
        parent_id = effective_parent.span_id if effective_parent else None

        # Spans share the trace-id of their root so the backend can
        # group them. Root spans get a fresh trace id.
        if effective_parent is not None:
            trace_id = effective_parent._trace.trace_id
            resolved_agent = agent_id or effective_parent._trace.agent_id
            resolved_session = session_id or effective_parent._trace.session_id
            resolved_user = user_id or effective_parent._trace.user_id
        else:
            trace_id = None
            resolved_agent = agent_id
            resolved_session = session_id
            resolved_user = user_id

        trace = self._exe.start_trace(
            trace_id=trace_id,
            agent_id=resolved_agent,
            session_id=resolved_session,
            user_id=resolved_user,
        )
        if metadata:
            trace.add_metadata(metadata)

        span_id = f"sp_{uuid.uuid4().hex[:16]}"
        span = Span(
            trace,
            span_id=span_id,
            parent_span_id=parent_id,
            kind=kind,
            name=name,
            tree=self,
        )
        stack.append(span)
        return span

    def _pop(self, span: Span) -> None:
        """Remove ``span`` from the stack. Tolerates out-of-order finishes
        (can happen if a framework fires its end-callback on a different
        thread or out of sequence) by searching the stack."""
        stack = self._stack()
        try:
            stack.remove(span)
        except ValueError:
            # Already popped or never pushed — harmless.
            pass


def get_span_tree(exe: Any) -> SpanTree:
    """Return (creating if needed) the ``SpanTree`` attached to an
    ``Execlave`` client instance. Using a shared tree per client keeps
    the API surface on ``Execlave`` unchanged."""
    tree: SpanTree | None = getattr(exe, "_span_tree", None)
    if tree is None:
        tree = SpanTree(exe)
        # Attach directly; Execlave allows arbitrary attribute assignment.
        setattr(exe, "_span_tree", tree)
    return tree
