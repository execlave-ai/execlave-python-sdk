"""
Shared internal instrumentation layer.

Builds nested trace spans on top of the SDK's ``Trace`` primitive and
exposes a small set of event helpers (``record_llm_call``,
``record_tool_call``, ``record_agent_action``) that framework adapters
(LangChain, OpenAI Agents SDK, CrewAI) consume.

Design goals:
* Zero runtime deps beyond the base SDK.
* Thread-safe span stack per ``Execlave`` client so nested callbacks
  from a single agent run remain ordered even under concurrent tool calls.
* Policy enforcement delegated to ``Execlave.enforce_policy`` — this
  module does not re-implement decision logic.
* Failures in instrumentation must NEVER break the host application;
  every public helper swallows exceptions and logs at ``warning`` level.

Public API:
    Span             — lightweight handle with ``child()`` / ``finish()``
    SpanTree         — per-client context managing the current span stack
    record_llm_call  — wrap an LLM invocation with a span + enforcement
    record_tool_call — wrap a tool/function call with a span + enforcement
    record_agent_action — record an agent-level decision/step
"""

from .spans import Span, SpanTree, get_span_tree
from .events import (
    record_llm_call,
    record_tool_call,
    record_agent_action,
    record_retrieval,
)

__all__ = [
    "Span",
    "SpanTree",
    "get_span_tree",
    "record_llm_call",
    "record_tool_call",
    "record_agent_action",
    "record_retrieval",
]
