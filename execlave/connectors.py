"""Connector helpers for OpenAI and LangChain integrations.

These helpers provide a thin convenience layer that applies Execlave
pre-execution enforcement and tracing around third-party model calls.
"""

from __future__ import annotations

import warnings
from typing import Any

from .client import Execlave


def run_openai_chat(
    exe: Execlave,
    openai_client: Any,
    *,
    agent_id: str,
    input_text: str,
    model: str,
    messages: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    tools: list[str] | None = None,
    estimated_cost: float | None = None,
) -> Any:
    """Execute an OpenAI chat completion with enforcement + tracing."""
    exe.enforce_policy(
        agent_id,
        input_text,
        metadata=metadata,
        estimated_cost=estimated_cost,
        tools=tools,
    )

    trace = exe.start_trace(agent_id=agent_id, metadata={"connector": "openai", **(metadata or {})})
    trace.set_input(messages)
    trace.set_model(model)

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
        )

        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                trace.set_tokens(prompt_tokens, completion_tokens)

        trace.set_output(response)
        trace.finish("success")
        return response
    except Exception as exc:  # pragma: no cover - simple passthrough
        trace.finish("error", str(exc), exc.__class__.__name__)
        raise


def run_langchain(
    exe: Execlave,
    runnable: Any,
    *,
    agent_id: str,
    input_text: str,
    metadata: dict[str, Any] | None = None,
    tools: list[str] | None = None,
    estimated_cost: float | None = None,
) -> Any:
    """Execute a LangChain runnable with enforcement + tracing.

    .. deprecated:: 1.1.0
        Use :class:`execlave.integrations.langchain.ExeclaveCallbackHandler`
        instead. The callback handler supports the full LangChain callback
        surface (chains, tools, retrievers, agent actions) and nested
        spans. This thin wrapper remains for backward compatibility but
        will be removed in 2.0.0.
    """
    warnings.warn(
        "run_langchain() is deprecated and will be removed in execlave-sdk 2.0.0. "
        "Use execlave.integrations.langchain.ExeclaveCallbackHandler with "
        "chain.invoke(..., config={'callbacks': [handler]}) for full-fidelity "
        "LangChain instrumentation.",
        DeprecationWarning,
        stacklevel=2,
    )
    exe.enforce_policy(
        agent_id,
        input_text,
        metadata=metadata,
        estimated_cost=estimated_cost,
        tools=tools,
    )

    trace = exe.start_trace(agent_id=agent_id, metadata={"connector": "langchain", **(metadata or {})})
    trace.set_input(input_text)

    try:
        response = runnable.invoke(input_text)
        trace.set_output(response)
        trace.finish("success")
        return response
    except Exception as exc:  # pragma: no cover - simple passthrough
        trace.finish("error", str(exc), exc.__class__.__name__)
        raise
