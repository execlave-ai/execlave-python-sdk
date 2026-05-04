"""Tests for the OpenAI Agents SDK integration.

We fake the `agents.tracing.processor_interface.TracingProcessor` base
class and synthesise span objects shaped like the SDK's span-data so we
can run the processor without installing openai-agents.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from execlave.errors import PolicyBlockedError


# ---------------------------------------------------------------------------
# Fake openai-agents module structure BEFORE import. Runs at module load
# (collection time) — fixtures run too late.
# ---------------------------------------------------------------------------
def _install_fake_agents() -> None:
    fake_iface = types.ModuleType("agents.tracing.processor_interface")

    class TracingProcessor:
        def on_trace_start(self, trace): ...
        def on_trace_end(self, trace): ...
        def on_span_start(self, span): ...
        def on_span_end(self, span): ...
        def shutdown(self): ...
        def force_flush(self): ...

    fake_iface.TracingProcessor = TracingProcessor
    fake_tracing = types.ModuleType("agents.tracing")
    fake_tracing.TracingProcessor = TracingProcessor
    fake_tracing.processor_interface = fake_iface
    fake_pkg = types.ModuleType("agents")
    fake_pkg.tracing = fake_tracing

    sys.modules.setdefault("agents", fake_pkg)
    sys.modules.setdefault("agents.tracing", fake_tracing)
    sys.modules.setdefault("agents.tracing.processor_interface", fake_iface)
    sys.modules.pop("execlave.integrations.openai_agents", None)


_install_fake_agents()

from execlave.integrations.openai_agents import ExeclaveTracingProcessor  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — construct fake Agents-SDK spans.
# ---------------------------------------------------------------------------
class _AgentSpanData:
    """Matches what `type(span_data).__name__` would produce for
    ``AgentSpanData`` from the SDK."""

    def __init__(self, name="assistant"):
        self.name = name


_AgentSpanData.__name__ = "AgentSpanData"


class _FunctionSpanData:
    def __init__(self, name="search", input="q=x", output="results"):
        self.name = name
        self.input = input
        self.output = output


_FunctionSpanData.__name__ = "FunctionSpanData"


class _GenerationSpanData:
    def __init__(self, model="gpt-4o", output="hi", usage=None):
        self.model = model
        self.output = output
        self.usage = usage or {"prompt_tokens": 10, "completion_tokens": 20}


_GenerationSpanData.__name__ = "GenerationSpanData"


def _make_span(span_id, span_data, parent_id=None, error=None):
    s = MagicMock()
    s.span_id = span_id
    s.parent_id = parent_id
    s.span_data = span_data
    s.error = error
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestProcessor:
    def test_agent_span_opens_and_closes(self, ag_client):
        p = ExeclaveTracingProcessor(ag_client, agent_id="bot", enforce=False)
        agent_span = _make_span("sp1", _AgentSpanData(name="assistant"))
        p.on_span_start(agent_span)
        assert "sp1" in p._spans
        p.on_span_end(agent_span)
        assert "sp1" not in p._spans

    def test_generation_span_captures_model_and_tokens(self, ag_client):
        p = ExeclaveTracingProcessor(ag_client, agent_id="bot", enforce=False)
        s = _make_span("sp-g", _GenerationSpanData())
        p.on_span_start(s)
        our_span = p._spans["sp-g"]
        p.on_span_end(s)
        assert our_span._trace._model_name == "gpt-4o"
        assert our_span._trace._input_tokens == 10
        assert our_span._trace._output_tokens == 20

    def test_function_span_enforces_tool_allowlist(self, ag_client, monkeypatch):
        captured: dict = {}
        def fake_enforce(agent_id, input, **kw):
            captured["tools"] = kw.get("tools")
            return {"allowed": True}
        monkeypatch.setattr(ag_client, "enforce_policy", fake_enforce)

        p = ExeclaveTracingProcessor(ag_client, agent_id="bot", enforce=True)
        s = _make_span("sp-f", _FunctionSpanData(name="web_search"))
        p.on_span_start(s)
        assert captured["tools"] == ["web_search"]
        p.on_span_end(s)

    def test_function_span_block_halts_run(self, ag_client, monkeypatch):
        def raise_block(*a, **kw):
            raise PolicyBlockedError([{"policyType": "tool", "message": "no"}])
        monkeypatch.setattr(ag_client, "enforce_policy", raise_block)

        p = ExeclaveTracingProcessor(ag_client, agent_id="bot", enforce=True)
        s = _make_span("sp-f", _FunctionSpanData(name="rm_rf"))
        with pytest.raises(PolicyBlockedError):
            p.on_span_start(s)

    def test_nested_spans_track_parent(self, ag_client):
        p = ExeclaveTracingProcessor(ag_client, agent_id="bot", enforce=False)
        parent = _make_span("p1", _AgentSpanData())
        child = _make_span("c1", _GenerationSpanData(), parent_id="p1")
        p.on_span_start(parent)
        p.on_span_start(child)
        assert p._spans["c1"].parent_span_id == p._spans["p1"].span_id
        # Same underlying trace id so backend can group.
        assert p._spans["c1"]._trace.trace_id == p._spans["p1"]._trace.trace_id
        p.on_span_end(child)
        p.on_span_end(parent)

    def test_error_recorded(self, ag_client):
        p = ExeclaveTracingProcessor(ag_client, agent_id="bot", enforce=False)
        err = RuntimeError("upstream failure")
        s = _make_span("sp-e", _AgentSpanData(), error=err)
        p.on_span_start(s)
        our = p._spans["sp-e"]
        p.on_span_end(s)
        assert our._finished
        assert our._trace._status == "error"

    def test_missing_span_id_is_ignored(self, ag_client):
        p = ExeclaveTracingProcessor(ag_client, agent_id="bot", enforce=False)
        s = MagicMock(span_id=None, id=None, parent_id=None, span_data=None, error=None)
        p.on_span_start(s)  # should not raise
        p.on_span_end(s)
