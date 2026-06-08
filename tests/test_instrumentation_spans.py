"""Tests for execlave.instrumentation — nested spans + event helpers."""

from __future__ import annotations

import pytest

from execlave.errors import PolicyBlockedError
from execlave.instrumentation import (
    get_span_tree,
    record_agent_action,
    record_llm_call,
    record_retrieval,
    record_tool_call,
)
from execlave.instrumentation.spans import (
    SPAN_KIND_AGENT,
    SPAN_KIND_LLM,
    SPAN_KIND_RETRIEVER,
    SPAN_KIND_TOOL,
)


# ---------------------------------------------------------------------------
# SpanTree basics
# ---------------------------------------------------------------------------
class TestSpanTree:
    def test_start_root_span_has_no_parent(self, ag_client):
        tree = get_span_tree(ag_client)
        span = tree.start(kind=SPAN_KIND_LLM, name="gpt-4")
        assert span.parent_span_id is None
        assert span.span_id.startswith("sp_")
        assert span.kind == SPAN_KIND_LLM
        span.finish("success")

    def test_nested_spans_share_trace_id(self, ag_client):
        tree = get_span_tree(ag_client)
        root = tree.start(kind=SPAN_KIND_AGENT, name="root")
        child = root.child(kind=SPAN_KIND_LLM, name="gpt-4")
        # Same trace_id across root + child so backend groups them.
        assert child._trace.trace_id == root._trace.trace_id
        assert child.parent_span_id == root.span_id
        child.finish("success")
        root.finish("success")

    def test_current_span_reflects_stack(self, ag_client):
        tree = get_span_tree(ag_client)
        assert tree.current() is None
        a = tree.start(kind=SPAN_KIND_AGENT, name="a")
        assert tree.current() is a
        b = a.child(kind=SPAN_KIND_TOOL, name="tool")
        assert tree.current() is b
        b.finish("success")
        assert tree.current() is a
        a.finish("success")
        assert tree.current() is None

    def test_finish_is_idempotent(self, ag_client):
        tree = get_span_tree(ag_client)
        span = tree.start(kind=SPAN_KIND_LLM, name="gpt-4")
        span.finish("success")
        span.finish("success")  # no raise
        assert span._finished is True

    def test_metadata_is_recorded(self, ag_client):
        tree = get_span_tree(ag_client)
        span = tree.start(kind=SPAN_KIND_TOOL, name="search")
        # spans seed spanId/parentSpanId/spanKind into trace metadata
        assert span._trace._metadata["spanId"] == span.span_id
        assert span._trace._metadata["spanKind"] == SPAN_KIND_TOOL
        assert span._trace._metadata["spanName"] == "search"
        span.finish("success")

    def test_context_manager_marks_error_on_exception(self, ag_client):
        tree = get_span_tree(ag_client)
        with pytest.raises(ValueError):
            with tree.start(kind=SPAN_KIND_TOOL, name="x") as span:
                raise ValueError("boom")
        assert span._finished

    def test_tree_cached_per_client(self, ag_client):
        assert get_span_tree(ag_client) is get_span_tree(ag_client)


# ---------------------------------------------------------------------------
# record_llm_call / record_tool_call
# ---------------------------------------------------------------------------
class TestRecordLlmCall:
    def test_enforces_and_yields_span(self, ag_client, monkeypatch):
        calls: list[tuple] = []

        def fake_enforce(agent_id, input, **kwargs):
            calls.append((agent_id, input, kwargs))
            return {"allowed": True, "mode": "monitor", "source": "cache"}

        monkeypatch.setattr(ag_client, "enforce_policy", fake_enforce)

        with record_llm_call(
            ag_client, agent_id="bot", model="gpt-4o", input_text="hello"
        ) as span:
            assert span.kind == SPAN_KIND_LLM
            assert span.name == "gpt-4o"

        assert calls == [("bot", "hello", {"metadata": None, "estimated_cost": None, "tools": None})]

    def test_skips_enforcement_when_no_input(self, ag_client, monkeypatch):
        calls = []
        monkeypatch.setattr(
            ag_client, "enforce_policy", lambda *a, **k: calls.append(a) or {"allowed": True}
        )
        with record_llm_call(ag_client, agent_id="bot", model="gpt-4o"):
            pass
        assert calls == []

    def test_policy_blocked_propagates(self, ag_client, monkeypatch):
        def raise_block(*a, **kw):
            raise PolicyBlockedError([{"policyType": "pii", "message": "blocked"}])

        monkeypatch.setattr(ag_client, "enforce_policy", raise_block)
        with pytest.raises(PolicyBlockedError):
            with record_llm_call(
                ag_client, agent_id="bot", model="gpt-4o", input_text="hello"
            ):
                pytest.fail("body should not run")

    def test_exception_in_body_marks_span_error(self, ag_client, monkeypatch):
        monkeypatch.setattr(ag_client, "enforce_policy", lambda *a, **k: {"allowed": True})
        with pytest.raises(RuntimeError):
            with record_llm_call(
                ag_client, agent_id="bot", model="gpt-4o", input_text="hi"
            ) as span:
                raise RuntimeError("boom")
        assert span._finished

    def test_transient_enforcement_error_is_swallowed(self, ag_client, monkeypatch):
        def raise_generic(*a, **kw):
            raise RuntimeError("network hiccup")

        monkeypatch.setattr(ag_client, "enforce_policy", raise_generic)
        # Should NOT raise — the span still opens + finishes.
        with record_llm_call(
            ag_client, agent_id="bot", model="gpt-4o", input_text="hi"
        ) as span:
            assert span.kind == SPAN_KIND_LLM


class TestRecordToolCall:
    def test_passes_tool_name_to_enforce(self, ag_client, monkeypatch):
        captured: dict = {}

        def fake_enforce(agent_id, input, **kwargs):
            captured["agent_id"] = agent_id
            captured["tools"] = kwargs.get("tools")
            captured["metadata"] = kwargs.get("metadata")
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake_enforce)

        with record_tool_call(ag_client, agent_id="bot", tool_name="web_search", input_text="hi"):
            pass

        assert captured["tools"] == ["web_search"]
        assert captured["metadata"] == {"toolName": "web_search"}

    def test_span_kind_is_tool(self, ag_client, monkeypatch):
        monkeypatch.setattr(ag_client, "enforce_policy", lambda *a, **k: {"allowed": True})
        with record_tool_call(ag_client, agent_id="bot", tool_name="calc", input_text="1+1") as s:
            assert s.kind == SPAN_KIND_TOOL
            assert s.name == "calc"


# ---------------------------------------------------------------------------
# record_agent_action / record_retrieval
# ---------------------------------------------------------------------------
class TestRecordAgentAndRetrieval:
    def test_agent_action_returns_open_span(self, ag_client):
        span = record_agent_action(ag_client, agent_id="bot", action="plan")
        assert span.kind == SPAN_KIND_AGENT
        assert span._finished is False
        span.finish("success")

    def test_retrieval_is_context_manager(self, ag_client):
        with record_retrieval(ag_client, agent_id="bot", query="docs about X") as span:
            assert span.kind == SPAN_KIND_RETRIEVER
            span.add_metadata({"topK": 5})
        assert span._finished
