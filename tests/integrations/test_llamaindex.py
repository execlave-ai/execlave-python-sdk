"""Tests for the LlamaIndex integration.

The integration imports ``BaseEventHandler`` from
``llama_index.core.instrumentation.event_handlers`` — we fake that module
at collection time so the test suite does not require llama-index.
"""

from __future__ import annotations

import sys
import types

import pytest

from execlave.errors import PolicyBlockedError


def _install_fake_llamaindex() -> None:
    pkg_core = types.ModuleType("llama_index.core")
    pkg_instr = types.ModuleType("llama_index.core.instrumentation")
    pkg_handlers = types.ModuleType(
        "llama_index.core.instrumentation.event_handlers"
    )

    class BaseEventHandler:
        @classmethod
        def class_name(cls) -> str:
            return "BaseEventHandler"

        def __init__(self) -> None:
            pass

        def handle(self, event):  # pragma: no cover — overridden
            return None

    pkg_handlers.BaseEventHandler = BaseEventHandler
    pkg = types.ModuleType("llama_index")
    pkg.core = pkg_core
    pkg_core.instrumentation = pkg_instr
    pkg_instr.event_handlers = pkg_handlers

    sys.modules.setdefault("llama_index", pkg)
    sys.modules.setdefault("llama_index.core", pkg_core)
    sys.modules.setdefault("llama_index.core.instrumentation", pkg_instr)
    sys.modules.setdefault(
        "llama_index.core.instrumentation.event_handlers", pkg_handlers
    )
    sys.modules.pop("execlave.integrations.llamaindex", None)


_install_fake_llamaindex()

from execlave.integrations.llamaindex import ExeclaveLlamaIndexHandler  # noqa: E402


def _event(cls_name: str, **fields):
    cls = type(cls_name, (), {})
    inst = cls()
    for k, v in fields.items():
        setattr(inst, k, v)
    return inst


class TestExeclaveLlamaIndexHandler:
    def test_requires_exe(self):
        with pytest.raises(ValueError):
            ExeclaveLlamaIndexHandler(None, agent_id="bot")  # type: ignore[arg-type]

    def test_requires_agent_id(self, ag_client):
        with pytest.raises(ValueError):
            ExeclaveLlamaIndexHandler(ag_client, agent_id="")

    def test_query_start_enforces(self, ag_client, monkeypatch):
        calls: list = []

        def fake(agent_id, input, **kw):
            calls.append((agent_id, input, kw))
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake)
        h = ExeclaveLlamaIndexHandler(ag_client, agent_id="bot")
        h.handle(_event("QueryStartEvent", span_id="s1", query="hello"))
        assert calls and calls[0][1] == "hello"

    def test_query_block_propagates(self, ag_client, monkeypatch):
        def raise_block(*a, **kw):
            raise PolicyBlockedError([{"policyType": "pii", "message": "no"}])

        monkeypatch.setattr(ag_client, "enforce_policy", raise_block)
        h = ExeclaveLlamaIndexHandler(ag_client, agent_id="bot")
        with pytest.raises(PolicyBlockedError):
            h.handle(_event("QueryStartEvent", span_id="s1", query="ssn=1"))

    def test_tool_call_enforces_with_allowlist(self, ag_client, monkeypatch):
        captured = {}

        def fake(agent_id, input, **kw):
            captured["tools"] = kw.get("tools")
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake)
        h = ExeclaveLlamaIndexHandler(ag_client, agent_id="bot")
        h.handle(
            _event(
                "AgentToolCallEvent",
                span_id="t1",
                tool_name="web_search",
                arguments={"q": "x"},
            )
        )
        assert captured["tools"] == ["web_search"]

    def test_query_end_closes_span(self, ag_client):
        h = ExeclaveLlamaIndexHandler(ag_client, agent_id="bot", enforce=False)
        h.handle(_event("QueryStartEvent", span_id="s1", query="hi"))
        assert "s1" in h._spans
        h.handle(_event("QueryEndEvent", span_id="s1", response="answer"))
        assert "s1" not in h._spans

    def test_unknown_event_ignored(self, ag_client):
        h = ExeclaveLlamaIndexHandler(ag_client, agent_id="bot", enforce=False)
        # should not raise
        h.handle(_event("SomethingElse", span_id="x"))

    def test_handle_swallows_non_enforcement_errors(self, ag_client, monkeypatch):
        h = ExeclaveLlamaIndexHandler(ag_client, agent_id="bot", enforce=False)

        def boom(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(h, "_handle", boom)
        # Must not raise
        h.handle(_event("QueryStartEvent", span_id="s1", query="x"))
