"""Tests for the MCP client integration.

No real ``mcp`` package needed — we duck-type a ClientSession.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from execlave.errors import PolicyBlockedError
from execlave.integrations.mcp import instrument_mcp_session


class _FakeContent:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeCallToolResult:
    def __init__(self, text: str, is_error: bool = False) -> None:
        self.content = [_FakeContent(text)]
        self.isError = is_error


class _FakeSession:
    def __init__(self, result: _FakeCallToolResult | Exception) -> None:
        self._result = result
        self.calls: list = []

    async def call_tool(self, name: str, arguments: dict | None = None):
        self.calls.append((name, arguments))
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture()
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestInstrumentMcpSession:
    def test_requires_session(self, ag_client):
        with pytest.raises(ValueError):
            instrument_mcp_session(None, ag_client, agent_id="bot")

    def test_requires_agent_id(self, ag_client):
        with pytest.raises(ValueError):
            instrument_mcp_session(_FakeSession(_FakeCallToolResult("x")), ag_client, agent_id="")

    def test_call_tool_enforces(self, ag_client, monkeypatch, event_loop):
        sess = _FakeSession(_FakeCallToolResult("ok"))
        captured = {}

        def fake(agent_id, input, **kw):
            captured["tools"] = kw.get("tools")
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake)
        instrument_mcp_session(sess, ag_client, agent_id="bot")
        event_loop.run_until_complete(sess.call_tool("search", {"q": "x"}))
        assert captured["tools"] == ["search"]
        assert sess.calls == [("search", {"q": "x"})]

    def test_block_halts_call(self, ag_client, monkeypatch, event_loop):
        sess = _FakeSession(_FakeCallToolResult("nope"))

        def raise_block(*a, **kw):
            raise PolicyBlockedError([{"policyType": "tool", "message": "no"}])

        monkeypatch.setattr(ag_client, "enforce_policy", raise_block)
        instrument_mcp_session(sess, ag_client, agent_id="bot")
        with pytest.raises(PolicyBlockedError):
            event_loop.run_until_complete(sess.call_tool("rm_rf", {"path": "/"}))
        assert sess.calls == []  # underlying never invoked

    def test_idempotent(self, ag_client):
        sess = _FakeSession(_FakeCallToolResult("x"))
        instrument_mcp_session(sess, ag_client, agent_id="bot", enforce=False)
        first = sess.call_tool
        instrument_mcp_session(sess, ag_client, agent_id="bot", enforce=False)
        assert sess.call_tool is first

    def test_session_missing_call_tool_rejected(self, ag_client):
        class _NoCallTool:
            pass

        with pytest.raises(TypeError):
            instrument_mcp_session(_NoCallTool(), ag_client, agent_id="bot")

    def test_error_result_marks_span_error(self, ag_client, event_loop):
        sess = _FakeSession(_FakeCallToolResult("err", is_error=True))
        instrument_mcp_session(sess, ag_client, agent_id="bot", enforce=False)
        result = event_loop.run_until_complete(sess.call_tool("t", {}))
        assert result.isError is True

    def test_underlying_exception_propagates(self, ag_client, event_loop):
        sess = _FakeSession(RuntimeError("transport gone"))
        instrument_mcp_session(sess, ag_client, agent_id="bot", enforce=False)
        with pytest.raises(RuntimeError):
            event_loop.run_until_complete(sess.call_tool("t", {}))
