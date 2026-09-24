"""Tests for the MCP client integration.

No real ``mcp`` package needed — we duck-type a ClientSession.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from execlave.errors import (
    PolicyBlockedError,
    CertificateMismatchError,
    ApprovalVerificationError,
)
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

    def test_seals_full_untruncated_tool_arguments_into_metadata(
        self, ag_client, monkeypatch, event_loop
    ):
        # `input` (_safe_str) truncates for policy/classifier read. The
        # certificate's digest must cover the WHOLE sealed action context —
        # metadata is never truncated — so a large argument blob's tail is
        # still bound: a change past input's truncation point must
        # invalidate the certificate, not silently escape it.
        sess = _FakeSession(_FakeCallToolResult("ok"))
        captured = {}

        def fake(agent_id, input, **kw):
            captured["metadata"] = kw.get("metadata")
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake)
        instrument_mcp_session(sess, ag_client, agent_id="bot")
        big_args = {"q": "x" * 5000}
        event_loop.run_until_complete(sess.call_tool("search", big_args))
        assert captured["metadata"] == {"toolArguments": big_args}

    def test_sanitizes_non_json_serializable_tool_arguments(
        self, ag_client, monkeypatch, event_loop
    ):
        # Regression: sealing raw arguments verbatim risked a circular
        # reference reaching json.dumps at the HTTP layer. json.dumps raises
        # TypeError for non-JSON-native types, which `requests` does NOT wrap
        # into a RequestException the way it does a circular-reference
        # ValueError -- an uncaught TypeError propagates past enforce_policy
        # entirely and is swallowed by the adapter's generic `except
        # Exception` as "non-fatal", silently allowing the call every time
        # for that payload shape. The adapter must sanitize before calling
        # enforce_policy so this never reaches the network layer at all.
        sess = _FakeSession(_FakeCallToolResult("ok"))
        captured = {}

        def fake(agent_id, input, **kw):
            captured["metadata"] = kw.get("metadata")
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake)
        instrument_mcp_session(sess, ag_client, agent_id="bot")

        circular: dict = {"cmd": "wire_funds"}
        circular["self"] = circular
        event_loop.run_until_complete(sess.call_tool("wire_funds", circular))

        assert captured["metadata"] is not None
        # Whatever was sealed must itself be JSON-serializable -- the real
        # risk surface -- otherwise this raises.
        json.dumps(captured["metadata"])

    def test_block_halts_call(self, ag_client, monkeypatch, event_loop):
        sess = _FakeSession(_FakeCallToolResult("nope"))

        def raise_block(*a, **kw):
            raise PolicyBlockedError([{"policyType": "tool", "message": "no"}])

        monkeypatch.setattr(ag_client, "enforce_policy", raise_block)
        instrument_mcp_session(sess, ag_client, agent_id="bot")
        with pytest.raises(PolicyBlockedError):
            event_loop.run_until_complete(sess.call_tool("rm_rf", {"path": "/"}))
        assert sess.calls == []  # underlying never invoked

    def test_certificate_mismatch_halts_call(self, ag_client, monkeypatch, event_loop):
        # Regression: CertificateMismatchError (post-approval action drift) was
        # missing from this adapter's enforcement-error allowlist, so a drift
        # rejection was swallowed as "non-fatal" and the tool call went through
        # anyway — fail-open on the exact case the certificate binding exists
        # to catch.
        sess = _FakeSession(_FakeCallToolResult("nope"))

        def raise_mismatch(*a, **kw):
            raise CertificateMismatchError("apr_1", "action_context_mismatch")

        monkeypatch.setattr(ag_client, "enforce_policy", raise_mismatch)
        instrument_mcp_session(sess, ag_client, agent_id="bot")
        with pytest.raises(CertificateMismatchError):
            event_loop.run_until_complete(sess.call_tool("wire_funds", {"amount": 999999}))
        assert sess.calls == []

    def test_unverifiable_approval_halts_call(self, ag_client, monkeypatch, event_loop):
        sess = _FakeSession(_FakeCallToolResult("nope"))

        def raise_unverifiable(*a, **kw):
            raise ApprovalVerificationError("apr_1", "ECONNRESET")

        monkeypatch.setattr(ag_client, "enforce_policy", raise_unverifiable)
        instrument_mcp_session(sess, ag_client, agent_id="bot")
        with pytest.raises(ApprovalVerificationError):
            event_loop.run_until_complete(sess.call_tool("wire_funds", {}))
        assert sess.calls == []

    def test_transient_enforcement_failure_still_fails_open(self, ag_client, monkeypatch, event_loop):
        # Not a governance decision — a transport blip talking to Execlave.
        # The adapter should log and continue, otherwise every Execlave
        # hiccup becomes an agent outage.
        sess = _FakeSession(_FakeCallToolResult("ok"))

        def raise_transient(*a, **kw):
            raise ConnectionError("socket hang up")

        monkeypatch.setattr(ag_client, "enforce_policy", raise_transient)
        instrument_mcp_session(sess, ag_client, agent_id="bot")
        result = event_loop.run_until_complete(sess.call_tool("search", {"q": "x"}))
        assert result.content[0].text == "ok"

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
