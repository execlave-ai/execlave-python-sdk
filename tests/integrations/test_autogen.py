"""Tests for the AutoGen integration. Duck-types ConversableAgent."""

from __future__ import annotations

import pytest

from execlave.errors import PolicyBlockedError
from execlave.integrations.autogen import instrument_autogen_agent


class _FakeAgent:
    name = "assistant"

    def __init__(self, reply):
        self._reply = reply
        self.calls: list = []

    def generate_reply(self, messages=None, **kw):
        self.calls.append(messages)
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply


class TestInstrumentAutogenAgent:
    def test_requires_agent(self, ag_client):
        with pytest.raises(ValueError):
            instrument_autogen_agent(None, ag_client, agent_id="bot")

    def test_requires_agent_id(self, ag_client):
        with pytest.raises(ValueError):
            instrument_autogen_agent(_FakeAgent("x"), ag_client, agent_id="")

    def test_rejects_agent_without_generate_reply(self, ag_client):
        class _NoReply:
            pass

        with pytest.raises(TypeError):
            instrument_autogen_agent(_NoReply(), ag_client, agent_id="bot")

    def test_enforces_on_last_user_message(self, ag_client, monkeypatch):
        agent = _FakeAgent("ok")
        captured = {}

        def fake(agent_id, input, **kw):
            captured["input"] = input
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake)
        instrument_autogen_agent(agent, ag_client, agent_id="bot")
        agent.generate_reply(
            messages=[
                {"role": "system", "content": "be nice"},
                {"role": "user", "content": "hello"},
            ]
        )
        assert captured["input"] == "hello"

    def test_block_halts_call(self, ag_client, monkeypatch):
        agent = _FakeAgent("ok")

        def raise_block(*a, **kw):
            raise PolicyBlockedError([{"policyType": "pii", "message": "no"}])

        monkeypatch.setattr(ag_client, "enforce_policy", raise_block)
        instrument_autogen_agent(agent, ag_client, agent_id="bot")
        with pytest.raises(PolicyBlockedError):
            agent.generate_reply(messages=[{"role": "user", "content": "ssn=1"}])
        assert agent.calls == []

    def test_tool_calls_enforce_with_allowlist(self, ag_client, monkeypatch):
        reply = {
            "tool_calls": [
                {"function": {"name": "web_search", "arguments": '{"q": "x"}'}},
            ],
            "content": None,
        }
        agent = _FakeAgent(reply)
        seen_tools: list = []

        def fake(agent_id, input, **kw):
            tools = kw.get("tools")
            if tools:
                seen_tools.extend(tools)
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake)
        instrument_autogen_agent(agent, ag_client, agent_id="bot")
        agent.generate_reply(messages=[{"role": "user", "content": "search"}])
        assert "web_search" in seen_tools

    def test_legacy_function_call_shape(self, ag_client, monkeypatch):
        reply = {"function_call": {"name": "lookup", "arguments": "{}"}}
        agent = _FakeAgent(reply)
        seen: list = []

        def fake(agent_id, input, **kw):
            if kw.get("tools"):
                seen.extend(kw["tools"])
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake)
        instrument_autogen_agent(agent, ag_client, agent_id="bot")
        agent.generate_reply(messages=[{"role": "user", "content": "find"}])
        assert "lookup" in seen

    def test_idempotent(self, ag_client):
        agent = _FakeAgent("ok")
        instrument_autogen_agent(agent, ag_client, agent_id="bot", enforce=False)
        first = agent.generate_reply
        instrument_autogen_agent(agent, ag_client, agent_id="bot", enforce=False)
        assert agent.generate_reply is first

    def test_underlying_exception_propagates(self, ag_client):
        agent = _FakeAgent(RuntimeError("boom"))
        instrument_autogen_agent(agent, ag_client, agent_id="bot", enforce=False)
        with pytest.raises(RuntimeError):
            agent.generate_reply(messages=[{"role": "user", "content": "x"}])
