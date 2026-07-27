"""Tests for the AutoGen integration. Duck-types ConversableAgent."""

from __future__ import annotations

import json

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

    def test_seals_the_full_conversation_into_metadata(self, ag_client, monkeypatch):
        # _last_user_text only looks at the LAST user/human message -- the
        # certificate's digest must cover the WHOLE sealed action context,
        # so the full message history goes in metadata too.
        agent = _FakeAgent("ok")
        captured = {}

        def fake(agent_id, input, **kw):
            captured["metadata"] = kw.get("metadata")
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake)
        instrument_autogen_agent(agent, ag_client, agent_id="bot")
        messages = [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hello"},
        ]
        agent.generate_reply(messages=messages)
        assert captured["metadata"] == {"messages": messages}

    def test_sanitizes_non_json_serializable_messages(self, ag_client, monkeypatch):
        # Regression: sealing raw messages verbatim risked a circular
        # reference reaching json.dumps at the HTTP layer, and json.dumps
        # raises TypeError for non-JSON-native types -- which `requests`
        # does NOT wrap into a RequestException the way it does a
        # circular-reference ValueError. An uncaught TypeError propagates
        # past enforce_policy entirely and is swallowed by the adapter's
        # generic except Exception as "non-fatal", silently allowing the
        # call.
        agent = _FakeAgent("ok")
        captured = {}

        def fake(agent_id, input, **kw):
            captured["metadata"] = kw.get("metadata")
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake)
        instrument_autogen_agent(agent, ag_client, agent_id="bot")
        circular_msg: dict = {"role": "user", "content": "hi"}
        circular_msg["self"] = circular_msg
        agent.generate_reply(messages=[circular_msg])

        assert captured["metadata"] is not None
        json.dumps(captured["metadata"])

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

    def test_seals_full_untruncated_tool_arguments_into_metadata(self, ag_client, monkeypatch):
        big_args = "x" * 5000
        reply = {
            "tool_calls": [
                {"function": {"name": "web_search", "arguments": big_args}},
            ],
            "content": None,
        }
        agent = _FakeAgent(reply)
        captured = {}

        def fake(agent_id, input, **kw):
            captured["metadata"] = kw.get("metadata")
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake)
        instrument_autogen_agent(agent, ag_client, agent_id="bot")
        agent.generate_reply(messages=[{"role": "user", "content": "search"}])
        assert captured["metadata"] == {"toolArguments": big_args}

    def test_sanitizes_non_json_serializable_tool_arguments(self, ag_client, monkeypatch):
        circular_args: dict = {"cmd": "wire_funds"}
        circular_args["self"] = circular_args
        reply = {
            "tool_calls": [{"function": {"name": "wire_funds", "arguments": circular_args}}],
            "content": None,
        }
        agent = _FakeAgent(reply)
        captured = {}

        def fake(agent_id, input, **kw):
            captured["metadata"] = kw.get("metadata")
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake)
        instrument_autogen_agent(agent, ag_client, agent_id="bot")
        agent.generate_reply(messages=[{"role": "user", "content": "go"}])

        assert captured["metadata"] is not None
        json.dumps(captured["metadata"])

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
