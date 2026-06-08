"""Tests for CrewAI integration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from execlave.errors import PolicyBlockedError
from execlave.integrations.crewai import instrument_crew


class _FakeAgent:
    def __init__(self):
        self.step_callback = None


class _FakeCrew:
    def __init__(self):
        self.step_callback = None
        self.task_callback = None
        self.agents: list = []


class TestInstrumentCrew:
    def test_sets_callbacks_when_unset(self, ag_client):
        crew = _FakeCrew()
        instrument_crew(crew, ag_client, agent_id="bot", enforce=False)
        assert callable(crew.step_callback)
        assert callable(crew.task_callback)
        assert getattr(crew, "_execlave_instrumented") is True

    def test_idempotent(self, ag_client):
        crew = _FakeCrew()
        instrument_crew(crew, ag_client, agent_id="bot")
        first = crew.step_callback
        instrument_crew(crew, ag_client, agent_id="bot")
        # Second call is a no-op — callback should be unchanged.
        assert crew.step_callback is first

    def test_preserves_user_step_callback(self, ag_client):
        crew = _FakeCrew()
        user_calls: list = []
        crew.step_callback = lambda step: user_calls.append(step)
        instrument_crew(crew, ag_client, agent_id="bot", enforce=False)

        step = MagicMock(tool=None, tool_input=None, output="x")
        crew.step_callback(step)
        assert user_calls == [step]

    def test_tool_step_enforces(self, ag_client, monkeypatch):
        crew = _FakeCrew()
        captured: dict = {}
        def fake_enforce(agent_id, input, **kw):
            captured["tools"] = kw.get("tools")
            return {"allowed": True}
        monkeypatch.setattr(ag_client, "enforce_policy", fake_enforce)

        instrument_crew(crew, ag_client, agent_id="bot", enforce=True)
        step = MagicMock(tool="search", tool_input="q", output="r")
        crew.step_callback(step)
        assert captured["tools"] == ["search"]

    def test_tool_step_block_halts(self, ag_client, monkeypatch):
        crew = _FakeCrew()
        def raise_block(*a, **kw):
            raise PolicyBlockedError([{"policyType": "tool", "message": "no"}])
        monkeypatch.setattr(ag_client, "enforce_policy", raise_block)

        instrument_crew(crew, ag_client, agent_id="bot", enforce=True)
        step = MagicMock(tool="rm_rf", tool_input="/", output=None)
        with pytest.raises(PolicyBlockedError):
            crew.step_callback(step)

    def test_task_callback_records_span(self, ag_client):
        crew = _FakeCrew()
        instrument_crew(crew, ag_client, agent_id="bot", enforce=False)
        task_out = MagicMock(description="Analyse sales", raw="result")
        # Should not raise.
        crew.task_callback(task_out)

    def test_inner_agents_instrumented_too(self, ag_client):
        crew = _FakeCrew()
        inner = _FakeAgent()
        crew.agents = [inner]
        instrument_crew(crew, ag_client, agent_id="bot", enforce=False)
        assert callable(inner.step_callback)

    def test_missing_crew_raises(self, ag_client):
        with pytest.raises(ValueError):
            instrument_crew(None, ag_client, agent_id="bot")

    def test_missing_agent_id_raises(self, ag_client):
        with pytest.raises(ValueError):
            instrument_crew(_FakeCrew(), ag_client, agent_id="")
