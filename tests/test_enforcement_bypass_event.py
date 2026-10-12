"""Structured reporting of ungoverned execution (F-SDK-12).

Under ``fail_open`` (the default) an enforcement outage is invisible: the call
returns ``allowed: True`` and the only trace is a log line. That window — the
agent running ungoverned — is what an auditor needs, so it is reported as a
structured event. Mirrors the JS SDK so the two do not drift.
"""

from unittest.mock import patch

import pytest
import requests

from execlave.client import Execlave


def make_client(events):
    return Execlave(
        api_key="exe_prod_test",
        base_url="https://api.test",
        enable_control_channel=False,
        async_mode=False,
        on_enforcement_bypassed=events.append,
    )


class _Resp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.ok = status_code < 400

    def json(self):
        return self._payload


def test_network_outage_is_reported():
    events = []
    exe = make_client(events)
    with patch.object(
        exe._session, "request", side_effect=requests.ConnectionError("connection reset")
    ):
        result = exe.enforce_policy(agent_id="agent-1", input="hello")

    assert result["allowed"] is True
    assert len(events) == 1
    assert events[0]["reason"] == "network_error"
    assert events[0]["agentId"] == "agent-1"
    assert "connection reset" in events[0]["message"]
    # Timestamped at the moment of the bypass so the ungoverned window can be
    # reconstructed afterwards.
    assert events[0]["timestamp"].endswith("+00:00")


def test_server_error_is_reported_with_status():
    events = []
    exe = make_client(events)
    with patch.object(exe._session, "request", return_value=_Resp(503)):
        result = exe.enforce_policy(agent_id="agent-2", input="hi")

    assert result["allowed"] is True
    assert events[0]["reason"] == "server_error"
    assert events[0]["status"] == 503


def test_every_bypass_is_reported_while_breaker_is_open():
    events = []
    exe = make_client(events)
    with patch.object(exe._session, "request", side_effect=requests.ConnectionError("down")):
        for i in range(5):
            exe.enforce_policy(agent_id="agent-3", input=f"call {i}")

    # Going quiet once the breaker opens would hide the longest ungoverned
    # stretch, which is the part that matters most.
    assert len(events) == 5
    assert any(e["reason"] == "circuit_breaker_open" for e in events)


def test_no_event_on_a_governed_allow():
    events = []
    exe = make_client(events)
    with patch.object(exe._session, "request", return_value=_Resp(200, {"allowed": True})):
        exe.enforce_policy(agent_id="agent-4", input="hi")

    assert events == []


def test_listener_exception_does_not_break_enforcement():
    def boom(_event):
        raise RuntimeError("listener exploded")

    exe = Execlave(
        api_key="exe_prod_test",
        base_url="https://api.test",
        enable_control_channel=False,
        async_mode=False,
        on_enforcement_bypassed=boom,
    )
    with patch.object(exe._session, "request", side_effect=requests.ConnectionError("down")):
        # A faulty observer must never break the path it observes.
        result = exe.enforce_policy(agent_id="agent-5", input="hi")

    assert result["allowed"] is True


def test_works_without_a_listener():
    exe = Execlave(
        api_key="exe_prod_test",
        base_url="https://api.test",
        enable_control_channel=False,
        async_mode=False,
    )
    with patch.object(exe._session, "request", side_effect=requests.ConnectionError("down")):
        assert exe.enforce_policy(agent_id="agent-6", input="hi")["allowed"] is True


def _quota_body(message="Plan limit reached"):
    return {
        "error": {
            "resource": "maxTracesPerMonth",
            "current": 10000,
            "max": 10000,
            "message": message,
        }
    }


def test_plan_limit_fail_open_is_reported():
    """Quota exhaustion is the bypass most likely to fire in normal operation.

    "Continuing unmonitored" is a governance gap even though the cause is
    commercial rather than an outage. It used to return allowed=True with no
    event, so a customer who wired the callback got silence on the most common
    case — false assurance, worse than no callback at all. Parity with the JS SDK.
    """
    events = []
    exe = make_client(events)
    exe.plan_limit_behavior = "fail_open"
    with patch.object(exe._session, "request", return_value=_Resp(402, _quota_body())):
        result = exe.enforce_policy(agent_id="agent-7", input="hi")

    assert result["allowed"] is True
    assert len(events) == 1
    assert events[0]["reason"] == "plan_limit_exceeded"
    assert events[0]["source"] == "fail_open_plan_limit"
    assert events[0]["agentId"] == "agent-7"
    assert events[0]["status"] == 402
    assert "Plan limit reached" in events[0]["message"]
    assert events[0]["timestamp"].endswith("+00:00")


def test_plan_limit_result_carries_the_same_source_as_the_event():
    """A caller checking result["source"] must be able to tell this allow apart
    from a governed one, and correlate it with the event."""
    events = []
    exe = make_client(events)
    exe.plan_limit_behavior = "fail_open"
    with patch.object(exe._session, "request", return_value=_Resp(402, _quota_body())):
        result = exe.enforce_policy(agent_id="agent-7", input="hi")

    assert result["source"] == events[0]["source"] == "fail_open_plan_limit"


def test_every_plan_limit_bypass_is_reported():
    """Going quiet after the first 402 would hide the whole exhausted stretch."""
    events = []
    exe = make_client(events)
    exe.plan_limit_behavior = "fail_open"
    with patch.object(exe._session, "request", return_value=_Resp(402, _quota_body())):
        for i in range(3):
            exe.enforce_policy(agent_id="agent-7", input=f"call {i}")

    assert [e["reason"] for e in events] == ["plan_limit_exceeded"] * 3


def test_plan_limit_fail_closed_raises_and_reports_nothing():
    """fail_closed blocks the action, so nothing was bypassed."""
    from execlave.errors import PlanLimitExceededError

    events = []
    exe = make_client(events)
    exe.plan_limit_behavior = "fail_closed"
    with patch.object(exe._session, "request", return_value=_Resp(402, _quota_body())):
        with pytest.raises(PlanLimitExceededError):
            exe.enforce_policy(agent_id="agent-7", input="hi")

    assert events == []


def test_circuit_breaker_source_matches_the_js_sdk():
    """The JS SDK's EnforcementBypassEvent doc promises `source` matches
    EnforceResult.source for correlation, and both SDKs are supposed to speak
    the same vocabulary. Python said `circuit_breaker_fail_open`; JS says
    `fail_open_circuit_breaker`, like every other fail_open_* source."""
    events = []
    exe = make_client(events)
    with patch.object(exe._session, "request", side_effect=requests.ConnectionError("down")):
        results = [exe.enforce_policy(agent_id="agent-8", input=f"c{i}") for i in range(5)]

    breaker_events = [e for e in events if e["reason"] == "circuit_breaker_open"]
    assert breaker_events, "breaker never opened"
    assert breaker_events[0]["source"] == "fail_open_circuit_breaker"
    # The returned result and the event agree, so they can be correlated.
    assert results[-1]["source"] == breaker_events[-1]["source"]
