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
